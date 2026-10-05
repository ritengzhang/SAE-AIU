#!/usr/bin/env python3
"""
Batch runner for ALL SAE Bench Pythia models with ALL trainers.

This script runs analysis3 (feature firing analysis) on:
- All 7 SAE types
- All 3 widths (4k, 16k, 65k)
- All 6 trainers (trainer_0 through trainer_5)
Total: 7 × 3 × 6 = 126 analyses

Usage:
    # Run with skip_existing enabled (recommended for re-runs):
    python -m feature_firing_analysis.runners.run_all_saebench_pythia --device cuda:1 --output-dir /data/user/sae/feature_firing_analysis
    
    # Note: --output-dir MUST be provided for skip_existing to work correctly.
    # If --output-dir is not provided, skip_existing will be disabled and all analyses will re-run.
"""

import sys
import os
from pathlib import Path
import time
from datetime import datetime
import json
import shutil

# Add parent directories to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

# Import analysis functions
from feature_firing_analysis import (
    compute_firing_statistics,
    create_firing_frequency_distribution_plots,
    create_cumulative_magnitude_distribution_plots,
    create_firing_magnitude_plots,
    create_per_input_magnitude_analysis_simplified,
    create_feature_ranking_overlap_plots,
    create_top_features_frequency_plots,
    create_top_features_aggregation_by_architecture,
    create_top_features_aggregation_by_width,
)

from sae_family_load_convert.saebench import (
    load_saebench_sae,
    cleanup_sae_cache,
)
from sae_family_load_convert import (
    setup_activations_store,
    load_base_model,
)

from encoder_decoder_activation_relationship.cleanup_utils import (
    cleanup_model_cache,
)


def run_analysis_saebench_pythia(
    repo_id: str,
    sae_type_folder: str,
    layer_folder: str = "resid_post_layer_8",
    trainer_folder: str = "trainer_0",
    width_name: str = None,
    device: str = None,
    n_batches: int = 30,
    thresholds: list = None,
    output_dir: Path = None
):
    """
    Run feature firing analysis for SAE Bench Pythia SAE.
    
    Args:
        repo_id: HuggingFace repo ID
        sae_type_folder: SAE type folder name
        layer_folder: Layer folder name
        trainer_folder: Trainer folder name
        width_name: Width name (4k, 16k, 65k)
        device: Device to use (default: auto-detect cuda:0, requires GPU)
        n_batches: Number of batches for activation collection
        thresholds: List of thresholds for counting fires (default: [0.0, 0.01])
        output_dir: Output directory (default: /data/user/sae/feature_firing_analysis/)
    """
    if thresholds is None:
        thresholds = [0.0, 0.01]
    
    # Auto-detect device if not specified - require GPU
    if device is None:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/GPU is not available. GPU is required for analysis.")
        device = "cuda:0"
        print(f"Using GPU: {device}")
    
    print("="*70)
    print("FEATURE FIRING ANALYSIS - SAE BENCH PYTHIA")
    print("="*70)
    
    # Setup output directory: saebench_pythia/{width}/{sae_type}/{trainer}/
    if output_dir is None:
        output_dir = Path(os.environ.get("DATA_DIR", "/data/user")) / "sae" / "feature_firing_analysis"
    
    # Infer width from repo_id if not provided
    if width_name is None:
        if '2pow12' in repo_id or 'width-2pow12' in repo_id:
            width_name = '4k'
        elif '2pow14' in repo_id or 'width-2pow14' in repo_id:
            width_name = '16k'
        elif '2pow16' in repo_id or 'width-2pow16' in repo_id:
            width_name = '65k'
        else:
            width_name = 'unknown'
    
    # Create base output directory (we'll add trainer folder name after loading config)
    base_output_dir = output_dir / "saebench_pythia" / width_name / sae_type_folder
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load SAE
    print("\n[1/4] Loading SAE...")
    step_start = time.time()
    print(f"[TIMING] Started loading SAE at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sae, metadata = load_saebench_sae(
        repo_id=repo_id,
        sae_type_folder=sae_type_folder,
        layer_folder=layer_folder,
        trainer_folder=trainer_folder,
        device=device
    )
    
    # Get config.json path from cache (before cleanup)
    cache_path = Path(metadata['_cache_path'])
    config_json_path = cache_path / "config.json"
    
    # Load config to get key hyperparameter and create descriptive folder name
    with open(config_json_path, 'r') as f:
        sae_config = json.load(f)
    
    trainer_cfg = sae_config.get('trainer', {})
    
    # Extract key hyperparameter based on SAE type
    # Different SAE types have different key hyperparameters:
    # - TopK/BatchTopK/MatryoshkaBatchTopK: k (sparsity)
    # - Standard/GatedSAE: l1_penalty
    # - JumpRelu: target_l0
    # - PAnneal: sparsity_penalty
    key_param = None
    param_value = None
    
    # Priority order: check most specific first
    # Check for k (TopK, BatchTopK, MatryoshkaBatchTopK)
    if 'k' in trainer_cfg:
        key_param = 'k'
        param_value = trainer_cfg['k']
    # Check for target_l0 (JumpRelu)
    elif 'target_l0' in trainer_cfg:
        key_param = 'target_l0'
        param_value = trainer_cfg['target_l0']
    # Check for sparsity_penalty (PAnneal)
    elif 'sparsity_penalty' in trainer_cfg:
        key_param = 'sparsity_penalty'
        param_value = trainer_cfg['sparsity_penalty']
    # Check for l1_penalty (Standard, GatedSAE)
    elif 'l1_penalty' in trainer_cfg:
        key_param = 'l1_penalty'
        param_value = trainer_cfg['l1_penalty']
    
    # Create descriptive trainer folder name
    if key_param and param_value is not None:
        # Format: trainer_0_k20, trainer_1_target_l0_20, trainer_1_l1_penalty_0_012, etc.
        if key_param in ['l1_penalty', 'sparsity_penalty']:
            # Format decimal values nicely (replace . with _)
            param_str = str(param_value).replace('.', '_')
            # Remove trailing zeros after underscore
            if '_' in param_str:
                parts = param_str.split('_')
                if len(parts) == 2 and parts[1].endswith('0'):
                    parts[1] = parts[1].rstrip('0')
                    if not parts[1]:
                        param_str = parts[0]
                    else:
                        param_str = '_'.join(parts)
            trainer_folder_name = f"{trainer_folder}_{key_param}_{param_str}"
        elif key_param == 'target_l0':
            trainer_folder_name = f"{trainer_folder}_{key_param}_{param_value}"
        else:
            # For k, just append value
            trainer_folder_name = f"{trainer_folder}_{key_param}{param_value}"
    else:
        trainer_folder_name = trainer_folder
    
    # Create final output directory with descriptive trainer name
    output_dir = base_output_dir / trainer_folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Copy config.json to output folder
    output_config_path = output_dir / "config.json"
    shutil.copy2(config_json_path, output_config_path)
    if key_param:
        print(f"✓ Copied config.json to output folder ({key_param}={param_value})")
    else:
        print(f"✓ Copied config.json to output folder")
    
    step_elapsed = time.time() - step_start
    print(f"[TIMING] Finished loading SAE at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Took {step_elapsed/60:.2f} minutes ({step_elapsed:.2f} seconds)")
    
    # Step 2: Load base model
    print("\n[2/4] Loading base model...")
    step_start = time.time()
    print(f"[TIMING] Started loading base model at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    model = load_base_model(
        model_name=metadata['model_name'],
        device=device
    )
    
    step_elapsed = time.time() - step_start
    print(f"[TIMING] Finished loading base model at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Took {step_elapsed/60:.2f} minutes ({step_elapsed:.2f} seconds)")
    
    # Step 3: Setup activation store
    print("\n[3/4] Setting up activation store...")
    step_start = time.time()
    print(f"[TIMING] Started setting up activation store at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    store = setup_activations_store(
        model=model,
        sae=sae,
        metadata=metadata,
        device=device
    )
    step_elapsed = time.time() - step_start
    print(f"[TIMING] Finished setting up activation store at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Took {step_elapsed/60:.2f} minutes ({step_elapsed:.2f} seconds)")
    
    # Step 4: Compute firing statistics
    print("\n[4/4] Computing firing statistics...")
    step_start = time.time()
    print(f"[TIMING] Started computing firing statistics at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    firing_stats = compute_firing_statistics(
        sae=sae,
        store=store,
        n_batches=n_batches,
        thresholds=thresholds,
        device=device
    )
    step_elapsed = time.time() - step_start
    print(f"[TIMING] Finished computing firing statistics at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Took {step_elapsed/60:.2f} minutes ({step_elapsed:.2f} seconds)")
    
    # Create plots organized into 3 subfolders (like original analysis)
    # For single SAE, we need to wrap stats in a dict with experiment name
    sae_name = f"{sae_type_folder}_{trainer_folder_name}"
    stats_dict = {sae_name: firing_stats}
    
    # Analysis 1: Firing frequency distribution (subfolder: firing_frequency_distribution/)
    print("\n[Analysis 1] Creating firing frequency distribution plots...")
    step_start = time.time()
    print(f"[TIMING] Started Analysis 1 (firing frequency plots) at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    firing_freq_dir = output_dir / "firing_frequency_distribution"
    create_firing_frequency_distribution_plots(
        stats_dict=stats_dict,
        output_dir=firing_freq_dir
    )
    
    # Also create cumulative magnitude plots in the same folder
    create_cumulative_magnitude_distribution_plots(
        stats_dict=stats_dict,
        output_dir=firing_freq_dir
    )
    
    # Also create top features frequency plots
    print("\n[Analysis 1b] Creating top features frequency plots...")
    create_top_features_frequency_plots(
        stats_dict=stats_dict,
        output_dir=firing_freq_dir
    )
    step_elapsed = time.time() - step_start
    print(f"[TIMING] Finished Analysis 1 at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Took {step_elapsed/60:.2f} minutes ({step_elapsed:.2f} seconds)")
    
    # Analysis 2: Firing frequency vs magnitude (subfolder: firing_frequency_magnitude/)
    print("\n[Analysis 2] Creating firing frequency vs magnitude plots...")
    step_start = time.time()
    print(f"[TIMING] Started Analysis 2 (magnitude plots) at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    magnitude_dir = output_dir / "firing_frequency_magnitude"
    create_firing_magnitude_plots(
        stats_dict=stats_dict,
        output_dir=magnitude_dir,
        top_k_groups=[10, 50, 100, 250, 500, 1000, 10000]
    )
    
    # Also create feature ranking overlap plots
    print("\n[Analysis 2b] Creating feature ranking overlap plots...")
    create_feature_ranking_overlap_plots(
        stats_dict=stats_dict,
        output_dir=magnitude_dir
    )
    step_elapsed = time.time() - step_start
    print(f"[TIMING] Finished Analysis 2 at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Took {step_elapsed/60:.2f} minutes ({step_elapsed:.2f} seconds)")
    
    # Analysis 3: Per-input magnitude analysis (subfolder: per_input_magnitude_analysis/)
    print("\n[Analysis 3] Creating per-input magnitude analysis plots...")
    step_start = time.time()
    print(f"[TIMING] Started Analysis 3 (per-input analysis) at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    per_input_dir = output_dir / "per_input_magnitude_analysis"
    create_per_input_magnitude_analysis_simplified(
        sae=sae,
        store=store,
        stats_dict=stats_dict,
        output_dir=per_input_dir,
        device=device,
        n_batches_for_collection=n_batches  # Use same batch count as other analyses
    )
    step_elapsed = time.time() - step_start
    print(f"[TIMING] Finished Analysis 3 at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Took {step_elapsed/60:.2f} minutes ({step_elapsed:.2f} seconds)")
    
    # Save full firing stats JSON in main output directory
    results_file = output_dir / "firing_stats.json"
    with open(results_file, 'w') as f:
        json.dump(firing_stats, f, indent=2)
    print(f"✓ Saved firing statistics to: {results_file}")
    
    # Clean up SAE cache after analysis
    print("\n[Cleanup] Removing cached SAE files...")
    cleanup_sae_cache(metadata['_cache_path'])
    
    # Clean up LLM model cache after analysis
    print("\n[Cleanup] Removing cached LLM model files...")
    cleanup_model_cache(metadata['model_name'])
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")


# SAE Bench Configuration
SAE_BENCH_CONFIG = {
    'pythia_160m': {
        'repos': {
            '4k': 'adamkarvonen/saebench_pythia-160m-deduped_width-2pow12_date-0108',
            '16k': 'adamkarvonen/saebench_pythia-160m-deduped_width-2pow14_date-0108',
            '65k': 'adamkarvonen/saebench_pythia-160m-deduped_width-2pow16_date-0108',
        },
        'layer': 'resid_post_layer_8',
        'sae_types': [
            'Standard_pythia-160m-deduped__0108',  # ReLU
            'TopK_pythia-160m-deduped__0108',
            'BatchTopK_pythia-160m-deduped__0108',
            'JumpRelu_pythia-160m-deduped__0108',
            'GatedSAE_pythia-160m-deduped__0108',
            'PAnneal_pythia-160m-deduped__0108',
            'MatryoshkaBatchTopK_pythia-160m-deduped__0108',
        ],
    },
}

# All trainers
TRAINERS = ["trainer_0", "trainer_1", "trainer_2", "trainer_3", "trainer_4", "trainer_5"]


def run_all_analyses(
    device: str = None,
    n_batches: int = 30,
    thresholds: list = None,
    output_dir: Path = None,
    skip_existing: bool = True
):
    """
    Run analysis3 on all SAE Bench Pythia models with all trainers.
    
    Args:
        device: Device to use (default: auto-detect cuda:0)
        n_batches: Number of batches for activation collection
        thresholds: List of thresholds for counting fires (default: [0.0, 0.01])
        output_dir: Output directory
        skip_existing: Skip analyses that already have results
    """
    if thresholds is None:
        thresholds = [0.0, 0.01]
    
    config = SAE_BENCH_CONFIG['pythia_160m']
    
    total_analyses = len(config['sae_types']) * len(config['repos']) * len(TRAINERS)
    completed = 0
    skipped = 0
    failed = []
    
    print("="*70)
    print("BATCH ANALYSIS: ALL SAE BENCH PYTHIA MODELS (FEATURE FIRING)")
    print("="*70)
    print(f"\nTotal analyses to run: {total_analyses}")
    print(f"  - SAE types: {len(config['sae_types'])}")
    print(f"  - Widths: {len(config['repos'])}")
    print(f"  - Trainers: {len(TRAINERS)}")
    print(f"\nDevice: {device or 'auto-detect'}")
    print(f"n_batches: {n_batches}, thresholds: {thresholds}")
    
    start_time = time.time()
    
    for width_name, repo_id in config['repos'].items():
        print("\n" + "="*70)
        print(f"PROCESSING {width_name.upper()} WIDTH")
        print("="*70)
        
        for sae_type in config['sae_types']:
            print(f"\n--- {sae_type} ---")
            
            for trainer in TRAINERS:
                # Check if already exists
                if skip_existing and output_dir:
                    check_dir = output_dir / "saebench_pythia" / width_name / sae_type / trainer
                    # Check for one of the expected output files
                    if (check_dir / "firing_frequency_distribution" / "1_firing_frequency_distribution_thresh_0_00.png").exists():
                        print(f"  ⏭️  {trainer}: Already exists, skipping...")
                        skipped += 1
                        continue
                
                print(f"\n  🔄 {trainer}...")
                try:
                    run_analysis_saebench_pythia(
                        repo_id=repo_id,
                        sae_type_folder=sae_type,
                        layer_folder=config['layer'],
                        trainer_folder=trainer,
                        width_name=width_name,
                        device=device,
                        n_batches=n_batches,
                        thresholds=thresholds,
                        output_dir=output_dir
                    )
                    completed += 1
                    print(f"  ✅ {trainer}: Completed ({completed}/{total_analyses})")
                except Exception as e:
                    error_msg = str(e)[:200]
                    print(f"  ❌ {trainer}: Failed - {error_msg}")
                    failed.append({
                        'width': width_name,
                        'sae_type': sae_type,
                        'trainer': trainer,
                        'error': error_msg
                    })
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("BATCH ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nTotal analyses: {total_analyses}")
    print(f"✅ Completed: {completed}")
    print(f"⏭️  Skipped: {skipped}")
    print(f"❌ Failed: {len(failed)}")
    print(f"⏱️  Time elapsed: {elapsed_time/3600:.2f} hours ({elapsed_time/60:.1f} minutes)")
    
    if failed:
        print("\nFailed analyses:")
        for f in failed:
            print(f"  - {f['width']} {f['sae_type']} {f['trainer']}: {f['error']}")
    
    # Create aggregation plots
    if completed > 0:
        print("\n" + "="*70)
        print("CREATING AGGREGATION PLOTS")
        print("="*70)
        model_family = "saebench_pythia"
        base_dir = output_dir if output_dir else Path(os.environ.get("DATA_DIR", "/data/user")) / "sae" / "feature_firing_analysis"
        aggregation_output_dir = base_dir / model_family / "aggregation_final"
        
        try:
            create_top_features_aggregation_by_architecture(
                model_family=model_family,
                base_dir=base_dir,
                output_dir=aggregation_output_dir
            )
        except Exception as e:
            print(f"  ⚠️  Error creating aggregation by architecture: {e}")
        
        try:
            create_top_features_aggregation_by_width(
                model_family=model_family,
                base_dir=base_dir,
                output_dir=aggregation_output_dir
            )
        except Exception as e:
            print(f"  ⚠️  Error creating aggregation by width: {e}")
    
    return completed, skipped, len(failed)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run analysis3 on all SAE Bench Pythia models")
    parser.add_argument('--device', type=str, default=None, help='Device to use (default: auto-detect cuda:0)')
    parser.add_argument('--n-batches', type=int, default=30, help='Number of batches for activation collection')
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.0, 0.01],
                        help='Thresholds for counting fires (default: 0.0 0.01)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory. IMPORTANT: Must be provided for skip_existing to work. '
                            'If not provided, skip_existing is disabled and all analyses will re-run.')
    parser.add_argument('--no-skip-existing', action='store_true', help='Re-run analyses even if results exist')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    run_all_analyses(
        device=args.device,
        n_batches=args.n_batches,
        thresholds=args.thresholds,
        output_dir=output_dir,
        skip_existing=not args.no_skip_existing
    )

