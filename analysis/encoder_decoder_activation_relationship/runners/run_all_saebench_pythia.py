#!/usr/bin/env python3
"""
Batch runner for ALL SAE Bench Pythia models with ALL trainers.

This script runs analysis2 on:
- All 7 SAE types
- All 3 widths (4k, 16k, 65k)
- All 6 trainers (trainer_0 through trainer_5)
Total: 7 × 3 × 6 = 126 analyses

Usage:
    # Run with skip_existing enabled (recommended for re-runs):
    python -m encoder_decoder_activation_relationship.runners.run_all_saebench_pythia --device cuda:1 --output-dir /data/user/sae/encoder_decoder_activation_relationship
    
    # Note: --output-dir MUST be provided for skip_existing to work correctly.
    # If --output-dir is not provided, skip_existing will be disabled and all analyses will re-run.
"""

import sys
import os
from pathlib import Path
import time

# Add parent directories to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

# Import analysis functions
from encoder_decoder_activation_relationship import (
    analyze_activated_features_angular_distances,
    create_activated_features_plot,
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


def collect_activations(store, n_batches: int = 50):
    """Collect activations from the activation store."""
    print(f"\n📊 Collecting {n_batches} batches of activations...")
    all_activations = []
    
    for batch_idx in range(n_batches):
        batch_activations = store.next_batch()
        all_activations.append(batch_activations.cpu().numpy())
    
    import numpy as np
    all_activations = np.concatenate(all_activations, axis=0)
    print(f"✓ Collected {all_activations.shape[0]:,} activation vectors")
    
    return all_activations


def run_analysis_saebench_pythia(
    repo_id: str,
    sae_type_folder: str,
    layer_folder: str = "resid_post_layer_8",
    trainer_folder: str = "trainer_0",
    width_name: str = None,
    device: str = None,
    n_batches: int = 50,
    n_samples: int = 5000,
    output_dir: Path = None
):
    """
    Run encoder-decoder-activation analysis for SAE Bench Pythia SAE.
    
    Args:
        repo_id: HuggingFace repo ID
        sae_type_folder: SAE type folder name
        layer_folder: Layer folder name
        trainer_folder: Trainer folder name
        device: Device to use (default: auto-detect cuda:0, requires GPU)
        n_batches: Number of batches for activation collection
        n_samples: Number of samples for analysis
        output_dir: Output directory (default: /data/user/sae/encoder_decoder_activation_relationship/)
    """
    # Auto-detect device if not specified - require GPU
    if device is None:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/GPU is not available. GPU is required for analysis.")
        device = "cuda:0"
        print(f"Using GPU: {device}")
    
    print("="*70)
    print("ENCODER-DECODER-ACTIVATION ANALYSIS - SAE BENCH PYTHIA")
    print("="*70)
    
    # Setup output directory: saebench_pythia/{width}/{sae_type}/{trainer}/
    if output_dir is None:
        output_dir = Path(os.environ.get("DATA_DIR", "/data/user")) / "sae" / "encoder_decoder_activation_relationship"
    
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
    import json
    import shutil
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
    
    # Step 2: Load base model
    print("\n[2/4] Loading base model...")
    model = load_base_model(
        model_name=metadata['model_name'],
        device=device
    )
    
    # Step 3: Setup activation store and collect activations
    print("\n[3/4] Collecting activations...")
    store = setup_activations_store(
        model=model,
        sae=sae,
        metadata=metadata,
        device=device
    )
    
    activations = collect_activations(store, n_batches=n_batches)
    
    # Step 4: Run analysis
    print("\n[4/4] Running angular distance analysis...")
    
    # Get encoder and decoder vectors
    encoder_vectors = sae.W_enc.detach().cpu().numpy().T  # (d_sae, d_in)
    decoder_vectors = sae.W_dec.detach().cpu().numpy()     # (d_sae, d_in)
    
    # Get SAE architecture
    if hasattr(sae.cfg, 'architecture') and callable(getattr(sae.cfg, 'architecture')):
        sae_architecture = sae.cfg.architecture()
    else:
        sae_architecture = 'standard'
    
    # Analyze activated features
    results = analyze_activated_features_angular_distances(
        sae=sae,
        activations=activations,
        encoder_vectors=encoder_vectors,
        decoder_vectors=decoder_vectors,
        sae_architecture=sae_architecture,
        n_samples=n_samples
    )
    
    # Create plot
    metadata['experiment_folder'] = f"{sae_type_folder}_{layer_folder}_{trainer_folder}"
    metadata['description'] = f"SAE Bench Pythia: {sae_type_folder}"
    
    plot_path = output_dir / "activated_features_analysis.png"
    create_activated_features_plot(
        results=results,
        output_path=plot_path,
        metadata=metadata
    )
    
    # Save results JSON
    import json
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results['summary'], f, indent=2)
    print(f"✓ Saved results to: {results_file}")
    
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
    n_batches: int = 50,
    n_samples: int = 5000,
    output_dir: Path = None,
    skip_existing: bool = True
):
    """
    Run analysis2 on all SAE Bench Pythia models with all trainers.
    
    Args:
        device: Device to use (default: auto-detect cuda:0)
        n_batches: Number of batches for activation collection
        n_samples: Number of samples for analysis
        output_dir: Output directory
        skip_existing: Skip analyses that already have results
    """
    config = SAE_BENCH_CONFIG['pythia_160m']
    
    total_analyses = len(config['sae_types']) * len(config['repos']) * len(TRAINERS)
    completed = 0
    skipped = 0
    failed = []
    
    print("="*70)
    print("BATCH ANALYSIS: ALL SAE BENCH PYTHIA MODELS")
    print("="*70)
    print(f"\nTotal analyses to run: {total_analyses}")
    print(f"  - SAE types: {len(config['sae_types'])}")
    print(f"  - Widths: {len(config['repos'])}")
    print(f"  - Trainers: {len(TRAINERS)}")
    print(f"\nDevice: {device or 'auto-detect'}")
    print(f"n_batches: {n_batches}, n_samples: {n_samples}")
    
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
                    if (check_dir / "activated_features_analysis.png").exists():
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
                        n_samples=n_samples,
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
    
    return completed, skipped, len(failed)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run analysis2 on all SAE Bench Pythia models")
    parser.add_argument('--device', type=str, default=None, help='Device to use (default: auto-detect cuda:0)')
    parser.add_argument('--n-batches', type=int, default=50, help='Number of batches for activation collection')
    parser.add_argument('--n-samples', type=int, default=5000, help='Number of samples for analysis')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory. IMPORTANT: Must be provided for skip_existing to work. '
                            'If not provided, skip_existing is disabled and all analyses will re-run.')
    parser.add_argument('--no-skip-existing', action='store_true', help='Re-run analyses even if results exist')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    run_all_analyses(
        device=args.device,
        n_batches=args.n_batches,
        n_samples=args.n_samples,
        output_dir=output_dir,
        skip_existing=not args.no_skip_existing
    )

