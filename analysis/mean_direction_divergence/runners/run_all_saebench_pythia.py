#!/usr/bin/env python3
"""
Batch runner for ALL SAE Bench Pythia models with ALL trainers.

This script runs analysis5 (mean direction divergence) on:
- All 7 SAE types
- All 3 widths (4k, 16k, 65k)
- All 6 trainers (trainer_0 through trainer_5)
Total: 7 × 3 × 6 = 126 analyses

Usage:
    # Run with skip_existing enabled (recommended for re-runs):
    python -m mean_direction_divergence.runners.run_all_saebench_pythia --device cuda:1 --output-dir <DATA_DIR>/sae/mean_direction_divergence
    
    # Note: --output-dir MUST be provided for skip_existing to work correctly.
    # If --output-dir is not provided, skip_existing will be disabled and all analyses will re-run.
"""

import sys
import os
from pathlib import Path
import time
import json
import shutil
import numpy as np

# Add parent directories to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

# Import analysis functions
from mean_direction_divergence import (
    compute_mean_direction_from_activations,
    compute_angular_divergences,
    create_divergence_plots,
)
from mean_direction_divergence.mean_direction_analysis import (
    compute_angular_distances_from_mean,
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


# Add cleanup_repo_cache function
def cleanup_repo_cache(repo_id: str) -> None:
    """
    Clean up entire HuggingFace repo cache including blobs after all SAEs from a repo are done.
    
    Args:
        repo_id: HuggingFace repo ID
    """
    cache_base = Path.home() / ".cache" / "huggingface" / "hub"
    
    # Convert repo_id to cache directory name format
    repo_cache_name = f"models--{repo_id.replace('/', '--')}"
    repo_cache_dir = cache_base / repo_cache_name
    
    if repo_cache_dir.exists():
        try:
            shutil.rmtree(repo_cache_dir)
            print(f"✓ Cleaned up repo cache (including blobs): {repo_cache_dir}")
        except Exception as e:
            print(f"⚠️  Failed to clean up repo cache {repo_cache_dir}: {e}")
    else:
        print(f"⚠️  Repo cache does not exist: {repo_cache_dir}")


def collect_original_activations(store, n_batches: int = 30):
    """
    Collect original input activations (before SAE encoding).
    
    Args:
        store: ActivationsStore for collecting activations
        n_batches: Number of batches to collect
    
    Returns:
        all_activations: Original activations (n_samples, d_in)
    """
    print(f"\n📊 Collecting {n_batches} batches of original activations...")
    all_activations = []
    
    for batch_idx in range(n_batches):
        batch_activations = store.next_batch()  # [batch_size, d_in] - original activations
        all_activations.append(batch_activations.cpu().numpy())
        
        # Periodic memory cleanup
        if (batch_idx + 1) % 10 == 0:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
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
    n_batches: int = 30,
    output_dir: Path = None,
    model=None  # Optional: reuse existing model
):
    """
    Run mean direction divergence analysis for SAE Bench Pythia SAE.
    
    Args:
        repo_id: HuggingFace repo ID
        sae_type_folder: SAE type folder name
        layer_folder: Layer folder name
        trainer_folder: Trainer folder name
        width_name: Width name (4k, 16k, 65k)
        device: Device to use (default: auto-detect cuda:0, requires GPU)
        n_batches: Number of batches for activation collection
        output_dir: Output directory (default: {DATA_DIR}/sae/mean_direction_divergence/, where DATA_DIR defaults to /data/user)
        model: Optional pre-loaded model to reuse (saves GPU memory)
    """
    import torch
    import gc
    
    # Auto-detect device if not specified - require GPU
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/GPU is not available. GPU is required for analysis.")
        device = "cuda:0"
        print(f"Using GPU: {device}")
    
    print("="*70)
    print("MEAN DIRECTION DIVERGENCE ANALYSIS - SAE BENCH PYTHIA")
    print("="*70)
    
    # Setup output directory: saebench_pythia/{width}/{sae_type}/{trainer}/
    if output_dir is None:
        data_dir = os.environ.get("DATA_DIR", "/data/user")
        output_dir = Path(data_dir) / "sae" / "mean_direction_divergence"
    
    # Create base output directory
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
    trainer_folder_name = trainer_folder
    if config_json_path.exists():
        with open(config_json_path, 'r') as f:
            sae_config = json.load(f)
        
        trainer_cfg = sae_config.get('trainer', {})
        
        # Extract key hyperparameter
        key_param = None
        param_value = None
        
        if 'k' in trainer_cfg:
            key_param = 'k'
            param_value = trainer_cfg['k']
        elif 'l1_penalty' in trainer_cfg:
            key_param = 'l1_penalty'
            param_value = trainer_cfg['l1_penalty']
        
        # Create descriptive trainer folder name
        if key_param and param_value is not None:
            if key_param == 'l1_penalty':
                param_str = str(param_value).replace('.', '_')
                if '_' in param_str:
                    parts = param_str.split('_')
                    if len(parts) == 2 and parts[1].endswith('0'):
                        parts[1] = parts[1].rstrip('0')
                        if not parts[1]:
                            param_str = parts[0]
                        else:
                            param_str = '_'.join(parts)
                trainer_folder_name = f"{trainer_folder}_{key_param}_{param_str}"
            else:
                trainer_folder_name = f"{trainer_folder}_{key_param}{param_value}"
    
    # Create final output directory with descriptive trainer name
    output_dir_final = base_output_dir / trainer_folder_name
    output_dir_final.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir_final}")
    
    # Copy config.json to output folder if exists
    if config_json_path.exists():
        output_config_path = output_dir_final / "config.json"
        shutil.copy2(config_json_path, output_config_path)
        print(f"✓ Copied config.json to output folder")
    
    # Step 2: Load base model (reuse if provided)
    print("\n[2/4] Loading base model...")
    model_loaded_here = False
    if model is None:
        # Clear GPU cache before loading to avoid OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            model = load_base_model(
                model_name=metadata['model_name'],
                device=device
            )
            model_loaded_here = True
        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
    else:
        print("  ✓ Reusing existing model (saves GPU memory)")
    
    # Step 3: Setup activation store and collect original activations
    print("\n[3/4] Collecting original activations...")
    store = setup_activations_store(
        model=model,
        sae=sae,
        metadata=metadata,
        device=device,
        width_name=width_name
    )
    
    original_activations = collect_original_activations(store, n_batches=n_batches)
    
    # Step 4: Run mean direction divergence analysis
    print("\n[4/4] Running mean direction divergence analysis...")
    
    try:
        # Compute mean direction from original activations
        mean_direction, mean_resultant_length = compute_mean_direction_from_activations(original_activations)
        
        # Get encoder and decoder vectors
        encoder_vectors = sae.W_enc.detach().cpu().numpy().T  # (d_sae, d_in)
        decoder_vectors = sae.W_dec.detach().cpu().numpy()     # (d_sae, d_in)
        
        # Compute angular divergences
        results, encoder_distances, decoder_distances = compute_angular_divergences(
            encoder_weights=encoder_vectors,
            decoder_weights=decoder_vectors,
            mean_direction=mean_direction
        )
        
        # Compute activation distances for comparison
        activation_distances = compute_angular_distances_from_mean(original_activations, mean_direction)
        
        # Compute activation statistics
        activation_stats = {
            'mean': float(np.mean(activation_distances)),
            'median': float(np.median(activation_distances)),
            'std': float(np.std(activation_distances)),
            'min': float(np.min(activation_distances)),
            'max': float(np.max(activation_distances)),
            'percentile_25': float(np.percentile(activation_distances, 25)),
            'percentile_75': float(np.percentile(activation_distances, 75)),
            'percentile_90': float(np.percentile(activation_distances, 90)),
            'percentile_95': float(np.percentile(activation_distances, 95)),
            'percentile_99': float(np.percentile(activation_distances, 99)),
        }
        
        # Add activation statistics to results
        results['activation'] = activation_stats
        print(f"✓ Activation: mean={activation_stats['mean']:.1f}°, median={activation_stats['median']:.1f}°")
        
        # Create output subdirectory
        analysis_output_dir = output_dir_final / "mean_direction_divergence"
        
        # Create plots
        metadata['experiment_folder'] = f"{sae_type_folder}_{width_name}_{trainer_folder_name}"
        metadata['description'] = f"saebench_pythia: {sae_type_folder}"
        
        create_divergence_plots(
            encoder_distances=encoder_distances,
            decoder_distances=decoder_distances,
            activation_distances=activation_distances,
            results=results,
            output_dir=analysis_output_dir,
            metadata=metadata
        )
        
        # Save results JSON
        results_file = analysis_output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'mean_resultant_length': float(mean_resultant_length),
                'divergence_statistics': results,
            }, f, indent=2)
        print(f"✓ Saved results to: {results_file}")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        print(f"⚠️  Output directory may be incomplete: {output_dir_final}")
        raise
    
    # Clean up SAE cache after analysis
    print("\n[Cleanup] Removing cached SAE files...")
    cleanup_sae_cache(metadata['_cache_path'])
    
    # Clean up GPU memory (but keep model if we loaded it here and it should be reused)
    if torch.cuda.is_available():
        del store
        del sae
        # Only delete model if we loaded it here (not if it was passed in for reuse)
        if model_loaded_here:
            del model
            model = None
        gc.collect()  # Explicit garbage collection
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Wait for all operations to complete
        print(f"✓ Cleared GPU cache (device: {device})")
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir_final}")
    
    # Return model if we loaded it (for reuse)
    return model if model_loaded_here else None


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
    output_dir: Path = None,
    skip_existing: bool = True
):
    """
    Run analysis5 on all SAE Bench Pythia models with all trainers.
    
    Args:
        device: Device to use (default: auto-detect cuda:0)
        n_batches: Number of batches for activation collection
        output_dir: Output directory
        skip_existing: Skip analyses that already have results
    """
    import torch
    import gc
    
    config = SAE_BENCH_CONFIG['pythia_160m']
    
    total_analyses = len(config['sae_types']) * len(config['repos']) * len(TRAINERS)
    completed = 0
    skipped = 0
    failed = []
    
    print("="*70)
    print("BATCH ANALYSIS: ALL SAE BENCH PYTHIA MODELS (MEAN DIRECTION DIVERGENCE)")
    print("="*70)
    print(f"\nTotal analyses to run: {total_analyses}")
    print(f"  - SAE types: {len(config['sae_types'])}")
    print(f"  - Widths: {len(config['repos'])}")
    print(f"  - Trainers: {len(TRAINERS)}")
    print(f"\nDevice: {device or 'auto-detect'}")
    print(f"n_batches: {n_batches}")
    
    start_time = time.time()
    
    # Load model once per width and reuse (saves GPU memory)
    model_cache = {}  # {width_name: model}
    
    for width_name, repo_id in config['repos'].items():
        print("\n" + "="*70)
        print(f"PROCESSING {width_name.upper()} WIDTH")
        print("="*70)
        
        # Load model once for this width (all SAEs use same base model)
        print(f"\n📦 Loading base model for {width_name} width (will reuse for all SAEs)...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear before loading
        
        # Get model name from first SAE (all same model)
        try:
            first_sae, first_sae_metadata = load_saebench_sae(
                repo_id=repo_id,
                sae_type_folder=config['sae_types'][0],
                layer_folder=config['layer'],
                trainer_folder='trainer_0',
                device=device
            )
            model_name = first_sae_metadata['model_name']
            model_cache[width_name] = load_base_model(model_name, device=device)
            print(f"✓ Loaded model: {model_name} (will reuse for all {len(config['sae_types'])} SAE types)")
            # Clean up the test SAE
            cleanup_sae_cache(first_sae_metadata['_cache_path'])
            del first_sae
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"⚠️  Failed to load model for {width_name}: {e}")
            model_cache[width_name] = None
        
        for sae_type in config['sae_types']:
            print(f"\n--- {sae_type} ---")
            
            for trainer in TRAINERS:
                # Check if already exists
                if skip_existing and output_dir:
                    sae_type_dir = output_dir / "saebench_pythia" / width_name / sae_type
                    if sae_type_dir.exists():
                        found_existing = False
                        for existing_folder in sae_type_dir.iterdir():
                            if existing_folder.is_dir() and existing_folder.name.startswith(trainer + '_'):
                                png_exists = (existing_folder / "mean_direction_divergence" / "1_encoder_decoder_divergence_comparison.png").exists()
                                json_exists = (existing_folder / "mean_direction_divergence" / "results.json").exists()
                                if png_exists and json_exists:
                                    print(f"  ⏭️  {trainer}: Already exists ({existing_folder.name}), skipping...")
                                    skipped += 1
                                    found_existing = True
                                    break
                        if found_existing:
                            continue
                
                print(f"\n  🔄 {trainer}...")
                try:
                    # Aggressive memory cleanup before each analysis
                    if torch.cuda.is_available():
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    # Reuse model for this width
                    reused_model = model_cache.get(width_name)
                    run_analysis_saebench_pythia(
                        repo_id=repo_id,
                        sae_type_folder=sae_type,
                        layer_folder=config['layer'],
                        trainer_folder=trainer,
                        width_name=width_name,
                        device=device,
                        n_batches=n_batches,
                        output_dir=output_dir,
                        model=reused_model  # Reuse model to save GPU memory
                    )
                    completed += 1
                    print(f"  ✅ {trainer}: Completed ({completed}/{total_analyses})")
                    
                    # Aggressive cleanup after each successful analysis
                    if torch.cuda.is_available():
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except Exception as e:
                    error_msg = str(e)[:200]
                    print(f"  ❌ {trainer}: Failed - {error_msg}")
                    failed.append({
                        'sae_type': sae_type,
                        'width': width_name,
                        'trainer': trainer,
                        'error': error_msg
                    })
                    # Aggressive cleanup after failure too
                    if torch.cuda.is_available():
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
        
        # Clean up model and repo cache for this width after all SAEs processed
        if width_name in model_cache and model_cache[width_name] is not None:
            print(f"\n🧹 Cleaning up model and repo cache for {width_name} width...")
            repo_id = config['repos'][width_name]
            
            del model_cache[width_name]
            model_cache[width_name] = None
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            cleanup_model_cache('EleutherAI/pythia-160m-deduped')
            cleanup_repo_cache(repo_id)
            print(f"✓ Cleaned up model, repo cache (including blobs), and GPU cache for {width_name}")
    
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
            print(f"  - {f['sae_type']}/{f['width']}/{f['trainer']}: {f['error']}")
    
    return completed, skipped, len(failed)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run analysis5 on all SAE Bench Pythia models")
    parser.add_argument('--device', type=str, default=None, help='Device to use (default: auto-detect cuda:0)')
    parser.add_argument('--n-batches', type=int, default=30, help='Number of batches for activation collection (default: 30)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory. IMPORTANT: Must be provided for skip_existing to work. '
                            'If not provided, skip_existing is disabled and all analyses will re-run.')
    parser.add_argument('--no-skip-existing', action='store_true', help='Re-run analyses even if results exist')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    run_all_analyses(
        device=args.device,
        n_batches=args.n_batches,
        output_dir=output_dir,
        skip_existing=not args.no_skip_existing
    )

