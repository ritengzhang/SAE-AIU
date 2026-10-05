#!/usr/bin/env python3
"""
Batch runner for ALL canrager TopK+Vanilla SAEs with ALL trainers.

This script runs analysis5 (mean direction divergence) on:
- All 2 architectures (TopK, Vanilla)
- All 3 widths (4k, 16k, 65k)
- All 3 layers (5, 12, 19)
- All 6 trainers (trainer_0 through trainer_5)
Total: 2 × 3 × 3 × 6 = 108 analyses
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

from sae_family_load_convert.canrager_topk_vanilla import (
    load_canrager_sae,
    cleanup_sae_cache,
    cleanup_repo_cache,
    get_repo_id,
)

from sae_family_load_convert import (
    setup_activations_store,
    load_base_model,
)

from encoder_decoder_activation_relationship.cleanup_utils import (
    cleanup_model_cache,
)


def collect_original_activations(store, n_batches: int = 15):
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


def run_analysis_canrager(
    architecture: str,
    width_name: str,
    layer: int,
    trainer_folder: str = "trainer_0",
    device: str = None,
    n_batches: int = 15,
    output_dir: Path = None,
    model=None  # Optional: reuse existing model
):
    """
    Run mean direction divergence analysis for canrager SAE.
    
    Args:
        architecture: 'topk' or 'vanilla'
        width_name: '4k', '16k', or '65k'
        layer: Layer number (5, 12, or 19)
        trainer_folder: Trainer folder (e.g., "trainer_0")
        device: Device to use (default: auto-detect cuda:0, requires GPU)
        n_batches: Number of batches for activation collection
        output_dir: Output directory (default: /data/user/sae/mean_direction_divergence/)
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
    print("MEAN DIRECTION DIVERGENCE ANALYSIS - CANRAGER TOPK+VANILLA")
    print("="*70)
    
    # Setup output directory: canrager/{architecture}/{width}/layer_{layer}/{trainer}/
    if output_dir is None:
        output_dir = Path(os.environ.get("DATA_DIR", "/data/user")) / "sae" / "mean_direction_divergence"
    
    # Create base output directory (we'll add trainer folder name after loading config)
    base_output_dir = output_dir / "canrager" / architecture / width_name / f"layer_{layer}"
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load SAE
    print("\n[1/4] Loading SAE...")
    sae, metadata = load_canrager_sae(
        architecture=architecture,
        width_name=width_name,
        layer=layer,
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
    key_param = None
    param_value = None
    
    # Priority order: check most specific first
    if 'k' in trainer_cfg:
        key_param = 'k'
        param_value = trainer_cfg['k']
    elif 'target_l0' in trainer_cfg:
        key_param = 'target_l0'
        param_value = trainer_cfg['target_l0']
    elif 'sparsity_penalty' in trainer_cfg:
        key_param = 'sparsity_penalty'
        param_value = trainer_cfg['sparsity_penalty']
    elif 'l1_penalty' in trainer_cfg:
        key_param = 'l1_penalty'
        param_value = trainer_cfg['l1_penalty']
    
    # Create descriptive trainer folder name
    if key_param and param_value is not None:
        if key_param in ['l1_penalty', 'sparsity_penalty']:
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
        elif key_param == 'target_l0':
            trainer_folder_name = f"{trainer_folder}_{key_param}_{param_value}"
        else:
            trainer_folder_name = f"{trainer_folder}_{key_param}{param_value}"
    else:
        trainer_folder_name = trainer_folder
    
    # Create final output directory with descriptive trainer name
    output_dir_final = base_output_dir / trainer_folder_name
    output_dir_final.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir_final}")
    
    # Copy config.json to output folder
    output_config_path = output_dir_final / "config.json"
    shutil.copy2(config_json_path, output_config_path)
    if key_param:
        print(f"✓ Copied config.json to output folder ({key_param}={param_value})")
    else:
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
        from mean_direction_divergence.mean_direction_analysis import compute_angular_distances_from_mean
        activation_distances = compute_angular_distances_from_mean(original_activations, mean_direction)
        
        # Create output subdirectory
        analysis_output_dir = output_dir_final / "mean_direction_divergence"
        
        # Create plots
        metadata['experiment_folder'] = f"{architecture}_layer_{layer}_{trainer_folder_name}"
        metadata['description'] = f"canrager: {architecture}"
        
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


# canrager Configuration
CANRAGER_CONFIG = {
    'architectures': ['topk', 'vanilla'],
    'widths': ['4k', '16k', '65k'],
    'layers': [5, 12, 19],
    'trainers': ["trainer_0", "trainer_1", "trainer_2", "trainer_3", "trainer_4", "trainer_5"],
    'model_name': 'google/gemma-2-2b',
}

# Total: 2 architectures × 3 widths × 3 layers × 6 trainers = 108 analyses


def run_all_analyses(
    device: str = None,
    n_batches: int = 15,
    output_dir: Path = None,
    skip_existing: bool = True
):
    """
    Run analysis5 on all canrager TopK+Vanilla SAEs.
    
    Args:
        device: Device to use (default: auto-detect cuda:0)
        n_batches: Number of batches for activation collection
        output_dir: Output directory
        skip_existing: Skip analyses that already have results
    """
    import torch
    import gc
    
    config = CANRAGER_CONFIG
    
    total_analyses = len(config['architectures']) * len(config['widths']) * len(config['layers']) * len(config['trainers'])
    completed = 0
    skipped = 0
    failed = []
    
    print("="*70)
    print("BATCH ANALYSIS: ALL CANRAGER TOPK+VANILLA SAEs (MEAN DIRECTION DIVERGENCE)")
    print("="*70)
    print(f"\nTotal analyses to run: {total_analyses}")
    print(f"  - Architectures: {len(config['architectures'])} ({', '.join(config['architectures'])})")
    print(f"  - Widths: {len(config['widths'])} ({', '.join(config['widths'])})")
    print(f"  - Layers: {len(config['layers'])} ({', '.join(map(str, config['layers']))})")
    print(f"  - Trainers: {len(config['trainers'])}")
    print(f"\nDevice: {device or 'auto-detect'}")
    print(f"n_batches: {n_batches}")
    
    start_time = time.time()
    
    # Load model once per width and reuse (saves GPU memory for Gemma 2B)
    model_cache = {}  # {width_name: model}
    
    for width_name in config['widths']:
        print("\n" + "="*70)
        print(f"PROCESSING {width_name.upper()} WIDTH")
        print("="*70)
        
        # Load model once for this width (all SAEs use same base model)
        print(f"\n📦 Loading base model for {width_name} width (will reuse for all SAEs)...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear before loading
        
        try:
            model_cache[width_name] = load_base_model(config['model_name'], device=device)
            print(f"✓ Loaded model: {config['model_name']} (will reuse for all architectures/layers)")
        except Exception as e:
            print(f"⚠️  Failed to load model for {width_name}: {e}")
            model_cache[width_name] = None
        
        for architecture in config['architectures']:
            print(f"\n--- {architecture.upper()} Architecture ---")
            
            for layer in config['layers']:
                print(f"\n  Layer {layer}:")
                
                for trainer in config['trainers']:
                    # Check if already exists
                    if skip_existing and output_dir:
                        check_dir = output_dir / "canrager" / architecture / width_name / f"layer_{layer}"
                        if check_dir.exists():
                            found_existing = False
                            for existing_folder in check_dir.iterdir():
                                if existing_folder.is_dir() and existing_folder.name.startswith(trainer + '_'):
                                    # Check for both PNG and results.json to ensure analysis completed
                                    png_exists = (existing_folder / "mean_direction_divergence" / "1_encoder_decoder_divergence_comparison.png").exists()
                                    json_exists = (existing_folder / "mean_direction_divergence" / "results.json").exists()
                                    if png_exists and json_exists:
                                        print(f"    ⏭️  {trainer}: Already exists ({existing_folder.name}), skipping...")
                                        skipped += 1
                                        found_existing = True
                                        break
                            if found_existing:
                                continue
                    
                    print(f"    🔄 {trainer}...")
                    try:
                        # Aggressive memory cleanup before each analysis
                        if torch.cuda.is_available():
                            gc.collect()
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                            if device:
                                memory_allocated = torch.cuda.memory_allocated(device)
                                memory_reserved = torch.cuda.memory_reserved(device)
                                memory_total = torch.cuda.get_device_properties(device).total_memory
                                memory_free = memory_total - memory_reserved
                                print(f"      💾 GPU Memory: {memory_free/1024**3:.2f} GB free / {memory_total/1024**3:.2f} GB total")
                        
                        # Reuse model for this width
                        reused_model = model_cache.get(width_name)
                        run_analysis_canrager(
                            architecture=architecture,
                            width_name=width_name,
                            layer=layer,
                            trainer_folder=trainer,
                            device=device,
                            n_batches=n_batches,
                            output_dir=output_dir,
                            model=reused_model  # Reuse model to save GPU memory
                        )
                        completed += 1
                        print(f"    ✅ {trainer}: Completed ({completed}/{total_analyses})")
                        
                        # Aggressive cleanup after each successful analysis
                        if torch.cuda.is_available():
                            gc.collect()
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                    except Exception as e:
                        error_msg = str(e)[:200]
                        print(f"    ❌ {trainer}: Failed - {error_msg}")
                        failed.append({
                            'architecture': architecture,
                            'width': width_name,
                            'layer': layer,
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
            repo_id = get_repo_id(width_name)
            
            del model_cache[width_name]
            model_cache[width_name] = None
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            cleanup_model_cache(config['model_name'])
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
            print(f"  - {f['architecture']}/{f['width']}/layer_{f['layer']}/{f['trainer']}: {f['error']}")
    
    return completed, skipped, len(failed)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run analysis5 on all canrager TopK+Vanilla SAEs")
    parser.add_argument('--device', type=str, default=None, help='Device to use (default: auto-detect cuda:0)')
    parser.add_argument('--n-batches', type=int, default=15, help='Number of batches for activation collection (default: 15)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--no-skip-existing', action='store_true', help='Re-run analyses even if results exist')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    run_all_analyses(
        device=args.device,
        n_batches=args.n_batches,
        output_dir=output_dir,
        skip_existing=not args.no_skip_existing
    )

