#!/usr/bin/env python3
"""
Sparsity Analysis Runner for canrager TopK+Vanilla SAEs

This script calculates sparsity metrics for each canrager SAE:
1. Average active features (simple counting)
2. Effective active features (features summing to 75% and 90% of total activation)

Results are saved in results.json with config information.

Structure:
- 2 architectures: TopK, Vanilla
- 3 widths: 4k, 16k, 65k
- 3 layers: 5, 12, 19
- 6 trainers: trainer_0 through trainer_5
Total: 2 × 3 × 3 × 6 = 108 analyses
"""

import sys
import os
from pathlib import Path
import json
import time

# Set cache directories (can be overridden via CACHE_DIR env var)
cache_base = os.environ.get("CACHE_DIR", os.environ.get("DATA_DIR", "/data/user"))
cache_dir = Path(cache_base) / "cache"
hf_cache = cache_dir / "huggingface"
os.environ['HF_HOME'] = str(hf_cache)
os.environ['HF_DATASETS_CACHE'] = str(hf_cache / "datasets")
os.environ['TRANSFORMERS_CACHE'] = str(hf_cache / "transformers")
os.environ['XDG_CACHE_HOME'] = str(cache_dir)
# Ensure cache directories exist
hf_cache.mkdir(parents=True, exist_ok=True)
(hf_cache / "datasets").mkdir(parents=True, exist_ok=True)
(hf_cache / "transformers").mkdir(parents=True, exist_ok=True)

# Add parent directories to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

# Import analysis functions
from aiu_bench.sparsity_dead_neuron import calculate_sparsity_metrics, collect_activations

from sae_family_load_convert import (
    setup_activations_store,
    load_base_model,
)

from sae_family_load_convert.canrager_topk_vanilla import (
    load_canrager_sae,
    cleanup_sae_cache,
    cleanup_repo_cache,
)

from analysis2_encoder_decoder_activation_relationship.cleanup_utils import (
    cleanup_model_cache,
)


def run_analysis_canrager(
    architecture: str,
    width_name: str,
    layer: int,
    trainer_folder: str = "trainer_0",
    device: str = None,
    n_batches: int = 15,
    output_dir: Path = None,
    model_cache: dict = None  # For reusing model across trainers
):
    """
    Run sparsity analysis for canrager SAE.
    
    Args:
        architecture: 'topk' or 'vanilla'
        width_name: '4k', '16k', or '65k'
        layer: Layer number (5, 12, or 19)
        trainer_folder: Trainer folder (e.g., "trainer_0")
        device: Device to use (default: auto-detect cuda:0, requires GPU)
        n_batches: Number of batches for activation collection
        output_dir: Output directory (default: {DATA_DIR}/sae/benchmark/sparsity_dead_neuron/, where DATA_DIR defaults to /data/user)
        model_cache: Dictionary to cache model by width_name (for memory efficiency)
    """
    # Auto-detect device if not specified - require GPU
    if device is None:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/GPU is not available. GPU is required for analysis.")
        device = "cuda:0"
        print(f"Using GPU: {device}")
    
    print("="*70)
    print("SPARSITY ANALYSIS - CANRAGER TOPK+VANILLA")
    print("="*70)
    
    # Setup output directory: canrager/{architecture}/{width}/{layer}/{trainer}/
    if output_dir is None:
        data_dir = os.environ.get("DATA_DIR", "/data/user")
        output_dir = Path(data_dir) / "sae" / "benchmark" / "sparsity_dead_neuron"
    
    # Create base output directory
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
    
    # Create descriptive folder name
    if key_param and param_value is not None:
        if key_param in ['l1_penalty', 'sparsity_penalty']:
            folder_name = f"{trainer_folder}_{key_param}_{param_value:.6f}".replace('.', '_')
        else:
            folder_name = f"{trainer_folder}_{key_param}_{int(param_value)}"
    else:
        folder_name = trainer_folder
    
    # Final output directory
    output_dir_final = base_output_dir / folder_name
    output_dir_final.mkdir(parents=True, exist_ok=True)
    
    # Step 2: Load base model (reuse if cached)
    print("\n[2/4] Loading base model...")
    model_loaded_here = False
    if model_cache is not None and width_name in model_cache:
        model = model_cache[width_name]
        print(f"  ✓ Reusing cached model for {width_name}")
    else:
        model = load_base_model(metadata['model_name'], device=device)
        model_loaded_here = True
        if model_cache is not None:
            model_cache[width_name] = model
    
    # Step 3: Setup activation store and collect activations
    print("\n[3/4] Collecting activations...")
    store = setup_activations_store(
        model=model,
        sae=sae,
        metadata=metadata,
        device=device,
        width_name=width_name
    )
    
    activations = collect_activations(store, n_batches=n_batches)
    
    # Step 4: Calculate sparsity metrics
    print("\n[4/4] Calculating sparsity metrics...")
    
    sparsity_results = calculate_sparsity_metrics(
        sae=sae,
        activations=activations,
        n_batches=n_batches,
        activation_threshold=1e-5,
        percentiles=[75.0, 90.0]
    )
    
    # Prepare results JSON with config and sparsity metrics
    results = {
        'config': {
            'architecture': architecture,
            'width': width_name,
            'layer': layer,
            'trainer': trainer_folder,
            'model_name': metadata.get('model_name', 'unknown'),
            'hook_name': metadata.get('hook_name', 'unknown'),
            'd_in': sparsity_results['d_in'],
            'd_sae': sparsity_results['d_sae'],
        },
        'trainer_config': trainer_cfg,
        'sparsity_metrics': {
            'average_active_features': sparsity_results['average_active_features'],
            'effective_active_features_75': sparsity_results['effective_active_features_75'],
            'effective_active_features_90': sparsity_results['effective_active_features_90'],
            'n_samples': sparsity_results['n_samples'],
        }
    }
    
    # Add dead neuron metrics if calculated
    if 'dead_neuron_metrics' in sparsity_results:
        results['dead_neuron_metrics'] = sparsity_results['dead_neuron_metrics']
    
    # Add sparsity parameter info if found
    if key_param and param_value is not None:
        results['config']['sparsity_parameter'] = key_param
        results['config']['sparsity_parameter_value'] = param_value
    
    # Save results JSON
    results_file = output_dir_final / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved results to: {results_file}")
    
    # Clean up SAE cache after analysis
    print("\n[Cleanup] Removing cached SAE files...")
    cleanup_sae_cache(metadata['_cache_path'])
    
    # Aggressive GPU cleanup
    import torch
    import gc
    if torch.cuda.is_available():
        del store
        del sae
        del activations
        # Only delete model if we loaded it here (not if it was passed in for reuse)
        if model_loaded_here:
            del model
            model = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
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
    Run sparsity analysis on all canrager TopK+Vanilla SAEs.
    
    Args:
        device: Device to use (default: auto-detect cuda:0)
        n_batches: Number of batches for activation collection
        output_dir: Output directory
        skip_existing: Skip analyses that already have results.json
    """
    config = CANRAGER_CONFIG
    
    total_analyses = len(config['architectures']) * len(config['widths']) * len(config['layers']) * len(config['trainers'])
    completed = 0
    skipped = 0
    failed = []
    
    print("="*70)
    print("BATCH SPARSITY ANALYSIS: ALL CANRAGER TOPK+VANILLA SAEs")
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
        
        for architecture in config['architectures']:
            print(f"\n--- {architecture.upper()} Architecture ---")
            
            for layer in config['layers']:
                print(f"\n  Layer {layer}:")
                
                for trainer in config['trainers']:
                    # Check if already exists
                    if skip_existing and output_dir:
                        check_dir = output_dir / "canrager" / architecture / width_name / f"layer_{layer}"
                        # Check if results.json exists in trainer folder or any subfolder starting with trainer name
                        found_existing = False
                        if check_dir.exists():
                            # Check exact trainer folder first
                            trainer_dir = check_dir / trainer
                            if trainer_dir.exists() and (trainer_dir / "results.json").exists():
                                found_existing = True
                            else:
                                # Check subfolders starting with trainer name
                                for d in check_dir.iterdir():
                                    if d.is_dir() and d.name.startswith(trainer) and (d / "results.json").exists():
                                        found_existing = True
                                        break
                        if found_existing:
                            print(f"    ⏭️  {trainer}: Already exists, skipping...")
                            skipped += 1
                            continue
                    
                    print(f"    🔄 {trainer}...")
                    try:
                        run_analysis_canrager(
                            architecture=architecture,
                            width_name=width_name,
                            layer=layer,
                            trainer_folder=trainer,
                            device=device,
                            n_batches=n_batches,
                            output_dir=output_dir,
                            model_cache=model_cache
                        )
                        completed += 1
                        print(f"    ✅ {trainer}: Completed ({completed}/{total_analyses})")
                        
                        # Aggressive cleanup after each successful analysis
                        import torch
                        import gc
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
                        import torch
                        import gc
                        if torch.cuda.is_available():
                            gc.collect()
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
        
        # Clean up model and repo cache for this width after all SAEs processed
        if width_name in model_cache and model_cache[width_name] is not None:
            print(f"\n🧹 Cleaning up model and repo cache for {width_name} width...")
            import torch
            import gc
            from sae_family_load_convert.canrager_topk_vanilla import get_repo_id
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


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run sparsity analysis on canrager TopK+Vanilla SAEs"
    )
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (default: auto-detect cuda:0)')
    parser.add_argument('--n-batches', type=int, default=15,
                       help='Number of batches for activation collection (default: 15)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: /data/user/sae/benchmark/sparsity_dead_neuron)')
    parser.add_argument('--no-skip-existing', action='store_true',
                       help='Do not skip existing results')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    run_all_analyses(
        device=args.device,
        n_batches=args.n_batches,
        output_dir=output_dir,
        skip_existing=not args.no_skip_existing
    )

