#!/usr/bin/env python3
"""
Completeness Analysis Runner for canrager TopK+Vanilla SAEs

This script calculates completeness metrics for each canrager SAE:
1. Mean Squared Error (MSE) - direct reconstruction fidelity
2. Cross-Entropy Loss Recovery - model performance preservation
3. KL Divergence - model behavior distribution preservation

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

# Add parent directories to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

# Import analysis functions
from aiu_bench.completeness.completeness_analysis import (
    calculate_completeness_metrics,
    collect_activations,
)

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
, eval_batch_size_prompts: int = 8):
    """
    Run completeness analysis for canrager SAE.
    
    Args:
        architecture: 'topk' or 'vanilla'
        width_name: '4k', '16k', or '65k'
        layer: Layer number (5, 12, or 19)
        trainer_folder: Trainer folder (e.g., "trainer_0")
        device: Device to use (default: auto-detect cuda:0, requires GPU)
        n_batches: Number of batches for activation collection
        output_dir: Output directory (default: {DATA_DIR}/sae/benchmark/completeness/, where DATA_DIR defaults to /data/user)
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
    print("COMPLETENESS ANALYSIS - CANRAGER TOPK+VANILLA")
    print("="*70)
    
    # Setup output directory: canrager/{architecture}/{width}/{layer}/{trainer}/
    if output_dir is None:
        data_dir = os.environ.get("DATA_DIR", "/data/user")
        output_dir = Path(data_dir) / "sae" / "benchmark" / "completeness"
    
    # Create base output directory
    base_output_dir = output_dir / "canrager" / architecture / width_name / f"layer_{layer}"
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load SAE
    print("\n[1/5] Loading SAE...")
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
    print("\n[2/5] Loading base model...")
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
    print("\n[3/5] Collecting activations...")
    store = setup_activations_store(
        model=model,
        sae=sae,
        metadata=metadata,
        device=device,
        width_name=width_name
    )
    
    activations = collect_activations(store, n_batches=n_batches)
    
    # Step 4: Calculate completeness metrics
    print("\n[4/5] Calculating completeness metrics...")
    
    # Get hook_name from metadata
    hook_name = metadata.get('hook_name', 'unknown')
    hook_head_index = metadata.get('hook_head_index', None)
    
    completeness_results = calculate_completeness_metrics(
        sae=sae,
        model=model,
        activations=activations,
        activation_store=store,
        hook_name=hook_name,
        hook_head_index=hook_head_index,
        n_batches=n_batches,
        eval_batch_size_prompts=eval_batch_size_prompts,
        ignore_tokens=set(),
    )
    
    # Prepare results JSON with config and completeness metrics
    results = {
        'config': {
            'architecture': architecture,
            'width': width_name,
            'layer': layer,
            'trainer': trainer_folder,
            'model_name': metadata.get('model_name', 'unknown'),
            'hook_name': hook_name,
            'd_in': completeness_results['d_in'],
            'd_sae': completeness_results['d_sae'],
        },
        'trainer_config': trainer_cfg,
        'completeness_metrics': {
            'mse': completeness_results['mse'],
            # CE loss and KL divergence metrics commented out (MSE only for now)
            # 'ce_loss_without_sae': completeness_results.get('ce_loss_without_sae'),
            # 'ce_loss_with_sae': completeness_results.get('ce_loss_with_sae'),
            # 'ce_loss_with_ablation': completeness_results.get('ce_loss_with_ablation'),
            # 'ce_loss_score': completeness_results.get('ce_loss_score'),
            # 'kl_div_with_sae': completeness_results.get('kl_div_with_sae'),
            # 'kl_div_with_ablation': completeness_results.get('kl_div_with_ablation'),
            # 'kl_div_score': completeness_results.get('kl_div_score'),
            'n_samples': completeness_results['n_samples'],
        }
    }
    
    # Add sparsity parameter info if found
    if key_param and param_value is not None:
        results['config']['sparsity_parameter'] = key_param
        results['config']['sparsity_parameter_value'] = param_value
    
    # Save results JSON (merge with existing if present)
    results_file = output_dir_final / "results.json"
    if results_file.exists():
        # Load existing results and merge
        with open(results_file, 'r') as f:
            existing_results = json.load(f)
        # Merge completeness_metrics (new metrics override old ones, but keep existing ones)
        if 'completeness_metrics' in existing_results:
            existing_results['completeness_metrics'].update(results['completeness_metrics'])
            results['completeness_metrics'] = existing_results['completeness_metrics']
        # Merge config (new config overrides old one)
        if 'config' in existing_results:
            existing_results['config'].update(results['config'])
            results['config'] = existing_results['config']
        # Keep trainer_config from existing if present
        if 'trainer_config' in existing_results and 'trainer_config' not in results:
            results['trainer_config'] = existing_results['trainer_config']
        print(f"  ℹ️  Merged with existing results.json")
    
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


def has_all_metrics(results_file: Path) -> bool:
    """
    Check if results.json has all required completeness metrics.
    Since CE/KL computation is currently commented out, only check for MSE.
    Returns True if MSE is present, False otherwise.
    """
    if not results_file.exists():
        return False
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        metrics = results.get('completeness_metrics', {})
        # Only check for MSE since CE/KL computation is commented out
        return 'mse' in metrics and metrics['mse'] is not None
    except (json.JSONDecodeError, KeyError, Exception):
        return False


def run_all_analyses(
        device=None,
        n_batches=30,
        eval_batch_size_prompts=8,
        output_dir=None,
        skip_existing=True
    ):
    """
    Run completeness analysis on all canrager TopK+Vanilla SAEs.
    
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
    print("BATCH COMPLETENESS ANALYSIS: ALL CANRAGER TOPK+VANILLA SAEs")
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
                    # Check if already exists with all metrics
                    if skip_existing and output_dir:
                        check_dir = output_dir / "canrager" / architecture / width_name / f"layer_{layer}"
                        found_complete = False
                        # Check trainer folder directly
                        trainer_dir = check_dir / trainer
                        if (trainer_dir / "results.json").exists() and has_all_metrics(trainer_dir / "results.json"):
                            print(f"    ⏭️  {trainer}: Already exists with all metrics, skipping...")
                            skipped += 1
                            found_complete = True
                        else:
                            # Check subdirectories starting with trainer name
                            for subdir in check_dir.iterdir():
                                if subdir.is_dir() and subdir.name.startswith(trainer):
                                    results_file = subdir / "results.json"
                                    if results_file.exists() and has_all_metrics(results_file):
                                        print(f"    ⏭️  {trainer}: Already exists with all metrics, skipping...")
                                        skipped += 1
                                        found_complete = True
                                        break
                        if found_complete:
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
                        , eval_batch_size_prompts=eval_batch_size_prompts)
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
        description="Run completeness analysis on canrager TopK+Vanilla SAEs"
    )
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (default: auto-detect cuda:0)')
    parser.add_argument('--n-batches', type=int, default=15,
                       help='Number of batches for activation collection and downstream metrics (default: 15)')
    parser.add_argument('--eval-batch-size-prompts', type=int, default=8,
                       help='Batch size for downstream evaluation (default: 8)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: {DATA_DIR}/sae/benchmark/completeness, where DATA_DIR defaults to /data/user)')
    parser.add_argument('--no-skip-existing', action='store_true',
                       help='Do not skip existing results')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    run_all_analyses(
        device=args.device,
        n_batches=args.n_batches,
        eval_batch_size_prompts=args.eval_batch_size_prompts,
        output_dir=output_dir,
        skip_existing=not args.no_skip_existing
    )

