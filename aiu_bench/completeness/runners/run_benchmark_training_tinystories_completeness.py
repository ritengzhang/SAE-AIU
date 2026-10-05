#!/usr/bin/env python3
"""
Completeness Analysis Runner for Benchmark Training TinyStories

This script calculates completeness metrics for each benchmark training TinyStories run:
1. Mean Squared Error (MSE) - direct reconstruction fidelity
2. Cross-Entropy Loss Recovery - model performance preservation
3. KL Divergence - model behavior distribution preservation

Results are saved in results.json with config information.

Output structure: benchmark_training_tinystories/{width}/{architecture}/{combo_id}/results.json

Usage:
    python -m aiu_bench.completeness.runners.run_benchmark_training_tinystories_completeness --device cuda:1 --output-dir <DATA_DIR>/sae/benchmark/completeness
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
    setup_activations_store_for_benchmark,
)

# Import benchmark training utilities
from benchmark_training.load_benchmark_sae import (
    load_benchmark_sae,
    load_base_model_for_llm,
    parse_combo_id,
)
from benchmark_training.utils import load_combinations, filter_combinations_for_llm
from benchmark_config import OUTPUT_BASE_DIR

# Import SAELens utilities
SAELENS_PATH = parent_dir / "SAELens-main"
if str(SAELENS_PATH) not in sys.path:
    sys.path.insert(0, str(SAELENS_PATH))

from analysis2_encoder_decoder_activation_relationship.cleanup_utils import (
    cleanup_model_cache,
)


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


def run_analysis_benchmark_training_tinystories(
    combo: dict,
    device: str = None,
    n_batches: int = 30,
    eval_batch_size_prompts: int = 8,
    output_dir: Path = None,
    model=None  # Optional: reuse existing model
):
    """
    Run completeness analysis for a benchmark training TinyStories SAE.
    
    Args:
        combo: Combination dictionary from combinations.json
        device: Device to use (default: auto-detect cuda:0)
        n_batches: Number of batches for activation collection and downstream metrics
        eval_batch_size_prompts: Batch size for downstream evaluation
        output_dir: Output directory (default: {DATA_DIR}/sae/benchmark/completeness/, where DATA_DIR defaults to /data/user)
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
    print("COMPLETENESS ANALYSIS - BENCHMARK TRAINING TINYSTORIES")
    print("="*70)
    
    # Setup output directory
    if output_dir is None:
        data_dir = os.environ.get("DATA_DIR", "/data/user")
        output_dir = Path(data_dir) / "sae" / "benchmark" / "completeness"
    
    # Create combo_id
    combo_id = (
        f"{combo['architecture']}_L{combo['layer']}_"
        f"{combo['width']}_{combo['sparsity_param']}{combo['sparsity_value']}"
    )
    
    # Parse combo to get width and architecture for output structure
    parsed = parse_combo_id(combo_id)
    width_name = parsed['width']
    architecture = parsed['architecture']
    
    # Create output directory: benchmark_training_tinystories/{width}/{architecture}/{combo_id}/
    base_output_dir = output_dir / "benchmark_training_tinystories" / width_name / architecture
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    output_dir_final = base_output_dir / combo_id
    output_dir_final.mkdir(parents=True, exist_ok=True)
    
    # Check if already exists with all metrics
    results_file = output_dir_final / "results.json"
    if results_file.exists() and has_all_metrics(results_file):
        print(f"\n⏭️  Skipping (already exists with all metrics): {combo_id}")
        return model
    
    # Step 1: Load SAE
    print(f"\n[1/5] Loading SAE: {combo_id}...")
    sae, metadata = load_benchmark_sae(
        combo_id=combo_id,
        llm_name="tiny-stories",
        device=device
    )
    
    # Copy config.json to output folder if it exists
    sae_path = Path(metadata['sae_path'])
    config_json_path = sae_path / "cfg.json"
    if config_json_path.exists():
        import shutil
        output_config_path = output_dir_final / "config.json"
        shutil.copy2(config_json_path, output_config_path)
        print(f"✓ Copied config.json to output folder")
    
    # Step 2: Load base model (or reuse)
    print("\n[2/5] Loading base model...")
    if model is None:
        model = load_base_model_for_llm("tiny-stories", device=device)
    else:
        print("  Reusing existing model")
    
    # Step 3: Setup activation store
    print("\n[3/5] Setting up activation store...")
    store = setup_activations_store_for_benchmark(
        model=model,
        sae=sae,
        metadata=metadata,
        device=device,
        width_name=width_name
    )
    
    # Step 4: Collect activations for MSE calculation
    print("\n[4/5] Collecting activations for MSE...")
    activations = collect_activations(store, n_batches=n_batches)
    
    # Step 5: Calculate completeness metrics
    print("\n[5/5] Calculating completeness metrics...")
    
    # Get hook_name from metadata
    hook_name = metadata.get('hook_name', 'blocks.0.hook_mlp_out')
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
        ignore_tokens=set(),  # Can be customized
    )
    
    # Prepare results dictionary
    results = {
        'config': {
            'llm': 'tiny-stories',
            'combo_id': combo_id,
            'architecture': architecture,
            'layer': combo['layer'],
            'width': width_name,
            'd_sae': combo['d_sae'],
            'sparsity_param': combo['sparsity_param'],
            'sparsity_value': combo['sparsity_value'],
            'model_name': metadata.get('model_name', 'tiny-stories-1L-21M'),
            'hook_name': hook_name,
            'hook_head_index': hook_head_index,
            'd_in': completeness_results['d_in'],
        },
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
    
    # Save results JSON (merge with existing if present)
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
        print(f"  ℹ️  Merged with existing results.json")
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved results to: {results_file}")
    
    # Cleanup
    print("\n[Cleanup] Cleaning up...")
    del store
    del sae
    del activations
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"✓ Cleared GPU cache (device: {device})")
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir_final}")
    
    return model  # Return model for reuse


def run_all_analyses(
    device: str = None,
    n_batches: int = 30,
    eval_batch_size_prompts: int = 8,
    output_dir: Path = None,
    skip_existing: bool = True
):
    """
    Run completeness analysis on all benchmark training TinyStories combinations.
    
    Args:
        device: Device to use (default: auto-detect cuda:0)
        n_batches: Number of batches for activation collection and downstream metrics
        eval_batch_size_prompts: Batch size for downstream evaluation
        output_dir: Output directory
        skip_existing: Skip analyses that already have results.json
    """
    # Load all combinations
    combinations = load_combinations()
    tinystories_combinations = filter_combinations_for_llm(combinations, "tiny-stories")
    
    total_analyses = len(tinystories_combinations)
    completed = 0
    skipped = 0
    failed = []
    
    print("="*70)
    print("BATCH COMPLETENESS ANALYSIS: ALL BENCHMARK TRAINING TINYSTORIES MODELS")
    print("="*70)
    print(f"\nTotal analyses to run: {total_analyses}")
    print(f"Device: {device or 'auto-detect'}")
    print(f"n_batches: {n_batches}")
    print(f"eval_batch_size_prompts: {eval_batch_size_prompts}")
    
    start_time = time.time()
    
    # Load model once and reuse
    print("\n[Initial] Loading base model (will reuse for all analyses)...")
    if device is None:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/GPU is not available. GPU is required for analysis.")
        device = "cuda:0"
    
    model = load_base_model_for_llm("tiny-stories", device=device)
    
    try:
        for i, combo in enumerate(tinystories_combinations, 1):
            combo_id = (
                f"{combo['architecture']}_L{combo['layer']}_"
                f"{combo['width']}_{combo['sparsity_param']}{combo['sparsity_value']}"
            )
            
            # Check if already exists with all metrics
            if skip_existing and output_dir:
                parsed = parse_combo_id(combo_id)
                width_name = parsed['width']
                architecture = parsed['architecture']
                check_dir = output_dir / "benchmark_training_tinystories" / width_name / architecture / combo_id
                results_file = check_dir / "results.json"
                if results_file.exists() and has_all_metrics(results_file):
                    print(f"\n[{i}/{total_analyses}] ⏭️  Skipping (already exists with all metrics): {combo_id}")
                    skipped += 1
                    continue
            
            print(f"\n[{i}/{total_analyses}] 🔄 Processing: {combo_id}")
            try:
                model = run_analysis_benchmark_training_tinystories(
                    combo=combo,
                    device=device,
                    n_batches=n_batches,
                    eval_batch_size_prompts=eval_batch_size_prompts,
                    output_dir=output_dir,
                    model=model  # Reuse model
                )
                completed += 1
                print(f"  ✅ Completed ({completed}/{total_analyses})")
                
                # Aggressive cleanup after each successful analysis
                import torch
                import gc
                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
            except Exception as e:
                error_msg = str(e)[:200]
                print(f"  ❌ Failed: {error_msg}")
                failed.append({
                    'combo_id': combo_id,
                    'error': error_msg
                })
                
                # Aggressive cleanup after failure too
                import torch
                import gc
                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
    finally:
        # Cleanup model at the end
        print("\n🧹 Final cleanup...")
        del model
        import torch
        import gc
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        cleanup_model_cache("tiny-stories-1L-21M")
        print("✓ Cleaned up model cache and GPU cache")
    
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
            print(f"  - {f['combo_id']}: {f['error']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run completeness analysis on benchmark training TinyStories models"
    )
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (default: auto-detect cuda:0)')
    parser.add_argument('--n-batches', type=int, default=30,
                       help='Number of batches for activation collection and downstream metrics (default: 30)')
    parser.add_argument('--eval-batch-size-prompts', type=int, default=8,
                       help='Batch size for downstream evaluation (default: 8)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: /data/user/sae/benchmark/completeness). '
                            'IMPORTANT: Must be provided for skip_existing to work.')
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
