#!/usr/bin/env python3
"""
Sparsity Analysis Runner for Benchmark Training OPT-125m

This script calculates sparsity metrics for each benchmark training OPT-125m run:
1. Average active features (simple counting)
2. Effective active features (features summing to 75% and 90% of total activation)

Results are saved in results.json with config information.

Output structure: benchmark_training_opt125m/{width}/{architecture}/{combo_id}/results.json

Usage:
    python -m aiu_bench.sparsity.runners.run_benchmark_training_opt125m_sparsity --device cuda:1 --output-dir /data/user/sae/benchmark/sparsity
"""

import sys
import os
from pathlib import Path
import json
import time

# Set cache directories to /data/user/cache instead of home folder
os.environ['HF_HOME'] = '/data/user/cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/data/user/cache/huggingface/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/data/user/cache/huggingface/transformers'
os.environ['XDG_CACHE_HOME'] = '/data/user/cache'
# Ensure cache directories exist
Path('/data/user/cache/huggingface').mkdir(parents=True, exist_ok=True)
Path('/data/user/cache/huggingface/datasets').mkdir(parents=True, exist_ok=True)
Path('/data/user/cache/huggingface/transformers').mkdir(parents=True, exist_ok=True)

# Add parent directories to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

# Import analysis functions
from aiu_bench.sparsity_dead_neuron import (
    calculate_sparsity_metrics,
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


def run_analysis_benchmark_training_opt125m(
    combo: dict,
    device: str = None,
    n_batches: int = 30,
    output_dir: Path = None,
    model=None  # Optional: reuse existing model
):
    """
    Run sparsity analysis for a benchmark training OPT-125m SAE.
    
    Args:
        combo: Combination dictionary from combinations.json
        device: Device to use (default: auto-detect cuda:0)
        n_batches: Number of batches for activation collection
        output_dir: Output directory (default: /data/user/sae/benchmark/sparsity_dead_neuron/)
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
    print("SPARSITY ANALYSIS - BENCHMARK TRAINING OPT-125M")
    print("="*70)
    
    # Setup output directory
    if output_dir is None:
        data_dir = os.environ.get("DATA_DIR", "/data/user")
        output_dir = Path(data_dir) / "sae" / "benchmark" / "sparsity_dead_neuron"
    
    # Create combo_id
    combo_id = (
        f"{combo['architecture']}_L{combo['layer']}_"
        f"{combo['width']}_{combo['sparsity_param']}{combo['sparsity_value']}"
    )
    
    # Parse combo to get width and architecture for output structure
    parsed = parse_combo_id(combo_id)
    width_name = parsed['width']
    architecture = parsed['architecture']
    
    # Create output directory: benchmark_training_opt125m/{width}/{architecture}/{combo_id}/
    base_output_dir = output_dir / "benchmark_training_opt125m" / width_name / architecture
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    output_dir_final = base_output_dir / combo_id
    output_dir_final.mkdir(parents=True, exist_ok=True)
    
    # Check if already exists
    results_file = output_dir_final / "results.json"
    if results_file.exists():
        print(f"\n⏭️  Skipping (already exists): {combo_id}")
        return model
    
    # Step 1: Load SAE
    print(f"\n[1/4] Loading SAE: {combo_id}...")
    sae, metadata = load_benchmark_sae(
        combo_id=combo_id,
        llm_name="opt-125m",
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
    print("\n[2/4] Loading base model...")
    if model is None:
        model = load_base_model_for_llm("opt-125m", device=device)
    else:
        print("  Reusing existing model")
    
    # Step 3: Setup activation store and collect activations
    print("\n[3/4] Collecting activations...")
    store = setup_activations_store_for_benchmark(
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
    
    # Prepare results JSON
    results = {
        'config': {
            'llm': 'opt-125m',
            'combo_id': combo_id,
            'architecture': architecture,
            'layer': combo['layer'],
            'width': width_name,
            'd_sae': combo['d_sae'],
            'sparsity_param': combo['sparsity_param'],
            'sparsity_value': combo['sparsity_value'],
            'model_name': metadata.get('model_name', 'opt-125m-1L-21M'),
            'hook_name': metadata.get('hook_name', 'blocks.6.hook_mlp_out'),
            'd_in': sparsity_results['d_in'],
        },
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
    
    # Save results JSON
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
    output_dir: Path = None,
    skip_existing: bool = True
):
    """
    Run sparsity analysis on all benchmark training OPT-125m combinations.
    
    Args:
        device: Device to use (default: auto-detect cuda:0)
        n_batches: Number of batches for activation collection
        output_dir: Output directory
        skip_existing: Skip analyses that already have results.json
    """
    # Load all combinations
    combinations = load_combinations()
    opt125m_combinations = filter_combinations_for_llm(combinations, "opt-125m")
    
    total_analyses = len(opt125m_combinations)
    completed = 0
    skipped = 0
    failed = []
    
    print("="*70)
    print("BATCH SPARSITY ANALYSIS: ALL BENCHMARK TRAINING OPT-125M MODELS")
    print("="*70)
    print(f"\nTotal analyses to run: {total_analyses}")
    print(f"Device: {device or 'auto-detect'}")
    print(f"n_batches: {n_batches}")
    
    start_time = time.time()
    
    # Load model once and reuse
    print("\n[Initial] Loading base model (will reuse for all analyses)...")
    if device is None:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/GPU is not available. GPU is required for analysis.")
        device = "cuda:0"
    
    model = load_base_model_for_llm("opt-125m", device=device)
    
    try:
        for i, combo in enumerate(opt125m_combinations, 1):
            combo_id = (
                f"{combo['architecture']}_L{combo['layer']}_"
                f"{combo['width']}_{combo['sparsity_param']}{combo['sparsity_value']}"
            )
            
            # Check if already exists
            if skip_existing and output_dir:
                parsed = parse_combo_id(combo_id)
                width_name = parsed['width']
                architecture = parsed['architecture']
                check_dir = output_dir / "benchmark_training_opt125m" / width_name / architecture / combo_id
                if (check_dir / "results.json").exists():
                    print(f"\n[{i}/{total_analyses}] ⏭️  Skipping (already exists): {combo_id}")
                    skipped += 1
                    continue
            
            print(f"\n[{i}/{total_analyses}] 🔄 Processing: {combo_id}")
            try:
                model = run_analysis_benchmark_training_opt125m(
                    combo=combo,
                    device=device,
                    n_batches=n_batches,
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
        cleanup_model_cache("opt-125m-1L-21M")
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
        description="Run sparsity analysis on benchmark training OPT-125m models"
    )
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (default: auto-detect cuda:0)')
    parser.add_argument('--n-batches', type=int, default=30,
                       help='Number of batches for activation collection (default: 30)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: /data/user/sae/benchmark/sparsity). '
                            'IMPORTANT: Must be provided for skip_existing to work.')
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
