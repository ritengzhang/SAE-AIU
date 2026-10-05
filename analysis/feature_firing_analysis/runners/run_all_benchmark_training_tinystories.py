#!/usr/bin/env python3
"""
Batch runner for ALL Benchmark Training TinyStories models for Feature Firing Analysis.

This script runs analysis3 (feature firing) on all 63 TinyStories combinations.

Usage:
    python -m feature_firing_analysis.runners.run_all_benchmark_training_tiny-stories --device cuda:1 --output-dir /data/user/sae/feature_firing_analysis
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
from feature_firing_analysis import (
    compute_firing_statistics,
    create_firing_frequency_distribution_plots,
    create_cumulative_magnitude_distribution_plots,
    create_firing_magnitude_plots,
    create_per_input_magnitude_analysis_simplified,
    create_feature_ranking_overlap_plots,
    create_top_features_frequency_plots,
)

# Import benchmark training utilities
from benchmark_training.load_benchmark_sae import (
    load_benchmark_sae,
    load_base_model_for_llm,
    parse_combo_id,
)
from benchmark_training.utils import load_combinations, filter_combinations_for_llm

# Import SAELens utilities
SAELENS_PATH = parent_dir / "SAELens-main"
if str(SAELENS_PATH) not in sys.path:
    sys.path.insert(0, str(SAELENS_PATH))

from sae_lens.training.activations_store import ActivationsStore
from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.saes.standard_sae import StandardTrainingSAEConfig

from encoder_decoder_activation_relationship.cleanup_utils import (
    cleanup_model_cache,
)


def setup_activations_store_for_benchmark(model, sae, metadata: dict, device: str = None, width_name: str = None) -> ActivationsStore:
    """Setup activations store for benchmark training SAEs."""
    if device is None:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/GPU is not available. GPU is required.")
        device = "cuda:0"
    
    print("Setting up activations store...")
    
    dummy_sae_cfg = StandardTrainingSAEConfig(
        d_in=sae.cfg.d_in,
        d_sae=sae.cfg.d_sae,
        device=device,
        dtype="float32",
    )
    
    # Adaptive buffer sizing
    if width_name == '16k':
        n_batches_in_buffer = 8
        store_batch_size_prompts = 4
    elif width_name == '65k':
        n_batches_in_buffer = 8
        store_batch_size_prompts = 4
    else:
        n_batches_in_buffer = 16
        store_batch_size_prompts = 8
    
    cfg = LanguageModelSAERunnerConfig(
        sae=dummy_sae_cfg,
        model_name=metadata.get('model_name', 'tiny-stories'),
        hook_name=metadata.get('hook_name', 'blocks.0.hook_mlp_out'),
        dataset_path=metadata.get('dataset_path', 'apollo-research/Skylion007-openwebtext-tokenizer-tiny-stories'),
        is_dataset_tokenized=True,
        streaming=True,
        train_batch_size_tokens=4096,
        context_size=512,
        n_batches_in_buffer=n_batches_in_buffer,
        training_tokens=4096 * 100,
        store_batch_size_prompts=store_batch_size_prompts,
        device=device,
        seed=42,
        dtype="float32",
    )
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            store = ActivationsStore.from_config(model, cfg)
            print(f"✓ Activations store ready")
            return store
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  ⚠️  Attempt {attempt + 1} failed: {str(e)[:100]}...")
                time.sleep(5)
            else:
                raise


def run_analysis_benchmark_training_tinystories(
    combo: dict,
    device: str = None,
    n_batches: int = 30,
    thresholds: list = None,
    output_dir: Path = None,
    model=None
):
    """Run feature firing analysis for a benchmark training TinyStories SAE."""
    import torch
    import gc
    
    if thresholds is None:
        thresholds = [0.0, 0.01]
    
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/GPU is not available. GPU is required.")
        device = "cuda:0"
        print(f"Using GPU: {device}")
    
    print("="*70)
    print("FEATURE FIRING ANALYSIS - BENCHMARK TRAINING TinyStories")
    print("="*70)
    
    if output_dir is None:
        output_dir = Path(os.environ.get("DATA_DIR", "/data/user")) / "sae" / "feature_firing_analysis"
    
    # Create combo_id
    combo_id = (
        f"{combo['architecture']}_L{combo['layer']}_"
        f"{combo['width']}_{combo['sparsity_param']}{combo['sparsity_value']}"
    )
    
    parsed = parse_combo_id(combo_id)
    width_name = parsed['width']
    architecture = parsed['architecture']
    
    # Output directory: benchmark_training_tiny-stories/{width}/{architecture}/{combo_id}/
    base_output_dir = output_dir / "benchmark_training_tiny-stories" / width_name / architecture
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
    sae, metadata = load_benchmark_sae(combo_id=combo_id, llm_name="tiny-stories", device=device)
    
    # Copy config.json
    sae_path = Path(metadata['sae_path'])
    config_json_path = sae_path / "cfg.json"
    if config_json_path.exists():
        output_config_path = output_dir_final / "config.json"
        shutil.copy2(config_json_path, output_config_path)
        print(f"✓ Copied config.json")
    
    # Step 2: Load base model (or reuse)
    print("\n[2/4] Loading base model...")
    if model is None:
        model = load_base_model_for_llm("tiny-stories", device=device)
    else:
        print("  Reusing existing model")
    
    # Step 3: Setup activation store
    print("\n[3/4] Setting up activation store...")
    store = setup_activations_store_for_benchmark(
        model=model, sae=sae, metadata=metadata, device=device, width_name=width_name
    )
    
    # Step 4: Compute firing statistics
    print("\n[4/4] Computing firing statistics...")
    firing_stats = compute_firing_statistics(
        sae=sae,
        store=store,
        n_batches=n_batches,
        thresholds=thresholds,
        device=device
    )
    
    # Create plots
    sae_name = combo_id
    stats_dict = {sae_name: firing_stats}
    
    # Firing frequency distribution plots
    print("\n[Analysis 1] Creating firing frequency distribution plots...")
    firing_freq_dir = output_dir_final / "firing_frequency_distribution"
    create_firing_frequency_distribution_plots(stats_dict=stats_dict, output_dir=firing_freq_dir)
    create_cumulative_magnitude_distribution_plots(stats_dict=stats_dict, output_dir=firing_freq_dir)
    create_top_features_frequency_plots(stats_dict=stats_dict, output_dir=firing_freq_dir)
    
    # Firing magnitude plots
    print("\n[Analysis 2] Creating firing magnitude plots...")
    magnitude_dir = output_dir_final / "firing_magnitude"
    create_firing_magnitude_plots(stats_dict=stats_dict, output_dir=magnitude_dir)
    
    # Per-input magnitude analysis
    print("\n[Analysis 3] Creating per-input magnitude analysis...")
    per_input_dir = output_dir_final / "per_input_magnitude"
    create_per_input_magnitude_analysis_simplified(
        sae=sae,
        store=store,
        stats_dict=stats_dict,
        output_dir=per_input_dir,
        device=device,
        n_batches_for_collection=n_batches
    )
    
    # Feature ranking overlap
    print("\n[Analysis 4] Creating feature ranking overlap plots...")
    overlap_dir = output_dir_final / "feature_ranking_overlap"
    create_feature_ranking_overlap_plots(stats_dict=stats_dict, output_dir=overlap_dir)
    
    # Save results JSON
    with open(results_file, 'w') as f:
        json.dump({
            'config': {
                'combo_id': combo_id,
                'architecture': architecture,
                'width': width_name,
                'sparsity_param': combo['sparsity_param'],
                'sparsity_value': combo['sparsity_value'],
            },
            'firing_stats': firing_stats
        }, f, indent=2)
    print(f"✓ Saved results to: {results_file}")
    
    # Cleanup
    print("\n[Cleanup] Cleaning up...")
    del store
    del sae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir_final}")
    return model


def run_all_analyses(
    device: str = None,
    n_batches: int = 30,
    thresholds: list = None,
    output_dir: Path = None,
    skip_existing: bool = True
):
    """Run feature firing analysis on all benchmark training TinyStories combinations."""
    combinations = load_combinations()
    tinystories_combinations = filter_combinations_for_llm(combinations, "tiny-stories")
    
    total_analyses = len(tinystories_combinations)
    completed = 0
    skipped = 0
    failed = []
    
    print("="*70)
    print("BATCH FEATURE FIRING ANALYSIS: ALL BENCHMARK TRAINING TinyStories MODELS")
    print("="*70)
    print(f"\nTotal analyses to run: {total_analyses}")
    print(f"Device: {device or 'auto-detect'}")
    
    start_time = time.time()
    
    if device is None:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/GPU is not available. GPU is required.")
        device = "cuda:0"
    
    model = load_base_model_for_llm("tiny-stories", device=device)
    
    try:
        for i, combo in enumerate(tinystories_combinations, 1):
            combo_id = (
                f"{combo['architecture']}_L{combo['layer']}_"
                f"{combo['width']}_{combo['sparsity_param']}{combo['sparsity_value']}"
            )
            
            if skip_existing and output_dir:
                parsed = parse_combo_id(combo_id)
                width_name = parsed['width']
                architecture = parsed['architecture']
                check_dir = output_dir / "benchmark_training_tiny-stories" / width_name / architecture / combo_id
                if (check_dir / "results.json").exists():
                    print(f"\n[{i}/{total_analyses}] ⏭️  Skipping (already exists): {combo_id}")
                    skipped += 1
                    continue
            
            print(f"\n[{i}/{total_analyses}] 🔄 Processing: {combo_id}")
            try:
                model = run_analysis_benchmark_training_tinystories(
                    combo=combo,
                    device=device,
                    n_batches=n_batches,
                    thresholds=thresholds,
                    output_dir=output_dir,
                    model=model
                )
                completed += 1
                print(f"  ✅ Completed ({completed}/{total_analyses})")
                
                import torch
                import gc
                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
            except Exception as e:
                error_msg = str(e)[:200]
                print(f"  ❌ Failed: {error_msg}")
                failed.append({'combo_id': combo_id, 'error': error_msg})
                
                import torch
                import gc
                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
    finally:
        print("\n🧹 Final cleanup...")
        del model
        import torch
        import gc
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        cleanup_model_cache("tiny-stories")
    
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
    
    parser = argparse.ArgumentParser(description="Run feature firing analysis on benchmark training TinyStories models")
    parser.add_argument('--device', type=str, default=None, help='Device to use (default: auto-detect cuda:0)')
    parser.add_argument('--n-batches', type=int, default=30, help='Number of batches (default: 30)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--no-skip-existing', action='store_true', help='Do not skip existing results')
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    run_all_analyses(
        device=args.device,
        n_batches=args.n_batches,
        output_dir=output_dir,
        skip_existing=not args.no_skip_existing
    )
