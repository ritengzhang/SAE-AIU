#!/usr/bin/env python3
"""
Batch runner for ALL Benchmark Training OPT-125m models for Frequency Quantile Redundancy Analysis.

This script runs analysis6 (frequency quantile redundancy) on all 63 OPT-125m combinations.

Usage:
    python -m aiu_bench.redundancy.runners.run_all_benchmark_training_opt125m --device cuda:1 --output-dir /data/user/sae/benchmark/redundancy
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
from aiu_bench.redundancy import (
    split_features_by_frequency_quantiles,
    analyze_quantile_redundancy,
    create_8_panel_quantile_heatmaps,
    create_quantile_comparison_barplot,
    compare_redundant_vs_nonredundant_by_quantile,
    create_redundant_vs_nonredundant_plot_by_quantile,
    # Dominance-based functions
    compute_dominance_scores,
    split_features_by_dominance_quantiles,
    create_8_panel_quantile_heatmaps_dominance,
    create_quantile_comparison_barplot_dominance,
    compare_redundant_vs_nonredundant_by_quantile_dominance,
    create_redundant_vs_nonredundant_plot_by_quantile_dominance,
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

from analysis2_encoder_decoder_activation_relationship.cleanup_utils import (
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
        model_name=metadata.get('model_name', 'opt-125m'),
        hook_name=metadata.get('hook_name', 'blocks.6.hook_mlp_out'),
        dataset_path=metadata.get('dataset_path', 'apollo-research/Skylion007-openwebtext-tokenizer-opt-125m'),
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


def collect_activations_and_compute_firing_stats(store, sae, n_batches: int = 3):
    """Collect activations and compute firing statistics."""
    print(f"\n📊 Collecting {n_batches} batches of activations and computing firing stats...")
    
    import torch
    feature_firing_counts = np.zeros(sae.cfg.d_sae, dtype=np.int64)
    total_activations = 0
    
    for batch_idx in range(n_batches):
        batch_activations = store.next_batch()
        batch_size = batch_activations.shape[0]
        total_activations += batch_size
        
        with torch.no_grad():
            feature_acts = sae.encode(batch_activations)
        
        feature_acts_np = feature_acts.cpu().numpy()
        fired = (feature_acts_np > 0).astype(np.int64)
        feature_firing_counts += fired.sum(axis=0)
    
    feature_rank_by_firing = np.argsort(feature_firing_counts)[::-1].tolist()
    
    firing_stats = {
        'feature_rank_by_firing': feature_rank_by_firing,
        'feature_firing_counts': feature_firing_counts.tolist(),
        'total_activations': total_activations,
    }
    
    print(f"✓ Collected {total_activations:,} activation vectors")
    
    return firing_stats


def run_analysis_benchmark_training_opt125m(
    combo: dict,
    device: str = None,
    n_batches: int = 3,
    top_n_per_quantile: int = 3000,
    similarity_thresholds: list = None,
    output_dir: Path = None,
    model=None
):
    """Run frequency quantile redundancy analysis for a benchmark training OPT-125m SAE."""
    import torch
    import gc
    
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/GPU is not available. GPU is required.")
        device = "cuda:0"
        print(f"Using GPU: {device}")
    
    print("="*70)
    print("FREQUENCY QUANTILE REDUNDANCY ANALYSIS - BENCHMARK TRAINING OPT-125m")
    print("="*70)
    
    if output_dir is None:
        data_dir = os.environ.get("DATA_DIR", "/data/user")
        output_dir = Path(data_dir) / "sae" / "benchmark" / "redundancy"
    
    combo_id = (
        f"{combo['architecture']}_L{combo['layer']}_"
        f"{combo['width']}_{combo['sparsity_param']}{combo['sparsity_value']}"
    )
    
    parsed = parse_combo_id(combo_id)
    width_name = parsed['width']
    architecture = parsed['architecture']
    
    base_output_dir = output_dir / "benchmark_training_opt-125m" / width_name / architecture
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    output_dir_final = base_output_dir / combo_id
    output_dir_final.mkdir(parents=True, exist_ok=True)
    
    # Check if both frequency and dominance results exist
    frequency_results_file = output_dir_final / "frequency_quantile_redundancy" / "results.json"
    dominance_results_file = output_dir_final / "dominance_quantile_redundancy" / "results.json"
    if frequency_results_file.exists() and dominance_results_file.exists():
        print(f"\n⏭️  Skipping (already exists): {combo_id}")
        return model
    
    # Step 1: Load SAE
    print(f"\n[1/5] Loading SAE: {combo_id}...")
    sae, metadata = load_benchmark_sae(combo_id=combo_id, llm_name="opt-125m", device=device)
    
    # Extract decoder weights
    if hasattr(sae, 'W_dec'):
        decoder_weights = sae.W_dec.detach().cpu().numpy()
    else:
        raise ValueError("Could not find decoder weights in SAE")
    
    # Copy config.json
    sae_path = Path(metadata['sae_path'])
    config_json_path = sae_path / "cfg.json"
    if config_json_path.exists():
        output_config_path = output_dir_final / "config.json"
        shutil.copy2(config_json_path, output_config_path)
        print(f"✓ Copied config.json")
    
    # Step 2: Load base model (or reuse)
    print("\n[2/5] Loading base model...")
    if model is None:
        model = load_base_model_for_llm("opt-125m", device=device)
    else:
        print("  Reusing existing model")
    
    # Step 3: Compute firing statistics
    print("\n[3/5] Collecting activations and computing firing stats...")
    store = setup_activations_store_for_benchmark(
        model=model, sae=sae, metadata=metadata, device=device, width_name=width_name
    )
    
    firing_stats = collect_activations_and_compute_firing_stats(store, sae, n_batches=n_batches)
    
    # Step 4: Split features by frequency quantiles
    print("\n[4/5] Splitting features by frequency quantiles...")
    quantiles = split_features_by_frequency_quantiles(
        firing_stats=firing_stats,
        top_n_per_quantile=3000  # Default from SAEBench
    )
    
    # Step 5: Analyze quantile redundancy
    print("\n[5/5] Analyzing quantile redundancy...")
    
    # Get encoder weights
    if hasattr(sae, 'W_enc'):
        encoder_weights = sae.W_enc.detach().cpu().numpy()  # [d_in, d_sae]
    else:
        raise ValueError("Could not find encoder weights")
    
    # Analyze each quantile
    similarity_thresholds = [0.7, 0.9]  # Default from function signature
    quantile_results = {}
    for quantile in [1, 2, 3, 4]:
        print(f"\n  Analyzing Q{quantile}...")
        quantile_results[quantile] = analyze_quantile_redundancy(
            decoder_weights=decoder_weights,
            encoder_weights=encoder_weights,
            quantile_features=quantiles[quantile],
            similarity_thresholds=similarity_thresholds
        )
    
    # Create output subdirectory for frequency (like dominance has)
    frequency_output_dir = output_dir_final / "frequency_quantile_redundancy"
    frequency_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plots
    print("\n[Plots] Creating visualization plots...")
    
    # 8-panel quantile heatmaps
    create_8_panel_quantile_heatmaps(
        quantile_results=quantile_results,
        output_dir=frequency_output_dir,
        sae_name=combo_id
    )
    
    # Quantile comparison barplot
    create_quantile_comparison_barplot(
        quantile_results=quantile_results,
        similarity_thresholds=similarity_thresholds,
        output_dir=frequency_output_dir,
        sae_name=combo_id
    )
    
    # Compare redundant vs non-redundant by quantile
    comparison_results = compare_redundant_vs_nonredundant_by_quantile(
        decoder_weights=decoder_weights,
        encoder_weights=encoder_weights,
        firing_stats=firing_stats,
        similarity_thresholds=similarity_thresholds,
        top_n_per_quantile=3000,
        sample_size=1000
    )
    
    create_redundant_vs_nonredundant_plot_by_quantile(
        comparison_results=comparison_results,
        similarity_thresholds=similarity_thresholds,
        output_dir=frequency_output_dir,
        sae_name=combo_id
    )
    
    # Save results JSON
    results_file = frequency_output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'config': {
                'combo_id': combo_id,
                'architecture': architecture,
                'width': width_name,
                'sparsity_param': combo['sparsity_param'],
                'sparsity_value': combo['sparsity_value'],
            },
            'quantile_results': {
                str(q): {
                    'n_features': quantile_results[q]['n_features'],
                    'redundant_pairs': quantile_results[q]['redundant_pairs'],
                    'avg_encoder_pairwise_similarity': quantile_results[q]['avg_encoder_pairwise_similarity'],
                    'avg_decoder_pairwise_similarity': quantile_results[q]['avg_decoder_pairwise_similarity'],
                }
                for q in [1, 2, 3, 4]
            },
            'redundant_vs_nonredundant_comparison_by_quantile': comparison_results,
        }, f, indent=2, default=str)
    print(f"✓ Saved results to: {results_file}")
    
    # ========================================================================
    # DOMINANCE QUANTILE REDUNDANCY ANALYSIS
    # ========================================================================
    print("\n" + "="*70)
    print("RUNNING DOMINANCE QUANTILE REDUNDANCY ANALYSIS")
    print("="*70)
    
    # Step 1: Compute dominance scores
    print("\n[1/5] Computing dominance scores...")
    # Reset store to beginning for dominance computation
    store = setup_activations_store_for_benchmark(
        model=model, sae=sae, metadata=metadata, device=device, width_name=width_name
    )
    
    dominance_scores, dominance_stats = compute_dominance_scores(
        sae=sae,
        store=store,
        n_batches=n_batches,
        use_absolute_value=True,
        device=device
    )
    
    # Step 2: Split features by dominance quantiles
    print("\n[2/5] Splitting features by dominance quantiles...")
    dominance_quantiles = split_features_by_dominance_quantiles(
        dominance_scores=dominance_scores,
        top_n_per_quantile=3000
    )
    
    # Step 3: Run dominance quantile redundancy analysis
    print("\n[3/5] Running dominance quantile redundancy analysis...")
    
    # Analyze each quantile
    dominance_quantile_results = {}
    for quantile in [1, 2, 3, 4]:
        print(f"\n  Analyzing Q{quantile}...")
        dominance_quantile_results[quantile] = analyze_quantile_redundancy(
            decoder_weights=decoder_weights,
            encoder_weights=encoder_weights,
            quantile_features=dominance_quantiles[quantile],
            similarity_thresholds=similarity_thresholds
        )
    
    # Create output subdirectory for dominance
    dominance_output_dir = output_dir_final / "dominance_quantile_redundancy"
    
    # Create plots
    # Plot 1: 8-panel heatmaps
    create_8_panel_quantile_heatmaps_dominance(
        quantile_results=dominance_quantile_results,
        output_dir=dominance_output_dir,
        sae_name=combo_id
    )
    
    # Plot 2: Quantile comparison barplot
    create_quantile_comparison_barplot_dominance(
        quantile_results=dominance_quantile_results,
        similarity_thresholds=similarity_thresholds,
        output_dir=dominance_output_dir,
        sae_name=combo_id
    )
    
    # Compare redundant vs non-redundant by quantile
    dominance_comparison_by_quantile = compare_redundant_vs_nonredundant_by_quantile_dominance(
        decoder_weights=decoder_weights,
        encoder_weights=encoder_weights,
        dominance_scores=dominance_scores,
        similarity_thresholds=similarity_thresholds,
        top_n_per_quantile=3000,
        sample_size=1000
    )
    
    # Plot 3: Redundant vs non-redundant by quantile
    create_redundant_vs_nonredundant_plot_by_quantile_dominance(
        comparison_results=dominance_comparison_by_quantile,
        similarity_thresholds=similarity_thresholds,
        output_dir=dominance_output_dir,
        sae_name=combo_id
    )
    
    # Save results JSON
    dominance_results_file = dominance_output_dir / "results.json"
    with open(dominance_results_file, 'w') as f:
        json.dump({
            'config': {
                'combo_id': combo_id,
                'architecture': architecture,
                'width': width_name,
                'sparsity_param': combo['sparsity_param'],
                'sparsity_value': combo['sparsity_value'],
            },
            'quantile_results': {
                str(q): {
                    'n_features': dominance_quantile_results[q]['n_features'],
                    'redundant_pairs': dominance_quantile_results[q]['redundant_pairs'],
                    'avg_encoder_pairwise_similarity': dominance_quantile_results[q]['avg_encoder_pairwise_similarity'],
                }
                for q in [1, 2, 3, 4]
            },
            'redundant_vs_nonredundant_comparison_by_quantile': dominance_comparison_by_quantile,
            'dominance_stats': dominance_stats,
        }, f, indent=2, default=str)
    print(f"✓ Saved dominance results to: {dominance_results_file}")
    
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
    n_batches: int = 3,
    top_n_per_quantile: int = 3000,
    similarity_thresholds: list = None,
    output_dir: Path = None,
    skip_existing: bool = True
):
    """Run frequency quantile redundancy analysis on all benchmark training OPT-125m combinations."""
    combinations = load_combinations()
    opt125m_combinations = filter_combinations_for_llm(combinations, "opt-125m")
    
    total_analyses = len(opt125m_combinations)
    completed = 0
    skipped = 0
    failed = []
    
    print("="*70)
    print("BATCH FREQUENCY QUANTILE REDUNDANCY ANALYSIS: ALL BENCHMARK TRAINING OPT-125m MODELS")
    print("="*70)
    print(f"\nTotal analyses to run: {total_analyses}")
    print(f"Device: {device or 'auto-detect'}")
    
    start_time = time.time()
    
    if device is None:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/GPU is not available. GPU is required.")
        device = "cuda:0"
    
    model = load_base_model_for_llm("opt-125m", device=device)
    
    try:
        for i, combo in enumerate(opt125m_combinations, 1):
            combo_id = (
                f"{combo['architecture']}_L{combo['layer']}_"
                f"{combo['width']}_{combo['sparsity_param']}{combo['sparsity_value']}"
            )
            
            if skip_existing and output_dir:
                parsed = parse_combo_id(combo_id)
                width_name = parsed['width']
                architecture = parsed['architecture']
                check_dir = output_dir / "benchmark_training_opt-125m" / width_name / architecture / combo_id
                # Check for new structure (subfolders)
                frequency_results = check_dir / "frequency_quantile_redundancy" / "results.json"
                dominance_results = check_dir / "dominance_quantile_redundancy" / "results.json"
                # Also check for old structure (root results.json)
                old_results = check_dir / "results.json"
                if (frequency_results.exists() and dominance_results.exists()) or old_results.exists():
                    print(f"\n[{i}/{total_analyses}] ⏭️  Skipping (already exists): {combo_id}")
                    skipped += 1
                    continue
            
            print(f"\n[{i}/{total_analyses}] 🔄 Processing: {combo_id}")
            try:
                model = run_analysis_benchmark_training_opt125m(
                    combo=combo,
                    device=device,
                    n_batches=n_batches,
                    top_n_per_quantile=top_n_per_quantile,
                    similarity_thresholds=similarity_thresholds,
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
        cleanup_model_cache("opt-125m")
    
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
    
    parser = argparse.ArgumentParser(description="Run frequency quantile redundancy analysis on benchmark training OPT-125m models")
    parser.add_argument('--device', type=str, default=None, help='Device to use (default: auto-detect cuda:0)')
    parser.add_argument('--n-batches', type=int, default=3, help='Number of batches (default: 3)')
    parser.add_argument('--top-n-per-quantile', type=int, default=3000, help='Top N features per quantile (default: 3000)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--no-skip-existing', action='store_true', help='Do not skip existing results')
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else None
    similarity_thresholds = [0.7, 0.9]  # Default thresholds
    
    run_all_analyses(
        device=args.device,
        n_batches=args.n_batches,
        top_n_per_quantile=args.top_n_per_quantile,
        similarity_thresholds=similarity_thresholds,
        output_dir=output_dir,
        skip_existing=not args.no_skip_existing
    )
