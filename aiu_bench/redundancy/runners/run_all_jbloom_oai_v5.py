#!/usr/bin/env python3
"""
Batch runner for ALL jbloom OAI-v5 SAEs.

This script runs analysis6 (frequency quantile redundancy) on:
- All 8 repos (32k/128k × 4 hook types)
- All 12 layers (0-11)
Total: 8 × 12 = 96 analyses

All SAEs use the same base model (GPT-2 Small), so we load it once and reuse.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List
import time
import numpy as np
import json

# Add parent directories to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

# Import analysis functions
from aiu_bench.redundancy import (
    split_features_by_frequency_quantiles,
    analyze_quantile_redundancy,
    create_8_panel_quantile_heatmaps,
    create_quantile_comparison_barplot,
    compare_redundant_vs_nonredundant,
    create_redundant_vs_nonredundant_plot,
    # Dominance-based functions
    compute_dominance_scores,
    split_features_by_dominance_quantiles,
    create_8_panel_quantile_heatmaps_dominance,
    create_quantile_comparison_barplot_dominance,
    compare_redundant_vs_nonredundant_dominance,
    create_redundant_vs_nonredundant_plot_dominance,
    compare_redundant_vs_nonredundant_by_quantile_dominance,
    create_redundant_vs_nonredundant_plot_by_quantile_dominance,
)

from sae_family_load_convert.jbloom_oai_v5 import (
    load_jbloom_oai_v5_sae,
    cleanup_sae_cache,
    cleanup_repo_cache,
)

from sae_family_load_convert import (
    setup_activations_store,
    load_base_model,
)

from analysis2_encoder_decoder_activation_relationship.cleanup_utils import (
    cleanup_model_cache,
)


def collect_activations_and_compute_firing_stats(store, sae, n_batches: int = 3):
    """
    Collect activations and compute firing statistics.
    
    Returns:
        firing_stats: Dict with ranking keys:
            - 'feature_rank_by_firing'
            - 'feature_rank_by_magnitude_mass'
            - 'feature_rank_by_magnitude_median'
    """
    print(f"\n📊 Collecting {n_batches} batches of activations and computing firing stats...")
    
    import torch
    feature_firing_counts = np.zeros(sae.cfg.d_sae, dtype=np.int64)
    feature_magnitude_sums = np.zeros(sae.cfg.d_sae, dtype=np.float64)
    feature_magnitude_medians = np.zeros(sae.cfg.d_sae, dtype=np.float64)
    total_activations = 0
    
    # Store all activations per feature for median computation
    all_activations_per_feature = [[] for _ in range(sae.cfg.d_sae)]
    
    for batch_idx in range(n_batches):
        batch_activations = store.next_batch()  # [batch_size, d_in]
        batch_size = batch_activations.shape[0]
        total_activations += batch_size
        
        # Encode to get feature activations
        with torch.no_grad():
            feature_acts = sae.encode(batch_activations)  # [batch_size, d_sae]
        
        # Count non-zero activations (firing)
        feature_acts_np = feature_acts.cpu().numpy()
        fired = (feature_acts_np > 0).astype(np.int64)
        feature_firing_counts += fired.sum(axis=0)
        
        # Sum absolute magnitudes (magnitude mass)
        abs_acts = np.abs(feature_acts_np)
        feature_magnitude_sums += abs_acts.sum(axis=0)
        
        # Store activations for median computation
        for feat_idx in range(sae.cfg.d_sae):
            feat_acts = feature_acts_np[:, feat_idx]
            non_zero_acts = feat_acts[feat_acts > 0]
            if len(non_zero_acts) > 0:
                all_activations_per_feature[feat_idx].extend(non_zero_acts.tolist())
    
    # Compute medians
    for feat_idx in range(sae.cfg.d_sae):
        if len(all_activations_per_feature[feat_idx]) > 0:
            feature_magnitude_medians[feat_idx] = np.median(all_activations_per_feature[feat_idx])
        else:
            feature_magnitude_medians[feat_idx] = 0.0
    
    # Rank features by different methods (highest first)
    feature_rank_by_firing = np.argsort(feature_firing_counts)[::-1].tolist()
    feature_rank_by_magnitude_mass = np.argsort(feature_magnitude_sums)[::-1].tolist()
    feature_rank_by_magnitude_median = np.argsort(feature_magnitude_medians)[::-1].tolist()
    
    firing_stats = {
        'feature_rank_by_firing': feature_rank_by_firing,
        'feature_rank_by_magnitude_mass': feature_rank_by_magnitude_mass,
        'feature_rank_by_magnitude_median': feature_rank_by_magnitude_median,
        'feature_firing_counts': feature_firing_counts.tolist(),
        'feature_magnitude_sums': feature_magnitude_sums.tolist(),
        'feature_magnitude_medians': feature_magnitude_medians.tolist(),
        'total_activations': total_activations,
    }
    
    print(f"✓ Collected {total_activations:,} activation vectors")
    print(f"✓ Computed firing stats for {sae.cfg.d_sae:,} features")
    
    return firing_stats


def run_analysis_jbloom_oai_v5(
    repo_id: str,
    layer: int,
    device: str = None,
    n_batches: int = 3,
    top_n_per_quantile: int = 500,
    similarity_thresholds: List[float] = None,
    output_dir: Path = None,
    model=None  # Optional: reuse existing model
):
    """
    Run frequency quantile redundancy analysis for jbloom OAI-v5 SAE.
    
    Args:
        repo_id: HuggingFace repo ID
        layer: Layer number (0-11)
        device: Device to use (default: auto-detect cuda:0, requires GPU)
        n_batches: Number of batches for activation collection
        top_n_per_quantile: Number of top features per quantile (default: 500)
        similarity_thresholds: List of similarity thresholds (default: [0.7, 0.9])
        output_dir: Output directory (default: {DATA_DIR}/sae/benchmark/redundancy/, where DATA_DIR defaults to /data/user)
        model: Optional pre-loaded model to reuse (saves GPU memory)
    """
    if similarity_thresholds is None:
        similarity_thresholds = [0.7, 0.9]
    
    import torch
    import gc
    
    # Auto-detect device if not specified - require GPU
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/GPU is not available. GPU is required for analysis.")
        device = "cuda:0"
        print(f"Using GPU: {device}")
    
    print("="*70)
    print("FREQUENCY QUANTILE REDUNDANCY ANALYSIS - JBLOOM OAI-V5")
    print("="*70)
    
    # Setup output directory: jbloom_oai_v5/{repo_name}/layer_{layer}/
    if output_dir is None:
        data_dir = os.environ.get("DATA_DIR", "/data/user")
        output_dir = Path(data_dir) / "sae" / "benchmark" / "redundancy"
    
    # Parse repo_id to get repo name
    repo_name = repo_id.split('/')[-1]  # e.g., "GPT2-Small-OAI-v5-32k-resid-post-SAEs"
    
    # Create output directory
    output_dir_final = output_dir / "jbloom_oai_v5" / repo_name / f"layer_{layer}"
    output_dir_final.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir_final}")
    
    # Step 1: Load SAE
    print("\n[1/5] Loading SAE...")
    sae, metadata = load_jbloom_oai_v5_sae(
        repo_id=repo_id,
        layer=layer,
        device=device
    )
    
    # Step 2: Load base model (reuse if provided)
    print("\n[2/5] Loading base model...")
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
    
    # Step 3: Setup activation store and collect activations + compute firing stats
    print("\n[3/5] Collecting activations and computing firing statistics...")
    store = setup_activations_store(
        model=model,
        sae=sae,
        metadata=metadata,
        device=device
    )
    
    firing_stats = collect_activations_and_compute_firing_stats(store, sae, n_batches=n_batches)
    
    # Step 4: Split features by frequency quantiles
    print("\n[4/5] Splitting features by frequency quantiles...")
    quantiles = split_features_by_frequency_quantiles(
        firing_stats=firing_stats,
        top_n_per_quantile=top_n_per_quantile
    )
    
    # Step 5: Run quantile redundancy analysis
    print("\n[5/5] Running quantile redundancy analysis...")
    
    # Get decoder and encoder weights
    decoder_weights = sae.W_dec.detach().cpu().numpy()  # [d_sae, d_in]
    encoder_weights = sae.W_enc.detach().cpu().numpy()  # [d_in, d_sae]
    
    # Analyze each quantile
    quantile_results = {}
    for quantile in [1, 2, 3, 4]:
        print(f"\n  Analyzing Q{quantile}...")
        quantile_results[quantile] = analyze_quantile_redundancy(
            decoder_weights=decoder_weights,
            encoder_weights=encoder_weights,
            quantile_features=quantiles[quantile],
            similarity_thresholds=similarity_thresholds
        )
    
    # Create output subdirectory
    analysis_output_dir = output_dir_final / "frequency_quantile_redundancy"
    
    # Create plots
    sae_name = f"{repo_name}_layer_{layer}"
    
    # Plot 1: 8-panel heatmaps
    create_8_panel_quantile_heatmaps(
        quantile_results=quantile_results,
        output_dir=analysis_output_dir,
        sae_name=sae_name
    )
    
    # Plot 2: Quantile comparison barplot
    create_quantile_comparison_barplot(
        quantile_results=quantile_results,
        similarity_thresholds=similarity_thresholds,
        output_dir=analysis_output_dir,
        sae_name=sae_name
    )
    
    # Compare redundant vs non-redundant pairs
    comparison_results = compare_redundant_vs_nonredundant(
        decoder_weights=decoder_weights,
        encoder_weights=encoder_weights,
        firing_stats=firing_stats,
        similarity_thresholds=similarity_thresholds,
        top_n=10000,
        sample_size=1000
    )
    
    # Plot 3: Redundant vs non-redundant comparison
    create_redundant_vs_nonredundant_plot(
        comparison_results=comparison_results,
        similarity_thresholds=similarity_thresholds,
        output_dir=analysis_output_dir,
        sae_name=sae_name
    )
    
    # Save results JSON (summary statistics only, full pairs are in redundant_pairs.json)
    results_file = analysis_output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'quantile_results': {
                str(q): {
                    'n_features': quantile_results[q]['n_features'],
                    'redundant_pairs': quantile_results[q]['redundant_pairs'],  # Counts only
                    'avg_encoder_pairwise_similarity': quantile_results[q]['avg_encoder_pairwise_similarity'],
                    'avg_decoder_pairwise_similarity': quantile_results[q]['avg_decoder_pairwise_similarity'],
                }
                for q in [1, 2, 3, 4]
            },
            'redundant_vs_nonredundant_comparison': {
                str(threshold): {
                    'n_redundant_pairs': comparison_results[threshold]['n_redundant_pairs'],
                    'n_non_redundant_pairs_sampled': comparison_results[threshold]['n_non_redundant_pairs_sampled'],
                    'redundant_encoder_similarity': comparison_results[threshold]['redundant_encoder_similarity'],
                    'non_redundant_encoder_similarity': comparison_results[threshold]['non_redundant_encoder_similarity'],
                    'note': 'Full redundant pairs with metadata are saved in redundant_pairs.json'
                }
                for threshold in similarity_thresholds
            },
            'firing_stats_summary': {
                'total_features': sae.cfg.d_sae,
                'total_activations': firing_stats['total_activations'],
            }
        }, f, indent=2)
    print(f"✓ Saved results to: {results_file}")
    
    # Save redundant pairs JSON with full metadata
    redundant_pairs_file = analysis_output_dir / "redundant_pairs.json"
    redundant_pairs_data = {}
    for threshold in similarity_thresholds:
        redundant_pairs_data[str(threshold)] = {
            'n_pairs': comparison_results[threshold]['n_redundant_pairs'],
            'pairs': comparison_results[threshold]['redundant_pairs'],  # ALL pairs, ranked by encoder difference (big to low)
            'note': 'Pairs are ranked by encoder_difference (big to low), i.e., most different encoders first. '
                   'Each pair includes: feature_id_1, feature_id_2, decoder_similarity, encoder_similarity, '
                   'encoder_difference, quantile_1, quantile_2'
        }
    with open(redundant_pairs_file, 'w') as f:
        json.dump(redundant_pairs_data, f, indent=2)
    print(f"✓ Saved {sum(comparison_results[t]['n_redundant_pairs'] for t in similarity_thresholds)} redundant pairs to: {redundant_pairs_file}")
    
    # ========================================================================
    # DOMINANCE QUANTILE REDUNDANCY ANALYSIS
    # ========================================================================
    print("\n" + "="*70)
    print("RUNNING DOMINANCE QUANTILE REDUNDANCY ANALYSIS")
    print("="*70)
    
    # Step 1: Compute dominance scores
    print("\n[1/5] Computing dominance scores...")
    # Reset store to beginning for dominance computation
    store = setup_activations_store(
        model=model,
        sae=sae,
        metadata=metadata,
        device=device
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
        top_n_per_quantile=top_n_per_quantile
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
    
    # Create output subdirectory
    dominance_analysis_output_dir = output_dir_final / "dominance_quantile_redundancy"
    
    # Create plots
    # Plot 1: 8-panel heatmaps
    create_8_panel_quantile_heatmaps_dominance(
        quantile_results=dominance_quantile_results,
        output_dir=dominance_analysis_output_dir,
        sae_name=sae_name
    )
    
    # Plot 2: Quantile comparison barplot
    create_quantile_comparison_barplot_dominance(
        quantile_results=dominance_quantile_results,
        similarity_thresholds=similarity_thresholds,
        output_dir=dominance_analysis_output_dir,
        sae_name=sae_name
    )
    
    # Compare redundant vs non-redundant pairs
    dominance_comparison_results = compare_redundant_vs_nonredundant_dominance(
        decoder_weights=decoder_weights,
        encoder_weights=encoder_weights,
        dominance_scores=dominance_scores,
        similarity_thresholds=similarity_thresholds,
        top_n=10000,
        sample_size=1000
    )
    
    # Plot 3: Redundant vs non-redundant comparison
    create_redundant_vs_nonredundant_plot_dominance(
        comparison_results=dominance_comparison_results,
        similarity_thresholds=similarity_thresholds,
        output_dir=dominance_analysis_output_dir,
        sae_name=sae_name
    )
    
    # Save results JSON
    dominance_results_file = dominance_analysis_output_dir / "results.json"
    with open(dominance_results_file, 'w') as f:
        json.dump({
            'quantile_results': {
                str(q): {
                    'n_features': dominance_quantile_results[q]['n_features'],
                    'redundant_pairs': dominance_quantile_results[q]['redundant_pairs'],  # Counts only
                    'avg_encoder_pairwise_similarity': dominance_quantile_results[q]['avg_encoder_pairwise_similarity'],
                }
                for q in [1, 2, 3, 4]
            },
            'redundant_vs_nonredundant_comparison': {
                str(threshold): {
                    'n_redundant_pairs': dominance_comparison_results[threshold]['n_redundant_pairs'],
                    'n_non_redundant_pairs_sampled': dominance_comparison_results[threshold]['n_non_redundant_pairs_sampled'],
                    'redundant_encoder_similarity': dominance_comparison_results[threshold]['redundant_encoder_similarity'],
                    'non_redundant_encoder_similarity': dominance_comparison_results[threshold]['non_redundant_encoder_similarity'],
                    'note': 'Full redundant pairs with metadata are saved in redundant_pairs.json'
                }
                for threshold in similarity_thresholds
            },
            'dominance_stats': dominance_stats,
        }, f, indent=2)
    print(f"✓ Saved dominance results to: {dominance_results_file}")
    
    # Save redundant pairs JSON with full metadata
    dominance_redundant_pairs_file = dominance_analysis_output_dir / "redundant_pairs.json"
    dominance_redundant_pairs_data = {}
    for threshold in similarity_thresholds:
        dominance_redundant_pairs_data[str(threshold)] = {
            'n_pairs': dominance_comparison_results[threshold]['n_redundant_pairs'],
            'pairs': dominance_comparison_results[threshold]['redundant_pairs'],  # ALL pairs, ranked by encoder difference (big to low)
            'note': 'Pairs are ranked by encoder_difference (big to low), i.e., most different encoders first. '
                   'Each pair includes: feature_id_1, feature_id_2, decoder_similarity, encoder_similarity, '
                   'encoder_difference, quantile_1, quantile_2'
        }
    with open(dominance_redundant_pairs_file, 'w') as f:
        json.dump(dominance_redundant_pairs_data, f, indent=2)
    print(f"✓ Saved {sum(dominance_comparison_results[t]['n_redundant_pairs'] for t in similarity_thresholds)} dominance redundant pairs to: {dominance_redundant_pairs_file}")
    
    # Clean up SAE cache after analysis
    print("\n[Cleanup] Removing cached SAE files...")
    cache_path_str = metadata.get('_cache_path')
    if cache_path_str:
        cleanup_sae_cache(cache_path=cache_path_str)
    else:
        cleanup_sae_cache(metadata=metadata)
    
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


# jbloom OAI-v5 Configuration
JBLOOM_OAI_V5_CONFIG = {
    'repos': {
        '32k_resid_post': 'jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs',
        '128k_resid_post': 'jbloom/GPT2-Small-OAI-v5-128k-resid-post-SAEs',
        '32k_resid_mid': 'jbloom/GPT2-Small-OAI-v5-32k-resid-mid-SAEs',
        '128k_resid_mid': 'jbloom/GPT2-Small-OAI-v5-128k-resid-mid-SAEs',
        '32k_mlp_out': 'jbloom/GPT2-Small-OAI-v5-32k-mlp-out-SAEs',
        '128k_mlp_out': 'jbloom/GPT2-Small-OAI-v5-128k-mlp-out-SAEs',
        '32k_attn_out': 'jbloom/GPT2-Small-OAI-v5-32k-attn-out-SAEs',
        '128k_attn_out': 'jbloom/GPT2-Small-OAI-v5-128k-attn-out-SAEs',
    },
    'layers': list(range(12)),  # 0-11
    'model_name': 'gpt2-small',
}


def run_all_analyses(
    device: str = None,
    n_batches: int = 3,
    top_n_per_quantile: int = 500,
    similarity_thresholds: List[float] = None,
    output_dir: Path = None,
    skip_existing: bool = True
):
    """
    Run analysis6 on all jbloom OAI-v5 SAEs.
    
    Args:
        device: Device to use (default: auto-detect cuda:0)
        n_batches: Number of batches for activation collection
        top_n_per_quantile: Number of top features per quantile
        similarity_thresholds: List of similarity thresholds (default: [0.7, 0.9])
        output_dir: Output directory
        skip_existing: Skip analyses that already have results
    """
    if similarity_thresholds is None:
        similarity_thresholds = [0.7, 0.9]
    
    import torch
    import gc
    
    total_analyses = len(JBLOOM_OAI_V5_CONFIG['repos']) * len(JBLOOM_OAI_V5_CONFIG['layers'])
    completed = 0
    skipped = 0
    failed = []
    
    print("="*70)
    print("BATCH ANALYSIS: ALL JBLOOM OAI-V5 SAEs (FREQUENCY QUANTILE REDUNDANCY)")
    print("="*70)
    print(f"\nTotal analyses to run: {total_analyses}")
    print(f"  - Repos: {len(JBLOOM_OAI_V5_CONFIG['repos'])}")
    print(f"  - Layers: {len(JBLOOM_OAI_V5_CONFIG['layers'])}")
    print(f"\nDevice: {device or 'auto-detect'}")
    print(f"n_batches: {n_batches}, top_n_per_quantile: {top_n_per_quantile}")
    
    start_time = time.time()
    
    # Load model once and reuse (all SAEs use same base model: GPT-2 Small)
    print("\n📦 Loading base model (will reuse for all SAEs)...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear before loading
    
    try:
        model = load_base_model(
            model_name=JBLOOM_OAI_V5_CONFIG['model_name'],
            device=device
        )
        print(f"✓ Loaded model: {JBLOOM_OAI_V5_CONFIG['model_name']} (will reuse for all {total_analyses} SAEs)")
    except Exception as e:
        print(f"⚠️  Failed to load model: {e}")
        model = None
    
    for repo_name, repo_id in JBLOOM_OAI_V5_CONFIG['repos'].items():
        print("\n" + "="*70)
        print(f"PROCESSING {repo_name.upper()}")
        print("="*70)
        
        for layer in JBLOOM_OAI_V5_CONFIG['layers']:
            # Check if already exists
            if skip_existing and output_dir:
                layer_dir = output_dir / "jbloom_oai_v5" / repo_id.split('/')[-1] / f"layer_{layer}"
                # Check for new structure (subfolders)
                freq_png_exists = (layer_dir / "frequency_quantile_redundancy" / "1_8_panel_quantile_heatmaps.png").exists()
                freq_json_exists = (layer_dir / "frequency_quantile_redundancy" / "results.json").exists()
                dom_json_exists = (layer_dir / "dominance_quantile_redundancy" / "results.json").exists()
                # Also check for old structure (root results.json)
                old_results = layer_dir / "results.json"
                if (freq_png_exists and freq_json_exists and dom_json_exists) or old_results.exists():
                    print(f"  ⏭️  Layer {layer}: Already exists, skipping...")
                    skipped += 1
                    continue
            
            print(f"\n  🔄 Layer {layer}...")
            try:
                # Aggressive memory cleanup before each analysis
                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Reuse model for all SAEs
                run_analysis_jbloom_oai_v5(
                    repo_id=repo_id,
                    layer=layer,
                    device=device,
                    n_batches=n_batches,
                    top_n_per_quantile=top_n_per_quantile,
                    similarity_thresholds=similarity_thresholds,
                    output_dir=output_dir,
                    model=model  # Reuse model to save GPU memory
                )
                completed += 1
                print(f"  ✅ Layer {layer}: Completed ({completed}/{total_analyses})")
                
                # Aggressive cleanup after each successful analysis
                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception as e:
                error_msg = str(e)[:200]
                print(f"  ❌ Layer {layer}: Failed - {error_msg}")
                failed.append({
                    'repo': repo_name,
                    'layer': layer,
                    'error': error_msg
                })
                # Aggressive cleanup after failure too
                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
        
        # Clean up repo cache after all layers processed
        print(f"\n🧹 Cleaning up repo cache for {repo_name}...")
        cleanup_repo_cache(repo_id)
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print(f"✓ Cleaned up repo cache (including blobs) for {repo_name}")
    
    # Clean up model cache after all repos processed
    if model is not None:
        print("\n🧹 Cleaning up model cache...")
        del model
        model = None
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        cleanup_model_cache(JBLOOM_OAI_V5_CONFIG['model_name'])
        print(f"✓ Cleaned up model cache and GPU cache")
    
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
            print(f"  - {f['repo']}/layer_{f['layer']}: {f['error']}")
    
    return completed, skipped, len(failed)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run analysis6 on all jbloom OAI-v5 SAEs")
    parser.add_argument('--device', type=str, default=None, help='Device to use (default: auto-detect cuda:0)')
    parser.add_argument('--n-batches', type=int, default=3, help='Number of batches for activation collection (default: 3)')
    parser.add_argument('--top-n-per-quantile', type=int, default=500, help='Number of top features per quantile (default: 500)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--no-skip-existing', action='store_true', help='Re-run analyses even if results exist')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    run_all_analyses(
        device=args.device,
        n_batches=args.n_batches,
        top_n_per_quantile=args.top_n_per_quantile,
        output_dir=output_dir,
        skip_existing=not args.no_skip_existing
    )

