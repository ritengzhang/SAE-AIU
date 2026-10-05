#!/usr/bin/env python3
"""
Batch runner for ALL SAE Bench Gemma models with ALL trainers.

This script runs analysis6 (frequency quantile redundancy) on:
- All 7 SAE types
- All 3 widths (4k, 16k, 65k)
- All 6 trainers (trainer_0 through trainer_5)
Total: 7 × 3 × 6 = 126 analyses
"""

import sys
import os
from pathlib import Path
from typing import Dict, List
import time
import numpy as np
import json
import shutil

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

from sae_family_load_convert.saebench_gemma import (
    load_saebench_gemma_sae,
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


def run_analysis_saebench_gemma(
    repo_id: str,
    sae_type_folder: str,
    layer_folder: str = "resid_post_layer_12",
    trainer_folder: str = "trainer_0",
    width_name: str = None,
    device: str = None,
    n_batches: int = 3,
    top_n_per_quantile: int = 500,
    similarity_thresholds: List[float] = None,
    output_dir: Path = None,
    model=None  # Optional: reuse existing model
):
    """
    Run frequency quantile redundancy analysis for SAE Bench Gemma SAE.
    
    Args:
        repo_id: HuggingFace repo ID
        sae_type_folder: SAE type folder name
        layer_folder: Layer folder name
        trainer_folder: Trainer folder name
        width_name: Width name (4k, 16k, 65k)
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
    print("FREQUENCY QUANTILE REDUNDANCY ANALYSIS - SAE BENCH GEMMA")
    print("="*70)
    
    # Setup output directory: saebench_gemma/{width}/{sae_type}/{trainer}/
    if output_dir is None:
        data_dir = os.environ.get("DATA_DIR", "/data/user")
        output_dir = Path(data_dir) / "sae" / "benchmark" / "redundancy"
    
    # Create base output directory
    base_output_dir = output_dir / "saebench_gemma" / width_name / sae_type_folder
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load SAE
    print("\n[1/5] Loading SAE...")
    sae, metadata = load_saebench_gemma_sae(
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
        device=device,
        width_name=width_name
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
    sae_name = f"{sae_type_folder}_{width_name}_{trainer_folder_name}"
    
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
        device=device,
        width_name=width_name
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
    
    # Compare by quantile
    dominance_comparison_by_quantile = compare_redundant_vs_nonredundant_by_quantile_dominance(
        decoder_weights=decoder_weights,
        encoder_weights=encoder_weights,
        dominance_scores=dominance_scores,
        similarity_thresholds=similarity_thresholds,
        top_n_per_quantile=top_n_per_quantile,
        sample_size=1000
    )
    
    # Plot 4: Redundant vs non-redundant by quantile
    create_redundant_vs_nonredundant_plot_by_quantile_dominance(
        comparison_results=dominance_comparison_by_quantile,
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


# SAE Bench Gemma Configuration
SAE_BENCH_GEMMA_CONFIG = {
    'gemma_2_2b': {
        'repos': {
            '4k': 'adamkarvonen/saebench_gemma-2-2b_width-2pow12_date-0108',
            '16k': 'canrager/saebench_gemma-2-2b_width-2pow14_date-0107',
            '65k': 'canrager/saebench_gemma-2-2b_width-2pow16_date-0107',
        },
        'layer': 'resid_post_layer_12',
        'sae_types': [
            'Standard_gemma-2-2b__0108',  # ReLU
            'TopK_gemma-2-2b__0108',
            'BatchTopK_gemma-2-2b__0108',
            'JumpRelu_gemma-2-2b__0108',
            'GatedSAE_gemma-2-2b__0108',
            'PAnneal_gemma-2-2b__0108',
            'MatryoshkaBatchTopK_gemma-2-2b__0108',
        ],
    },
}

# All trainers
TRAINERS = ["trainer_0", "trainer_1", "trainer_2", "trainer_3", "trainer_4", "trainer_5"]


def run_all_analyses(
    device: str = None,
    n_batches: int = 3,
    top_n_per_quantile: int = 500,
    similarity_thresholds: List[float] = None,
    output_dir: Path = None,
    skip_existing: bool = True
):
    """
    Run analysis6 on all SAE Bench Gemma models with all trainers.
    
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
    
    config = SAE_BENCH_GEMMA_CONFIG['gemma_2_2b']
    
    total_analyses = len(config['sae_types']) * len(config['repos']) * len(TRAINERS)
    completed = 0
    skipped = 0
    failed = []
    
    print("="*70)
    print("BATCH ANALYSIS: ALL SAE BENCH GEMMA MODELS (FREQUENCY QUANTILE REDUNDANCY)")
    print("="*70)
    print(f"\nTotal analyses to run: {total_analyses}")
    print(f"  - SAE types: {len(config['sae_types'])}")
    print(f"  - Widths: {len(config['repos'])}")
    print(f"  - Trainers: {len(TRAINERS)}")
    print(f"\nDevice: {device or 'auto-detect'}")
    print(f"n_batches: {n_batches}, top_n_per_quantile: {top_n_per_quantile}")
    
    start_time = time.time()
    
    # Load model once per width and reuse (saves GPU memory for Gemma 2B)
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
            first_sae, first_sae_metadata = load_saebench_gemma_sae(
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
            cleanup_sae_cache(cache_path=first_sae_metadata['_cache_path'])
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
                    sae_type_dir = output_dir / "saebench_gemma" / width_name / sae_type
                    if sae_type_dir.exists():
                        found_existing = False
                        for existing_folder in sae_type_dir.iterdir():
                            if existing_folder.is_dir() and existing_folder.name.startswith(trainer + '_'):
                                # Check for new structure (subfolders)
                                freq_png_exists = (existing_folder / "frequency_quantile_redundancy" / "1_8_panel_quantile_heatmaps.png").exists()
                                freq_json_exists = (existing_folder / "frequency_quantile_redundancy" / "results.json").exists()
                                dom_json_exists = (existing_folder / "dominance_quantile_redundancy" / "results.json").exists()
                                # Also check for old structure (root results.json)
                                old_results = existing_folder / "results.json"
                                if (freq_png_exists and freq_json_exists and dom_json_exists) or old_results.exists():
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
                    run_analysis_saebench_gemma(
                        repo_id=repo_id,
                        sae_type_folder=sae_type,
                        layer_folder=config['layer'],
                        trainer_folder=trainer,
                        width_name=width_name,
                        device=device,
                        n_batches=n_batches,
                        top_n_per_quantile=top_n_per_quantile,
                        similarity_thresholds=similarity_thresholds,
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
            cleanup_model_cache('google/gemma-2-2b')
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
    
    parser = argparse.ArgumentParser(description="Run analysis6 on all SAE Bench Gemma models")
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

