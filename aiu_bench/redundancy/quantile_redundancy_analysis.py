#!/usr/bin/env python3
"""
Frequency Quantile Redundancy Analysis

This module analyzes feature redundancy patterns across frequency quantiles.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


def compute_decoder_similarity_matrix(
    decoder_weights: np.ndarray,
    feature_indices: Optional[List[int]] = None
) -> np.ndarray:
    """
    Compute pairwise cosine similarity between decoder weights.
    
    Args:
        decoder_weights: [d_sae, d_in] decoder weight matrix
        feature_indices: Optional list of feature indices to compute similarity for.
                       If None, computes for all features.
        
    Returns:
        Similarity matrix [n_features, n_features]
    """
    if feature_indices is None:
        weights = decoder_weights  # [d_sae, d_in]
    else:
        weights = decoder_weights[feature_indices]  # [n_selected, d_in]
    
    # Compute cosine similarity: normalize weights first
    weights_norm = weights / (np.linalg.norm(weights, axis=1, keepdims=True) + 1e-8)
    similarity_matrix = np.dot(weights_norm, weights_norm.T)  # [n_features, n_features]
    
    return similarity_matrix


def compute_encoder_similarity_matrix(
    encoder_weights: np.ndarray,
    feature_indices: Optional[List[int]] = None
) -> np.ndarray:
    """
    Compute pairwise cosine similarity between encoder weights.
    
    Args:
        encoder_weights: [d_in, d_sae] encoder weight matrix (will be transposed internally)
        feature_indices: Optional list of feature indices to compute similarity for.
                       If None, computes for all features.
        
    Returns:
        Similarity matrix [n_features, n_features]
    """
    # Transpose encoder weights so each row is a feature's encoder vector: [d_sae, d_in]
    encoder_weights_T = encoder_weights.T  # [d_sae, d_in]
    
    if feature_indices is None:
        weights = encoder_weights_T  # [d_sae, d_in]
    else:
        weights = encoder_weights_T[feature_indices]  # [n_selected, d_in]
    
    # Compute cosine similarity: normalize weights first
    weights_norm = weights / (np.linalg.norm(weights, axis=1, keepdims=True) + 1e-8)
    similarity_matrix = np.dot(weights_norm, weights_norm.T)  # [n_features, n_features]
    
    return similarity_matrix


def split_features_by_frequency_quantiles(
    firing_stats: Dict,
    top_n_per_quantile: int = 3000
) -> Dict[int, List[int]]:
    """
    Split features into 4 equal quantiles by firing frequency and select top N per quantile.
    
    Args:
        firing_stats: Dictionary with 'feature_rank_by_firing' key
        top_n_per_quantile: Number of top features to select per quantile (default: 3000)
    
    Returns:
        Dictionary mapping quantile (1-4) to list of feature indices
        Q1 = most frequent, Q4 = least frequent
    """
    feature_rank = np.array(firing_stats['feature_rank_by_firing'])  # Ranked by frequency (descending)
    total_features = len(feature_rank)
    
    # Split into 4 equal quantiles
    quantile_size = total_features // 4
    quantiles = {}
    
    quantiles[1] = feature_rank[:quantile_size].tolist()  # Most frequent
    quantiles[2] = feature_rank[quantile_size:2*quantile_size].tolist()
    quantiles[3] = feature_rank[2*quantile_size:3*quantile_size].tolist()
    quantiles[4] = feature_rank[3*quantile_size:].tolist()  # Least frequent
    
    # Select top N per quantile (already sorted by frequency within quantile)
    for q in quantiles:
        quantiles[q] = quantiles[q][:min(top_n_per_quantile, len(quantiles[q]))]
    
    print(f"✓ Split {total_features} features into 4 quantiles:")
    for q in [1, 2, 3, 4]:
        print(f"  Q{q}: {len(quantiles[q])} features (top {top_n_per_quantile} selected)")
    
    return quantiles


def compute_dominance_scores(
    sae,
    store,
    n_batches: int,
    use_absolute_value: bool = True,
    device: str = "cpu"
) -> Tuple[np.ndarray, Dict]:
    """
    Compute mean dominance scores for all features using the same formula as dominance_aware_auto_inter_score.
    
    Dominance = |f_i(φ(x))| / Σ_j |f_j(φ(x))| for each token position
    Mean dominance = average across ALL token positions (including where feature didn't fire)
    
    Args:
        sae: SAE object
        store: Activation store
        n_batches: Number of batches to process
        use_absolute_value: Whether to use absolute values for activations (default: True)
        device: Device to use
    
    Returns:
        Tuple of:
        - dominance_scores: Array of mean dominance per feature [d_sae]
        - dominance_stats: Dictionary with statistics
    """
    import torch
    
    d_sae = sae.cfg.d_sae
    dominance_sum = np.zeros(d_sae, dtype=np.float64)
    dominance_count = np.zeros(d_sae, dtype=np.int64)
    total_sequences = 0
    
    print(f"  Computing dominance scores from {n_batches} batches...")
    
    for batch_idx in range(n_batches):
        batch_activations = store.next_batch()
        if batch_activations is None:
            break
        
        batch_size = batch_activations.shape[0]
        total_sequences += batch_size
        
        with torch.no_grad():
            batch_activations_tensor = torch.tensor(batch_activations, device=device)
            sae_acts = sae.encode(batch_activations_tensor)  # [batch_size, d_sae]
            
            if use_absolute_value:
                sae_acts = torch.abs(sae_acts)
            
            sae_acts_np = sae_acts.cpu().numpy()  # [batch_size, d_sae]
        
        # For each input sequence
        for seq_idx in range(batch_size):
            input_acts = sae_acts_np[seq_idx]  # [d_sae]
            total_act = np.sum(input_acts)
            
            if total_act > 0:
                # For ALL features (even if activation is 0) - compute dominance across all inputs
                for feat_idx in range(d_sae):
                    # Dominance = this feature's activation / total activation (can be 0 if feature didn't fire)
                    dominance = input_acts[feat_idx] / total_act
                    dominance_sum[feat_idx] += dominance
                    dominance_count[feat_idx] += 1  # Count ALL positions
            else:
                # If total_act == 0, all features have 0 activation, dominance is undefined
                # But we still count the position for consistency (dominance = 0)
                for feat_idx in range(d_sae):
                    dominance_count[feat_idx] += 1  # Count position, dominance remains 0 (sum stays 0)
    
    # Compute average dominance: sum / count
    # dominance_count now represents ALL token positions processed (not just firings)
    dominance_scores = np.zeros(d_sae, dtype=np.float64)
    for feat_idx in range(d_sae):
        if dominance_count[feat_idx] > 0:
            # Average across ALL input positions (includes positions where feature didn't fire)
            dominance_scores[feat_idx] = dominance_sum[feat_idx] / dominance_count[feat_idx]
    
    dominance_stats = {
        'total_sequences': total_sequences,
        'total_positions': int(dominance_count[0]) if d_sae > 0 else 0,
        'dominance_range': (float(np.min(dominance_scores)), float(np.max(dominance_scores))),
        'mean_dominance': float(np.mean(dominance_scores)),
    }
    
    print(f"  ✓ Computed dominance scores: range={dominance_stats['dominance_range']}, mean={dominance_stats['mean_dominance']:.6f}")
    
    return dominance_scores, dominance_stats


def split_features_by_dominance_quantiles(
    dominance_scores: np.ndarray,
    top_n_per_quantile: int = 3000
) -> Dict[int, List[int]]:
    """
    Split features into 4 equal quantiles by mean dominance and select top N per quantile.
    
    Uses the same dominance ranking logic as dominance_aware_auto_inter_score.
    
    Args:
        dominance_scores: Array of mean dominance per feature [d_sae]
        top_n_per_quantile: Number of top features to select per quantile (default: 3000)
    
    Returns:
        Dictionary mapping quantile (1-4) to list of feature indices
        Q1 = highest dominance, Q4 = lowest dominance
    """
    # Sort features by dominance (descending - highest dominance first)
    feature_indices = np.arange(len(dominance_scores))
    sorted_indices = feature_indices[np.argsort(dominance_scores)[::-1]]  # Descending order
    
    total_features = len(sorted_indices)
    
    # Split into 4 equal quantiles
    quantile_size = total_features // 4
    quantiles = {}
    
    quantiles[1] = sorted_indices[:quantile_size].tolist()  # Highest dominance
    quantiles[2] = sorted_indices[quantile_size:2*quantile_size].tolist()
    quantiles[3] = sorted_indices[2*quantile_size:3*quantile_size].tolist()
    quantiles[4] = sorted_indices[3*quantile_size:].tolist()  # Lowest dominance
    
    # Select top N per quantile (already sorted by dominance within quantile)
    for q in quantiles:
        quantiles[q] = quantiles[q][:min(top_n_per_quantile, len(quantiles[q]))]
    
    print(f"✓ Split {total_features} features into 4 dominance quantiles:")
    for q in [1, 2, 3, 4]:
        if len(quantiles[q]) > 0:
            q_dominance_range = (
                float(np.min(dominance_scores[quantiles[q]])),
                float(np.max(dominance_scores[quantiles[q]]))
            )
            print(f"  Q{q}: {len(quantiles[q])} features, dominance range: {q_dominance_range}")
        else:
            print(f"  Q{q}: {len(quantiles[q])} features")
    
    return quantiles


def analyze_quantile_redundancy(
    decoder_weights: np.ndarray,
    encoder_weights: np.ndarray,
    quantile_features: List[int],
    similarity_thresholds: List[float] = [0.7, 0.9]
) -> Dict:
    """
    Analyze redundancy for a single quantile.
    
    Args:
        decoder_weights: [d_sae, d_in] decoder weight matrix
        encoder_weights: [d_in, d_sae] encoder weight matrix
        quantile_features: List of feature indices in this quantile
        similarity_thresholds: List of thresholds for redundancy counting
    
    Returns:
        Dictionary with redundancy statistics
    """
    n_features = len(quantile_features)
    
    # Compute similarity matrices
    decoder_similarity = compute_decoder_similarity_matrix(decoder_weights, quantile_features)
    encoder_similarity = compute_encoder_similarity_matrix(encoder_weights, quantile_features)
    
    # Remove diagonal (self-similarity = 1.0)
    mask = ~np.eye(n_features, dtype=bool)
    decoder_similarity_no_diag = decoder_similarity[mask]
    encoder_similarity_no_diag = encoder_similarity[mask]
    
    # Count redundant pairs for each threshold
    redundant_counts = {}
    for threshold in similarity_thresholds:
        # Count pairs above threshold (upper triangle only to avoid double counting)
        upper_triangle = np.triu(decoder_similarity, k=1)
        count = np.sum(upper_triangle >= threshold)
        redundant_counts[threshold] = int(count)
    
    # Compute average encoder pairwise cosine similarity
    avg_encoder_similarity = float(np.mean(encoder_similarity_no_diag))
    
    # Compute average decoder pairwise cosine similarity
    avg_decoder_similarity = float(np.mean(decoder_similarity_no_diag))
    
    results = {
        'n_features': n_features,
        'redundant_pairs': redundant_counts,
        'avg_encoder_pairwise_similarity': avg_encoder_similarity,
        'avg_decoder_pairwise_similarity': avg_decoder_similarity,
        'decoder_similarity_matrix': decoder_similarity.tolist(),
        'encoder_similarity_matrix': encoder_similarity.tolist(),
    }
    
    return results


def create_8_panel_quantile_heatmaps(
    quantile_results: Dict[int, Dict],
    output_dir: Path,
    sae_name: str
):
    """
    Create 8-panel heatmap (4 quantiles × decoder/encoder).
    
    Args:
        quantile_results: Dictionary mapping quantile (1-4) to results dict
        output_dir: Output directory
        sae_name: Name of the SAE for title
    """
    print(f"\n{'='*70}")
    print("CREATING 8-PANEL QUANTILE HEATMAPS")
    print(f"{'='*70}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    fig.suptitle(f'Decoder vs Encoder Similarity by Frequency Quantile\n{sae_name}', 
                 fontsize=16, fontweight='bold')
    
    for quantile in [1, 2, 3, 4]:
        q_idx = quantile - 1  # 0-indexed
        
        results = quantile_results[quantile]
        decoder_sim = np.array(results['decoder_similarity_matrix'])
        encoder_sim = np.array(results['encoder_similarity_matrix'])
        
        # Decoder heatmap (left column)
        ax = axes[q_idx, 0]
        im = ax.imshow(decoder_sim, cmap='viridis', aspect='auto', vmin=0, vmax=1)
        ax.set_title(f'Q{quantile} Decoder Similarity (n={results["n_features"]})', 
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature Index (ranked by frequency)', fontsize=10)
        ax.set_ylabel('Feature Index (ranked by frequency)', fontsize=10)
        plt.colorbar(im, ax=ax, label='Cosine Similarity')
        
        # Encoder heatmap (right column)
        ax = axes[q_idx, 1]
        im = ax.imshow(encoder_sim, cmap='viridis', aspect='auto', vmin=0, vmax=1)
        ax.set_title(f'Q{quantile} Encoder Similarity (n={results["n_features"]})', 
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature Index (ranked by frequency)', fontsize=10)
        ax.set_ylabel('Feature Index (ranked by frequency)', fontsize=10)
        plt.colorbar(im, ax=ax, label='Cosine Similarity')
    
    plt.tight_layout()
    plot_file = output_dir / "1_8_panel_quantile_heatmaps.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot_file.name}")


def create_quantile_comparison_barplot(
    quantile_results: Dict[int, Dict],
    similarity_thresholds: List[float] = [0.7, 0.9],
    output_dir: Path = None,
    sae_name: str = ""
):
    """
    Create bar plot with dual y-axis showing decoder redundancy and encoder cosine similarity.
    
    Args:
        quantile_results: Dictionary mapping quantile (1-4) to results dict
        similarity_thresholds: List of thresholds for redundancy counting
        output_dir: Output directory
        sae_name: Name of the SAE for title
    """
    print(f"\n{'='*70}")
    print("CREATING QUANTILE COMPARISON BARPLOT")
    print(f"{'='*70}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    
    quantiles = [1, 2, 3, 4]
    x = np.arange(len(quantiles))
    width = 0.35
    
    # Prepare data
    decoder_counts_07 = [quantile_results[q]['redundant_pairs'].get(0.7, 0) for q in quantiles]
    decoder_counts_09 = [quantile_results[q]['redundant_pairs'].get(0.9, 0) for q in quantiles]
    encoder_avg_sims = [quantile_results[q]['avg_encoder_pairwise_similarity'] for q in quantiles]
    
    # Left y-axis: Decoder redundancy counts (bars)
    bars1 = ax1.bar(x - width/2, decoder_counts_07, width, label='Decoder Redundancy (≥0.7)', 
                    color='steelblue', alpha=0.7, edgecolor='black')
    bars2 = ax1.bar(x + width/2, decoder_counts_09, width, label='Decoder Redundancy (≥0.9)', 
                    color='coral', alpha=0.7, edgecolor='black')
    
    ax1.set_xlabel('Frequency Quantile', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Redundant Feature Pairs', fontsize=12, fontweight='bold', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Q{q}' for q in quantiles])
    ax1.grid(axis='y', alpha=0.3)
    
    # Right y-axis: Average encoder cosine similarity (line)
    ax2 = ax1.twinx()
    line = ax2.plot(x, encoder_avg_sims, 'o-', color='green', linewidth=2, markersize=10, 
                    label='Avg Encoder Pairwise Similarity')
    ax2.set_ylabel('Average Encoder Pairwise Cosine Similarity', fontsize=12, fontweight='bold', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom', fontsize=9)
    
    for i, sim in enumerate(encoder_avg_sims):
        ax2.text(i, sim, f'{sim:.3f}', ha='center', va='bottom', fontsize=9, color='green')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax1.set_title(f'Quantile Redundancy Comparison: {sae_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_file = output_dir / "2_quantile_redundancy_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot_file.name}")


def get_feature_quantile_mapping(firing_stats: Dict, total_features: int) -> Dict[int, int]:
    """
    Create mapping from feature ID to quantile (1-4).
    
    Args:
        firing_stats: Dictionary with 'feature_rank_by_firing' key
        total_features: Total number of features
    
    Returns:
        Dictionary mapping feature_id -> quantile (1-4)
    """
    feature_rank = np.array(firing_stats['feature_rank_by_firing'])  # Ranked by frequency (descending)
    quantile_size = total_features // 4
    
    quantile_map = {}
    for idx, feature_id in enumerate(feature_rank):
        if idx < quantile_size:
            quantile_map[feature_id] = 1
        elif idx < 2 * quantile_size:
            quantile_map[feature_id] = 2
        elif idx < 3 * quantile_size:
            quantile_map[feature_id] = 3
        else:
            quantile_map[feature_id] = 4
    
    return quantile_map


def compare_redundant_vs_nonredundant(
    decoder_weights: np.ndarray,
    encoder_weights: np.ndarray,
    firing_stats: Dict,
    similarity_thresholds: List[float] = [0.7, 0.9],
    top_n: int = 10000,
    sample_size: int = 1000
) -> Dict:
    """
    Compare encoder similarity between redundant and non-redundant feature pairs.
    
    For redundant pairs (decoder sim >= threshold), ranks them by encoder difference
    (big to low, i.e., low encoder sim = high difference first).
    
    Args:
        decoder_weights: [d_sae, d_in] decoder weight matrix
        encoder_weights: [d_in, d_sae] encoder weight matrix
        firing_stats: Dictionary with 'feature_rank_by_firing' key
        similarity_thresholds: List of thresholds for redundancy identification
        top_n: Number of top features to analyze
        sample_size: Number of non-redundant pairs to sample
    
    Returns:
        Dictionary with comparison statistics and ranked redundant pairs with full metadata
    """
    print(f"\n{'='*70}")
    print("COMPARING REDUNDANT VS NON-REDUNDANT PAIRS")
    print(f"{'='*70}")
    
    # Get top N features
    feature_rank = np.array(firing_stats['feature_rank_by_firing'])
    top_n_features = feature_rank[:min(top_n, len(feature_rank))].tolist()
    n_features = len(top_n_features)
    
    print(f"  Analyzing top {n_features} features...")
    
    # Create quantile mapping for all features (not just top_n)
    total_features = len(feature_rank)
    quantile_map = get_feature_quantile_mapping(firing_stats, total_features)
    
    # Compute decoder similarity matrix
    decoder_similarity = compute_decoder_similarity_matrix(decoder_weights, top_n_features)
    encoder_similarity = compute_encoder_similarity_matrix(encoder_weights, top_n_features)
    
    results = {}
    
    for threshold in similarity_thresholds:
        # Find redundant pairs (decoder similarity >= threshold)
        upper_triangle = np.triu(decoder_similarity, k=1)
        redundant_mask = upper_triangle >= threshold
        
        # Get redundant pair indices with full metadata
        redundant_pairs_data = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if redundant_mask[i, j]:
                    feat_id_1 = top_n_features[i]
                    feat_id_2 = top_n_features[j]
                    decoder_sim = float(decoder_similarity[i, j])
                    encoder_sim = float(encoder_similarity[i, j])
                    quantile_1 = quantile_map.get(feat_id_1, 0)
                    quantile_2 = quantile_map.get(feat_id_2, 0)
                    
                    redundant_pairs_data.append({
                        'feature_id_1': int(feat_id_1),
                        'feature_id_2': int(feat_id_2),
                        'decoder_similarity': decoder_sim,
                        'encoder_similarity': encoder_sim,
                        'encoder_difference': 1.0 - encoder_sim,  # Higher = more different
                        'quantile_1': int(quantile_1),
                        'quantile_2': int(quantile_2),
                    })
        
        # Sort by encoder difference (big to low, i.e., most different first)
        redundant_pairs_data.sort(key=lambda x: x['encoder_difference'], reverse=True)
        
        # Sample non-redundant pairs
        non_redundant_mask = upper_triangle < threshold
        non_redundant_pairs = []
        non_redundant_encoder_sims = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if non_redundant_mask[i, j]:
                    non_redundant_pairs.append((top_n_features[i], top_n_features[j]))
                    non_redundant_encoder_sims.append(encoder_similarity[i, j])
        
        # Sample non-redundant pairs
        if len(non_redundant_pairs) > sample_size:
            sample_indices = np.random.choice(len(non_redundant_pairs), sample_size, replace=False)
            sampled_non_redundant_pairs = [non_redundant_pairs[i] for i in sample_indices]
            sampled_non_redundant_encoder_sims = [non_redundant_encoder_sims[i] for i in sample_indices]
        else:
            sampled_non_redundant_pairs = non_redundant_pairs
            sampled_non_redundant_encoder_sims = non_redundant_encoder_sims
        
        # Compute statistics
        redundant_encoder_sims = np.array([p['encoder_similarity'] for p in redundant_pairs_data])
        sampled_non_redundant_encoder_sims = np.array(sampled_non_redundant_encoder_sims)
        
        results[threshold] = {
            'n_redundant_pairs': len(redundant_pairs_data),
            'n_non_redundant_pairs_sampled': len(sampled_non_redundant_pairs),
            'redundant_pairs': redundant_pairs_data,  # ALL pairs, ranked by encoder difference (big to low)
            'redundant_encoder_similarity': {
                'mean': float(np.mean(redundant_encoder_sims)) if len(redundant_encoder_sims) > 0 else 0.0,
                'median': float(np.median(redundant_encoder_sims)) if len(redundant_encoder_sims) > 0 else 0.0,
                'std': float(np.std(redundant_encoder_sims)) if len(redundant_encoder_sims) > 0 else 0.0,
            },
            'non_redundant_encoder_similarity': {
                'mean': float(np.mean(sampled_non_redundant_encoder_sims)) if len(sampled_non_redundant_encoder_sims) > 0 else 0.0,
                'median': float(np.median(sampled_non_redundant_encoder_sims)) if len(sampled_non_redundant_encoder_sims) > 0 else 0.0,
                'std': float(np.std(sampled_non_redundant_encoder_sims)) if len(sampled_non_redundant_encoder_sims) > 0 else 0.0,
            }
        }
        
        print(f"  Threshold {threshold}:")
        print(f"    Redundant pairs: {len(redundant_pairs_data)}")
        print(f"    Redundant encoder sim: mean={results[threshold]['redundant_encoder_similarity']['mean']:.3f}, "
              f"median={results[threshold]['redundant_encoder_similarity']['median']:.3f}")
        print(f"    Non-redundant encoder sim: mean={results[threshold]['non_redundant_encoder_similarity']['mean']:.3f}, "
              f"median={results[threshold]['non_redundant_encoder_similarity']['median']:.3f}")
        if len(redundant_pairs_data) > 0:
            print(f"    Top 3 pairs by encoder difference:")
            for idx, pair in enumerate(redundant_pairs_data[:3]):
                print(f"      {idx+1}. Features ({pair['feature_id_1']}, {pair['feature_id_2']}): "
                      f"dec_sim={pair['decoder_similarity']:.3f}, enc_sim={pair['encoder_similarity']:.3f}, "
                      f"enc_diff={pair['encoder_difference']:.3f}, Q({pair['quantile_1']}, {pair['quantile_2']})")
    
    return results


def create_redundant_vs_nonredundant_plot(
    comparison_results: Dict,
    similarity_thresholds: List[float] = [0.7, 0.9],
    output_dir: Path = None,
    sae_name: str = ""
):
    """
    Create comparison plot for redundant vs non-redundant pairs.
    
    Args:
        comparison_results: Results from compare_redundant_vs_nonredundant
        similarity_thresholds: List of thresholds
        output_dir: Output directory
        sae_name: Name of the SAE for title
    """
    print(f"\n{'='*70}")
    print("CREATING REDUNDANT VS NON-REDUNDANT COMPARISON PLOT")
    print(f"{'='*70}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, len(similarity_thresholds), figsize=(6*len(similarity_thresholds), 6))
    if len(similarity_thresholds) == 1:
        axes = [axes]
    
    for idx, threshold in enumerate(similarity_thresholds):
        ax = axes[idx]
        
        results = comparison_results[threshold]
        
        # Prepare data
        redundant_mean = results['redundant_encoder_similarity']['mean']
        redundant_median = results['redundant_encoder_similarity']['median']
        non_redundant_mean = results['non_redundant_encoder_similarity']['mean']
        non_redundant_median = results['non_redundant_encoder_similarity']['median']
        
        x = ['Redundant\nPairs', 'Non-Redundant\nPairs']
        means = [redundant_mean, non_redundant_mean]
        medians = [redundant_median, non_redundant_median]
        
        x_pos = np.arange(len(x))
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, means, width, label='Mean', 
                       color='steelblue', alpha=0.7, edgecolor='black')
        bars2 = ax.bar(x_pos + width/2, medians, width, label='Median', 
                       color='coral', alpha=0.7, edgecolor='black')
        
        ax.set_ylabel('Encoder Cosine Similarity', fontsize=12, fontweight='bold')
        ax.set_title(f'Threshold {threshold} (n_redundant={results["n_redundant_pairs"]})', 
                     fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    fig.suptitle(f'Redundant vs Non-Redundant Feature Pairs: {sae_name}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_file = output_dir / "3_redundant_vs_nonredundant_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot_file.name}")


def compare_redundant_vs_nonredundant_by_quantile(
    decoder_weights: np.ndarray,
    encoder_weights: np.ndarray,
    firing_stats: Dict,
    similarity_thresholds: List[float] = [0.7, 0.9],
    top_n_per_quantile: int = 3000,
    sample_size: int = 1000
) -> Dict:
    """
    Compare encoder similarity between redundant and non-redundant feature pairs, 
    analyzed separately for each quantile.
    
    Args:
        decoder_weights: [d_sae, d_in] decoder weight matrix
        encoder_weights: [d_in, d_sae] encoder weight matrix
        firing_stats: Dictionary with 'feature_rank_by_firing' key
        similarity_thresholds: List of thresholds for redundancy identification
        top_n_per_quantile: Number of top features per quantile to analyze
        sample_size: Number of non-redundant pairs to sample per quantile
    
    Returns:
        Dictionary with comparison statistics per quantile
    """
    print(f"\n{'='*70}")
    print("COMPARING REDUNDANT VS NON-REDUNDANT PAIRS BY QUANTILE")
    print(f"{'='*70}")
    
    # Split features by quantiles
    quantiles = split_features_by_frequency_quantiles(firing_stats, top_n_per_quantile)
    
    results = {}
    
    for threshold in similarity_thresholds:
        results[threshold] = {}
        
        for quantile in [1, 2, 3, 4]:
            quantile_features = quantiles[quantile]
            n_features = len(quantile_features)
            
            if n_features == 0:
                continue
            
            print(f"\n  Analyzing Q{quantile} ({n_features} features) at threshold {threshold}...")
            
            # Compute similarity matrices for this quantile
            decoder_similarity = compute_decoder_similarity_matrix(decoder_weights, quantile_features)
            encoder_similarity = compute_encoder_similarity_matrix(encoder_weights, quantile_features)
            
            # Find redundant pairs (decoder similarity >= threshold)
            upper_triangle = np.triu(decoder_similarity, k=1)
            redundant_mask = upper_triangle >= threshold
            
            # Get redundant pairs
            redundant_pairs_data = []
            redundant_encoder_sims = []
            
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    if redundant_mask[i, j]:
                        feat_id_1 = quantile_features[i]
                        feat_id_2 = quantile_features[j]
                        decoder_sim = float(decoder_similarity[i, j])
                        encoder_sim = float(encoder_similarity[i, j])
                        
                        redundant_pairs_data.append({
                            'feature_id_1': int(feat_id_1),
                            'feature_id_2': int(feat_id_2),
                            'decoder_similarity': decoder_sim,
                            'encoder_similarity': encoder_sim,
                            'encoder_difference': 1.0 - encoder_sim,
                            'quantile_1': quantile,
                            'quantile_2': quantile,
                        })
                        redundant_encoder_sims.append(encoder_sim)
            
            # Get non-redundant pairs
            non_redundant_mask = upper_triangle < threshold
            non_redundant_encoder_sims = []
            
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    if non_redundant_mask[i, j]:
                        non_redundant_encoder_sims.append(encoder_similarity[i, j])
            
            # Sample non-redundant pairs
            if len(non_redundant_encoder_sims) > sample_size:
                sample_indices = np.random.choice(len(non_redundant_encoder_sims), sample_size, replace=False)
                sampled_non_redundant_encoder_sims = [non_redundant_encoder_sims[i] for i in sample_indices]
            else:
                sampled_non_redundant_encoder_sims = non_redundant_encoder_sims
            
            redundant_encoder_sims = np.array(redundant_encoder_sims)
            sampled_non_redundant_encoder_sims = np.array(sampled_non_redundant_encoder_sims)
            
            results[threshold][quantile] = {
                'n_redundant_pairs': len(redundant_pairs_data),
                'n_non_redundant_pairs_sampled': len(sampled_non_redundant_encoder_sims),
                'redundant_pairs': redundant_pairs_data,
                'redundant_encoder_similarity': {
                    'mean': float(np.mean(redundant_encoder_sims)) if len(redundant_encoder_sims) > 0 else 0.0,
                    'median': float(np.median(redundant_encoder_sims)) if len(redundant_encoder_sims) > 0 else 0.0,
                    'std': float(np.std(redundant_encoder_sims)) if len(redundant_encoder_sims) > 0 else 0.0,
                },
                'non_redundant_encoder_similarity': {
                    'mean': float(np.mean(sampled_non_redundant_encoder_sims)) if len(sampled_non_redundant_encoder_sims) > 0 else 0.0,
                    'median': float(np.median(sampled_non_redundant_encoder_sims)) if len(sampled_non_redundant_encoder_sims) > 0 else 0.0,
                    'std': float(np.std(sampled_non_redundant_encoder_sims)) if len(sampled_non_redundant_encoder_sims) > 0 else 0.0,
                }
            }
            
            print(f"    Q{quantile}: {len(redundant_pairs_data)} redundant pairs")
            if len(redundant_encoder_sims) > 0:
                print(f"      Redundant encoder sim: mean={results[threshold][quantile]['redundant_encoder_similarity']['mean']:.3f}")
            print(f"      Non-redundant encoder sim: mean={results[threshold][quantile]['non_redundant_encoder_similarity']['mean']:.3f}")
    
    return results


def create_redundant_vs_nonredundant_plot_by_quantile(
    comparison_results: Dict,
    similarity_thresholds: List[float] = [0.7, 0.9],
    output_dir: Path = None,
    sae_name: str = ""
):
    """
    Create comparison plot for redundant vs non-redundant pairs, per quantile.
    One PNG per threshold, showing all 4 quantiles together.
    
    Args:
        comparison_results: Results from compare_redundant_vs_nonredundant_by_quantile
        similarity_thresholds: List of thresholds
        output_dir: Output directory
        sae_name: Name of the SAE for title
    """
    print(f"\n{'='*70}")
    print("CREATING REDUNDANT VS NON-REDUNDANT COMPARISON PLOT BY QUANTILE")
    print(f"{'='*70}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create one plot per threshold
    for threshold in similarity_thresholds:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        quantiles = [1, 2, 3, 4]
        x = np.arange(len(quantiles))
        width = 0.35
        
        redundant_means = []
        non_redundant_means = []
        redundant_medians = []
        non_redundant_medians = []
        n_redundant_list = []
        
        for quantile in quantiles:
            if quantile in comparison_results[threshold]:
                results = comparison_results[threshold][quantile]
                redundant_means.append(results['redundant_encoder_similarity']['mean'])
                non_redundant_means.append(results['non_redundant_encoder_similarity']['mean'])
                redundant_medians.append(results['redundant_encoder_similarity']['median'])
                non_redundant_medians.append(results['non_redundant_encoder_similarity']['median'])
                n_redundant_list.append(results['n_redundant_pairs'])
            else:
                redundant_means.append(0.0)
                non_redundant_means.append(0.0)
                redundant_medians.append(0.0)
                non_redundant_medians.append(0.0)
                n_redundant_list.append(0)
        
        # Plot means as bars
        bars1 = ax.bar(x - width/2, redundant_means, width, label='Redundant (Mean)', 
                       color='steelblue', alpha=0.7, edgecolor='black')
        bars2 = ax.bar(x + width/2, non_redundant_means, width, label='Non-Redundant (Mean)', 
                       color='coral', alpha=0.7, edgecolor='black')
        
        # Plot medians as lines with markers
        ax.plot(x - width/2, redundant_medians, 'o-', color='darkblue', linewidth=2, 
                markersize=8, label='Redundant (Median)', alpha=0.8)
        ax.plot(x + width/2, non_redundant_medians, 's-', color='darkred', linewidth=2, 
                markersize=8, label='Non-Redundant (Median)', alpha=0.8)
        
        ax.set_xlabel('Frequency Quantile', fontsize=12, fontweight='bold')
        ax.set_ylabel('Encoder Cosine Similarity', fontsize=12, fontweight='bold')
        ax.set_title(f'Redundant vs Non-Redundant Pairs by Quantile (Threshold {threshold})\n{sae_name}', 
                     fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Q{q}\n(n={n_redundant_list[i]})' for i, q in enumerate(quantiles)])
        ax.legend(fontsize=10, loc='best')
        ax.grid(alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for i, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Add median value labels
        for i, (red_med, nonred_med) in enumerate(zip(redundant_medians, non_redundant_medians)):
            if red_med > 0:
                ax.text(i - width/2, red_med, f'{red_med:.3f}', 
                       ha='center', va='bottom', fontsize=7, color='darkblue')
            if nonred_med > 0:
                ax.text(i + width/2, nonred_med, f'{nonred_med:.3f}', 
                       ha='center', va='bottom', fontsize=7, color='darkred')
        
        plt.tight_layout()
        plot_file = output_dir / f"3_redundant_vs_nonredundant_comparison_by_quantile_threshold_{threshold}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {plot_file.name}")


# ============================================================================
# Dominance-based quantile functions (mirror of frequency-based functions)
# ============================================================================

def create_8_panel_quantile_heatmaps_dominance(
    quantile_results: Dict[int, Dict],
    output_dir: Path,
    sae_name: str
):
    """
    Create 8-panel heatmap (4 quantiles × decoder/encoder) for dominance-based quantiles.
    
    Args:
        quantile_results: Dictionary mapping quantile (1-4) to results dict
        output_dir: Output directory
        sae_name: Name of the SAE for title
    """
    print(f"\n{'='*70}")
    print("CREATING 8-PANEL DOMINANCE QUANTILE HEATMAPS")
    print(f"{'='*70}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    fig.suptitle(f'Decoder vs Encoder Similarity by Dominance Quantile\n{sae_name}', 
                 fontsize=16, fontweight='bold')
    
    for quantile in [1, 2, 3, 4]:
        q_idx = quantile - 1  # 0-indexed
        
        results = quantile_results[quantile]
        decoder_sim = np.array(results['decoder_similarity_matrix'])
        encoder_sim = np.array(results['encoder_similarity_matrix'])
        
        # Decoder heatmap (left column)
        ax = axes[q_idx, 0]
        im = ax.imshow(decoder_sim, cmap='viridis', aspect='auto', vmin=0, vmax=1)
        ax.set_title(f'Q{quantile} Decoder Similarity (n={results["n_features"]})', 
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature Index (ranked by dominance)', fontsize=10)
        ax.set_ylabel('Feature Index (ranked by dominance)', fontsize=10)
        plt.colorbar(im, ax=ax, label='Cosine Similarity')
        
        # Encoder heatmap (right column)
        ax = axes[q_idx, 1]
        im = ax.imshow(encoder_sim, cmap='viridis', aspect='auto', vmin=0, vmax=1)
        ax.set_title(f'Q{quantile} Encoder Similarity (n={results["n_features"]})', 
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature Index (ranked by dominance)', fontsize=10)
        ax.set_ylabel('Feature Index (ranked by dominance)', fontsize=10)
        plt.colorbar(im, ax=ax, label='Cosine Similarity')
    
    plt.tight_layout()
    plot_file = output_dir / "1_8_panel_quantile_heatmaps.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot_file.name}")


def create_quantile_comparison_barplot_dominance(
    quantile_results: Dict[int, Dict],
    similarity_thresholds: List[float] = [0.7, 0.9],
    output_dir: Path = None,
    sae_name: str = ""
):
    """
    Create bar plot with dual y-axis showing decoder redundancy and encoder cosine similarity for dominance quantiles.
    
    Args:
        quantile_results: Dictionary mapping quantile (1-4) to results dict
        similarity_thresholds: List of thresholds for redundancy counting
        output_dir: Output directory
        sae_name: Name of the SAE for title
    """
    print(f"\n{'='*70}")
    print("CREATING DOMINANCE QUANTILE COMPARISON BARPLOT")
    print(f"{'='*70}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    
    quantiles = [1, 2, 3, 4]
    x = np.arange(len(quantiles))
    width = 0.35
    
    # Prepare data
    decoder_counts_07 = [quantile_results[q]['redundant_pairs'].get(0.7, 0) for q in quantiles]
    decoder_counts_09 = [quantile_results[q]['redundant_pairs'].get(0.9, 0) for q in quantiles]
    encoder_avg_sims = [quantile_results[q]['avg_encoder_pairwise_similarity'] for q in quantiles]
    
    # Left y-axis: Decoder redundancy counts (bars)
    bars1 = ax1.bar(x - width/2, decoder_counts_07, width, label='Decoder Redundancy (≥0.7)', 
                    color='steelblue', alpha=0.7, edgecolor='black')
    bars2 = ax1.bar(x + width/2, decoder_counts_09, width, label='Decoder Redundancy (≥0.9)', 
                    color='coral', alpha=0.7, edgecolor='black')
    
    ax1.set_xlabel('Dominance Quantile', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Redundant Feature Pairs', fontsize=12, fontweight='bold', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Q{q}' for q in quantiles])
    ax1.grid(axis='y', alpha=0.3)
    
    # Right y-axis: Average encoder cosine similarity (line)
    ax2 = ax1.twinx()
    line = ax2.plot(x, encoder_avg_sims, 'o-', color='green', linewidth=2, markersize=10, 
                    label='Avg Encoder Pairwise Similarity')
    ax2.set_ylabel('Average Encoder Pairwise Cosine Similarity', fontsize=12, fontweight='bold', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom', fontsize=9)
    
    for i, sim in enumerate(encoder_avg_sims):
        ax2.text(i, sim, f'{sim:.3f}', ha='center', va='bottom', fontsize=9, color='green')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax1.set_title(f'Dominance Quantile Redundancy Comparison: {sae_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_file = output_dir / "2_quantile_redundancy_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot_file.name}")


def get_feature_quantile_mapping_dominance(dominance_scores: np.ndarray, total_features: int) -> Dict[int, int]:
    """
    Create mapping from feature ID to dominance quantile (1-4).
    
    Args:
        dominance_scores: Array of mean dominance per feature
        total_features: Total number of features
    
    Returns:
        Dictionary mapping feature_id -> quantile (1-4)
    """
    # Sort features by dominance (descending)
    feature_indices = np.arange(len(dominance_scores))
    sorted_indices = feature_indices[np.argsort(dominance_scores)[::-1]]
    quantile_size = total_features // 4
    
    quantile_map = {}
    for idx, feature_id in enumerate(sorted_indices):
        if idx < quantile_size:
            quantile_map[feature_id] = 1
        elif idx < 2 * quantile_size:
            quantile_map[feature_id] = 2
        elif idx < 3 * quantile_size:
            quantile_map[feature_id] = 3
        else:
            quantile_map[feature_id] = 4
    
    return quantile_map


def compare_redundant_vs_nonredundant_dominance(
    decoder_weights: np.ndarray,
    encoder_weights: np.ndarray,
    dominance_scores: np.ndarray,
    similarity_thresholds: List[float] = [0.7, 0.9],
    top_n: int = 10000,
    sample_size: int = 1000
) -> Dict:
    """
    Compare encoder similarity between redundant and non-redundant feature pairs, using dominance ranking.
    
    Args:
        decoder_weights: [d_sae, d_in] decoder weight matrix
        encoder_weights: [d_in, d_sae] encoder weight matrix
        dominance_scores: Array of mean dominance per feature
        similarity_thresholds: List of thresholds for redundancy identification
        top_n: Number of top features to analyze
        sample_size: Number of non-redundant pairs to sample
    
    Returns:
        Dictionary with comparison statistics and ranked redundant pairs with full metadata
    """
    print(f"\n{'='*70}")
    print("COMPARING REDUNDANT VS NON-REDUNDANT PAIRS (DOMINANCE-BASED)")
    print(f"{'='*70}")
    
    # Get top N features by dominance
    feature_indices = np.arange(len(dominance_scores))
    sorted_indices = feature_indices[np.argsort(dominance_scores)[::-1]]
    top_n_features = sorted_indices[:min(top_n, len(sorted_indices))].tolist()
    n_features = len(top_n_features)
    
    print(f"  Analyzing top {n_features} features by dominance...")
    
    # Create quantile mapping for all features
    total_features = len(dominance_scores)
    quantile_map = get_feature_quantile_mapping_dominance(dominance_scores, total_features)
    
    # Compute decoder similarity matrix
    decoder_similarity = compute_decoder_similarity_matrix(decoder_weights, top_n_features)
    encoder_similarity = compute_encoder_similarity_matrix(encoder_weights, top_n_features)
    
    results = {}
    
    for threshold in similarity_thresholds:
        # Find redundant pairs (decoder similarity >= threshold)
        upper_triangle = np.triu(decoder_similarity, k=1)
        redundant_mask = upper_triangle >= threshold
        
        # Get redundant pair indices with full metadata
        redundant_pairs_data = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if redundant_mask[i, j]:
                    feat_id_1 = top_n_features[i]
                    feat_id_2 = top_n_features[j]
                    decoder_sim = float(decoder_similarity[i, j])
                    encoder_sim = float(encoder_similarity[i, j])
                    quantile_1 = quantile_map.get(feat_id_1, 0)
                    quantile_2 = quantile_map.get(feat_id_2, 0)
                    
                    redundant_pairs_data.append({
                        'feature_id_1': int(feat_id_1),
                        'feature_id_2': int(feat_id_2),
                        'decoder_similarity': decoder_sim,
                        'encoder_similarity': encoder_sim,
                        'encoder_difference': 1.0 - encoder_sim,  # Higher = more different
                        'quantile_1': int(quantile_1),
                        'quantile_2': int(quantile_2),
                    })
        
        # Sort by encoder difference (big to low, i.e., most different first)
        redundant_pairs_data.sort(key=lambda x: x['encoder_difference'], reverse=True)
        
        # Sample non-redundant pairs
        non_redundant_mask = upper_triangle < threshold
        non_redundant_pairs = []
        non_redundant_encoder_sims = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if non_redundant_mask[i, j]:
                    non_redundant_pairs.append((top_n_features[i], top_n_features[j]))
                    non_redundant_encoder_sims.append(encoder_similarity[i, j])
        
        # Sample non-redundant pairs
        if len(non_redundant_pairs) > sample_size:
            sample_indices = np.random.choice(len(non_redundant_pairs), sample_size, replace=False)
            sampled_non_redundant_pairs = [non_redundant_pairs[i] for i in sample_indices]
            sampled_non_redundant_encoder_sims = [non_redundant_encoder_sims[i] for i in sample_indices]
        else:
            sampled_non_redundant_pairs = non_redundant_pairs
            sampled_non_redundant_encoder_sims = non_redundant_encoder_sims
        
        # Compute statistics
        redundant_encoder_sims = np.array([p['encoder_similarity'] for p in redundant_pairs_data])
        sampled_non_redundant_encoder_sims = np.array(sampled_non_redundant_encoder_sims)
        
        results[threshold] = {
            'n_redundant_pairs': len(redundant_pairs_data),
            'n_non_redundant_pairs_sampled': len(sampled_non_redundant_pairs),
            'redundant_pairs': redundant_pairs_data,  # ALL pairs, ranked by encoder difference (big to low)
            'redundant_encoder_similarity': {
                'mean': float(np.mean(redundant_encoder_sims)) if len(redundant_encoder_sims) > 0 else 0.0,
                'median': float(np.median(redundant_encoder_sims)) if len(redundant_encoder_sims) > 0 else 0.0,
                'std': float(np.std(redundant_encoder_sims)) if len(redundant_encoder_sims) > 0 else 0.0,
            },
            'non_redundant_encoder_similarity': {
                'mean': float(np.mean(sampled_non_redundant_encoder_sims)) if len(sampled_non_redundant_encoder_sims) > 0 else 0.0,
                'median': float(np.median(sampled_non_redundant_encoder_sims)) if len(sampled_non_redundant_encoder_sims) > 0 else 0.0,
                'std': float(np.std(sampled_non_redundant_encoder_sims)) if len(sampled_non_redundant_encoder_sims) > 0 else 0.0,
            }
        }
        
        print(f"  Threshold {threshold}:")
        print(f"    Redundant pairs: {len(redundant_pairs_data)}")
        print(f"    Redundant encoder sim: mean={results[threshold]['redundant_encoder_similarity']['mean']:.3f}, "
              f"median={results[threshold]['redundant_encoder_similarity']['median']:.3f}")
        print(f"    Non-redundant encoder sim: mean={results[threshold]['non_redundant_encoder_similarity']['mean']:.3f}, "
              f"median={results[threshold]['non_redundant_encoder_similarity']['median']:.3f}")
        if len(redundant_pairs_data) > 0:
            print(f"    Top 3 pairs by encoder difference:")
            for idx, pair in enumerate(redundant_pairs_data[:3]):
                print(f"      {idx+1}. Features ({pair['feature_id_1']}, {pair['feature_id_2']}): "
                      f"dec_sim={pair['decoder_similarity']:.3f}, enc_sim={pair['encoder_similarity']:.3f}, "
                      f"enc_diff={pair['encoder_difference']:.3f}, Q({pair['quantile_1']}, {pair['quantile_2']})")
    
    return results


def create_redundant_vs_nonredundant_plot_dominance(
    comparison_results: Dict,
    similarity_thresholds: List[float] = [0.7, 0.9],
    output_dir: Path = None,
    sae_name: str = ""
):
    """
    Create comparison plot for redundant vs non-redundant pairs (dominance-based).
    
    Args:
        comparison_results: Results from compare_redundant_vs_nonredundant_dominance
        similarity_thresholds: List of thresholds
        output_dir: Output directory
        sae_name: Name of the SAE for title
    """
    print(f"\n{'='*70}")
    print("CREATING REDUNDANT VS NON-REDUNDANT COMPARISON PLOT (DOMINANCE-BASED)")
    print(f"{'='*70}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, len(similarity_thresholds), figsize=(6*len(similarity_thresholds), 6))
    if len(similarity_thresholds) == 1:
        axes = [axes]
    
    for idx, threshold in enumerate(similarity_thresholds):
        ax = axes[idx]
        
        results = comparison_results[threshold]
        
        # Prepare data
        redundant_mean = results['redundant_encoder_similarity']['mean']
        redundant_median = results['redundant_encoder_similarity']['median']
        non_redundant_mean = results['non_redundant_encoder_similarity']['mean']
        non_redundant_median = results['non_redundant_encoder_similarity']['median']
        
        x = ['Redundant\nPairs', 'Non-Redundant\nPairs']
        means = [redundant_mean, non_redundant_mean]
        medians = [redundant_median, non_redundant_median]
        
        x_pos = np.arange(len(x))
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, means, width, label='Mean', 
                       color='steelblue', alpha=0.7, edgecolor='black')
        bars2 = ax.bar(x_pos + width/2, medians, width, label='Median', 
                       color='coral', alpha=0.7, edgecolor='black')
        
        ax.set_ylabel('Encoder Cosine Similarity', fontsize=12, fontweight='bold')
        ax.set_title(f'Threshold {threshold} (n_redundant={results["n_redundant_pairs"]})', 
                     fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    fig.suptitle(f'Redundant vs Non-Redundant Feature Pairs (Dominance-Based): {sae_name}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_file = output_dir / "3_redundant_vs_nonredundant_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot_file.name}")


def compare_redundant_vs_nonredundant_by_quantile_dominance(
    decoder_weights: np.ndarray,
    encoder_weights: np.ndarray,
    dominance_scores: np.ndarray,
    similarity_thresholds: List[float] = [0.7, 0.9],
    top_n_per_quantile: int = 3000,
    sample_size: int = 1000
) -> Dict:
    """
    Compare encoder similarity between redundant and non-redundant feature pairs, 
    analyzed separately for each dominance quantile.
    
    Args:
        decoder_weights: [d_sae, d_in] decoder weight matrix
        encoder_weights: [d_in, d_sae] encoder weight matrix
        dominance_scores: Array of mean dominance per feature
        similarity_thresholds: List of thresholds for redundancy identification
        top_n_per_quantile: Number of top features per quantile to analyze
        sample_size: Number of non-redundant pairs to sample per quantile
    
    Returns:
        Dictionary with comparison statistics per quantile
    """
    print(f"\n{'='*70}")
    print("COMPARING REDUNDANT VS NON-REDUNDANT PAIRS BY DOMINANCE QUANTILE")
    print(f"{'='*70}")
    
    # Split features by dominance quantiles
    quantiles = split_features_by_dominance_quantiles(dominance_scores, top_n_per_quantile)
    
    results = {}
    
    for threshold in similarity_thresholds:
        results[threshold] = {}
        
        for quantile in [1, 2, 3, 4]:
            quantile_features = quantiles[quantile]
            n_features = len(quantile_features)
            
            if n_features == 0:
                continue
            
            print(f"\n  Analyzing Q{quantile} ({n_features} features) at threshold {threshold}...")
            
            # Compute similarity matrices for this quantile
            decoder_similarity = compute_decoder_similarity_matrix(decoder_weights, quantile_features)
            encoder_similarity = compute_encoder_similarity_matrix(encoder_weights, quantile_features)
            
            # Find redundant pairs (decoder similarity >= threshold)
            upper_triangle = np.triu(decoder_similarity, k=1)
            redundant_mask = upper_triangle >= threshold
            
            # Get redundant pairs
            redundant_pairs_data = []
            redundant_encoder_sims = []
            
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    if redundant_mask[i, j]:
                        feat_id_1 = quantile_features[i]
                        feat_id_2 = quantile_features[j]
                        decoder_sim = float(decoder_similarity[i, j])
                        encoder_sim = float(encoder_similarity[i, j])
                        
                        redundant_pairs_data.append({
                            'feature_id_1': int(feat_id_1),
                            'feature_id_2': int(feat_id_2),
                            'decoder_similarity': decoder_sim,
                            'encoder_similarity': encoder_sim,
                            'encoder_difference': 1.0 - encoder_sim,
                            'quantile_1': quantile,
                            'quantile_2': quantile,
                        })
                        redundant_encoder_sims.append(encoder_sim)
            
            # Get non-redundant pairs
            non_redundant_mask = upper_triangle < threshold
            non_redundant_encoder_sims = []
            
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    if non_redundant_mask[i, j]:
                        non_redundant_encoder_sims.append(encoder_similarity[i, j])
            
            # Sample non-redundant pairs
            if len(non_redundant_encoder_sims) > sample_size:
                sample_indices = np.random.choice(len(non_redundant_encoder_sims), sample_size, replace=False)
                sampled_non_redundant_encoder_sims = [non_redundant_encoder_sims[i] for i in sample_indices]
            else:
                sampled_non_redundant_encoder_sims = non_redundant_encoder_sims
            
            redundant_encoder_sims = np.array(redundant_encoder_sims)
            sampled_non_redundant_encoder_sims = np.array(sampled_non_redundant_encoder_sims)
            
            results[threshold][quantile] = {
                'n_redundant_pairs': len(redundant_pairs_data),
                'n_non_redundant_pairs_sampled': len(sampled_non_redundant_encoder_sims),
                'redundant_pairs': redundant_pairs_data,
                'redundant_encoder_similarity': {
                    'mean': float(np.mean(redundant_encoder_sims)) if len(redundant_encoder_sims) > 0 else 0.0,
                    'median': float(np.median(redundant_encoder_sims)) if len(redundant_encoder_sims) > 0 else 0.0,
                    'std': float(np.std(redundant_encoder_sims)) if len(redundant_encoder_sims) > 0 else 0.0,
                },
                'non_redundant_encoder_similarity': {
                    'mean': float(np.mean(sampled_non_redundant_encoder_sims)) if len(sampled_non_redundant_encoder_sims) > 0 else 0.0,
                    'median': float(np.median(sampled_non_redundant_encoder_sims)) if len(sampled_non_redundant_encoder_sims) > 0 else 0.0,
                    'std': float(np.std(sampled_non_redundant_encoder_sims)) if len(sampled_non_redundant_encoder_sims) > 0 else 0.0,
                }
            }
            
            print(f"    Q{quantile}: {len(redundant_pairs_data)} redundant pairs")
            if len(redundant_encoder_sims) > 0:
                print(f"      Redundant encoder sim: mean={results[threshold][quantile]['redundant_encoder_similarity']['mean']:.3f}")
            print(f"      Non-redundant encoder sim: mean={results[threshold][quantile]['non_redundant_encoder_similarity']['mean']:.3f}")
    
    return results


def create_redundant_vs_nonredundant_plot_by_quantile_dominance(
    comparison_results: Dict,
    similarity_thresholds: List[float] = [0.7, 0.9],
    output_dir: Path = None,
    sae_name: str = ""
):
    """
    Create comparison plot for redundant vs non-redundant pairs, per dominance quantile.
    One PNG per threshold, showing all 4 quantiles together.
    
    Args:
        comparison_results: Results from compare_redundant_vs_nonredundant_by_quantile_dominance
        similarity_thresholds: List of thresholds
        output_dir: Output directory
        sae_name: Name of the SAE for title
    """
    print(f"\n{'='*70}")
    print("CREATING REDUNDANT VS NON-REDUNDANT COMPARISON PLOT BY DOMINANCE QUANTILE")
    print(f"{'='*70}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create one plot per threshold
    for threshold in similarity_thresholds:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        quantiles = [1, 2, 3, 4]
        x = np.arange(len(quantiles))
        width = 0.35
        
        redundant_means = []
        non_redundant_means = []
        redundant_medians = []
        non_redundant_medians = []
        n_redundant_list = []
        
        for quantile in quantiles:
            if quantile in comparison_results[threshold]:
                results = comparison_results[threshold][quantile]
                redundant_means.append(results['redundant_encoder_similarity']['mean'])
                non_redundant_means.append(results['non_redundant_encoder_similarity']['mean'])
                redundant_medians.append(results['redundant_encoder_similarity']['median'])
                non_redundant_medians.append(results['non_redundant_encoder_similarity']['median'])
                n_redundant_list.append(results['n_redundant_pairs'])
            else:
                redundant_means.append(0.0)
                non_redundant_means.append(0.0)
                redundant_medians.append(0.0)
                non_redundant_medians.append(0.0)
                n_redundant_list.append(0)
        
        # Plot means as bars
        bars1 = ax.bar(x - width/2, redundant_means, width, label='Redundant (Mean)', 
                       color='steelblue', alpha=0.7, edgecolor='black')
        bars2 = ax.bar(x + width/2, non_redundant_means, width, label='Non-Redundant (Mean)', 
                       color='coral', alpha=0.7, edgecolor='black')
        
        # Plot medians as lines with markers
        ax.plot(x - width/2, redundant_medians, 'o-', color='darkblue', linewidth=2, 
                markersize=8, label='Redundant (Median)', alpha=0.8)
        ax.plot(x + width/2, non_redundant_medians, 's-', color='darkred', linewidth=2, 
                markersize=8, label='Non-Redundant (Median)', alpha=0.8)
        
        ax.set_xlabel('Dominance Quantile', fontsize=12, fontweight='bold')
        ax.set_ylabel('Encoder Cosine Similarity', fontsize=12, fontweight='bold')
        ax.set_title(f'Redundant vs Non-Redundant Pairs by Dominance Quantile (Threshold {threshold})\n{sae_name}', 
                     fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Q{q}\n(n={n_redundant_list[i]})' for i, q in enumerate(quantiles)])
        ax.legend(fontsize=10, loc='best')
        ax.grid(alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for i, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Add median value labels
        for i, (red_med, nonred_med) in enumerate(zip(redundant_medians, non_redundant_medians)):
            if red_med > 0:
                ax.text(i - width/2, red_med, f'{red_med:.3f}', 
                       ha='center', va='bottom', fontsize=7, color='darkblue')
            if nonred_med > 0:
                ax.text(i + width/2, nonred_med, f'{nonred_med:.3f}', 
                       ha='center', va='bottom', fontsize=7, color='darkred')
        
        plt.tight_layout()
        plot_file = output_dir / f"3_redundant_vs_nonredundant_comparison_by_quantile_threshold_{threshold}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {plot_file.name}")

