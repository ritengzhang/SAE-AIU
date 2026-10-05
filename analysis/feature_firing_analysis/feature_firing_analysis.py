#!/usr/bin/env python3
"""
Feature Firing Analysis Helper Functions

Core analysis functions for analyzing feature firing patterns in SAEs.
Extracted from feature_firing_analysis_no_inter_sae_bench.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm.auto import tqdm
import torch
import time
from datetime import datetime
from collections import defaultdict
import re

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


def compute_firing_statistics(
    sae,
    store,
    n_batches: int = 30,
    thresholds: List[float] = [0.0, 0.01],
    device: str = "cpu"
) -> Dict:
    """
    Compute firing frequency and magnitude statistics for all features.
    
    Args:
        sae: SAE object with encode() method
        store: ActivationsStore for collecting activations
        n_batches: Number of batches to analyze
        thresholds: List of thresholds for counting fires (e.g., [0.0, 0.01])
        device: Device to use
        
    Returns:
        Dictionary with firing statistics for each threshold
    """
    print(f"\n{'='*70}")
    print(f"COMPUTING FIRING STATISTICS ({n_batches} batches)")
    print(f"{'='*70}")
    
    d_sae = sae.cfg.d_sae
    
    # Initialize counters for each threshold
    firing_counts = {thresh: np.zeros(d_sae, dtype=np.int64) for thresh in thresholds}
    activation_sums = np.zeros(d_sae, dtype=np.float64)
    activation_sums_squared = np.zeros(d_sae, dtype=np.float64)
    total_tokens = 0
    
    # Storage for magnitude statistics (only activations > 0)
    all_activations = {i: [] for i in range(d_sae)}
    
    print("Collecting firing data...")
    start_time = time.time()
    print(f"[TIMING] Started collecting firing data at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
        batch_activations = store.next_batch()
        batch_size = batch_activations.shape[0]
        
        with torch.no_grad():
            batch_activations = batch_activations.to(device)
            sae_acts = sae.encode(batch_activations)  # [batch_size, d_sae]
        
        sae_acts_np = sae_acts.cpu().numpy()
        
        # Count fires for each threshold
        for thresh in thresholds:
            if thresh == 0.0:
                fires = (sae_acts_np > thresh).astype(np.int64)
            else:
                fires = (np.abs(sae_acts_np) > thresh).astype(np.int64)
            firing_counts[thresh] += fires.sum(axis=0)
        
        # Accumulate for magnitude statistics
        activation_sums += sae_acts_np.sum(axis=0)
        activation_sums_squared += (sae_acts_np ** 2).sum(axis=0)
        
        # Accumulate absolute magnitude sums
        abs_magnitude_sums = np.abs(sae_acts_np).sum(axis=0)
        if batch_idx == 0:
            total_abs_magnitude_sums = abs_magnitude_sums.copy()
        else:
            total_abs_magnitude_sums += abs_magnitude_sums
        
        # Store activations > 0 for magnitude statistics
        for feat_idx in range(d_sae):
            feat_acts = sae_acts_np[:, feat_idx]
            positive_acts = feat_acts[feat_acts > 0]
            if len(positive_acts) > 0:
                all_activations[feat_idx].extend(positive_acts.tolist())
        
        total_tokens += batch_size
    
    elapsed = time.time() - start_time
    print(f"[TIMING] Finished collecting firing data at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Took {elapsed/60:.2f} minutes ({elapsed:.2f} seconds)")
    
    # Compute magnitude statistics
    print("Computing magnitude statistics...")
    start_time = time.time()
    print(f"[TIMING] Started computing magnitude stats at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    magnitude_stats = {}
    for feat_idx in tqdm(range(d_sae), desc="Computing magnitude stats"):
        acts = all_activations[feat_idx]
        if len(acts) > 0:
            acts_array = np.array(acts)
            magnitude_stats[feat_idx] = {
                'mean': float(np.mean(acts_array)),
                'median': float(np.median(acts_array)),
                'std': float(np.std(acts_array)),
                'max': float(np.max(acts_array)),
                'min': float(np.min(acts_array)),
                'n_activations': len(acts),
            }
        else:
            magnitude_stats[feat_idx] = {
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'max': 0.0,
                'min': 0.0,
                'n_activations': 0,
            }
    
    elapsed = time.time() - start_time
    print(f"[TIMING] Finished computing magnitude stats at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Took {elapsed/60:.2f} minutes ({elapsed:.2f} seconds)")
    
    # Rank features by firing count (threshold 0.0)
    firing_counts_0 = firing_counts[0.0]
    feature_rank_by_firing = np.argsort(firing_counts_0)[::-1]
    features_that_fire = np.where(firing_counts_0 > 0)[0]
    
    # Compute cumulative magnitude distribution
    abs_magnitude_sums = total_abs_magnitude_sums
    feature_rank_by_magnitude = np.argsort(abs_magnitude_sums)[::-1]
    total_magnitude = float(np.sum(abs_magnitude_sums))
    
    cumulative_magnitude = np.cumsum(abs_magnitude_sums[feature_rank_by_magnitude])
    cumulative_magnitude_pct = cumulative_magnitude / total_magnitude if total_magnitude > 0 else np.zeros_like(cumulative_magnitude)
    feature_pct = np.arange(1, d_sae + 1) / d_sae
    
    results = {
        'firing_counts': {thresh: firing_counts[thresh].tolist() for thresh in thresholds},
        'firing_frequencies': {thresh: (firing_counts[thresh] / total_tokens).tolist() 
                               for thresh in thresholds},
        'magnitude_stats': magnitude_stats,
        'feature_rank_by_firing': feature_rank_by_firing.tolist(),
        'features_that_fire': features_that_fire.tolist(),
        'total_tokens': total_tokens,
        'n_batches': n_batches,
        'd_sae': d_sae,
        'thresholds': thresholds,
        'abs_magnitude_sums': abs_magnitude_sums.tolist(),
        'feature_rank_by_magnitude': feature_rank_by_magnitude.tolist(),
        'total_magnitude': float(total_magnitude),
        'cumulative_magnitude_pct': cumulative_magnitude_pct.tolist(),
        'feature_pct': feature_pct.tolist(),
    }
    
    print(f"✓ Computed firing statistics")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Features that fire (threshold 0.0): {len(features_that_fire):,}/{d_sae:,}")
    print(f"  Top firing feature: {feature_rank_by_firing[0]} "
          f"({firing_counts_0[feature_rank_by_firing[0]]:,} fires)")
    
    return results


def create_firing_frequency_distribution_plots(
    stats_dict: Dict[str, Dict],
    output_dir: Path
):
    """Create firing frequency distribution plots."""
    print(f"\n{'='*70}")
    print("CREATING FIRING FREQUENCY DISTRIBUTION PLOTS")
    print(f"{'='*70}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    first_exp = next(iter(stats_dict.values()))
    thresholds = first_exp['thresholds']
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    for thresh in thresholds:
        thresh_str = f"{thresh:.2f}".replace('.', '_')
        
        all_freqs = {}
        for exp_name, stats in stats_dict.items():
            freqs = np.array(stats['firing_frequencies'][thresh])
            freqs_nonzero = freqs[freqs > 0]
            all_freqs[exp_name] = freqs_nonzero
        
        n_experiments = len(stats_dict)
        fig = plt.figure(figsize=(16, 6 + 3 * n_experiments))
        gs = fig.add_gridspec(1 + n_experiments, 1, height_ratios=[1] + [1] * n_experiments, hspace=0.3)
        fig.suptitle(f'Firing Frequency Distribution (Threshold = {thresh})', 
                     fontsize=16, fontweight='bold')
        
        # Histogram comparison
        ax1 = fig.add_subplot(gs[0])
        for idx, (exp_name, freqs_nonzero) in enumerate(all_freqs.items()):
            color = colors[idx % len(colors)]
            ax1.hist(freqs_nonzero, bins=100, alpha=0.6, 
                    label=exp_name, color=color, edgecolor='black', density=False)
        ax1.set_xlabel('Firing Frequency (fraction of tokens)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Number of Features', fontsize=11, fontweight='bold')
        ax1.set_title('Firing Frequency Distribution (Count of Features)', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.set_yscale('log')
        ax1.grid(alpha=0.3)
        
        # Rank plots
        for plot_idx, (exp_name, freqs_nonzero) in enumerate(all_freqs.items()):
            ax = fig.add_subplot(gs[1 + plot_idx])
            sorted_freqs = np.sort(freqs_nonzero)[::-1]
            color = colors[plot_idx % len(colors)]
            ax.plot(np.arange(len(sorted_freqs)), sorted_freqs, '-', 
                   label=exp_name, linewidth=2, color=color)
            ax.set_xlabel('Feature Rank (by firing frequency, highest to lowest)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Firing Frequency', fontsize=11, fontweight='bold')
            ax.set_title(f'Firing Frequency Rank Plot - {exp_name}', fontsize=12, fontweight='bold')
            ax.legend()
            ax.set_yscale('log')
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plot_file = output_dir / f"1_firing_frequency_distribution_thresh_{thresh_str}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {plot_file.name}")
    
    # Save results
    results_dict = {
        'experiments': list(stats_dict.keys()),
        'stats': {
            exp_name: {
                'firing_counts': {str(k): v for k, v in stats['firing_counts'].items()},
                'firing_frequencies': {str(k): v for k, v in stats['firing_frequencies'].items()},
                'total_tokens': stats['total_tokens'],
                'features_that_fire': stats['features_that_fire'],
            }
            for exp_name, stats in stats_dict.items()
        },
    }
    
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"  ✓ Saved results to: {results_file}")


def create_cumulative_magnitude_distribution_plots(
    stats_dict: Dict[str, Dict],
    output_dir: Path
):
    """Create cumulative magnitude distribution plots showing top N% features vs magnitude %.
    
    Creates 3 versions:
    1. Ranked by total magnitude (sum across all inputs)
    2. Ranked by firing frequency
    3. Ranked by magnitude median
    """
    print(f"\n{'='*70}")
    print("CREATING CUMULATIVE MAGNITUDE DISTRIBUTION PLOTS")
    print(f"{'='*70}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    # Plot 1: Ranked by total magnitude (original)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for idx, (exp_name, stats) in enumerate(stats_dict.items()):
        if 'cumulative_magnitude_pct' not in stats or 'feature_pct' not in stats:
            print(f"  ⚠️  Skipping {exp_name}: missing cumulative magnitude data")
            continue
        
        cumulative_pct = np.array(stats['cumulative_magnitude_pct'])
        feature_pct = np.array(stats['feature_pct'])
        color = colors[idx % len(colors)]
        
        ax.plot(feature_pct * 100, cumulative_pct * 100, 
               label=exp_name, linewidth=2, color=color, alpha=0.8)
        
        # Add diagonal reference line (if all features contribute equally)
        if idx == 0:
            ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, linewidth=1, label='Equal distribution')
    
    ax.set_xlabel('Top N% of Features (ranked by total magnitude)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative % of Total Magnitude', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Magnitude Distribution: Ranked by Total Magnitude', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3, which='both')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    
    # Add some reference lines for common percentiles
    for pct in [10, 25, 50]:
        ax.axvline(pct, color='gray', linestyle=':', alpha=0.3, linewidth=1)
        ax.axhline(pct, color='gray', linestyle=':', alpha=0.3, linewidth=1)
    
    plt.tight_layout()
    plot_file = output_dir / "cumulative_magnitude_distribution_by_magnitude_sum.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot_file.name}")
    
    # Also save with old name for backward compatibility
    import shutil
    shutil.copy2(plot_file, output_dir / "cumulative_magnitude_distribution.png")
    
    # Plot 2: Ranked by firing frequency
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for idx, (exp_name, stats) in enumerate(stats_dict.items()):
        if 'feature_rank_by_firing' not in stats or 'abs_magnitude_sums' not in stats:
            print(f"  ⚠️  Skipping {exp_name}: missing frequency ranking data")
            continue
        
        firing_rank = np.array(stats['feature_rank_by_firing'])  # Ranked by frequency (highest first)
        abs_magnitude_sums = np.array(stats['abs_magnitude_sums'])
        d_sae = stats['d_sae']
        
        # Reorder magnitude sums by frequency rank
        magnitude_by_freq_rank = abs_magnitude_sums[firing_rank]
        total_magnitude = float(np.sum(abs_magnitude_sums))
        
        # Compute cumulative distribution
        cumulative_magnitude = np.cumsum(magnitude_by_freq_rank)
        cumulative_magnitude_pct = cumulative_magnitude / total_magnitude if total_magnitude > 0 else np.zeros_like(cumulative_magnitude)
        feature_pct = np.arange(1, d_sae + 1) / d_sae
        
        color = colors[idx % len(colors)]
        ax.plot(feature_pct * 100, cumulative_magnitude_pct * 100, 
               label=exp_name, linewidth=2, color=color, alpha=0.8)
        
        if idx == 0:
            ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, linewidth=1, label='Equal distribution')
    
    ax.set_xlabel('Top N% of Features (ranked by firing frequency)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative % of Total Magnitude', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Magnitude Distribution: Ranked by Firing Frequency', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3, which='both')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    
    for pct in [10, 25, 50]:
        ax.axvline(pct, color='gray', linestyle=':', alpha=0.3, linewidth=1)
        ax.axhline(pct, color='gray', linestyle=':', alpha=0.3, linewidth=1)
    
    plt.tight_layout()
    plot_file = output_dir / "cumulative_magnitude_distribution_by_frequency.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot_file.name}")
    
    # Plot 3: Ranked by magnitude median
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for idx, (exp_name, stats) in enumerate(stats_dict.items()):
        if 'magnitude_stats' not in stats or 'abs_magnitude_sums' not in stats:
            print(f"  ⚠️  Skipping {exp_name}: missing magnitude stats data")
            continue
        
        magnitude_stats = stats['magnitude_stats']
        abs_magnitude_sums = np.array(stats['abs_magnitude_sums'])
        d_sae = stats['d_sae']
        
        # Get median magnitude for each feature
        median_magnitudes = np.array([
            magnitude_stats.get(f, {}).get('median', 0.0) 
            for f in range(d_sae)
        ])
        
        # Rank by median magnitude
        median_rank = np.argsort(median_magnitudes)[::-1]  # Highest median first
        
        # Reorder magnitude sums by median rank
        magnitude_by_median_rank = abs_magnitude_sums[median_rank]
        total_magnitude = float(np.sum(abs_magnitude_sums))
        
        # Compute cumulative distribution
        cumulative_magnitude = np.cumsum(magnitude_by_median_rank)
        cumulative_magnitude_pct = cumulative_magnitude / total_magnitude if total_magnitude > 0 else np.zeros_like(cumulative_magnitude)
        feature_pct = np.arange(1, d_sae + 1) / d_sae
        
        color = colors[idx % len(colors)]
        ax.plot(feature_pct * 100, cumulative_magnitude_pct * 100, 
               label=exp_name, linewidth=2, color=color, alpha=0.8)
        
        if idx == 0:
            ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, linewidth=1, label='Equal distribution')
    
    ax.set_xlabel('Top N% of Features (ranked by magnitude median)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative % of Total Magnitude', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Magnitude Distribution: Ranked by Magnitude Median', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3, which='both')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    
    for pct in [10, 25, 50]:
        ax.axvline(pct, color='gray', linestyle=':', alpha=0.3, linewidth=1)
        ax.axhline(pct, color='gray', linestyle=':', alpha=0.3, linewidth=1)
    
    plt.tight_layout()
    plot_file = output_dir / "cumulative_magnitude_distribution_by_magnitude_median.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot_file.name}")


def create_firing_magnitude_plots(
    stats_dict: Dict[str, Dict],
    output_dir: Path,
    top_k_groups: List[int] = [10, 50, 100, 250, 500, 1000, 10000]
):
    """Create firing frequency vs magnitude plots (Analysis 3) for multiple experiments."""
    print(f"\n{'='*70}")
    print("CREATING FIRING FREQUENCY vs MAGNITUDE PLOTS")
    print(f"{'='*70}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Limit top_k_groups to available features (use minimum across all experiments)
    max_features = min(stats['d_sae'] for stats in stats_dict.values())
    top_k_groups = [k for k in top_k_groups if k <= max_features]
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    # Plot 1: Magnitude by Top-K Groups
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Magnitude Statistics by Top-K Firing Frequency Groups', 
                 fontsize=16, fontweight='bold')
    
    # Collect data for all experiments
    all_medians = {}
    all_means = {}
    
    for exp_name, stats in stats_dict.items():
        rank = np.array(stats['feature_rank_by_firing'])
        medians = []
        means = []
        
        for k in top_k_groups:
            topk = rank[:k]
            mags = [stats['magnitude_stats'][f]['median'] for f in topk 
                   if stats['magnitude_stats'][f]['n_activations'] > 0]
            mags_mean = [stats['magnitude_stats'][f]['mean'] for f in topk 
                        if stats['magnitude_stats'][f]['n_activations'] > 0]
            
            medians.append(np.median(mags) if mags else 0.0)
            means.append(np.mean(mags_mean) if mags_mean else 0.0)
        
        all_medians[exp_name] = medians
        all_means[exp_name] = means
    
    # Plot medians
    ax = axes[0]
    for idx, (exp_name, medians) in enumerate(all_medians.items()):
        color = colors[idx % len(colors)]
        ax.plot(top_k_groups, medians, 'o-', color=color, linewidth=2,
                markersize=8, label=exp_name, alpha=0.8)
    ax.set_xlabel('Top-K Features (by firing frequency)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Median Activation Magnitude', fontsize=11, fontweight='bold')
    ax.set_title('Median Magnitude by Top-K Groups', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(alpha=0.3, which='both')
    
    # Plot means
    ax = axes[1]
    for idx, (exp_name, means) in enumerate(all_means.items()):
        color = colors[idx % len(colors)]
        ax.plot(top_k_groups, means, 's--', color=color, linewidth=2,
                markersize=8, label=exp_name, alpha=0.8)
    ax.set_xlabel('Top-K Features (by firing frequency)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean Activation Magnitude', fontsize=11, fontweight='bold')
    ax.set_title('Mean Magnitude by Top-K Groups', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(alpha=0.3, which='both')
    
    plt.tight_layout()
    plot_file = output_dir / "1_magnitude_by_topk_groups.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot_file.name}")
    
    # Plot 1b: Magnitude by bins (split entire feature space into bins)
    print("\nCreating magnitude by bins plot...")
    n_bins = 50
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Magnitude Statistics by Feature Bins (Entire Feature Space)', 
                 fontsize=16, fontweight='bold')
    
    # Collect data for all experiments
    all_medians_bins = {}
    all_means_bins = {}
    
    for exp_name, stats in stats_dict.items():
        rank = np.array(stats['feature_rank_by_firing'])
        d_sae = stats['d_sae']
        
        # Split into bins
        bin_size = d_sae // n_bins
        medians_bins = []
        means_bins = []
        bin_labels = []
        
        for bin_idx in range(n_bins):
            start_idx = bin_idx * bin_size
            if bin_idx == n_bins - 1:
                # Last bin gets remainder
                end_idx = d_sae
            else:
                end_idx = (bin_idx + 1) * bin_size
            
            bin_features = rank[start_idx:end_idx]
            mags = [stats['magnitude_stats'][f]['median'] for f in bin_features 
                   if stats['magnitude_stats'][f]['n_activations'] > 0]
            mags_mean = [stats['magnitude_stats'][f]['mean'] for f in bin_features 
                        if stats['magnitude_stats'][f]['n_activations'] > 0]
            
            medians_bins.append(np.median(mags) if mags else 0.0)
            means_bins.append(np.mean(mags_mean) if mags_mean else 0.0)
            bin_labels.append(f'{start_idx}-{end_idx-1}')
        
        all_medians_bins[exp_name] = medians_bins
        all_means_bins[exp_name] = means_bins
    
    # Plot medians
    ax = axes[0]
    x_positions = np.arange(n_bins)
    for idx, (exp_name, medians) in enumerate(all_medians_bins.items()):
        color = colors[idx % len(colors)]
        ax.plot(x_positions, medians, 'o-', color=color, linewidth=2,
                markersize=4, label=exp_name, alpha=0.8)
    ax.set_xlabel('Feature Bin (ranked by firing frequency)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Median Activation Magnitude', fontsize=11, fontweight='bold')
    ax.set_title('Median Magnitude by Feature Bins', fontsize=12, fontweight='bold')
    ax.set_xticks(x_positions[::5])  # Show every 5th bin label
    ax.set_xticklabels([bin_labels[i] for i in x_positions[::5]], rotation=45, ha='right')
    ax.legend()
    ax.grid(alpha=0.3, which='both')
    
    # Plot means
    ax = axes[1]
    for idx, (exp_name, means) in enumerate(all_means_bins.items()):
        color = colors[idx % len(colors)]
        ax.plot(x_positions, means, 's--', color=color, linewidth=2,
                markersize=4, label=exp_name, alpha=0.8)
    ax.set_xlabel('Feature Bin (ranked by firing frequency)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean Activation Magnitude', fontsize=11, fontweight='bold')
    ax.set_title('Mean Magnitude by Feature Bins', fontsize=12, fontweight='bold')
    ax.set_xticks(x_positions[::5])
    ax.set_xticklabels([bin_labels[i] for i in x_positions[::5]], rotation=45, ha='right')
    ax.legend()
    ax.grid(alpha=0.3, which='both')
    
    plt.tight_layout()
    plot_file = output_dir / "1_magnitude_by_bin.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot_file.name}")
    
    # Plot 2: Heatmap-style visualization
    # Get all features ranked by firing frequency, compute magnitude
    all_magnitudes = {}
    for exp_name, stats in stats_dict.items():
        rank = np.array(stats['feature_rank_by_firing'])
        magnitudes = np.array([stats['magnitude_stats'][f]['median'] 
                              for f in rank])
        all_magnitudes[exp_name] = magnitudes
    
    # Ensure non-negative (magnitude when activated should be >= 0)
    for exp_name in all_magnitudes:
        all_magnitudes[exp_name] = np.maximum(all_magnitudes[exp_name], 0)
    
    # Find global max for consistent scaling across all experiments
    global_magnitude_max = max(np.max(mags) for mags in all_magnitudes.values())
    n_features = len(list(all_magnitudes.values())[0])
    
    if n_features < 70000:
        # Show all features as continuous heatmap
        n_experiments = len(stats_dict)
        fig, axes = plt.subplots(n_experiments, 1, figsize=(16, 4 * n_experiments))
        if n_experiments == 1:
            axes = [axes]
        fig.suptitle('Magnitude Heatmap by Firing Frequency Rank (All Features)', 
                     fontsize=16, fontweight='bold')
        
        for idx, (exp_name, magnitudes) in enumerate(all_magnitudes.items()):
            ax = axes[idx]
            im = ax.imshow(magnitudes.reshape(1, -1), aspect='auto', cmap='viridis',
                          interpolation='nearest', vmin=0, vmax=global_magnitude_max)
            ax.set_title(exp_name, fontsize=12, fontweight='bold')
            ax.set_xlabel('Feature Rank (by firing frequency)', fontsize=11)
            ax.set_ylabel('')
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, label='Median Activation Magnitude')
        
        plt.tight_layout()
        plot_file = output_dir / "2_magnitude_heatmap_all_features.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {plot_file.name}")
    else:
        # Use 3-window approach
        window_size = n_features // 10  # Show 10% of features per window
        n_experiments = len(stats_dict)
        
        fig, axes = plt.subplots(3, n_experiments, figsize=(4 * n_experiments, 12))
        if n_experiments == 1:
            axes = axes.reshape(-1, 1)
        fig.suptitle('Magnitude Heatmap by Firing Frequency Rank (3 Windows)', 
                     fontsize=16, fontweight='bold')
        
        windows = [
            (0, window_size, 'Top'),
            (n_features // 2 - window_size // 2, n_features // 2 + window_size // 2, 'Middle'),
            (n_features - window_size, n_features, 'Bottom'),
        ]
        
        # For each window, use consistent scaling across experiments
        for win_idx, (start, end, label) in enumerate(windows):
            window_magnitudes = {exp: mags[start:end] for exp, mags in all_magnitudes.items()}
            window_max = max(np.max(mags) for mags in window_magnitudes.values())
            
            for exp_idx, (exp_name, window_mags) in enumerate(window_magnitudes.items()):
                ax = axes[win_idx, exp_idx]
                im = ax.imshow(window_mags.reshape(1, -1), aspect='auto', cmap='viridis',
                              interpolation='nearest', vmin=0, vmax=window_max)
                ax.set_title(f'{exp_name} - {label} Window', fontsize=11, fontweight='bold')
                ax.set_xlabel(f'Feature Rank [{start}:{end}]', fontsize=10)
                ax.set_ylabel('')
                ax.set_yticks([])
                plt.colorbar(im, ax=ax, label='Magnitude')
        
        plt.tight_layout()
        plot_file = output_dir / "2_magnitude_heatmap_3_windows.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {plot_file.name}")
    
    # Plot 3: Sampled heatmap with dense top sampling
    sample_indices = []
    # Dense top sampling: 1 from each of first 100 features
    for i in range(0, min(100, n_features), 5):
        sample_indices.append(i)
    # Then sample 1 from every 10
    for i in range(100, n_features, 10):
        sample_indices.append(i)
    sample_indices = sorted(sample_indices)
    
    all_sampled = {exp: mags[sample_indices] for exp, mags in all_magnitudes.items()}
    sampled_max = max(np.max(mags) for mags in all_sampled.values())
    
    n_experiments = len(stats_dict)
    fig, axes = plt.subplots(n_experiments, 1, figsize=(16, 4 * n_experiments))
    if n_experiments == 1:
        axes = [axes]
    fig.suptitle('Magnitude Heatmap by Firing Frequency Rank (Sampled, Dense Top)', 
                 fontsize=16, fontweight='bold')
    
    for idx, (exp_name, sampled) in enumerate(all_sampled.items()):
        ax = axes[idx]
        im = ax.imshow(sampled.reshape(1, -1), aspect='auto', cmap='viridis',
                      interpolation='nearest', vmin=0, vmax=sampled_max)
        ax.set_title(f'{exp_name} - {len(sample_indices)} sampled features', 
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Sampled Feature Rank', fontsize=11)
        ax.set_ylabel('')
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, label='Median Activation Magnitude')
    
    plt.tight_layout()
    plot_file = output_dir / "3_magnitude_heatmap_sampled.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot_file.name}")
    
    # Save results
    results_dict = {
        'experiments': list(stats_dict.keys()),
        'top_k_groups': top_k_groups,
        'magnitude_by_topk': {
            exp_name: {
                'medians': medians,
                'means': means,
            }
            for exp_name, (medians, means) in zip(stats_dict.keys(), 
                                                  zip(all_medians.values(), all_means.values()))
        },
    }
    
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"  ✓ Saved results to: {results_file}")


def create_feature_ranking_overlap_plots(
    stats_dict: Dict[str, Dict],
    output_dir: Path
):
    """Create plots showing overlap between different feature ranking methods.
    
    Compares:
    1. Magnitude median vs Frequency
    2. Frequency vs Magnitude sum
    3. Magnitude sum vs Magnitude median
    
    For each comparison, shows overlap counts for top 500, top 100, and top 30 features.
    """
    print(f"\n{'='*70}")
    print("CREATING FEATURE RANKING OVERLAP PLOTS")
    print(f"{'='*70}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # For single SAE, we only have one experiment
    exp_name = next(iter(stats_dict.keys()))
    stats = stats_dict[exp_name]
    
    d_sae = stats['d_sae']
    
    # Extract rankings
    rank_by_firing = np.array(stats['feature_rank_by_firing'])  # Frequency ranking
    rank_by_magnitude_sum = np.array(stats['feature_rank_by_magnitude'])  # Magnitude sum ranking
    
    # Compute magnitude median ranking (efficiently using numpy argsort)
    magnitude_medians = np.array([stats['magnitude_stats'][i]['median'] for i in range(d_sae)])
    rank_by_magnitude_median = np.argsort(magnitude_medians)[::-1]  # Descending order
    
    # Top K values to analyze
    top_ks = [500, 100, 30]
    
    # Compute overlaps for each top K
    overlaps = {
        'magnitude_median_vs_frequency': {},
        'frequency_vs_magnitude_sum': {},
        'magnitude_sum_vs_magnitude_median': {},
    }
    
    for k in top_ks:
        if k > d_sae:
            k = d_sae
        
        # Top K features for each ranking
        top_k_firing = set(rank_by_firing[:k])
        top_k_magnitude_sum = set(rank_by_magnitude_sum[:k])
        top_k_magnitude_median = set(rank_by_magnitude_median[:k])
        
        # Compute overlaps
        overlaps['magnitude_median_vs_frequency'][k] = len(top_k_magnitude_median & top_k_firing)
        overlaps['frequency_vs_magnitude_sum'][k] = len(top_k_firing & top_k_magnitude_sum)
        overlaps['magnitude_sum_vs_magnitude_median'][k] = len(top_k_magnitude_sum & top_k_magnitude_median)
    
    # Create 3 bar charts
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # blue, orange, green
    
    # Plot 1: Magnitude median vs Frequency
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(range(len(top_ks)), 
                  [overlaps['magnitude_median_vs_frequency'][k] for k in top_ks],
                  color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Top-K Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Overlapping Features', fontsize=12, fontweight='bold')
    ax.set_title(f'Feature Overlap: Magnitude Median vs Frequency\n{exp_name}', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(top_ks)))
    ax.set_xticklabels([f'Top {k}' for k in top_ks])
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim(bottom=0)
    
    # Add value labels on bars
    for bar, k in zip(bars, top_ks):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/k*100:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plot_file = output_dir / "feature_ranking_overlap_magnitude_median_vs_frequency.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot_file.name}")
    
    # Plot 2: Frequency vs Magnitude sum
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(range(len(top_ks)), 
                  [overlaps['frequency_vs_magnitude_sum'][k] for k in top_ks],
                  color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Top-K Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Overlapping Features', fontsize=12, fontweight='bold')
    ax.set_title(f'Feature Overlap: Frequency vs Magnitude Sum\n{exp_name}', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(top_ks)))
    ax.set_xticklabels([f'Top {k}' for k in top_ks])
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim(bottom=0)
    
    # Add value labels on bars
    for bar, k in zip(bars, top_ks):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/k*100:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plot_file = output_dir / "feature_ranking_overlap_frequency_vs_magnitude_sum.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot_file.name}")
    
    # Plot 3: Magnitude sum vs Magnitude median
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(range(len(top_ks)), 
                  [overlaps['magnitude_sum_vs_magnitude_median'][k] for k in top_ks],
                  color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Top-K Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Overlapping Features', fontsize=12, fontweight='bold')
    ax.set_title(f'Feature Overlap: Magnitude Sum vs Magnitude Median\n{exp_name}', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(top_ks)))
    ax.set_xticklabels([f'Top {k}' for k in top_ks])
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim(bottom=0)
    
    # Add value labels on bars
    for bar, k in zip(bars, top_ks):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/k*100:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plot_file = output_dir / "feature_ranking_overlap_magnitude_sum_vs_magnitude_median.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot_file.name}")
    
    # Save overlap data to results.json (append to existing if present)
    results_file = output_dir / "results.json"
    overlap_results = {
        'feature_ranking_overlaps': {
            'top_ks': top_ks,
            'magnitude_median_vs_frequency': overlaps['magnitude_median_vs_frequency'],
            'frequency_vs_magnitude_sum': overlaps['frequency_vs_magnitude_sum'],
            'magnitude_sum_vs_magnitude_median': overlaps['magnitude_sum_vs_magnitude_median'],
        }
    }
    
    # Try to load existing results and merge
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                existing_results = json.load(f)
            existing_results.update(overlap_results)
            overlap_results = existing_results
        except Exception as e:
            print(f"  ⚠️  Could not merge with existing results.json: {e}")
    
    with open(results_file, 'w') as f:
        json.dump(overlap_results, f, indent=2)
    print(f"  ✓ Saved overlap results to: {results_file}")


def create_per_input_magnitude_analysis_simplified(
    sae,
    store,
    stats_dict: Dict[str, Dict],
    output_dir: Path,
    device: str = "cpu",
    n_batches_for_collection: int = 30  # Default to 30 to match other analyses
):
    """Create per-input magnitude analysis plots (simplified version that works with sae/store directly).
    
    Analyzes dominance patterns:
    1. Top 10 globally frequent features dominance per input
    2. Top 10 globally high magnitude features dominance per input
    3. Top 30 globally frequent features dominance per input
    4. Top 30 globally high magnitude features dominance per input
    5. Dominance-based ranking: ranks features by how often they appear in top-K per input
    """
    print(f"\n{'='*70}")
    print("CREATING PER-INPUT MAGNITUDE ANALYSIS PLOTS")
    print(f"{'='*70}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # For single SAE, we only have one experiment
    exp_name = next(iter(stats_dict.keys()))
    stats = stats_dict[exp_name]
    
    # Color for plots
    color = 'blue'
    
    # Get top frequency and magnitude features (global)
    firing_rank = np.array(stats['feature_rank_by_firing'])
    magnitude_rank = np.array(stats['feature_rank_by_magnitude'])
    
    # Get top 10 globally frequent and magnitude features
    top10_globally_frequent = firing_rank[:10]
    top10_globally_magnitude = magnitude_rank[:10]
    
    # Get top 30 globally frequent and magnitude features
    top30_globally_frequent = firing_rank[:30]
    top30_globally_magnitude = magnitude_rank[:30]
    
    # Storage for dominance plots (fast - no ranking needed)
    all_top10_globally_frequent_dominance = []
    all_top10_globally_magnitude_dominance = []
    all_top30_globally_frequent_dominance = []
    all_top30_globally_magnitude_dominance = []
    
    # Storage for dominance-based ranking: accumulate actual dominance (fraction of magnitude) for each feature
    # For ALL features across ALL inputs (not just top-K per input)
    d_sae = stats['d_sae']
    dominance_sum = np.zeros(d_sae, dtype=np.float64)  # Sum of per-input dominance percentages for ALL features
    
    print(f"  Processing {n_batches_for_collection} batches...")
    start_time = time.time()
    print(f"[TIMING] Started per-input magnitude analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for batch_idx in tqdm(range(n_batches_for_collection), desc=f"  {exp_name}"):
        batch_start = time.time()
        batch_activations = store.next_batch()
        batch_size = batch_activations.shape[0]
        
        with torch.no_grad():
            batch_activations = batch_activations.to(device)
            sae_acts = sae.encode(batch_activations)  # [batch_size, d_sae]
            sae_acts_np = sae_acts.cpu().numpy()
        
        # Process each input in batch
        for seq_idx in range(batch_size):
            input_acts = np.abs(sae_acts_np[seq_idx])
            total_magnitude = np.sum(input_acts)
            
            # Compute dominance plots (no ranking needed - FAST!)
            # These just sum activations of globally-top features
            top10_glob_freq_magnitude = np.sum(input_acts[top10_globally_frequent])
            top10_glob_mag_magnitude = np.sum(input_acts[top10_globally_magnitude])
            top30_glob_freq_magnitude = np.sum(input_acts[top30_globally_frequent])
            top30_glob_mag_magnitude = np.sum(input_acts[top30_globally_magnitude])
            
            if total_magnitude > 0:
                all_top10_globally_frequent_dominance.append(float(top10_glob_freq_magnitude / total_magnitude))
                all_top10_globally_magnitude_dominance.append(float(top10_glob_mag_magnitude / total_magnitude))
                all_top30_globally_frequent_dominance.append(float(top30_glob_freq_magnitude / total_magnitude))
                all_top30_globally_magnitude_dominance.append(float(top30_glob_mag_magnitude / total_magnitude))
            else:
                all_top10_globally_frequent_dominance.append(0.0)
                all_top10_globally_magnitude_dominance.append(0.0)
                all_top30_globally_frequent_dominance.append(0.0)
                all_top30_globally_magnitude_dominance.append(0.0)
            
            # Compute dominance-based ranking: accumulate per-input dominance percentage for ALL features
            if total_magnitude > 0:
                # Compute dominance fraction for each feature: feature_magnitude / total_magnitude
                # This is the per-input dominance percentage for ALL features
                dominance_fractions = input_acts / total_magnitude
                
                # Accumulate dominance for ALL features (not just top-K)
                dominance_sum += dominance_fractions
        
        if (batch_idx + 1) % 10 == 0:
            batch_elapsed = time.time() - batch_start
            total_elapsed = time.time() - start_time
            print(f"[TIMING] Completed batch {batch_idx+1}/{n_batches_for_collection} - Batch took {batch_elapsed:.2f}s, Total elapsed: {total_elapsed/60:.2f} minutes")
            torch.cuda.empty_cache()
    
    elapsed = time.time() - start_time
    print(f"[TIMING] Finished per-input magnitude analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Took {elapsed/60:.2f} minutes ({elapsed:.2f} seconds)")
    print(f"  ✓ Collected data from {len(all_top10_globally_frequent_dominance)} inputs")
    
    # Compute dominance-based ranking: rank ALL features by sum of per-input dominance percentages
    feature_rank_by_dominance = np.argsort(dominance_sum)[::-1]
    
    # Get top 10/30 dominance-based ranked features
    top10_dominance_ranked = feature_rank_by_dominance[:10]
    top30_dominance_ranked = feature_rank_by_dominance[:30]
    
    print(f"  ✓ Computed dominance-based ranking (top feature: sum={dominance_sum[feature_rank_by_dominance[0]]:.4f})")
    
    # Now compute dominance scores for these features (second pass through batches)
    print(f"\n  Computing dominance scores for dominance-ranked features (second pass)...")
    all_top10_dominance_ranked_dominance = []
    all_top30_dominance_ranked_dominance = []
    
    # Reset store to beginning if possible
    if hasattr(store, 'reset_input_dataset'):
        store.reset_input_dataset()
        if hasattr(store, 'iterable_sequences') and hasattr(store, '_iterate_tokenized_sequences'):
            store.iterable_sequences = store._iterate_tokenized_sequences()
        print(f"  ✓ Reset store for second pass")
    
    for batch_idx in tqdm(range(n_batches_for_collection), desc=f"  {exp_name} (pass 2)"):
        batch_activations = store.next_batch()
        batch_size = batch_activations.shape[0]
        
        with torch.no_grad():
            batch_activations = batch_activations.to(device)
            sae_acts = sae.encode(batch_activations)
            sae_acts_np = sae_acts.cpu().numpy()
        
        for seq_idx in range(batch_size):
            input_acts = np.abs(sae_acts_np[seq_idx])
            total_magnitude = np.sum(input_acts)
            
            if total_magnitude > 0:
                top10_dom_magnitude = np.sum(input_acts[top10_dominance_ranked])
                top30_dom_magnitude = np.sum(input_acts[top30_dominance_ranked])
                all_top10_dominance_ranked_dominance.append(float(top10_dom_magnitude / total_magnitude))
                all_top30_dominance_ranked_dominance.append(float(top30_dom_magnitude / total_magnitude))
            else:
                all_top10_dominance_ranked_dominance.append(0.0)
                all_top30_dominance_ranked_dominance.append(0.0)
    
    print(f"  ✓ Computed dominance scores for dominance-ranked features")
    
    # Plot 1: Top 10 globally frequent features dominance
    print("\nCreating top 10 globally frequent features dominance plots...")
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('Top 10 Globally Frequent Features Dominance per Input\n(Within Input\'s Top-K)', 
                 fontsize=16, fontweight='bold')
    
    if len(all_top10_globally_frequent_dominance) > 0:
        ax.hist(all_top10_globally_frequent_dominance, bins=50, alpha=0.7, color=color, edgecolor='black')
        mean_dom = np.mean(all_top10_globally_frequent_dominance)
        median_dom = np.median(all_top10_globally_frequent_dominance)
        ax.axvline(mean_dom, color=color, linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_dom:.2%}')
        ax.axvline(median_dom, color=color, linestyle=':', linewidth=2, 
                  label=f'Median: {median_dom:.2%}')
    ax.set_xlabel('Fraction of Total Magnitude from Top 10 Globally Frequent Features', 
                  fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Inputs', fontsize=11, fontweight='bold')
    ax.set_title(exp_name, fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plot_file = output_dir / "1_top10_globally_frequent_dominance.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot_file.name}")
    
    # Plot 2: Top 10 globally high magnitude features dominance
    print("\nCreating top 10 globally high magnitude features dominance plots...")
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('Top 10 Globally High Magnitude Features Dominance per Input\n(Within Input\'s Top-K)', 
                 fontsize=16, fontweight='bold')
    
    if len(all_top10_globally_magnitude_dominance) > 0:
        ax.hist(all_top10_globally_magnitude_dominance, bins=50, alpha=0.7, color=color, edgecolor='black')
        mean_dom = np.mean(all_top10_globally_magnitude_dominance)
        median_dom = np.median(all_top10_globally_magnitude_dominance)
        ax.axvline(mean_dom, color=color, linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_dom:.2%}')
        ax.axvline(median_dom, color=color, linestyle=':', linewidth=2, 
                  label=f'Median: {median_dom:.2%}')
    ax.set_xlabel('Fraction of Total Magnitude from Top 10 Globally High Magnitude Features', 
                  fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Inputs', fontsize=11, fontweight='bold')
    ax.set_title(exp_name, fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plot_file = output_dir / "2_top10_globally_magnitude_dominance.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot_file.name}")
    
    # Plot 3: Top 30 globally frequent features dominance
    print("\nCreating top 30 globally frequent features dominance plots...")
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('Top 30 Globally Frequent Features Dominance per Input\n(Within Input\'s Top-K)', 
                 fontsize=16, fontweight='bold')
    
    if len(all_top30_globally_frequent_dominance) > 0:
        ax.hist(all_top30_globally_frequent_dominance, bins=50, alpha=0.7, color=color, edgecolor='black')
        mean_dom = np.mean(all_top30_globally_frequent_dominance)
        median_dom = np.median(all_top30_globally_frequent_dominance)
        ax.axvline(mean_dom, color=color, linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_dom:.2%}')
        ax.axvline(median_dom, color=color, linestyle=':', linewidth=2, 
                  label=f'Median: {median_dom:.2%}')
    ax.set_xlabel('Fraction of Total Magnitude from Top 30 Globally Frequent Features', 
                  fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Inputs', fontsize=11, fontweight='bold')
    ax.set_title(exp_name, fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plot_file = output_dir / "3_top30_globally_frequent_dominance.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot_file.name}")
    
    # Plot 4: Top 30 globally high magnitude features dominance
    print("\nCreating top 30 globally high magnitude features dominance plots...")
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('Top 30 Globally High Magnitude Features Dominance per Input\n(Within Input\'s Top-K)', 
                 fontsize=16, fontweight='bold')
    
    if len(all_top30_globally_magnitude_dominance) > 0:
        ax.hist(all_top30_globally_magnitude_dominance, bins=50, alpha=0.7, color=color, edgecolor='black')
        mean_dom = np.mean(all_top30_globally_magnitude_dominance)
        median_dom = np.median(all_top30_globally_magnitude_dominance)
        ax.axvline(mean_dom, color=color, linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_dom:.2%}')
        ax.axvline(median_dom, color=color, linestyle=':', linewidth=2, 
                  label=f'Median: {median_dom:.2%}')
    ax.set_xlabel('Fraction of Total Magnitude from Top 30 Globally High Magnitude Features', 
                  fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Inputs', fontsize=11, fontweight='bold')
    ax.set_title(exp_name, fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plot_file = output_dir / "4_top30_globally_magnitude_dominance.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot_file.name}")
    
    # Plot 5: Top 10 dominance-based ranked features dominance
    print("\nCreating top 10 dominance-based ranked features dominance plots...")
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('Top 10 Dominance-Based Ranked Features Dominance per Input\n(Ranked by average dominance - fraction of magnitude)', 
                 fontsize=16, fontweight='bold')
    
    if len(all_top10_dominance_ranked_dominance) > 0:
        ax.hist(all_top10_dominance_ranked_dominance, bins=50, alpha=0.7, color=color, edgecolor='black')
        mean_dom = np.mean(all_top10_dominance_ranked_dominance)
        median_dom = np.median(all_top10_dominance_ranked_dominance)
        ax.axvline(mean_dom, color=color, linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_dom:.2%}')
        ax.axvline(median_dom, color=color, linestyle=':', linewidth=2, 
                  label=f'Median: {median_dom:.2%}')
    ax.set_xlabel('Fraction of Total Magnitude from Top 10 Dominance-Based Ranked Features', 
                  fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Inputs', fontsize=11, fontweight='bold')
    ax.set_title(exp_name, fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plot_file = output_dir / "5_top10_dominance_ranked_dominance.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot_file.name}")
    
    # Plot 6: Top 30 dominance-based ranked features dominance
    print("\nCreating top 30 dominance-based ranked features dominance plots...")
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('Top 30 Dominance-Based Ranked Features Dominance per Input\n(Ranked by average dominance - fraction of magnitude)', 
                 fontsize=16, fontweight='bold')
    
    if len(all_top30_dominance_ranked_dominance) > 0:
        ax.hist(all_top30_dominance_ranked_dominance, bins=50, alpha=0.7, color=color, edgecolor='black')
        mean_dom = np.mean(all_top30_dominance_ranked_dominance)
        median_dom = np.median(all_top30_dominance_ranked_dominance)
        ax.axvline(mean_dom, color=color, linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_dom:.2%}')
        ax.axvline(median_dom, color=color, linestyle=':', linewidth=2, 
                  label=f'Median: {median_dom:.2%}')
    ax.set_xlabel('Fraction of Total Magnitude from Top 30 Dominance-Based Ranked Features', 
                  fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Inputs', fontsize=11, fontweight='bold')
    ax.set_title(exp_name, fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plot_file = output_dir / "6_top30_dominance_ranked_dominance.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot_file.name}")
    
    # Compute feature ranking overlaps (same as in create_feature_ranking_overlap_plots)
    d_sae = stats['d_sae']
    rank_by_firing = np.array(stats['feature_rank_by_firing'])
    rank_by_magnitude_sum = np.array(stats['feature_rank_by_magnitude'])
    magnitude_medians = np.array([stats['magnitude_stats'][i]['median'] for i in range(d_sae)])
    rank_by_magnitude_median = np.argsort(magnitude_medians)[::-1]
    
    top_ks = [500, 100, 30]
    overlaps = {
        'magnitude_median_vs_frequency': {},
        'frequency_vs_magnitude_sum': {},
        'magnitude_sum_vs_magnitude_median': {},
        'dominance_top10_vs_frequency': {},
        'dominance_top10_vs_magnitude_sum': {},
        'dominance_top30_vs_frequency': {},
        'dominance_top30_vs_magnitude_sum': {},
    }
    
    for k in top_ks:
        if k > d_sae:
            k = d_sae
        top_k_firing = set(rank_by_firing[:k])
        top_k_magnitude_sum = set(rank_by_magnitude_sum[:k])
        top_k_magnitude_median = set(rank_by_magnitude_median[:k])
        top_k_dominance_10 = set(feature_rank_by_dominance[:min(10, k)])
        top_k_dominance_30 = set(feature_rank_by_dominance[:min(30, k)])
        
        overlaps['magnitude_median_vs_frequency'][k] = len(top_k_magnitude_median & top_k_firing)
        overlaps['frequency_vs_magnitude_sum'][k] = len(top_k_firing & top_k_magnitude_sum)
        overlaps['magnitude_sum_vs_magnitude_median'][k] = len(top_k_magnitude_sum & top_k_magnitude_median)
        overlaps['dominance_top10_vs_frequency'][k] = len(top_k_dominance_10 & top_k_firing)
        overlaps['dominance_top10_vs_magnitude_sum'][k] = len(top_k_dominance_10 & top_k_magnitude_sum)
        overlaps['dominance_top30_vs_frequency'][k] = len(top_k_dominance_30 & top_k_firing)
        overlaps['dominance_top30_vs_magnitude_sum'][k] = len(top_k_dominance_30 & top_k_magnitude_sum)
    
    # Save results
    results_dict = {
        'experiment': exp_name,
        'n_inputs_analyzed': len(all_top10_globally_frequent_dominance),
        'top10_globally_frequent_dominance_stats': {
            'mean': float(np.mean(all_top10_globally_frequent_dominance)) if len(all_top10_globally_frequent_dominance) > 0 else 0.0,
            'median': float(np.median(all_top10_globally_frequent_dominance)) if len(all_top10_globally_frequent_dominance) > 0 else 0.0,
        },
        'top10_globally_magnitude_dominance_stats': {
            'mean': float(np.mean(all_top10_globally_magnitude_dominance)) if len(all_top10_globally_magnitude_dominance) > 0 else 0.0,
            'median': float(np.median(all_top10_globally_magnitude_dominance)) if len(all_top10_globally_magnitude_dominance) > 0 else 0.0,
        },
        'top30_globally_frequent_dominance_stats': {
            'mean': float(np.mean(all_top30_globally_frequent_dominance)) if len(all_top30_globally_frequent_dominance) > 0 else 0.0,
            'median': float(np.median(all_top30_globally_frequent_dominance)) if len(all_top30_globally_frequent_dominance) > 0 else 0.0,
        },
        'top30_globally_magnitude_dominance_stats': {
            'mean': float(np.mean(all_top30_globally_magnitude_dominance)) if len(all_top30_globally_magnitude_dominance) > 0 else 0.0,
            'median': float(np.median(all_top30_globally_magnitude_dominance)) if len(all_top30_globally_magnitude_dominance) > 0 else 0.0,
        },
        'top10_dominance_ranked_dominance_stats': {
            'mean': float(np.mean(all_top10_dominance_ranked_dominance)) if len(all_top10_dominance_ranked_dominance) > 0 else 0.0,
            'median': float(np.median(all_top10_dominance_ranked_dominance)) if len(all_top10_dominance_ranked_dominance) > 0 else 0.0,
        },
        'top30_dominance_ranked_dominance_stats': {
            'mean': float(np.mean(all_top30_dominance_ranked_dominance)) if len(all_top30_dominance_ranked_dominance) > 0 else 0.0,
            'median': float(np.median(all_top30_dominance_ranked_dominance)) if len(all_top30_dominance_ranked_dominance) > 0 else 0.0,
        },
        'dominance_based_ranking': {
            'feature_rank_by_dominance': feature_rank_by_dominance.tolist(),  # Rank ALL features by sum of per-input dominance percentages
            'dominance_sum': dominance_sum.tolist(),  # Sum of per-input dominance percentages for ALL features
        },
        'feature_ranking_overlaps': {
            'top_ks': top_ks,
            'magnitude_median_vs_frequency': overlaps['magnitude_median_vs_frequency'],
            'frequency_vs_magnitude_sum': overlaps['frequency_vs_magnitude_sum'],
            'magnitude_sum_vs_magnitude_median': overlaps['magnitude_sum_vs_magnitude_median'],
            'dominance_top10_vs_frequency': overlaps['dominance_top10_vs_frequency'],
            'dominance_top10_vs_magnitude_sum': overlaps['dominance_top10_vs_magnitude_sum'],
            'dominance_top30_vs_frequency': overlaps['dominance_top30_vs_frequency'],
            'dominance_top30_vs_magnitude_sum': overlaps['dominance_top30_vs_magnitude_sum'],
        },
    }
    
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"  ✓ Saved results to: {results_file}")

# Top Features Frequency Analysis Functions
# These will be added to feature_firing_analysis.py

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import re


def normalize_sparsity_level(param_name: str, param_value: float) -> float:
    """
    Normalize sparsity parameter to a consistent "sparsity level" where:
    - Lower value = less sparse (more features active)
    - Higher value = more sparse (fewer features active)
    
    Parameter meanings:
    - k (TopK/BatchTopK): Higher k = LESS sparse (more features active) → invert
    - target_l0 (JumpReLU): Higher target_l0 = LESS sparse (more features active) → invert
    - l1_penalty: Higher penalty = MORE sparse → use directly
    - sparsity_penalty: Higher penalty = MORE sparse → use directly
    """
    if param_name in ['k', 'target_l0']:
        return -param_value  # Invert: higher k/l0 = less sparse
    elif param_name in ['l1_penalty', 'sparsity_penalty']:
        return param_value  # Higher penalty = more sparse
    return param_value


def extract_sparsity_parameter_from_config(config_path: Path) -> Tuple[Optional[str], Optional[float]]:
    """Extract sparsity parameter from config.json file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        trainer_cfg = config.get('trainer', {})
        
        if 'k' in trainer_cfg:
            return 'k', float(trainer_cfg['k'])
        elif 'target_l0' in trainer_cfg:
            return 'target_l0', float(trainer_cfg['target_l0'])
        elif 'sparsity_penalty' in trainer_cfg:
            return 'sparsity_penalty', float(trainer_cfg['sparsity_penalty'])
        elif 'l1_penalty' in trainer_cfg:
            return 'l1_penalty', float(trainer_cfg['l1_penalty'])
    except Exception:
        pass
    
    return None, None


def create_top_features_frequency_plots(
    stats_dict: Dict[str, Dict],
    output_dir: Path
):
    """
    Create line plots showing top 10 and top 30 features frequency percentages
    using 3 ranking methods: frequency, firing_magnitude_median, firing_magnitude_mean.
    
    Args:
        stats_dict: Dictionary with single SAE stats (key: experiment name, value: stats dict)
        output_dir: Output directory (should be firing_frequency_distribution folder)
    """
    print(f"\n{'='*70}")
    print("CREATING TOP FEATURES FREQUENCY PLOTS")
    print(f"{'='*70}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the single SAE stats
    exp_name = next(iter(stats_dict.keys()))
    stats = stats_dict[exp_name]
    
    d_sae = stats['d_sae']
    firing_frequencies = np.array(stats['firing_frequencies'][0.0])  # Use threshold 0.0
    magnitude_stats = stats['magnitude_stats']
    
    # Ranking methods
    ranking_methods = {
        'frequency': {
            'label': 'Firing Frequency',
            'rank': np.array(stats['feature_rank_by_firing']),
        },
        'firing_magnitude_median': {
            'label': 'Firing Magnitude Median',
            'rank': np.argsort([magnitude_stats.get(i, {}).get('median', 0.0) for i in range(d_sae)])[::-1],
        },
        'firing_magnitude_mean': {
            'label': 'Firing Magnitude Mean',
            'rank': np.argsort([magnitude_stats.get(i, {}).get('mean', 0.0) for i in range(d_sae)])[::-1],
        },
    }
    
    top_numbers = [10, 30]
    results_data = {}
    
    for ranking_method, method_info in ranking_methods.items():
        rank = method_info['rank']
        label = method_info['label']
        
        # Extract top 10 and top 30 features
        top_features_data = {}
        for top_n in top_numbers:
            top_indices = rank[:top_n]
            top_frequencies = firing_frequencies[top_indices]
            
            top_features_data[f'top{top_n}'] = {
                'feature_indices': top_indices.tolist(),
                'frequencies': top_frequencies.tolist(),
            }
        
        results_data[ranking_method] = top_features_data
        
        # Create line plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot top 10
        top10_indices = rank[:10]
        top10_freqs = firing_frequencies[top10_indices]
        ax.plot(range(1, 11), top10_freqs, 'o-', linewidth=2, markersize=6,
                label='Top 10', color='steelblue', alpha=0.8)
        
        # Plot top 30
        top30_indices = rank[:30]
        top30_freqs = firing_frequencies[top30_indices]
        ax.plot(range(1, 31), top30_freqs, 's--', linewidth=2, markersize=4,
                label='Top 30', color='coral', alpha=0.8)
        
        ax.set_xlabel('Feature Rank', fontsize=12, fontweight='bold')
        ax.set_ylabel('Firing Frequency (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Top Features Frequency Distribution\nRanked by {label}', 
                    fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_xlim(0.5, 30.5)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plot_file = output_dir / f"top_features_{ranking_method}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {plot_file.name}")
    
    # Save results JSON
    results_json = {
        'top_features': results_data
    }
    results_file = output_dir / "top_features_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"  ✓ Saved results to: {results_file}")


def load_all_top_features_results(
    model_family: str,
    base_dir: Path
) -> List[Dict]:
    """
    Load all top_features_results.json files from all SAEs.
    
    Handles different path structures:
    - saebench: {model_family}/{width}/{sae_type}/{trainer}/firing_frequency_distribution/top_features_results.json
    - canrager: {model_family}/{arch}/{width}/layer_{layer}/{trainer}/firing_frequency_distribution/top_features_results.json
    - jbloom: {model_family}/{width}/{hook_type}/layer_{layer}/{trainer}/firing_frequency_distribution/top_features_results.json
    
    Returns:
        List of dicts, each containing:
        {
            'width': str,
            'sae_type': str,
            'trainer': str,
            'config_path': Path,
            'top_features': dict,
            'architecture': str,
        }
    """
    base_path = base_dir / model_family
    if not base_path.exists():
        return []
    
    results_files = list(base_path.rglob("firing_frequency_distribution/top_features_results.json"))
    all_results = []
    
    # Import architecture detection function
    import sys
    parent_dir = base_dir.parent.parent
    sys.path.insert(0, str(parent_dir))
    from encoder_decoder_activation_relationship.runners.cross_architecture_utils import (
        get_architecture_type
    )
    
    for results_file in results_files:
        # Parse path based on structure
        parts = results_file.parts
        try:
            family_idx = parts.index(model_family)
            
            if len(parts) <= family_idx + 3:
                continue
            
            part1 = parts[family_idx + 1]
            part2 = parts[family_idx + 2]
            part3 = parts[family_idx + 3] if len(parts) > family_idx + 3 else None
            
            # saebench structure: {model_family}/{width}/{sae_type}/{trainer}/...
            # width ends with 'k' (e.g., "4k", "16k", "65k")
            if part1.endswith('k') or part1.isdigit():
                width = part1
                sae_type = part2
                trainer = part3
            # canrager structure: {model_family}/{arch}/{width}/layer_{layer}/{trainer}/...
            elif part3 and 'layer_' in part3:
                arch = part1  # e.g., "topk" or "vanilla"
                width = part2  # e.g., "16k"
                sae_type = arch  # Use arch as sae_type
                trainer = parts[family_idx + 4] if len(parts) > family_idx + 4 else "unknown"
            else:
                # Default: assume standard structure
                width = part1
                sae_type = part2
                trainer = part3 if part3 else "unknown"
        except (ValueError, IndexError):
            continue
        
        # Load results
        try:
            with open(results_file, 'r') as f:
                top_features_data = json.load(f)
        except Exception:
            continue
        
        # Get architecture type
        config_path = results_file.parent.parent / "config.json"
        architecture = get_architecture_type(sae_type, config_path if config_path.exists() else None)
        
        all_results.append({
            'width': width,
            'sae_type': sae_type,
            'trainer': trainer,
            'config_path': config_path,
            'top_features': top_features_data.get('top_features', {}),
            'architecture': architecture,
        })
    
    return all_results


def create_top_features_aggregation_by_architecture(
    model_family: str,
    base_dir: Path,
    output_dir: Path
):
    """
    Create aggregation plots grouped by architecture (Option A).
    
    For each architecture, creates 6 violin plots (3 ranking methods × 2 top numbers)
    showing frequency distribution across all widths/trainers, ordered by sparsity.
    """
    print(f"\n{'='*70}")
    print("CREATING TOP FEATURES AGGREGATION BY ARCHITECTURE")
    print(f"{'='*70}")
    
    # Load all results
    all_results = load_all_top_features_results(model_family, base_dir)
    if not all_results:
        print("  ⚠️  No results found!")
        return
    
    # Group by architecture
    by_architecture = defaultdict(list)
    for result in all_results:
        by_architecture[result['architecture']].append(result)
    
    ranking_methods = ['frequency', 'firing_magnitude_median', 'firing_magnitude_mean']
    ranking_labels = {
        'frequency': 'Firing Frequency',
        'firing_magnitude_median': 'Firing Magnitude Median',
        'firing_magnitude_mean': 'Firing Magnitude Mean',
    }
    top_numbers = [10, 30]
    
    width_colors = {
        '4k': 'steelblue',
        '16k': 'coral',
        '65k': 'mediumseagreen',
    }
    
    for architecture, arch_results in by_architecture.items():
        arch_dir = output_dir / "top_feature_firing_frequency_by_architecture" / architecture
        arch_dir.mkdir(parents=True, exist_ok=True)
        
        for ranking_method in ranking_methods:
            for top_n in top_numbers:
                # Collect data for this ranking method and top_n
                data_to_plot = []
                labels = []
                sparsity_levels = []
                
                for result in arch_results:
                    top_features = result['top_features'].get(ranking_method, {})
                    top_n_data = top_features.get(f'top{top_n}', {})
                    frequencies = top_n_data.get('frequencies', [])
                    
                    if not frequencies:
                        continue
                    
                    # Extract sparsity parameter
                    sparsity_param, sparsity_value = extract_sparsity_parameter_from_config(result['config_path'])
                    if sparsity_param is None:
                        # Try parsing from trainer folder name
                        trainer_name = result['trainer']
                        k_match = re.search(r'_k(\d+)$', trainer_name)
                        if k_match:
                            sparsity_param, sparsity_value = 'k', float(k_match.group(1))
                        else:
                            target_l0_match = re.search(r'_target_l0_(\d+)$', trainer_name)
                            if target_l0_match:
                                sparsity_param, sparsity_value = 'target_l0', float(target_l0_match.group(1))
                            else:
                                l1_match = re.search(r'_l1_penalty_(\d+)_(\d+)$', trainer_name)
                                if l1_match:
                                    value = float(f"{l1_match.group(1)}.{l1_match.group(2)}")
                                    sparsity_param, sparsity_value = 'l1_penalty', value
                    
                    if sparsity_param is None:
                        continue
                    
                    sparsity_level = normalize_sparsity_level(sparsity_param, sparsity_value)
                    
                    # Convert frequencies to numpy array for violin plot
                    freq_array = np.array(frequencies)
                    data_to_plot.append(freq_array)
                    labels.append(f"{result['width']}/{result['trainer']}")
                    sparsity_levels.append(sparsity_level)
                
                if not data_to_plot:
                    continue
                
                # Sort by sparsity level (increasing)
                sorted_indices = sorted(range(len(sparsity_levels)), key=lambda i: sparsity_levels[i])
                data_to_plot = [data_to_plot[i] for i in sorted_indices]
                labels = [labels[i] for i in sorted_indices]
                
                # Create violin plot
                fig, ax = plt.subplots(figsize=(14, 6))
                parts = ax.violinplot(data_to_plot, positions=range(len(data_to_plot)),
                                     widths=0.6, showmeans=True, showmedians=True)
                
                # Color violins by width
                for i, (pc, label) in enumerate(zip(parts['bodies'], labels)):
                    width = label.split('/')[0]
                    pc.set_facecolor(width_colors.get(width, 'gray'))
                    pc.set_alpha(0.7)
                
                ax.set_xlabel('Width / Trainer Config (sorted by sparsity)', fontsize=12, fontweight='bold')
                ax.set_ylabel('Firing Frequency (%)', fontsize=12, fontweight='bold')
                ax.set_title(f'{architecture}: Top {top_n} Features ({ranking_labels[ranking_method]})\nFrequency Distribution Across Widths/Trainers',
                            fontsize=14, fontweight='bold')
                ax.set_ylim(0, 1)
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
                ax.grid(alpha=0.3, axis='y')
                
                plt.tight_layout()
                plot_file = arch_dir / f"top{top_n}_{ranking_method}.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Saved: {plot_file.relative_to(output_dir)}")


def create_top_features_aggregation_by_width(
    model_family: str,
    base_dir: Path,
    output_dir: Path
):
    """
    Create aggregation plots grouped by ranking method and top number, split by width (Option B).
    
    For each ranking method × top number combination, creates 3 PNGs (one per width),
    each showing 30 violin plots (6 architectures × 5 trainers).
    """
    print(f"\n{'='*70}")
    print("CREATING TOP FEATURES AGGREGATION BY WIDTH")
    print(f"{'='*70}")
    
    # Load all results
    all_results = load_all_top_features_results(model_family, base_dir)
    if not all_results:
        print("  ⚠️  No results found!")
        return
    
    ranking_methods = ['frequency', 'firing_magnitude_median', 'firing_magnitude_mean']
    ranking_labels = {
        'frequency': 'Firing Frequency',
        'firing_magnitude_median': 'Firing Magnitude Median',
        'firing_magnitude_mean': 'Firing Magnitude Mean',
    }
    top_numbers = [10, 30]
    widths = ['4k', '16k', '65k']
    
    # Architecture colors
    arch_colors = {
        'ReLU': 'steelblue',
        'TopK': 'coral',
        'BatchTopK': 'mediumseagreen',
        'JumpReLU': 'purple',
        'GatedSAE': 'orange',
        'PAnneal': 'brown',
        'MatryoshkaBatchTopK': 'pink',
    }
    
    # Architecture order for consistent grouping
    arch_order = ['ReLU', 'TopK', 'BatchTopK', 'JumpReLU', 'GatedSAE', 'PAnneal', 'MatryoshkaBatchTopK']
    
    for ranking_method in ranking_methods:
        for top_n in top_numbers:
            folder_name = f"top{top_n}_{ranking_method}"
            folder_dir = output_dir / "top_feature_firing_frequency_by_width" / folder_name
            folder_dir.mkdir(parents=True, exist_ok=True)
            
            for width in widths:
                # Collect data for this width
                data_to_plot = []
                labels = []
                sparsity_levels = []
                
                # Group by architecture first
                arch_groups = defaultdict(list)
                for result in all_results:
                    if result['width'] != width:
                        continue
                    arch_groups[result['architecture']].append(result)
                
                # Process each architecture in order
                for arch in arch_order:
                    if arch not in arch_groups:
                        continue
                    
                    arch_results = arch_groups[arch]
                    # Sort within architecture by sparsity
                    arch_results_with_sparsity = []
                    for result in arch_results:
                        top_features = result['top_features'].get(ranking_method, {})
                        top_n_data = top_features.get(f'top{top_n}', {})
                        frequencies = top_n_data.get('frequencies', [])
                        
                        if not frequencies:
                            continue
                        
                        # Extract sparsity parameter
                        sparsity_param, sparsity_value = extract_sparsity_parameter_from_config(result['config_path'])
                        if sparsity_param is None:
                            # Try parsing from trainer folder name
                            trainer_name = result['trainer']
                            k_match = re.search(r'_k(\d+)$', trainer_name)
                            if k_match:
                                sparsity_param, sparsity_value = 'k', float(k_match.group(1))
                            else:
                                target_l0_match = re.search(r'_target_l0_(\d+)$', trainer_name)
                                if target_l0_match:
                                    sparsity_param, sparsity_value = 'target_l0', float(target_l0_match.group(1))
                                else:
                                    l1_match = re.search(r'_l1_penalty_(\d+)_(\d+)$', trainer_name)
                                    if l1_match:
                                        value = float(f"{l1_match.group(1)}.{l1_match.group(2)}")
                                        sparsity_param, sparsity_value = 'l1_penalty', value
                        
                        if sparsity_param is None:
                            continue
                        
                        sparsity_level = normalize_sparsity_level(sparsity_param, sparsity_value)
                        arch_results_with_sparsity.append((result, frequencies, sparsity_level))
                    
                    # Sort by sparsity level
                    arch_results_with_sparsity.sort(key=lambda x: x[2])
                    
                    # Add to plot data
                    for result, frequencies, sparsity_level in arch_results_with_sparsity:
                        freq_array = np.array(frequencies)
                        data_to_plot.append(freq_array)
                        labels.append(f"{arch}\n{result['trainer']}")
                        sparsity_levels.append(sparsity_level)
                
                if not data_to_plot:
                    continue
                
                # Create violin plot
                fig, ax = plt.subplots(figsize=(18, 8))
                positions = range(len(data_to_plot))
                parts = ax.violinplot(data_to_plot, positions=positions,
                                     widths=0.7, showmeans=True, showmedians=True)
                
                # Color violins by architecture
                for i, (pc, label) in enumerate(zip(parts['bodies'], labels)):
                    arch = label.split('\n')[0]
                    pc.set_facecolor(arch_colors.get(arch, 'gray'))
                    pc.set_alpha(0.7)
                
                ax.set_xlabel('Architecture / Trainer Config (sorted by sparsity)', fontsize=12, fontweight='bold')
                ax.set_ylabel('Firing Frequency (%)', fontsize=12, fontweight='bold')
                ax.set_title(f'Top {top_n} Features ({ranking_labels[ranking_method]}) - Width: {width}\nFrequency Distribution Across All Architectures',
                            fontsize=14, fontweight='bold')
                ax.set_ylim(0, 1)
                ax.set_xticks(positions)
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6)
                ax.grid(alpha=0.3, axis='y')
                
                plt.tight_layout()
                plot_file = folder_dir / f"{width}.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Saved: {plot_file.relative_to(output_dir)}")

