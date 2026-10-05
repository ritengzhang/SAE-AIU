"""
Visualization utilities for interpretability evaluation results.

Provides plotting functions for quantile-based analysis and results visualization.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any


def create_quantile_scores_plot(
    results: Dict[str, Any],
    final_sampled_features: List[int],
    quantile_stats: Dict[str, Dict],
    output_file: Path
) -> None:
    """
    Create quantile score plot showing token and sentence accuracy by quantile.
    
    Args:
        results: Results dictionary containing token_results, sentence_results, feature_firing_frequencies
        final_sampled_features: List of sampled feature indices
        quantile_stats: Dictionary mapping quantile names to statistics
        output_file: Path to save the plot
    """
    token_results = results.get('token_results', {})
    sentence_results = results.get('sentence_results', {})
    feature_firing = results.get('feature_firing_frequencies', {})
    
    # Group features by quantile based on firing frequency
    features_by_quantile = {
        'Q1 (0-25%)': [],
        'Q2 (25-50%)': [],
        'Q3 (50-75%)': [],
        'Q4 (75-100%)': []
    }
    
    # Sort features by firing frequency
    sorted_features = sorted(
        feature_firing.items(),
        key=lambda x: x[1]['firing_count'],
        reverse=True
    )
    n_features = len(sorted_features)
    quartile_size = n_features // 4
    
    for i, (feat_idx, stats) in enumerate(sorted_features):
        if i < quartile_size:
            quantile = 'Q1 (0-25%)'
        elif i < 2 * quartile_size:
            quantile = 'Q2 (25-50%)'
        elif i < 3 * quartile_size:
            quantile = 'Q3 (50-75%)'
        else:
            quantile = 'Q4 (75-100%)'
        features_by_quantile[quantile].append(int(feat_idx))
    
    # Collect all accuracies per quantile for box plots
    quantile_accuracies = {}
    for qname in ['Q1 (0-25%)', 'Q2 (25-50%)', 'Q3 (50-75%)', 'Q4 (75-100%)']:
        feat_indices = features_by_quantile[qname]
        # Handle both integer and string keys in token_results/sentence_results
        token_accs = []
        for f in feat_indices:
            # Try both integer and string key
            result = token_results.get(f, token_results.get(str(f), {}))
            if result and not result.get('error') and 'accuracy' in result:
                token_accs.append(result['accuracy'])
        
        sentence_accs = []
        for f in feat_indices:
            # Try both integer and string key
            result = sentence_results.get(f, sentence_results.get(str(f), {}))
            if result and not result.get('error') and 'accuracy' in result:
                sentence_accs.append(result['accuracy'])
        
        quantile_accuracies[qname] = {
            'token_accs': token_accs if token_accs else [0.0],
            'sentence_accs': sentence_accs if sentence_accs else [0.0],
            'n_features': len(feat_indices)
        }
    
    # Create plot - 1x2 layout (only comparison and firing frequency)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Quantile Performance Analysis', fontsize=16, fontweight='bold')
    
    quantiles = ['Q1 (0-25%)', 'Q2 (25-50%)', 'Q3 (50-75%)', 'Q4 (75-100%)']
    quantile_labels = ['Q1', 'Q2', 'Q3', 'Q4']
    x_pos = np.arange(len(quantiles))
    positions = np.arange(1, len(quantiles) + 1)
    width = 0.35
    
    # Prepare data for box plots
    token_box_data = [quantile_accuracies[q]['token_accs'] for q in quantiles]
    sentence_box_data = [quantile_accuracies[q]['sentence_accs'] for q in quantiles]
    
    # Plot 1: Comparison - Token vs Sentence accuracy (side by side box plots)
    ax = axes[0]
    pos1 = positions - width/2
    pos2 = positions + width/2
    
    bp1 = ax.boxplot(token_box_data, positions=pos1, widths=width*0.8, patch_artist=True,
                     showmeans=True, meanline=True,
                     boxprops=dict(facecolor='steelblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2),
                     meanprops=dict(color='green', linewidth=2, linestyle='--'),
                     whiskerprops=dict(color='black', linewidth=1.5),
                     capprops=dict(color='black', linewidth=1.5))
    
    bp2 = ax.boxplot(sentence_box_data, positions=pos2, widths=width*0.8, patch_artist=True,
                     showmeans=True, meanline=True,
                     boxprops=dict(facecolor='coral', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2),
                     meanprops=dict(color='green', linewidth=2, linestyle='--'),
                     whiskerprops=dict(color='black', linewidth=1.5),
                     capprops=dict(color='black', linewidth=1.5))
    
    ax.axhline(1/3, color='red', linestyle=':', linewidth=2, label='Random (1/3)', alpha=0.7)
    ax.set_xlabel('Quantile', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Token vs Sentence Accuracy Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(positions)
    ax.set_xticklabels(quantile_labels)
    # Create custom legend
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='steelblue', alpha=0.7, label='Token'),
        Rectangle((0, 0), 1, 1, facecolor='coral', alpha=0.7, label='Sentence'),
        Line2D([0], [0], color='red', linestyle=':', linewidth=2, label='Random (1/3)')
    ]
    ax.legend(handles=legend_elements)
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim([0, 1.0])
    
    # Plot 2: Firing frequency by quantile (keep as bar chart)
    ax = axes[1]
    firing_freqs = [quantile_stats[q]['mean_firing'] for q in quantiles]
    firing_pcts = [quantile_stats[q]['firing_percentage'] for q in quantiles]
    
    # Create dual y-axis
    ax1 = ax
    bars = ax1.bar(x_pos, firing_freqs, width, alpha=0.7, color='mediumseagreen',
                   edgecolor='black', label='Mean Firing Count')
    ax1.set_xlabel('Quantile', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Firing Count (sequences)', fontsize=12, fontweight='bold', color='darkgreen')
    ax1.tick_params(axis='y', labelcolor='darkgreen')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(quantile_labels)
    ax1.grid(alpha=0.3, axis='y')
    
    # Add percentage as second y-axis
    ax2 = ax1.twinx()
    line = ax2.plot(x_pos, firing_pcts, 'o-', color='purple', linewidth=2,
                    markersize=8, label='Firing Percentage')
    ax2.set_ylabel('Firing Percentage (%)', fontsize=12, fontweight='bold', color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    
    ax1.set_title('Firing Frequency by Quantile', fontsize=13, fontweight='bold')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved quantile scores plot to: {output_file}")


def create_dame_metric_plots(
    results: Dict[str, Any],
    final_sampled_features: List[int],
    quantile_stats: Dict[str, Dict],
    output_dir: Path
) -> None:
    """
    Create 4 separate PNG plots for DAME metrics:
    1. Perfect ranking accuracy
    2. Rank correlation
    3. Pairwise accuracy
    4. Rank distance
    
    Args:
        results: Results dictionary containing token_results, sentence_results, feature_firing_frequencies
        final_sampled_features: List of sampled feature indices
        quantile_stats: Dictionary mapping quantile names to statistics
        output_dir: Directory to save the 4 plots
    """
    token_results = results.get('token_results', {})
    sentence_results = results.get('sentence_results', {})
    feature_firing = results.get('feature_firing_frequencies', {})
    
    # Group features by quantile based on dominance (from quantile_stats)
    features_by_quantile = {
        'Q1 (0-25%)': [],
        'Q2 (25-50%)': [],
        'Q3 (50-75%)': [],
        'Q4 (75-100%)': []
    }
    
    # Map features to quantiles from quantile_stats
    quantile_names = ['Q1 (0-25%)', 'Q2 (25-50%)', 'Q3 (50-75%)', 'Q4 (75-100%)']
    quantile_features = []
    for qname in quantile_names:
        if qname in quantile_stats:
            # Get features in this quantile (they're sampled sequentially)
            n_in_quantile = quantile_stats[qname].get('n_sampled', 0)
            quantile_features.append(n_in_quantile)
        else:
            quantile_features.append(0)
    
    # Distribute final_sampled_features into quantiles
    idx = 0
    for q_idx, qname in enumerate(quantile_names):
        n_features = quantile_features[q_idx]
        if idx + n_features <= len(final_sampled_features):
            features_by_quantile[qname] = final_sampled_features[idx:idx+n_features]
            idx += n_features
    
    # Collect metrics per quantile
    quantile_metrics = {}
    for qname in quantile_names:
        feat_indices = features_by_quantile[qname]
        
        token_perfect = []
        token_corr = []
        token_pairwise = []
        token_dist = []
        
        sentence_perfect = []
        sentence_corr = []
        sentence_pairwise = []
        sentence_dist = []
        
        for f in feat_indices:
            # Token results
            result = token_results.get(f, token_results.get(str(f), {}))
            if result and not result.get('error'):
                if 'perfect_ranking_acc' in result:
                    token_perfect.append(result['perfect_ranking_acc'])
                if 'mean_rank_correlation' in result:
                    token_corr.append(result['mean_rank_correlation'])
                if 'mean_pairwise_acc' in result:
                    token_pairwise.append(result['mean_pairwise_acc'])
                if 'mean_rank_distance' in result:
                    token_dist.append(result['mean_rank_distance'])
            
            # Sentence results
            result = sentence_results.get(f, sentence_results.get(str(f), {}))
            if result and not result.get('error'):
                if 'perfect_ranking_acc' in result:
                    sentence_perfect.append(result['perfect_ranking_acc'])
                if 'mean_rank_correlation' in result:
                    sentence_corr.append(result['mean_rank_correlation'])
                if 'mean_pairwise_acc' in result:
                    sentence_pairwise.append(result['mean_pairwise_acc'])
                if 'mean_rank_distance' in result:
                    sentence_dist.append(result['mean_rank_distance'])
        
        quantile_metrics[qname] = {
            'token': {
                'perfect_ranking': token_perfect if token_perfect else [0.0],
                'correlation': token_corr if token_corr else [0.0],
                'pairwise': token_pairwise if token_pairwise else [0.0],
                'rank_distance': token_dist if token_dist else [0.0],
            },
            'sentence': {
                'perfect_ranking': sentence_perfect if sentence_perfect else [0.0],
                'correlation': sentence_corr if sentence_corr else [0.0],
                'pairwise': sentence_pairwise if sentence_pairwise else [0.0],
                'rank_distance': sentence_dist if sentence_dist else [0.0],
            }
        }
    
    quantile_labels = ['Q1', 'Q2', 'Q3', 'Q4']
    positions = np.arange(1, len(quantile_names) + 1)
    width = 0.35
    
    # Create 4 separate plots
    metrics = [
        ('perfect_ranking', 'Perfect Ranking Accuracy', 0.0, 1.0),
        ('correlation', 'Rank Correlation (Spearman)', -1.0, 1.0),
        ('pairwise', 'Pairwise Accuracy', 0.0, 1.0),
        ('rank_distance', 'Rank Distance', 0.0, 3.0),
    ]
    
    for metric_key, metric_title, y_min, y_max in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare box plot data
        token_box_data = [quantile_metrics[q]['token'][metric_key] for q in quantile_names]
        sentence_box_data = [quantile_metrics[q]['sentence'][metric_key] for q in quantile_names]
        
        pos1 = positions - width/2
        pos2 = positions + width/2
        
        bp1 = ax.boxplot(token_box_data, positions=pos1, widths=width*0.8, patch_artist=True,
                         showmeans=True, meanline=True,
                         boxprops=dict(facecolor='steelblue', alpha=0.7),
                         medianprops=dict(color='red', linewidth=2),
                         meanprops=dict(color='green', linewidth=2, linestyle='--'),
                         whiskerprops=dict(color='black', linewidth=1.5),
                         capprops=dict(color='black', linewidth=1.5))
        
        bp2 = ax.boxplot(sentence_box_data, positions=pos2, widths=width*0.8, patch_artist=True,
                         showmeans=True, meanline=True,
                         boxprops=dict(facecolor='coral', alpha=0.7),
                         medianprops=dict(color='red', linewidth=2),
                         meanprops=dict(color='green', linewidth=2, linestyle='--'),
                         whiskerprops=dict(color='black', linewidth=1.5),
                         capprops=dict(color='black', linewidth=1.5))
        
        if metric_key == 'perfect_ranking' or metric_key == 'pairwise':
            ax.axhline(1/3, color='red', linestyle=':', linewidth=2, label='Random (1/3)', alpha=0.7)
        
        ax.set_xlabel('Quantile', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_title, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_title} by Quantile', fontsize=14, fontweight='bold')
        ax.set_xticks(positions)
        ax.set_xticklabels(quantile_labels)
        
        from matplotlib.patches import Rectangle
        from matplotlib.lines import Line2D
        legend_elements = [
            Rectangle((0, 0), 1, 1, facecolor='steelblue', alpha=0.7, label='Token'),
            Rectangle((0, 0), 1, 1, facecolor='coral', alpha=0.7, label='Sentence'),
        ]
        if metric_key == 'perfect_ranking' or metric_key == 'pairwise':
            legend_elements.append(Line2D([0], [0], color='red', linestyle=':', linewidth=2, label='Random (1/3)'))
        ax.legend(handles=legend_elements)
        ax.grid(alpha=0.3, axis='y')
        ax.set_ylim([y_min, y_max])
        
        plt.tight_layout()
        output_file = output_dir / f"dame_{metric_key}_plot.png"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved {metric_title} plot to: {output_file}")
