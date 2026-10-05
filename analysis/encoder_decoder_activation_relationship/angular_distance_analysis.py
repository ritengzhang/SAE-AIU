#!/usr/bin/env python3
"""
Encoder-Decoder-Activation Angular Relationship Analysis

Core helper functions for analyzing angular relationships between:
- Encoder vectors and input activations
- Decoder vectors and input activations  
- Encoder and decoder vectors (same feature)

All analysis is performed ONLY on features that are actually fired/activated.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm.auto import tqdm

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


def normalize_to_sphere(activations: np.ndarray) -> np.ndarray:
    """
    Normalize activation vectors to unit length.
    
    Args:
        activations: (n, d) array of vectors
        
    Returns:
        Normalized vectors (n, d) with unit length
    """
    norms = np.linalg.norm(activations, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return activations / norms


def is_feature_activated(activation_value: float, sae_architecture: str, 
                        k_value: int = None, total_features: int = None) -> bool:
    """
    Determine if a feature is activated based on SAE architecture rules.
    
    Args:
        activation_value: The activation value
        sae_architecture: SAE architecture name
        k_value: For TopK architectures, the k value
        total_features: For batched_topk, total number of features
        
    Returns:
        True if activated, False otherwise
    """
    # For TopK with abs (topk_abs, batched_topk_abs, jumprelu_topk_abs):
    # Features are activated if they're in the top-k by ABSOLUTE value
    # This is already handled by the encode() function
    
    # For standard architectures (ReLU): activated if > threshold
    # Use small threshold to avoid numerical noise
    return abs(activation_value) > 1e-5


def analyze_activated_features_angular_distances(
    sae,
    activations: np.ndarray,
    encoder_vectors: np.ndarray,
    decoder_vectors: np.ndarray,
    sae_architecture: str = 'standard',
    n_samples: int = 5000,
    activation_threshold: float = 1e-5,
    percentiles: List[int] = None
) -> Dict:
    """
    Analyze angular distances of activated features to input activations.
    
    IMPORTANT: Only analyzes features that are actually fired/activated for each input.
    This is NOT random pairs - it's specifically for features that fire.
    
    Args:
        sae: SAE object with encode() method
        activations: (n_samples, d_in) array of input activations
        encoder_vectors: (d_sae, d_in) encoder weight matrix
        decoder_vectors: (d_sae, d_in) decoder weight matrix
        sae_architecture: SAE architecture name
        n_samples: Number of activations to sample (to avoid OOM)
        activation_threshold: Threshold for considering a feature activated
        percentiles: List of percentiles to analyze (default: [25, 50, 75, 100])
        
    Returns:
        Dictionary with percentile-based statistics:
        {
            'percentile_data': {
                percentile: {
                    'encoder_to_input': [list of angles],
                    'decoder_to_input': [list of angles],
                    'encoder_to_decoder': [list of angles],
                }
            },
            'summary': {
                'percentile_{percentile}': {
                    'encoder_to_input': {mean, median, std, ...},
                    'decoder_to_input': {mean, median, std, ...},
                    'encoder_to_decoder': {mean, median, std, ...},
                }
            }
        }
    """
    if percentiles is None:
        percentiles = [25, 50, 75, 100]
    
    print(f"  Analyzing activated features angular distances...")
    print(f"  Processing {min(n_samples, activations.shape[0])} sampled activations...")
    
    # Storage for results
    results = {
        p: {
            'encoder_to_input': [],
            'decoder_to_input': [],
            'encoder_to_decoder': [],
        }
        for p in percentiles
    }
    
    # Normalize encoder and decoder vectors for angular distance computation
    encoder_normalized = normalize_to_sphere(encoder_vectors)
    decoder_normalized = normalize_to_sphere(decoder_vectors)
    
    # Sample a subset of activations for analysis (to avoid OOM)
    max_samples = min(n_samples, activations.shape[0])
    if max_samples < activations.shape[0]:
        sample_indices = np.random.choice(activations.shape[0], max_samples, replace=False)
        sample_activations = activations[sample_indices]
    else:
        sample_activations = activations
    
    device = next(sae.parameters()).device if hasattr(sae, 'parameters') else 'cpu'
    
    for act_idx in tqdm(range(max_samples), desc="Analyzing activations", leave=False):
        activation = torch.from_numpy(sample_activations[act_idx:act_idx+1]).to(device)
        
        # Get SAE encoding
        with torch.no_grad():
            feature_acts = sae.encode(activation).squeeze(0).cpu().numpy()  # (d_sae,)
        
        # Find activated features (using abs for topk_abs architectures)
        active_mask = np.abs(feature_acts) > activation_threshold
        if not active_mask.any():
            continue
        
        # Get activated feature indices and values
        active_indices = np.where(active_mask)[0]
        active_values = feature_acts[active_mask]
        
        # Rank by absolute activation value
        ranked_indices = np.argsort(np.abs(active_values))[::-1]
        sorted_active_indices = active_indices[ranked_indices]
        
        # Normalize input activation
        input_normalized = normalize_to_sphere(sample_activations[act_idx:act_idx+1])[0]  # (d_in,)
        
        # For each percentile, get corresponding features
        for percentile in percentiles:
            # Get top percentile% of activated features
            n_features_to_take = max(1, int(len(sorted_active_indices) * percentile / 100))
            top_feature_indices = sorted_active_indices[:n_features_to_take]
            
            # Compute angular distances for these features
            for feat_idx in top_feature_indices:
                # Encoder to input
                enc_vec = encoder_normalized[feat_idx]
                dot_enc_input = np.clip(np.dot(enc_vec, input_normalized), -1.0, 1.0)
                ang_enc_input = float(np.degrees(np.arccos(dot_enc_input)))
                results[percentile]['encoder_to_input'].append(ang_enc_input)
                
                # Decoder to input
                dec_vec = decoder_normalized[feat_idx]
                dot_dec_input = np.clip(np.dot(dec_vec, input_normalized), -1.0, 1.0)
                ang_dec_input = float(np.degrees(np.arccos(dot_dec_input)))
                results[percentile]['decoder_to_input'].append(ang_dec_input)
                
                # Encoder to decoder (same feature)
                dot_enc_dec = np.clip(np.dot(enc_vec, dec_vec), -1.0, 1.0)
                ang_enc_dec = float(np.degrees(np.arccos(dot_enc_dec)))
                results[percentile]['encoder_to_decoder'].append(ang_enc_dec)
    
    # Compute summary statistics
    summary = {}
    for percentile in percentiles:
        summary[f'percentile_{percentile}'] = {}
        for dist_type in ['encoder_to_input', 'decoder_to_input', 'encoder_to_decoder']:
            dists = results[percentile][dist_type]
            if dists:
                summary[f'percentile_{percentile}'][dist_type] = {
                    'mean': float(np.mean(dists)),
                    'median': float(np.median(dists)),
                    'std': float(np.std(dists)),
                    'min': float(np.min(dists)),
                    'max': float(np.max(dists)),
                    'count': len(dists),
                }
            else:
                summary[f'percentile_{percentile}'][dist_type] = None
    
    return {
        'percentile_data': results,
        'summary': summary,
    }


def create_activated_features_plot(
    results: Dict,
    output_path: Path,
    metadata: Dict,
    percentiles: List[int] = None
):
    """
    Create visualization plot for activated features angular distance analysis.
    
    Creates a 2x2 subplot figure showing:
    - Top-left: Encoder → Input angular distances
    - Top-right: Decoder → Input angular distances
    - Bottom-left: Encoder → Decoder (same feature) angular distances
    - Bottom-right: Box plots summary across percentiles
    
    Args:
        results: Results dictionary from analyze_activated_features_angular_distances()
        output_path: Path to save the plot
        metadata: Dictionary with 'description' and 'experiment_folder' keys
        percentiles: List of percentiles to plot (default: [25, 50, 75, 100])
    """
    if percentiles is None:
        percentiles = [25, 50, 75, 100]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    exp_name = metadata.get('description', 'Unknown')
    exp_folder = metadata.get('experiment_folder', 'Unknown')
    fig.suptitle(
        f'Activated Features Angular Distance Analysis\n{exp_folder}: {exp_name}\n' +
        '(Features ranked by activation magnitude)',
        fontsize=16, fontweight='bold'
    )
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    labels = ['Top 25%', 'Top 50%', 'Top 75%', 'All (100%)']
    
    # Plot 1: Encoder to Input
    ax = axes[0, 0]
    for p, color, label in zip(percentiles, colors, labels):
        dists = results['percentile_data'][p]['encoder_to_input']
        if dists:
            ax.hist(dists, bins=40, alpha=0.5, color=color, label=label, edgecolor='black')
    ax.set_xlabel('Angular Distance (degrees)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.set_title('Encoder Vector → Input Activation', fontsize=12, fontweight='bold')
    ax.set_xlim([0, 180])
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: Decoder to Input
    ax = axes[0, 1]
    for p, color, label in zip(percentiles, colors, labels):
        dists = results['percentile_data'][p]['decoder_to_input']
        if dists:
            ax.hist(dists, bins=40, alpha=0.5, color=color, label=label, edgecolor='black')
    ax.set_xlabel('Angular Distance (degrees)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.set_title('Decoder Vector → Input Activation', fontsize=12, fontweight='bold')
    ax.set_xlim([0, 180])
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: Encoder to Decoder
    ax = axes[1, 0]
    for p, color, label in zip(percentiles, colors, labels):
        dists = results['percentile_data'][p]['encoder_to_decoder']
        if dists:
            ax.hist(dists, bins=40, alpha=0.5, color=color, label=label, edgecolor='black')
    ax.set_xlabel('Angular Distance (degrees)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.set_title('Encoder Vector → Decoder Vector (Same Feature)', fontsize=12, fontweight='bold')
    ax.set_xlim([0, 180])
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 4: Box plots showing distribution statistics for each percentile and type
    ax = axes[1, 1]
    
    dist_types = [
        ('encoder_to_input', 'Enc→Input', '#FF6B6B'),
        ('decoder_to_input', 'Dec→Input', '#4ECDC4'),
        ('encoder_to_decoder', 'Enc→Dec', '#45B7D1'),
    ]
    
    # Prepare data for box plots
    x_positions = []
    box_data = []
    box_colors = []
    mean_positions = []
    mean_values = []
    mean_colors = []
    
    for i, p in enumerate(percentiles):
        for j, (dist_type, dist_label, color) in enumerate(dist_types):
            dists = results['percentile_data'][p][dist_type]
            if dists and len(dists) > 0:
                # Position for this box
                x_pos = i * 4 + j  # Group by percentile, 3 boxes per group
                x_positions.append(x_pos)
                box_data.append(dists)
                box_colors.append(color)
                
                # Store mean for marking
                mean_val = np.mean(dists)
                mean_positions.append(x_pos)
                mean_values.append(mean_val)
                mean_colors.append(color)
    
    # Create box plots
    bp = ax.boxplot(box_data, positions=x_positions, widths=0.6, patch_artist=True,
                     showfliers=False,  # Don't show outliers to keep it clean
                     boxprops=dict(linewidth=1.5),
                     medianprops=dict(color='black', linewidth=2),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Mark the means with diamond markers
    ax.scatter(mean_positions, mean_values, marker='D', s=80, c=mean_colors, 
              edgecolors='black', linewidths=2, zorder=10, label='Mean')
    
    # Set x-axis labels (percentile groups)
    ax.set_xticks([i * 4 + 1 for i in range(len(percentiles))])
    ax.set_xticklabels([f'Top {p}%' for p in percentiles], fontsize=10, fontweight='bold')
    ax.set_ylabel('Angular Distance (degrees)', fontsize=11, fontweight='bold')
    ax.set_title('Statistics by Percentile\n(Box=Med/Min/Max, Diamond=Mean)', 
                fontweight='bold', fontsize=12)
    ax.set_xlim([-1, len(percentiles) * 4])
    ax.set_ylim([0, 180])
    ax.grid(axis='y', alpha=0.3)
    
    # Add legend for distance types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=dist_types[0][2], alpha=0.7, label=dist_types[0][1]),
        Patch(facecolor=dist_types[1][2], alpha=0.7, label=dist_types[1][1]),
        Patch(facecolor=dist_types[2][2], alpha=0.7, label=dist_types[2][1]),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path.name}")

