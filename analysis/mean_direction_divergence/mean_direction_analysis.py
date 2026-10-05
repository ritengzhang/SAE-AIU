#!/usr/bin/env python3
"""
Mean Direction Divergence Analysis

This module computes how encoder/decoder weights diverge from the original activation mean direction.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple
import json

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


def normalize_to_sphere(activations: np.ndarray) -> np.ndarray:
    """Normalize activation vectors to unit length."""
    norms = np.linalg.norm(activations, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return activations / norms


def compute_mean_direction_from_activations(activations: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute mean direction from original input activations.
    
    Args:
        activations: Original input activations (n_samples, d_in) - BEFORE SAE encoding
    
    Returns:
        mean_direction: Unit-normalized mean direction (d_in,)
        mean_resultant_length: Mean resultant length R (scalar)
    """
    print(f"\n📊 Computing mean direction from {activations.shape[0]:,} activation vectors...")
    
    # Normalize to unit sphere
    normalized_activations = normalize_to_sphere(activations)
    
    # Compute mean direction
    mean_vector = np.mean(normalized_activations, axis=0)
    mean_vector_norm = np.linalg.norm(mean_vector)
    mean_direction = mean_vector / mean_vector_norm if mean_vector_norm > 0 else mean_vector
    
    print(f"✓ Mean resultant length R: {mean_vector_norm:.4f}")
    
    return mean_direction, mean_vector_norm


def compute_angular_distances_from_mean(vectors: np.ndarray, mean_direction: np.ndarray) -> np.ndarray:
    """
    Compute angular distances (in degrees) from mean direction for each vector.
    
    Args:
        vectors: Vectors to compute distances for (n_vectors, d_in)
        mean_direction: Mean direction vector (d_in,)
    
    Returns:
        angular_distances: Angular distances in degrees (n_vectors,)
    """
    # Normalize vectors to unit length
    normalized_vectors = normalize_to_sphere(vectors)
    
    # Compute cosine similarities
    cosine_similarities = normalized_vectors @ mean_direction  # (n_vectors,)
    
    # Clip to valid range for arccos
    cosine_similarities = np.clip(cosine_similarities, -1.0, 1.0)
    
    # Convert to angular distances in degrees
    angular_distances_rad = np.arccos(cosine_similarities)
    angular_distances_deg = np.degrees(angular_distances_rad)
    
    return angular_distances_deg


def compute_angular_divergences(
    encoder_weights: np.ndarray,
    decoder_weights: np.ndarray,
    mean_direction: np.ndarray
) -> Dict:
    """
    Compute angular divergences for encoder and decoder weights from mean direction.
    
    Args:
        encoder_weights: Encoder weight matrix (d_sae, d_in) - already transposed
        decoder_weights: Decoder weight matrix (d_sae, d_in)
        mean_direction: Mean direction vector (d_in,)
    
    Returns:
        Dictionary with divergence statistics
    """
    print(f"\n📊 Computing angular divergences...")
    
    # Compute angular distances
    encoder_distances = compute_angular_distances_from_mean(encoder_weights, mean_direction)
    decoder_distances = compute_angular_distances_from_mean(decoder_weights, mean_direction)
    
    # Compute statistics
    results = {
        'encoder': {
            'mean': float(np.mean(encoder_distances)),
            'median': float(np.median(encoder_distances)),
            'std': float(np.std(encoder_distances)),
            'min': float(np.min(encoder_distances)),
            'max': float(np.max(encoder_distances)),
            'percentile_25': float(np.percentile(encoder_distances, 25)),
            'percentile_75': float(np.percentile(encoder_distances, 75)),
            'percentile_90': float(np.percentile(encoder_distances, 90)),
            'percentile_95': float(np.percentile(encoder_distances, 95)),
            'percentile_99': float(np.percentile(encoder_distances, 99)),
        },
        'decoder': {
            'mean': float(np.mean(decoder_distances)),
            'median': float(np.median(decoder_distances)),
            'std': float(np.std(decoder_distances)),
            'min': float(np.min(decoder_distances)),
            'max': float(np.max(decoder_distances)),
            'percentile_25': float(np.percentile(decoder_distances, 25)),
            'percentile_75': float(np.percentile(decoder_distances, 75)),
            'percentile_90': float(np.percentile(decoder_distances, 90)),
            'percentile_95': float(np.percentile(decoder_distances, 95)),
            'percentile_99': float(np.percentile(decoder_distances, 99)),
        }
    }
    
    print(f"✓ Encoder: mean={results['encoder']['mean']:.1f}°, median={results['encoder']['median']:.1f}°")
    print(f"✓ Decoder: mean={results['decoder']['mean']:.1f}°, median={results['decoder']['median']:.1f}°")
    
    return results, encoder_distances, decoder_distances


def create_divergence_plots(
    encoder_distances: np.ndarray,
    decoder_distances: np.ndarray,
    activation_distances: np.ndarray,
    results: Dict,
    output_dir: Path,
    metadata: Dict
):
    """
    Create plots showing encoder/decoder divergence from mean direction.
    
    Args:
        encoder_distances: Angular distances for encoder weights (d_sae,)
        decoder_distances: Angular distances for decoder weights (d_sae,)
        activation_distances: Angular distances for original activations (n_samples,)
        results: Statistics dictionary
        output_dir: Output directory
        metadata: Metadata dictionary
    """
    print(f"\n{'='*70}")
    print("CREATING DIVERGENCE PLOTS")
    print(f"{'='*70}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Comparison of angular distance distributions
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Angular Distance from Mean Direction\n{metadata.get("experiment_folder", "SAE")}', 
                 fontsize=16, fontweight='bold')
    
    # Left: Histogram comparison
    ax = axes[0]
    ax.hist(activation_distances, bins=50, alpha=0.5, label='Original Activations', 
            color='gray', edgecolor='black', density=True)
    ax.hist(encoder_distances, bins=50, alpha=0.5, label='Encoder Weights', 
            color='steelblue', edgecolor='black', density=True)
    ax.hist(decoder_distances, bins=50, alpha=0.5, label='Decoder Weights', 
            color='coral', edgecolor='black', density=True)
    ax.set_xlabel('Angular Distance from Mean (degrees)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title('Distribution Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Right: Box plot comparison
    ax = axes[1]
    data_to_plot = [activation_distances, encoder_distances, decoder_distances]
    labels = ['Original\nActivations', 'Encoder\nWeights', 'Decoder\nWeights']
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('gray')
    bp['boxes'][1].set_facecolor('steelblue')
    bp['boxes'][2].set_facecolor('coral')
    ax.set_ylabel('Angular Distance from Mean (degrees)', fontsize=12, fontweight='bold')
    ax.set_title('Box Plot Comparison', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_file = output_dir / "1_encoder_decoder_divergence_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot_file.name}")
    
    # Plot 2: Statistics comparison
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    metrics = ['mean', 'median', 'percentile_25', 'percentile_75', 'percentile_90', 'percentile_95', 'percentile_99']
    metric_labels = ['Mean', 'Median', 'P25', 'P75', 'P90', 'P95', 'P99']
    
    encoder_vals = [results['encoder'][m] for m in metrics]
    decoder_vals = [results['decoder'][m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, encoder_vals, width, label='Encoder', 
                   color='steelblue', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, decoder_vals, width, label='Decoder', 
                   color='coral', alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Statistic', fontsize=12, fontweight='bold')
    ax.set_ylabel('Angular Distance (degrees)', fontsize=12, fontweight='bold')
    ax.set_title(f'Encoder vs Decoder Divergence Statistics\n{metadata.get("experiment_folder", "SAE")}', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}°', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plot_file = output_dir / "2_divergence_statistics.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot_file.name}")

