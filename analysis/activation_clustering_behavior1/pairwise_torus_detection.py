#!/usr/bin/env python3
"""
Pairwise Angular Distance Analysis for Torus Geometry Detection

This module provides functions to analyze pairwise angular distances between vectors
to detect torus-like geometry in high-dimensional embeddings.

Key metrics:
1. ALL pairwise distances - overall distribution
2. MIN pairwise per vector - nearest neighbor distances (detects "holes")
3. MAX pairwise per vector - furthest point distances (detects boundedness)

Usage:
    from pairwise_torus_detection import compute_pairwise_angular_distances, plot_pairwise_torus_detection
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional


def compute_pairwise_angular_distances(
    normalized_vectors: np.ndarray,
    sample_size: int = 10000
) -> Dict:
    """
    Compute pairwise angular distance statistics for torus geometry detection.
    
    For computational efficiency, we sample a subset of vectors and compute:
    1. Distribution of ALL pairwise angular distances
    2. Distribution of MINIMUM pairwise distance per vector (nearest neighbor)
    3. Distribution of MAXIMUM pairwise distance per vector (furthest point)
    
    Args:
        normalized_vectors: (n_tokens, d_model) unit vectors
        sample_size: Number of vectors to sample for pairwise computation
        
    Returns:
        Dictionary with pairwise distance statistics:
        - sample_size: Number of vectors analyzed
        - total_pairs: Total number of unique pairs
        - all_pairwise: Stats for all pairwise distances
        - min_pairwise_per_vector: Stats for minimum distance per vector
        - max_pairwise_per_vector: Stats for maximum distance per vector
    """
    n_total = normalized_vectors.shape[0]
    
    # Sample if dataset is too large
    if n_total > sample_size:
        print(f"      Sampling {sample_size} of {n_total} vectors for pairwise analysis...")
        sample_indices = np.random.choice(n_total, size=sample_size, replace=False)
        sample_vectors = normalized_vectors[sample_indices]
    else:
        sample_vectors = normalized_vectors
        sample_size = n_total
    
    print(f"      Computing pairwise distances for {sample_size} vectors...")
    
    # Compute pairwise cosine similarities (dot products)
    # Shape: [sample_size, sample_size]
    cosine_sim_matrix = sample_vectors @ sample_vectors.T
    
    # Clip to valid range and convert to angular distances
    cosine_sim_matrix = np.clip(cosine_sim_matrix, -1.0, 1.0)
    angular_dist_matrix_rad = np.arccos(cosine_sim_matrix)
    angular_dist_matrix_deg = np.degrees(angular_dist_matrix_rad)
    
    # Mask out diagonal (self-similarity)
    mask = ~np.eye(sample_size, dtype=bool)
    
    # 1. ALL pairwise distances (upper triangle to avoid duplicates)
    upper_triangle_mask = np.triu(np.ones((sample_size, sample_size), dtype=bool), k=1)
    all_pairwise_dists = angular_dist_matrix_deg[upper_triangle_mask]
    
    # 2. MINIMUM pairwise distance per vector (nearest neighbor)
    # Set diagonal to inf so it's not selected as minimum
    angular_dist_for_min = angular_dist_matrix_deg.copy()
    np.fill_diagonal(angular_dist_for_min, np.inf)
    min_pairwise_per_vector = np.min(angular_dist_for_min, axis=1)
    
    # 3. MAXIMUM pairwise distance per vector (furthest point)
    # Set diagonal to -inf so it's not selected as maximum
    angular_dist_for_max = angular_dist_matrix_deg.copy()
    np.fill_diagonal(angular_dist_for_max, -np.inf)
    max_pairwise_per_vector = np.max(angular_dist_for_max, axis=1)
    
    # ========================================
    # IMPROVED TORUS DETECTION METRICS
    # ========================================
    # Compute distance to mean direction
    mean_vector = np.mean(normalized_vectors, axis=0)
    mean_vector_norm = np.linalg.norm(mean_vector)
    if mean_vector_norm > 1e-8:
        mean_direction = mean_vector / mean_vector_norm
        dot_products_to_mean = normalized_vectors @ mean_direction
        dot_products_to_mean = np.clip(dot_products_to_mean, -1.0, 1.0)
        distances_to_mean_deg = np.degrees(np.arccos(dot_products_to_mean))
        mean_distance_to_mean = float(np.mean(distances_to_mean_deg))
        min_distance_to_mean = float(np.min(distances_to_mean_deg))  # MIN distance to mean direction
    else:
        mean_distance_to_mean = 0.0
        min_distance_to_mean = 0.0
    
    # CORRECTED hole detection: min distance to mean direction - mean nearest neighbor
    mean_min_pairwise = float(np.mean(min_pairwise_per_vector))
    hole_indicator = min_distance_to_mean - mean_min_pairwise  # Large positive = hole!
    
    # Surface coverage percentiles for clustering assessment
    surface_coverage_90 = (1 - np.cos(np.radians(np.percentile(all_pairwise_dists, 90)))) / 2 * 100
    surface_coverage_95 = (1 - np.cos(np.radians(np.percentile(all_pairwise_dists, 95)))) / 2 * 100
    surface_coverage_99 = (1 - np.cos(np.radians(np.percentile(all_pairwise_dists, 99)))) / 2 * 100
    
    return {
        "sample_size": int(sample_size),
        "total_pairs": int(len(all_pairwise_dists)),
        
        # IMPROVED TORUS DETECTION METRICS
        "mean_min_pairwise_deg": mean_min_pairwise,
        "mean_distance_to_mean_direction_deg": mean_distance_to_mean,
        "min_distance_to_mean_direction_deg": min_distance_to_mean,
        "hole_indicator_deg": hole_indicator,  # Large positive value suggests hole
        "surface_coverage_90pct": float(surface_coverage_90),
        "surface_coverage_95pct": float(surface_coverage_95),
        "surface_coverage_99pct": float(surface_coverage_99),
        
        # ALL pairwise distances
        "all_pairwise": {
            "mean_deg": float(np.mean(all_pairwise_dists)),
            "median_deg": float(np.median(all_pairwise_dists)),
            "std_deg": float(np.std(all_pairwise_dists)),
            "min_deg": float(np.min(all_pairwise_dists)),
            "max_deg": float(np.max(all_pairwise_dists)),
            "percentile_25_deg": float(np.percentile(all_pairwise_dists, 25)),
            "percentile_75_deg": float(np.percentile(all_pairwise_dists, 75)),
            "distribution": all_pairwise_dists,  # Keep full array for plotting
        },
        
        # MINIMUM pairwise distance per vector (nearest neighbor)
        "min_pairwise_per_vector": {
            "mean_deg": float(np.mean(min_pairwise_per_vector)),
            "median_deg": float(np.median(min_pairwise_per_vector)),
            "std_deg": float(np.std(min_pairwise_per_vector)),
            "min_deg": float(np.min(min_pairwise_per_vector)),
            "max_deg": float(np.max(min_pairwise_per_vector)),
            "percentile_25_deg": float(np.percentile(min_pairwise_per_vector, 25)),
            "percentile_75_deg": float(np.percentile(min_pairwise_per_vector, 75)),
            "distribution": min_pairwise_per_vector,
        },
        
        # MAXIMUM pairwise distance per vector (furthest point)
        "max_pairwise_per_vector": {
            "mean_deg": float(np.mean(max_pairwise_per_vector)),
            "median_deg": float(np.median(max_pairwise_per_vector)),
            "std_deg": float(np.std(max_pairwise_per_vector)),
            "min_deg": float(np.min(max_pairwise_per_vector)),
            "max_deg": float(np.max(max_pairwise_per_vector)),
            "percentile_25_deg": float(np.percentile(max_pairwise_per_vector, 25)),
            "percentile_75_deg": float(np.percentile(max_pairwise_per_vector, 75)),
            "distribution": max_pairwise_per_vector,
        },
    }


def classify_geometry(pairwise_data: Dict) -> tuple[str, str, int]:
    """
    Classify the geometry based on improved torus detection metrics.
    
    NEW Torus Detection Logic:
    - HOLE: hole_indicator > 20° (min distance to mean >> mean nearest neighbor)
    - CLUSTERED: surface coverage 95% < 60% (points cluster together)
    - STRUCTURED: Low std (< 15°) suggests organized structure
    
    Args:
        pairwise_data: Dictionary from compute_pairwise_angular_distances
        
    Returns:
        Tuple of (geometry_type, confidence, torus_score)
    """
    hole_indicator = pairwise_data.get("hole_indicator_deg", 0)
    surface_coverage_95 = pairwise_data.get("surface_coverage_95pct", 100)
    all_std = pairwise_data["all_pairwise"]["std_deg"]
    
    # NEW torus indicators
    has_hole = hole_indicator > 20  # Large positive hole indicator
    is_clustered = surface_coverage_95 < 60  # Low surface coverage = clustered
    is_structured = all_std < 15  # Low std = organized structure
    
    torus_score = sum([has_hole, is_clustered, is_structured])
    
    if torus_score >= 2:
        geometry_type = "🔵 TORUS-LIKE"
        confidence = "High" if torus_score == 3 else "Moderate"
    elif surface_coverage_95 > 80 and all_std > 15:
        geometry_type = "🌐 UNIFORM/RANDOM"
        confidence = "High"
    else:
        geometry_type = "📊 CLUSTERED"
        confidence = "Moderate"
    
    return geometry_type, confidence, torus_score


def plot_pairwise_torus_detection(
    pairwise_data: Dict,
    output_path: Path,
    title: str = "Pairwise Angular Distance Analysis"
) -> None:
    """
    Create comprehensive pairwise distance visualization for torus detection.
    
    Args:
        pairwise_data: Dictionary from compute_pairwise_angular_distances
        output_path: Where to save the plot
        title: Title for the plot
    """
    # Create comprehensive pairwise distance visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{title}\n(Torus Detection)', fontsize=14, fontweight='bold')
    
    # Extract data
    all_pairwise = pairwise_data["all_pairwise"]["distribution"]
    min_pairwise = pairwise_data["min_pairwise_per_vector"]["distribution"]
    max_pairwise = pairwise_data["max_pairwise_per_vector"]["distribution"]
    
    # Plot 1: ALL pairwise distances
    ax = axes[0, 0]
    ax.hist(all_pairwise, bins=60, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(pairwise_data["all_pairwise"]["mean_deg"], color='red', linestyle='-', 
              linewidth=2, label=f'Mean: {pairwise_data["all_pairwise"]["mean_deg"]:.1f}°')
    ax.axvline(pairwise_data["all_pairwise"]["median_deg"], color='orange', linestyle='--',
              linewidth=2, label=f'Median: {pairwise_data["all_pairwise"]["median_deg"]:.1f}°')
    ax.set_xlabel('Angular Distance (degrees)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Density', fontsize=10, fontweight='bold')
    ax.set_title(f'ALL Pairwise Angular Distances\n({pairwise_data["total_pairs"]:,} pairs from {pairwise_data["sample_size"]:,} vectors)', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 180])
    
    # Plot 2: MINIMUM pairwise distance per vector (nearest neighbor)
    ax = axes[0, 1]
    ax.hist(min_pairwise, bins=50, density=True, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(pairwise_data["min_pairwise_per_vector"]["mean_deg"], color='red', linestyle='-',
              linewidth=2, label=f'Mean: {pairwise_data["min_pairwise_per_vector"]["mean_deg"]:.1f}°')
    ax.axvline(pairwise_data["min_pairwise_per_vector"]["median_deg"], color='orange', linestyle='--',
              linewidth=2, label=f'Median: {pairwise_data["min_pairwise_per_vector"]["median_deg"]:.1f}°')
    ax.set_xlabel('Angular Distance (degrees)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Density', fontsize=10, fontweight='bold')
    ax.set_title('MINIMUM Pairwise Distance per Vector\n(Nearest Neighbor Distance)', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 180])
    
    # Plot 3: MAXIMUM pairwise distance per vector (furthest point)
    ax = axes[0, 2]
    ax.hist(max_pairwise, bins=50, density=True, alpha=0.7, color='red', edgecolor='black')
    ax.axvline(pairwise_data["max_pairwise_per_vector"]["mean_deg"], color='darkred', linestyle='-',
              linewidth=2, label=f'Mean: {pairwise_data["max_pairwise_per_vector"]["mean_deg"]:.1f}°')
    ax.axvline(pairwise_data["max_pairwise_per_vector"]["median_deg"], color='orange', linestyle='--',
              linewidth=2, label=f'Median: {pairwise_data["max_pairwise_per_vector"]["median_deg"]:.1f}°')
    ax.set_xlabel('Angular Distance (degrees)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Density', fontsize=10, fontweight='bold')
    ax.set_title('MAXIMUM Pairwise Distance per Vector\n(Furthest Point Distance)', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 180])
    
    # Plot 4: Overlay comparison
    ax = axes[1, 0]
    ax.hist(all_pairwise, bins=60, density=True, alpha=0.4, color='steelblue', 
           edgecolor='black', label='All Pairwise')
    ax.hist(min_pairwise, bins=50, density=True, alpha=0.4, color='green',
           edgecolor='black', label='Min per Vector')
    ax.hist(max_pairwise, bins=50, density=True, alpha=0.4, color='red',
           edgecolor='black', label='Max per Vector')
    ax.set_xlabel('Angular Distance (degrees)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Density', fontsize=10, fontweight='bold')
    ax.set_title('Overlay Comparison\n(All three distributions)', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 180])
    
    # Plot 5: Summary statistics comparison
    ax = axes[1, 1]
    categories = ['Mean', 'Median', 'Min', 'Max']
    all_vals = [
        pairwise_data["all_pairwise"]["mean_deg"],
        pairwise_data["all_pairwise"]["median_deg"],
        pairwise_data["all_pairwise"]["min_deg"],
        pairwise_data["all_pairwise"]["max_deg"]
    ]
    min_vals = [
        pairwise_data["min_pairwise_per_vector"]["mean_deg"],
        pairwise_data["min_pairwise_per_vector"]["median_deg"],
        pairwise_data["min_pairwise_per_vector"]["min_deg"],
        pairwise_data["min_pairwise_per_vector"]["max_deg"]
    ]
    max_vals = [
        pairwise_data["max_pairwise_per_vector"]["mean_deg"],
        pairwise_data["max_pairwise_per_vector"]["median_deg"],
        pairwise_data["max_pairwise_per_vector"]["min_deg"],
        pairwise_data["max_pairwise_per_vector"]["max_deg"]
    ]
    
    x = np.arange(len(categories))
    width = 0.25
    ax.bar(x - width, all_vals, width, label='All Pairwise', color='steelblue', alpha=0.7)
    ax.bar(x, min_vals, width, label='Min per Vector', color='green', alpha=0.7)
    ax.bar(x + width, max_vals, width, label='Max per Vector', color='red', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Angular Distance (degrees)', fontsize=10, fontweight='bold')
    ax.set_title('Summary Statistics Comparison', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 6: Interpretation guide
    ax = axes[1, 2]
    ax.axis('off')
    
    # Get classification
    geometry_type, confidence, torus_score = classify_geometry(pairwise_data)
    
    min_min = pairwise_data["min_pairwise_per_vector"]["min_deg"]
    max_max = pairwise_data["max_pairwise_per_vector"]["max_deg"]
    all_mean = pairwise_data["all_pairwise"]["mean_deg"]
    all_std = pairwise_data["all_pairwise"]["std_deg"]
    
    # Recompute torus indicators with new thresholds
    min_mean_nn = pairwise_data["min_pairwise_per_vector"]["mean_deg"]
    has_hole = min_min > 30 and min_mean_nn > 40
    is_bounded = max_max < 170
    is_concentrated = all_std < 15
    
    # Extract new torus detection metrics
    mean_min_pw = pairwise_data.get("mean_min_pairwise_deg", min_mean_nn)
    mean_to_center = pairwise_data.get("mean_distance_to_mean_direction_deg", 0)
    min_to_center = pairwise_data.get("min_distance_to_mean_direction_deg", 0)
    hole_indicator = pairwise_data.get("hole_indicator_deg", 0)
    surf_90 = pairwise_data.get("surface_coverage_90pct", 0)
    surf_95 = pairwise_data.get("surface_coverage_95pct", 0)
    surf_99 = pairwise_data.get("surface_coverage_99pct", 0)
    
    interpretation = f"""
TORUS DETECTION ANALYSIS
Sample: {pairwise_data["sample_size"]:,} vectors

CORRECTED HOLE DETECTION:
  Mean nearest neighbor: {mean_min_pw:.1f}°
  Mean distance to mean: {mean_to_center:.1f}°
  MIN distance to mean: {min_to_center:.1f}°
  Hole indicator: {hole_indicator:.1f}°
  └─ (Large positive = hole! Min to mean >> Mean nearest neighbor)
  
CLUSTERING ASSESSMENT:
  Surface coverage:
    90%: {surf_90:.1f}%
    95%: {surf_95:.1f}%
    99%: {surf_99:.1f}%
  └─ (Low % = clustered)

PAIRWISE DISTANCE STATS:
  All Pairwise: {all_mean:.1f}° (std: {all_std:.1f}°)
  Min per Vector: {min_mean_nn:.1f}° (min: {min_min:.1f}°)
  Max per Vector: {max_max:.1f}° (max: {max_max:.1f}°)

NEW TORUS INDICATORS:
  Hole: {hole_indicator > 20} | Clustered: {surf_95 < 60}
  Structured: {all_std < 15}
  Score: {torus_score}/3

RESULT: {geometry_type}
Confidence: {confidence}
"""
    
    ax.text(0.05, 0.95, interpretation, transform=ax.transAxes,
            fontsize=8, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', 
                     facecolor='lightgreen' if torus_score >= 2 else 'lightyellow' if 'UNIFORM' in geometry_type else 'lightblue',
                     alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved {output_path.name}")
    print(f"      Geometry: {geometry_type} (Confidence: {confidence})")

