#!/usr/bin/env python3
"""
Within-Shell Uniformity Analysis

For each angular distance bin (shell layer), compute pairwise angular distances
to determine if vectors are uniformly distributed or clustered within that shell.

In high-dimensional space (d >> 100), vectors uniformly distributed on a 
(d-2)-sphere have mean pairwise angular distance ≈ 90°.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from pathlib import Path


def compute_pairwise_uniformity(normalized_vectors: np.ndarray,
                                original_magnitudes: np.ndarray,
                                angular_distances_rad: np.ndarray,
                                n_distance_bins: int = 10,
                                max_pairs_per_bin: int = 10000,
                                store_full_distribution: bool = False) -> Dict:
    """
    Analyze uniformity within each angular distance bin by computing pairwise distances.
    
    Args:
        normalized_vectors: (n_samples, d_model) unit vectors
        original_magnitudes: (n_samples,) original lengths
        angular_distances_rad: (n_samples,) angular distances from mean
        n_distance_bins: number of bins for angular distance
        max_pairs_per_bin: max number of random pairs to sample per bin (for efficiency)
        
    Returns:
        dict with uniformity analysis results
    """
    n_samples, d_model = normalized_vectors.shape
    
    print(f"   Computing within-shell uniformity (pairwise distances)...")
    
    # Convert to degrees
    angular_distances_deg = np.degrees(angular_distances_rad)
    
    # Create distance bins (percentile-based)
    distance_percentiles = np.linspace(0, 100, n_distance_bins + 1)
    distance_bin_edges = np.percentile(angular_distances_rad, distance_percentiles)
    
    # Analyze each distance bin
    bin_uniformity = []
    
    for i in range(n_distance_bins):
        lower_dist = distance_bin_edges[i]
        upper_dist = distance_bin_edges[i + 1]
        
        # Find vectors in this distance range
        if i == n_distance_bins - 1:
            dist_mask = (angular_distances_rad >= lower_dist) & (angular_distances_rad <= upper_dist)
        else:
            dist_mask = (angular_distances_rad >= lower_dist) & (angular_distances_rad < upper_dist)
        
        n_in_bin = np.sum(dist_mask)
        
        if n_in_bin < 2:
            continue
        
        bin_vectors = normalized_vectors[dist_mask]
        bin_magnitudes = original_magnitudes[dist_mask]
        bin_angles = angular_distances_deg[dist_mask]
        
        # Compute pairwise angular distances
        # For efficiency, sample if too many pairs
        if n_in_bin > 100:  # Sample for large bins
            n_pairs = min(max_pairs_per_bin, n_in_bin * (n_in_bin - 1) // 2)
            indices = np.random.choice(n_in_bin, size=min(200, n_in_bin), replace=False)
            sample_vectors = bin_vectors[indices]
        else:
            sample_vectors = bin_vectors
        
        # Compute all pairwise dot products
        dot_products = sample_vectors @ sample_vectors.T
        dot_products = np.clip(dot_products, -1.0, 1.0)
        
        # Get upper triangle (excluding diagonal)
        upper_tri_mask = np.triu(np.ones_like(dot_products, dtype=bool), k=1)
        pairwise_dots = dot_products[upper_tri_mask]
        
        # Convert to angles
        pairwise_angles_rad = np.arccos(pairwise_dots)
        pairwise_angles_deg = np.degrees(pairwise_angles_rad)
        
        # Statistics
        mean_pairwise = float(np.mean(pairwise_angles_deg))
        std_pairwise = float(np.std(pairwise_angles_deg))
        median_pairwise = float(np.median(pairwise_angles_deg))
        min_pairwise = float(np.min(pairwise_angles_deg))
        max_pairwise = float(np.max(pairwise_angles_deg))
        
        # Percentiles
        p25_pairwise = float(np.percentile(pairwise_angles_deg, 25))
        p75_pairwise = float(np.percentile(pairwise_angles_deg, 75))
        
        # Uniformity metric: how close to 90° (expected for uniform high-D)
        # Normalized deviation: 0 = perfectly uniform, 1 = maximally clustered
        uniformity_score = 1.0 - abs(mean_pairwise - 90.0) / 90.0
        
        bin_info = {
            "distance_bin_idx": i,
            "distance_percentile_range": f"{distance_percentiles[i]:.0f}-{distance_percentiles[i+1]:.0f}%",
            "distance_bin_edge_lower_deg": float(np.degrees(lower_dist)),
            "distance_bin_edge_upper_deg": float(np.degrees(upper_dist)),
            "distance_actual_min_deg": float(np.min(bin_angles)),
            "distance_actual_max_deg": float(np.max(bin_angles)),
            "distance_actual_mean_deg": float(np.mean(bin_angles)),
            "n_vectors": int(n_in_bin),
            "pct_of_total": float(n_in_bin / n_samples * 100),
            
            # Pairwise uniformity statistics
            "mean_pairwise_angle_deg": mean_pairwise,
            "std_pairwise_angle_deg": std_pairwise,
            "median_pairwise_angle_deg": median_pairwise,
            "min_pairwise_angle_deg": min_pairwise,
            "max_pairwise_angle_deg": max_pairwise,
            "p25_pairwise_angle_deg": p25_pairwise,
            "p75_pairwise_angle_deg": p75_pairwise,
            "uniformity_score": uniformity_score,
            
            # Magnitude statistics
            "magnitude_mean": float(np.mean(bin_magnitudes)),
            "magnitude_std": float(np.std(bin_magnitudes)),
        }
        
        # Store full distribution if requested
        if store_full_distribution:
            bin_info["pairwise_angles_deg"] = pairwise_angles_deg
        
        bin_uniformity.append(bin_info)
    
    return {
        "n_distance_bins": n_distance_bins,
        "d_model": d_model,
        "n_samples": n_samples,
        "bin_uniformity": bin_uniformity,
        "theoretical_uniform_angle": 90.0,  # Expected for high-D uniform distribution
    }


def plot_within_shell_uniformity(uniformity_data: Dict, output_path: Path, 
                                 title: str = "Within-Shell Uniformity Analysis"):
    """
    Plot uniformity analysis showing if vectors within each shell are uniformly distributed.
    
    Key question: Are vectors at similar distances from mean uniformly spread on the sphere,
    or are they clustered together?
    
    Answer: Mean pairwise angle ≈ 90° = uniform (expected in high-D)
            Mean pairwise angle < 90° = clustered
    """
    
    bin_data = uniformity_data["bin_uniformity"]
    
    if len(bin_data) == 0:
        print("   ⚠️  No bins to plot")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'{title}\nWithin-Shell Uniformity: Are vectors in the same shell uniformly distributed?',
                 fontsize=14, fontweight='bold')
    
    # Extract data
    bin_labels = [b["distance_percentile_range"] for b in bin_data]
    distance_centers = [b["distance_actual_mean_deg"] for b in bin_data]
    mean_pairwise = [b["mean_pairwise_angle_deg"] for b in bin_data]
    std_pairwise = [b["std_pairwise_angle_deg"] for b in bin_data]
    magnitudes = [b["magnitude_mean"] for b in bin_data]
    magnitude_stds = [b["magnitude_std"] for b in bin_data]
    
    x_pos = np.arange(len(bin_labels))
    
    # Plot 1: Mean pairwise angle per shell (MAIN METRIC)
    ax = axes[0]
    bars = ax.bar(x_pos, mean_pairwise, color='steelblue', alpha=0.7, edgecolor='black',
                  yerr=std_pairwise, capsize=5)
    ax.axhline(y=90, color='red', linestyle='--', linewidth=2, 
              label='Expected for Uniform (90°)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean Pairwise Angle (degrees)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Shell (percentile of distance from mean)', fontsize=11, fontweight='bold')
    ax.set_title('Are vectors within each shell uniformly distributed?\n' +
                 '90° = uniform, <90° = clustered', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 180])
    
    # Add interpretation labels with distance ranges
    for i, (bar, val, dist) in enumerate(zip(bars, mean_pairwise, distance_centers)):
        if val < 75:
            color = 'red'
            label = 'Clustered'
        elif val < 85:
            color = 'orange'
            label = 'Weak uniform'
        else:
            color = 'green'
            label = 'Uniform'
        ax.text(bar.get_x() + bar.get_width()/2, 5, f'{label}\n{dist:.1f}°', 
               ha='center', fontsize=6, color=color, fontweight='bold', rotation=90)
    
    # Plot 2: Standard deviation of pairwise angles (consistency of uniformity)
    ax = axes[1]
    ax.bar(x_pos, std_pairwise, color='coral', alpha=0.7, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Std Dev of Pairwise Angles (degrees)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Shell (percentile)', fontsize=11, fontweight='bold')
    ax.set_title('Consistency of Uniformity\n(Lower = more consistent)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 3: Magnitude distribution per shell
    ax = axes[2]
    ax.bar(x_pos, magnitudes, color='purple', alpha=0.7, edgecolor='black',
           yerr=magnitude_stds, capsize=5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Average Original Magnitude', fontsize=11, fontweight='bold')
    ax.set_xlabel('Shell (percentile)', fontsize=11, fontweight='bold')
    ax.set_title('Magnitude Distribution per Shell\n(Does magnitude correlate with uniformity?)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved within-shell uniformity plot: {output_path.name}")


def plot_pairwise_angle_distributions(uniformity_data: Dict, output_path: Path,
                                      title: str = "Pairwise Angle Distributions"):
    """
    Plot histograms of pairwise angles for each shell.
    
    KEY INSIGHT: For a shell at angular distance θ from mean direction:
    - Maximum possible pairwise angle = 2θ (points on opposite sides of shell)
    - If vectors are CLUSTERED at one point → pairwise angles peak at 0°
    - If vectors are UNIFORMLY spread on shell → pairwise angles peak near 2θ (≈85% of max)
    
    This is MORE ACCURATE than using 90° as a reference, which ignores shell geometry!
    """
    
    bin_data = uniformity_data["bin_uniformity"]
    
    if len(bin_data) == 0:
        print("   ⚠️  No bins to plot")
        return
    
    # Check if we have full distributions
    if "pairwise_angles_deg" not in bin_data[0]:
        print("   ⚠️  Pairwise angle distributions not available (need store_full_distribution=True)")
        return
    
    n_bins = len(bin_data)
    
    # Determine grid size
    n_cols = min(5, n_bins)
    n_rows = (n_bins + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3.5*n_rows))
    if n_bins == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    fig.suptitle(f'{title}\nPairwise Angle Distributions per Shell\n' +
                 '(0° = clustered at one point, peak near Max Possible = uniformly spread on shell)',
                 fontsize=14, fontweight='bold')
    
    # First pass: compute all histograms to find global max density for consistent y-axis
    max_density = 0
    all_hist_data = []
    for bin_info in bin_data:
        pairwise_angles = bin_info["pairwise_angles_deg"]
        counts, _ = np.histogram(pairwise_angles, bins=40, range=(0, 180), density=True)
        max_density = max(max_density, np.max(counts))
        all_hist_data.append((pairwise_angles, counts))
    
    # Second pass: plot with consistent axes
    for i, bin_info in enumerate(bin_data):
        ax = axes[i]
        
        pairwise_angles = bin_info["pairwise_angles_deg"]
        bin_label = bin_info["distance_percentile_range"]
        distance_mean = bin_info["distance_actual_mean_deg"]
        distance_min = bin_info["distance_actual_min_deg"]
        distance_max = bin_info["distance_actual_max_deg"]
        n_vectors = bin_info["n_vectors"]
        mean_pw = bin_info["mean_pairwise_angle_deg"]
        
        # Calculate MAXIMUM POSSIBLE pairwise angle for this shell
        # Two points on a spherical cap at distance θ from pole can be at most 2θ apart
        max_possible_pairwise = 2 * distance_max
        
        # For uniform distribution on a spherical cap in high dimensions,
        # the expected pairwise distance is closer to max than to 0
        # A good heuristic: uniform peak ≈ 0.8 to 0.9 of max_possible
        expected_uniform_peak = 0.85 * max_possible_pairwise
        
        # Plot histogram
        ax.hist(pairwise_angles, bins=40, range=(0, 180), color='steelblue', alpha=0.7, 
                edgecolor='black', density=True)
        
        # Mark key statistics
        ax.axvline(mean_pw, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_pw:.1f}°')
        ax.axvline(max_possible_pairwise, color='purple', linestyle='-', linewidth=2, 
                  label=f'Max possible: {max_possible_pairwise:.1f}°', alpha=0.8)
        ax.axvline(expected_uniform_peak, color='green', linestyle=':', linewidth=2, 
                  label=f'Uniform peak ≈{expected_uniform_peak:.1f}°', alpha=0.7)
        
        # Interpretation based on shell geometry
        if mean_pw < 0.5 * max_possible_pairwise:
            interpretation = 'Highly clustered'
            color = 'red'
        elif mean_pw < 0.7 * max_possible_pairwise:
            interpretation = 'Some clustering'
            color = 'orange'
        else:
            interpretation = 'Uniformly spread'
            color = 'green'
        
        ax.set_title(f'Shell {bin_label}\nθ: [{distance_min:.1f}°, {distance_max:.1f}°] (mean={distance_mean:.1f}°), n={n_vectors}',
                    fontsize=9)
        ax.set_xlabel('Pairwise Angle (degrees)', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.set_xlim([0, 180])
        ax.set_ylim([0, max_density * 1.1])  # Consistent y-axis with 10% padding
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(alpha=0.3)
        
        # Add interpretation text
        ax.text(0.02, 0.98, interpretation, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', 
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
               fontweight='bold')
    
    # Hide unused subplots
    for idx in range(n_bins, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved pairwise angle distributions: {output_path.name}")

