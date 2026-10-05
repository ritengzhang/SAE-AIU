"""
Redundancy Analysis

This analysis studies feature redundancy patterns across frequency quantiles.
Also includes dominance-based quantile redundancy analysis.
"""

from .quantile_redundancy_analysis import (
    split_features_by_frequency_quantiles,
    analyze_quantile_redundancy,
    create_8_panel_quantile_heatmaps,
    create_quantile_comparison_barplot,
    compare_redundant_vs_nonredundant,
    create_redundant_vs_nonredundant_plot,
    compare_redundant_vs_nonredundant_by_quantile,
    create_redundant_vs_nonredundant_plot_by_quantile,
    # Dominance-based functions
    compute_dominance_scores,
    split_features_by_dominance_quantiles,
    create_8_panel_quantile_heatmaps_dominance,
    create_quantile_comparison_barplot_dominance,
    compare_redundant_vs_nonredundant_dominance,
    create_redundant_vs_nonredundant_plot_dominance,
    compare_redundant_vs_nonredundant_by_quantile_dominance,
    create_redundant_vs_nonredundant_plot_by_quantile_dominance,
    get_feature_quantile_mapping_dominance,
)

__all__ = [
    'split_features_by_frequency_quantiles',
    'analyze_quantile_redundancy',
    'create_8_panel_quantile_heatmaps',
    'create_quantile_comparison_barplot',
    'compare_redundant_vs_nonredundant',
    'create_redundant_vs_nonredundant_plot',
    'compare_redundant_vs_nonredundant_by_quantile',
    'create_redundant_vs_nonredundant_plot_by_quantile',
    # Dominance-based functions
    'compute_dominance_scores',
    'split_features_by_dominance_quantiles',
    'create_8_panel_quantile_heatmaps_dominance',
    'create_quantile_comparison_barplot_dominance',
    'compare_redundant_vs_nonredundant_dominance',
    'create_redundant_vs_nonredundant_plot_dominance',
    'compare_redundant_vs_nonredundant_by_quantile_dominance',
    'create_redundant_vs_nonredundant_plot_by_quantile_dominance',
    'get_feature_quantile_mapping_dominance',
]

