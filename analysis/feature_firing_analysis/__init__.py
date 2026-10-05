"""
Feature firing analysis module.
"""

from .feature_firing_analysis import (
    compute_firing_statistics,
    create_firing_frequency_distribution_plots,
    create_cumulative_magnitude_distribution_plots,
    create_firing_magnitude_plots,
    create_per_input_magnitude_analysis_simplified,
    create_feature_ranking_overlap_plots,
    create_top_features_frequency_plots,
    create_top_features_aggregation_by_architecture,
    create_top_features_aggregation_by_width,
)

__all__ = [
    'compute_firing_statistics',
    'create_firing_frequency_distribution_plots',
    'create_cumulative_magnitude_distribution_plots',
    'create_firing_magnitude_plots',
    'create_per_input_magnitude_analysis_simplified',
    'create_feature_ranking_overlap_plots',
    'create_top_features_frequency_plots',
    'create_top_features_aggregation_by_architecture',
    'create_top_features_aggregation_by_width',
]

