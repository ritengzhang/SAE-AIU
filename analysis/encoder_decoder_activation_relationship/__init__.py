"""
Encoder-Decoder-Activation relationship analysis module.
"""

from .angular_distance_analysis import (
    normalize_to_sphere,
    is_feature_activated,
    analyze_activated_features_angular_distances,
    create_activated_features_plot,
)

__all__ = [
    'normalize_to_sphere',
    'is_feature_activated',
    'analyze_activated_features_angular_distances',
    'create_activated_features_plot',
]

