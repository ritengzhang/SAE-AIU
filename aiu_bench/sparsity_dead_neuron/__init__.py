"""
Sparsity and Dead Neuron Analysis Module

This module provides functions for calculating sparsity and dead neuron metrics for SAEs:
1. Average active features (simple counting)
2. Effective active features (features that sum to 75% and 90% of total activation)
3. Dead neuron counting (features that never fire or fire very rarely)
4. Common utilities for activation collection and store setup
"""

from .sparsity_and_deadneuron_analysis import (
    calculate_sparsity_metrics,
    calculate_average_active_features,
    calculate_effective_active_features,
    calculate_dead_neuron_metrics,
    collect_activations,
    setup_activations_store_for_benchmark,
)

__all__ = [
    'calculate_sparsity_metrics',
    'calculate_average_active_features',
    'calculate_effective_active_features',
    'calculate_dead_neuron_metrics',
    'collect_activations',
    'setup_activations_store_for_benchmark',
]

