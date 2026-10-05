"""
Core modules for interpretability evaluation.

These are the main classes used by all runners:
- ActivationCollector: Collects activation sequences
- ExplanationGenerator: Generates feature explanations
- TokenBasedEvaluator: Token-level multiple-choice evaluation
- SentenceBasedEvaluator: Sentence-level multiple-choice evaluation
"""

from .collection_stage import ActivationCollector, CollectionConfig
from .explanation_stage import ExplanationGenerator
from .evaluation_stage import TokenBasedEvaluator, SentenceBasedEvaluator
from .visualization import create_quantile_scores_plot

__all__ = [
    'ActivationCollector',
    'CollectionConfig',
    'ExplanationGenerator',
    'TokenBasedEvaluator',
    'SentenceBasedEvaluator',
    'create_quantile_scores_plot',
]
