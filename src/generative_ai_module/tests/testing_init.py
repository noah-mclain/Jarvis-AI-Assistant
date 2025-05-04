"""
Simplified __init__.py for testing purposes

This version only imports the essential components and avoids complex dependencies.
"""

# Import core components directly
from .evaluation_metrics import EvaluationMetrics, save_metrics
from .train_models import calculate_metrics

# Define minimal public API
__all__ = [
    'EvaluationMetrics',
    'save_metrics',
    'calculate_metrics',
]

# Version information
__version__ = '0.3.0'
__author__ = 'Jarvis AI Team'
__description__ = 'Generative AI Module for Jarvis Assistant - Testing Version' 