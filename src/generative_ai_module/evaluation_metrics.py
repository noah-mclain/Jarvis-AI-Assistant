"""
Evaluation Metrics Module (Compatibility Layer)

This module is a compatibility layer for the consolidated evaluation.py module.
It re-exports classes and functions from evaluation.py to maintain backward compatibility
with code that imports from evaluation_metrics.py.
"""

# Import everything from the consolidated evaluation module
from .evaluation import (
    EvaluationMetrics,
    evaluate_generation,
    calculate_bleu,
    calculate_rouge,
    evaluation_metrics
)

# Define these functions for backward compatibility
def save_metrics(metrics, output_file):
    """Save metrics to a JSON file"""
    import json
    import os
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Convert any non-serializable objects to strings
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, str, bool, list, dict, tuple)) or value is None:
            serializable_metrics[key] = value
        else:
            serializable_metrics[key] = str(value)
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    
    return True