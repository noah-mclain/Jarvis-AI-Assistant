"""
Import Fix Module

This module provides all the functions and classes that were missing or had issues in imports.
Simply import this module first to ensure all dependencies are properly available.
"""

import os
import sys
import torch
import numpy as np
import json
from datetime import datetime

# Add the parent directory to sys.path for proper imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Calculate metrics function
def calculate_metrics(model, data_batches, device):
    """Calculate metrics on a dataset (loss, perplexity, accuracy)"""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    total_correct = 0
    total_samples = 0
    
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for input_batch, target_batch in data_batches:
            # Move data to the model's device
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            
            # Forward pass
            output, _ = model(input_batch)
            
            # Calculate loss
            loss = criterion(output.view(-1, output.size(-1)), target_batch.view(-1))
            total_loss += loss.item()
            
            # Calculate accuracy
            predictions = output.argmax(dim=-1)
            correct = (predictions == target_batch).sum().item()
            total_correct += correct
            total_samples += target_batch.numel()
            
            total_batches += 1
    
    # Calculate metrics
    avg_loss = total_loss / max(1, total_batches)
    perplexity = np.exp(avg_loss)
    accuracy = total_correct / max(1, total_samples)
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'accuracy': accuracy
    }

def save_metrics(metrics, model_name, dataset_name, timestamp=None):
    """
    Save evaluation metrics to a JSON file.
    
    Args:
        metrics (dict): Dictionary of metrics
        model_name (str): Name of the model
        dataset_name (str): Name of the dataset
        timestamp (str, optional): Timestamp to use in the filename
    
    Returns:
        str: Path to the saved metrics file
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create metrics directory
    metrics_dir = "metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Create a clean filename
    model_name_clean = model_name.replace('/', '_')
    dataset_name_clean = dataset_name.replace('/', '_')
    
    filename = f"{model_name_clean}_{dataset_name_clean}_{timestamp}.json"
    filepath = os.path.join(metrics_dir, filename)
    
    # Add metadata
    metrics_with_meta = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "timestamp": timestamp,
        "metrics": metrics
    }
    
    # Save the metrics
    with open(filepath, 'w') as f:
        json.dump(metrics_with_meta, f, indent=2)
    
    print(f"Saved metrics to {filepath}")
    
    return filepath

# EvaluationMetrics class
class EvaluationMetrics:
    """Class for evaluating generative models"""
    
    def __init__(self, metrics_dir="evaluation_metrics", use_gpu=None):
        """Initialize the metrics"""
        self.metrics_dir = metrics_dir
        os.makedirs(metrics_dir, exist_ok=True)
        self.use_gpu = torch.cuda.is_available() if use_gpu is None else use_gpu
    
    def evaluate_generation(self, prompt, generated_text, reference_text=None, 
                          dataset_name="unknown", save_results=True):
        """Evaluate generated text against reference"""
        results = {
            "prompt": prompt,
            "generated_text": generated_text,
            "dataset": dataset_name,
            "timestamp": datetime.now().isoformat()
        }
        
        if reference_text:
            results["reference_text"] = reference_text
        
        # Save results if requested
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = os.path.join(self.metrics_dir, f"evaluation_{dataset_name}_{timestamp}.json")
            
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        return results

# Update the module's __all__ to include these functions
__all__ = [
    'calculate_metrics',
    'save_metrics',
    'EvaluationMetrics'
]

# Monkey patch the required modules
try:
    import src.generative_ai_module.evaluation_metrics
    sys.modules['src.generative_ai_module.evaluation_metrics'].calculate_metrics = calculate_metrics
    sys.modules['src.generative_ai_module.evaluation_metrics'].save_metrics = save_metrics
    sys.modules['src.generative_ai_module.evaluation_metrics'].EvaluationMetrics = EvaluationMetrics
except ImportError:
    # Module not imported yet, that's fine
    pass

# Fix the module import if this is imported directly
if __name__ != "__main__":
    # Add ourselves to sys.modules
    sys.modules['src.generative_ai_module.calculate_metrics'] = sys.modules[__name__]
    sys.modules['src.generative_ai_module.evaluate_generation'] = sys.modules[__name__] 