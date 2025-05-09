#!/usr/bin/env python3
"""
Final fix for import issues - direct implementation of required functions
"""

import os
import sys
import torch
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("jarvis_fix")

print("🔧 Loading direct implementations of required functions...")

# Direct implementation of calculate_metrics function
def calculate_metrics(model, data_batches, device):
    """Calculate metrics on a dataset (loss, perplexity, accuracy)"""
    print("📊 Using direct implementation of calculate_metrics")
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

# Direct implementation of save_metrics function
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
    print("📊 Using direct implementation of save_metrics")
    import json
    from datetime import datetime
    
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
    
    logger.info(f"Saved metrics to {filepath}")
    
    return filepath

# Simple EvaluationMetrics class
class EvaluationMetrics:
    """
    Minimal implementation of the EvaluationMetrics class for compatibility.
    """
    
    def __init__(self, metrics_dir="evaluation_metrics", use_gpu=None):
        """Initialize the metrics with minimal functionality"""
        print("📊 Using direct implementation of EvaluationMetrics")
        self.metrics_dir = metrics_dir
        os.makedirs(metrics_dir, exist_ok=True)
        self.use_gpu = torch.cuda.is_available() if use_gpu is None else use_gpu
    
    def evaluate_generation(self, prompt, generated_text, reference_text=None, 
                          dataset_name="unknown", save_results=True):
        """
        Simplified evaluate_generation method
        """
        print(f"Evaluating generation for dataset: {dataset_name}")
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
            import json
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = os.path.join(self.metrics_dir, f"evaluation_{dataset_name}_{timestamp}.json")
            
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Evaluation results saved to {results_path}")
        
        return results

print("✅ Successfully loaded direct implementations!")
print("✅ You can now import the following:")
print("   - calculate_metrics")
print("   - save_metrics")
print("   - EvaluationMetrics class")

# Test the implementations if run directly
if __name__ == "__main__":
    # Create a simple output directory
    os.makedirs("metrics", exist_ok=True)
    
    # Test save_metrics
    test_metrics = {
        "accuracy": 0.95,
        "loss": 0.05
    }
    
    metrics_path = save_metrics(test_metrics, "test_model", "test_dataset")
    print(f"Saved test metrics to: {metrics_path}")
    
    # Test EvaluationMetrics
    evaluator = EvaluationMetrics()
    results = evaluator.evaluate_generation(
        prompt="Test prompt",
        generated_text="Test generated text",
        dataset_name="test_dataset"
    )
    
    print("All functions tested successfully!") 