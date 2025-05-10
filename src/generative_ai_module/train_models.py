"""
Train Models Module (Compatibility Layer)

This module is a compatibility layer to maintain backward compatibility with
code that imports from train_models.py. It provides functions that were originally
defined in train_models.py but are now part of the consolidated_generation_pipeline.py
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple

# Import from consolidated modules
from .consolidated_generation_pipeline import ConsolidatedGenerationPipeline
from .evaluation_metrics import calculate_bleu, calculate_rouge, evaluate_generation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Compatibility functions
def calculate_metrics(reference_texts, generated_texts):
    """
    Calculate evaluation metrics between reference and generated texts.
    
    Args:
        reference_texts: List of reference texts
        generated_texts: List of generated texts
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Ensure inputs are lists
    if isinstance(reference_texts, str):
        reference_texts = [reference_texts]
    if isinstance(generated_texts, str):
        generated_texts = [generated_texts]
    
    # Ensure lists are of the same length
    if len(reference_texts) != len(generated_texts):
        if len(reference_texts) == 1:
            reference_texts = reference_texts * len(generated_texts)
        elif len(generated_texts) == 1:
            generated_texts = generated_texts * len(reference_texts)
        else:
            logger.warning("Reference and generated text lists have different lengths")
            min_len = min(len(reference_texts), len(generated_texts))
            reference_texts = reference_texts[:min_len]
            generated_texts = generated_texts[:min_len]
    
    # Calculate metrics using consolidated evaluation_metrics
    metrics = evaluate_generation(reference_texts, generated_texts)
    
    return metrics

def train_model(dataset, model_type="text", config=None, output_dir=None):
    """
    Train a model on a dataset.
    
    Args:
        dataset: Dataset to train on
        model_type: Type of model to train ("text" or "code")
        config: Configuration dictionary
        output_dir: Directory to save the model
        
    Returns:
        Dictionary with training information
    """
    # Create a consolidated generation pipeline
    pipeline = ConsolidatedGenerationPipeline(model_type=model_type, config=config)
    
    # Train the model
    training_info = pipeline.train(dataset)
    
    # Save the model if output_dir is provided
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f"{model_type}_model.pt")
        pipeline.save_model(model_path)
        training_info["model_path"] = model_path
    
    return training_info

def train_text_model(dataset, config=None, output_dir=None):
    """
    Train a text generation model.
    
    This is a specialized version of train_model specifically for text models.
    
    Args:
        dataset: Dataset to train on
        config: Configuration dictionary
        output_dir: Directory to save the model
        
    Returns:
        Dictionary with training information
    """
    return train_model(dataset, model_type="text", config=config, output_dir=output_dir)

def generate_with_model(model_path, seed_text, model_type="text", max_length=100, temperature=0.7):
    """
    Generate text with a trained model.
    
    Args:
        model_path: Path to the trained model
        seed_text: Seed text to start generation from
        model_type: Type of model ("text" or "code")
        max_length: Maximum length of generated text
        temperature: Temperature for sampling
        
    Returns:
        Generated text
    """
    # Create a consolidated generation pipeline
    pipeline = ConsolidatedGenerationPipeline(model_type=model_type, model_path=model_path)
    
    # Generate text
    if model_type == "code":
        generated_text = pipeline.generate_code(seed_text, max_length=max_length, temperature=temperature)
    else:
        generated_text = pipeline.generate_text(seed_text, max_length=max_length, temperature=temperature)
    
    return generated_text

# Add other functions for backward compatibility as needed
def evaluate_model(model_path, test_dataset, model_type="text"):
    """
    Evaluate a trained model on a test dataset.
    
    Args:
        model_path: Path to the trained model
        test_dataset: Test dataset for evaluation
        model_type: Type of model ("text" or "code")
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Create a consolidated generation pipeline
    pipeline = ConsolidatedGenerationPipeline(model_type=model_type, model_path=model_path)
    
    # Evaluate the model
    metrics = pipeline.evaluate(test_dataset)
    
    return metrics