#!/usr/bin/env python3
"""
Fix for the "too many values to unpack (expected 2)" error in model forward pass.

This module provides a direct fix for the tuple unpacking error that occurs when
the model's forward method returns a tuple with more than 2 elements, but the code
expects to unpack it into exactly 2 variables.

Usage:
    from src.generative_ai_module.fix_tuple_unpacking import apply_fix_to_model
    apply_fix_to_model(model)
"""

import logging
import torch
from typing import Any, Dict, Tuple, List, Union, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def apply_fix_to_model(model):
    """
    Apply the tuple unpacking fix to a specific model instance.
    
    Args:
        model: The model to patch
        
    Returns:
        The patched model
    """
    if model is None:
        logger.warning("No model provided to apply_fix_to_model")
        return model
    
    logger.info(f"Applying tuple unpacking fix to model: {type(model).__name__}")
    
    # Check if the model has a forward method
    if hasattr(model, 'forward'):
        # Store the original forward method
        original_forward = model.forward
        
        # Define a patched forward method
        def patched_forward(*args, **kwargs):
            """
            Patched forward method that ensures outputs are always ModelOutput objects.
            """
            # Always set return_dict=True to avoid tuple outputs
            if "return_dict" not in kwargs:
                kwargs["return_dict"] = True
                logger.info("Setting return_dict=True to avoid tuple unpacking issues")
            
            # Call the original forward method
            outputs = original_forward(*args, **kwargs)
            
            # Handle tuple outputs
            if isinstance(outputs, tuple):
                logger.info(f"Got tuple output with {len(outputs)} elements, converting to ModelOutput")
                
                # Import ModelOutput
                try:
                    from transformers.modeling_outputs import ModelOutput
                except ImportError:
                    logger.warning("Could not import ModelOutput, creating a custom version")
                    
                    # Create a simple ModelOutput-like class
                    class ModelOutput(dict):
                        """Simple ModelOutput-like class"""
                        def __init__(self, *args, **kwargs):
                            super().__init__(*args, **kwargs)
                            self.__dict__ = self
                
                # Convert tuple to a dictionary-like object
                outputs_dict = {}
                
                # Check if we have labels in the kwargs to determine if first element is loss
                has_labels = kwargs.get("labels") is not None
                
                if len(outputs) >= 1:
                    # First element is typically the loss or logits
                    if has_labels:
                        # If we have labels, first element is likely the loss
                        outputs_dict["loss"] = outputs[0]
                        if len(outputs) >= 2:
                            # Second element is likely the logits
                            outputs_dict["logits"] = outputs[1]
                    else:
                        # If no labels, first element is likely the logits
                        outputs_dict["logits"] = outputs[0]
                    
                    # Add any remaining elements with generic names
                    for i in range(1, len(outputs)):
                        if i == 1 and "logits" in outputs_dict:
                            continue  # Skip if we already assigned logits
                        outputs_dict[f"hidden_states_{i}"] = outputs[i]
                    
                    # Convert to ModelOutput
                    outputs = ModelOutput(outputs_dict)
                    logger.info(f"Converted tuple to ModelOutput with keys: {list(outputs_dict.keys())}")
            
            return outputs
        
        # Apply the patch
        model.forward = patched_forward
        logger.info(f"✅ Successfully patched {type(model).__name__}.forward")
    else:
        logger.warning(f"⚠️ Model {type(model).__name__} does not have a forward method")
    
    # Also patch compute_loss method if it exists
    if hasattr(model, 'compute_loss'):
        # Store the original compute_loss method
        original_compute_loss = model.compute_loss
        
        # Define a patched compute_loss method
        def patched_compute_loss(*args, **kwargs):
            """
            Patched compute_loss method that handles tuple outputs.
            """
            try:
                # Call the original compute_loss method
                outputs = original_compute_loss(*args, **kwargs)
                
                # If outputs is a tuple, extract the loss
                if isinstance(outputs, tuple):
                    logger.info(f"Got tuple output with {len(outputs)} elements from compute_loss")
                    # First element is typically the loss
                    if len(outputs) >= 1:
                        loss = outputs[0]
                        logger.info(f"Extracted loss from tuple: {loss.item() if hasattr(loss, 'item') else loss}")
                        return loss
                
                return outputs
            except Exception as e:
                logger.error(f"Error in patched compute_loss: {e}")
                # Fall back to original method
                return original_compute_loss(*args, **kwargs)
        
        # Apply the patch
        model.compute_loss = patched_compute_loss
        logger.info(f"✅ Successfully patched {type(model).__name__}.compute_loss")
    
    return model

def fix_model_class():
    """
    Apply the tuple unpacking fix to model classes in the transformers library.
    
    Returns:
        bool: True if the fix was applied successfully, False otherwise.
    """
    try:
        # Try to import PreTrainedModel
        from transformers import PreTrainedModel
        
        # Store the original forward method
        original_forward = PreTrainedModel.forward
        
        # Define a patched forward method
        def patched_forward(self, *args, **kwargs):
            """
            Patched forward method that ensures outputs are always ModelOutput objects.
            """
            # Always set return_dict=True to avoid tuple outputs
            if "return_dict" not in kwargs:
                kwargs["return_dict"] = True
                logger.info("Setting return_dict=True to avoid tuple unpacking issues")
            
            # Call the original forward method
            return original_forward(self, *args, **kwargs)
        
        # Apply the patch
        PreTrainedModel.forward = patched_forward
        logger.info("✅ Successfully patched PreTrainedModel.forward")
        
        # Try to import DeepSeekModel
        try:
            from transformers.models.deepseek.modeling_deepseek import DeepSeekModel
            
            # Store the original forward method
            original_deepseek_forward = DeepSeekModel.forward
            
            # Define a patched forward method
            def patched_deepseek_forward(self, *args, **kwargs):
                """
                Patched forward method for DeepSeekModel.
                """
                # Always set return_dict=True to avoid tuple outputs
                if "return_dict" not in kwargs:
                    kwargs["return_dict"] = True
                    logger.info("Setting return_dict=True for DeepSeekModel")
                
                # Call the original forward method
                return original_deepseek_forward(self, *args, **kwargs)
            
            # Apply the patch
            DeepSeekModel.forward = patched_deepseek_forward
            logger.info("✅ Successfully patched DeepSeekModel.forward")
        except ImportError:
            logger.warning("⚠️ Could not import DeepSeekModel")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to apply model class fix: {e}")
        return False

if __name__ == "__main__":
    logger.info("Applying tuple unpacking fix to model classes")
    success = fix_model_class()
    
    if success:
        logger.info("✅ Successfully applied tuple unpacking fix to model classes")
    else:
        logger.error("❌ Failed to apply tuple unpacking fix to model classes")
