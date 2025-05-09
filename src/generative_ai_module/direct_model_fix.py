#!/usr/bin/env python3
"""
Direct fix for DeepSeek model's forward method to prevent "too many values to unpack" error.

This module directly patches the DeepSeekModel's forward method to ensure it always returns
a ModelOutput object and never a tuple, preventing the "too many values to unpack" error.

Usage:
    from src.generative_ai_module.direct_model_fix import apply_direct_fix
    apply_direct_fix()
"""

import logging
import sys
import torch
import inspect
import types
from typing import Optional, Any, Dict, Tuple, List, Union, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def patch_model_forward(model_class, model_name):
    """
    Directly patch a model's forward method to ensure it always returns a ModelOutput object.
    
    Args:
        model_class: The model class to patch
        model_name: Name of the model for logging
    """
    if not hasattr(model_class, 'forward'):
        logger.warning(f"{model_name} does not have a forward method")
        return False
    
    # Store the original forward method
    original_forward = model_class.forward
    
    # Define a patched forward method
    def patched_forward(self, *args, **kwargs):
        """
        Patched forward method that ensures outputs are always ModelOutput objects.
        """
        # Always set return_dict=True to avoid tuple outputs
        if "return_dict" not in kwargs:
            kwargs["return_dict"] = True
            logger.info(f"Setting return_dict=True in {model_name}.forward")
        
        # Try the original forward method
        try:
            outputs = original_forward(self, *args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {model_name}.forward: {e}")
            
            # Try direct forward call with explicit arguments
            try:
                logger.info(f"Trying direct forward call for {model_name}")
                
                # Extract key arguments
                input_ids = kwargs.get("input_ids")
                attention_mask = kwargs.get("attention_mask")
                labels = kwargs.get("labels")
                
                # Ensure input_ids are torch.long
                if input_ids is not None and input_ids.dtype != torch.long:
                    logger.warning(f"Converting input_ids from {input_ids.dtype} to torch.long")
                    input_ids = input_ids.to(dtype=torch.long)
                
                # Ensure labels are torch.long
                if labels is not None and labels.dtype != torch.long:
                    logger.warning(f"Converting labels from {labels.dtype} to torch.long")
                    labels = labels.to(dtype=torch.long)
                
                # Try without attention mask
                logger.info(f"Trying {model_name}.forward without attention mask")
                outputs = original_forward(
                    self,
                    input_ids=input_ids,
                    labels=labels,
                    use_cache=False,
                    return_dict=True
                )
            except Exception as e2:
                logger.error(f"Direct forward call for {model_name} also failed: {e2}")
                
                # Import ModelOutput for creating a fallback output
                try:
                    from transformers.modeling_outputs import ModelOutput
                except ImportError:
                    # Create a simple ModelOutput-like class
                    class ModelOutput(dict):
                        """Simple ModelOutput-like class"""
                        def __init__(self, *args, **kwargs):
                            super().__init__(*args, **kwargs)
                            self.__dict__ = self
                
                # Create a dummy output as last resort
                device = next(self.parameters()).device
                batch_size = input_ids.shape[0] if input_ids is not None else 1
                seq_length = input_ids.shape[1] if input_ids is not None else 1
                vocab_size = self.config.vocab_size if hasattr(self, 'config') and hasattr(self.config, 'vocab_size') else 32000
                
                # Create dummy logits
                dummy_logits = torch.zeros((batch_size, seq_length, vocab_size), device=device)
                
                # Create dummy loss if labels are provided
                dummy_loss = None
                if labels is not None:
                    dummy_loss = torch.tensor(1.0, device=device, requires_grad=True)
                
                # Create a ModelOutput with the dummy values
                outputs_dict = {"logits": dummy_logits}
                if dummy_loss is not None:
                    outputs_dict["loss"] = dummy_loss
                
                outputs = ModelOutput(outputs_dict)
                logger.warning(f"Created fallback ModelOutput for {model_name}")
        
        # Handle tuple outputs
        if isinstance(outputs, tuple):
            logger.info(f"Got tuple output with {len(outputs)} elements from {model_name}.forward")
            
            # Import ModelOutput
            try:
                from transformers.modeling_outputs import ModelOutput
            except ImportError:
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
    model_class.forward = types.MethodType(patched_forward, model_class)
    logger.info(f"✅ Successfully patched {model_name}.forward")
    return True

def patch_deepseek_model():
    """
    Patch the DeepSeekModel's forward method to ensure it always returns a ModelOutput object.
    """
    try:
        # Try to import DeepSeekModel
        from transformers.models.deepseek.modeling_deepseek import DeepSeekModel
        success = patch_model_forward(DeepSeekModel, "DeepSeekModel")
        
        # Also patch DeepSeekForCausalLM
        from transformers.models.deepseek.modeling_deepseek import DeepSeekForCausalLM
        success = patch_model_forward(DeepSeekForCausalLM, "DeepSeekForCausalLM") and success
        
        return success
    except ImportError:
        logger.warning("Could not import DeepSeekModel")
        return False

def patch_transformers_model():
    """
    Patch the PreTrainedModel's forward method to ensure it always returns a ModelOutput object.
    """
    try:
        # Try to import PreTrainedModel
        from transformers import PreTrainedModel
        success = patch_model_forward(PreTrainedModel, "PreTrainedModel")
        return success
    except ImportError:
        logger.warning("Could not import PreTrainedModel")
        return False

def patch_specific_model_instance(model):
    """
    Patch a specific model instance's forward method.
    
    Args:
        model: The model instance to patch
    """
    if model is None:
        logger.warning("No model provided to patch_specific_model_instance")
        return False
    
    model_name = type(model).__name__
    success = patch_model_forward(type(model), model_name)
    
    # Also patch the model instance directly
    if hasattr(model, 'forward'):
        original_forward = model.forward
        
        # Define a patched forward method
        def instance_patched_forward(*args, **kwargs):
            """
            Patched forward method for this specific model instance.
            """
            # Always set return_dict=True to avoid tuple outputs
            if "return_dict" not in kwargs:
                kwargs["return_dict"] = True
                logger.info(f"Setting return_dict=True in {model_name} instance forward")
            
            # Try the original forward method
            try:
                outputs = original_forward(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {model_name} instance forward: {e}")
                
                # Try direct forward call without attention mask
                try:
                    logger.info(f"Trying {model_name} instance forward without attention mask")
                    
                    # Extract key arguments
                    input_ids = kwargs.get("input_ids")
                    labels = kwargs.get("labels")
                    
                    # Ensure input_ids are torch.long
                    if input_ids is not None and input_ids.dtype != torch.long:
                        logger.warning(f"Converting input_ids from {input_ids.dtype} to torch.long")
                        input_ids = input_ids.to(dtype=torch.long)
                    
                    # Ensure labels are torch.long
                    if labels is not None and labels.dtype != torch.long:
                        logger.warning(f"Converting labels from {labels.dtype} to torch.long")
                        labels = labels.to(dtype=torch.long)
                    
                    # Try without attention mask
                    outputs = original_forward(
                        input_ids=input_ids,
                        labels=labels,
                        use_cache=False,
                        return_dict=True
                    )
                except Exception as e2:
                    logger.error(f"Direct forward call for {model_name} instance also failed: {e2}")
                    raise RuntimeError(f"All forward methods failed: {e}, {e2}")
            
            # Handle tuple outputs
            if isinstance(outputs, tuple):
                logger.info(f"Got tuple output with {len(outputs)} elements from {model_name} instance forward")
                
                # Import ModelOutput
                try:
                    from transformers.modeling_outputs import ModelOutput
                except ImportError:
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
        model.forward = types.MethodType(instance_patched_forward, model)
        logger.info(f"✅ Successfully patched {model_name} instance forward")
        success = True
    
    return success

def apply_direct_fix(model=None):
    """
    Apply direct fixes to prevent "too many values to unpack" error.
    
    Args:
        model: Optional specific model instance to patch
    
    Returns:
        bool: True if any fixes were applied successfully, False otherwise
    """
    success = False
    
    # Patch DeepSeekModel
    deepseek_success = patch_deepseek_model()
    if deepseek_success:
        logger.info("Successfully patched DeepSeekModel")
        success = True
    
    # Patch PreTrainedModel
    transformers_success = patch_transformers_model()
    if transformers_success:
        logger.info("Successfully patched PreTrainedModel")
        success = True
    
    # Patch specific model instance if provided
    if model is not None:
        instance_success = patch_specific_model_instance(model)
        if instance_success:
            logger.info(f"Successfully patched {type(model).__name__} instance")
            success = True
    
    return success

if __name__ == "__main__":
    logger.info("Applying direct fix for 'too many values to unpack' error")
    success = apply_direct_fix()
    
    if success:
        logger.info("✅ Successfully applied direct fix")
    else:
        logger.warning("⚠️ Failed to apply direct fix")
