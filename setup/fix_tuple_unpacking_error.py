#!/usr/bin/env python3
"""
Fix for the "too many values to unpack (expected 2)" error in transformers models.

This script applies patches to fix the tuple unpacking error that occurs when
model outputs are returned as tuples with more than 2 elements, but the code
expects to unpack them into exactly 2 variables.

The fix ensures that model outputs are always returned as ModelOutput objects
with a dictionary-like interface, rather than tuples.
"""

import os
import sys
import logging
import importlib
from typing import Optional, Any, Dict, Tuple, List, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def fix_tuple_unpacking_error(model=None):
    """
    Apply fixes for the "too many values to unpack (expected 2)" error.

    This function patches the model's forward method to ensure it always returns
    a ModelOutput object with a dictionary-like interface, rather than a tuple.

    Args:
        model: The model to patch. If None, tries to patch the transformers library.

    Returns:
        bool: True if the fix was applied successfully, False otherwise.
    """
    import torch

    success = False

    # If a specific model is provided, patch it directly
    if model is not None:
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
            success = True
        else:
            logger.warning(f"⚠️ Model {type(model).__name__} does not have a forward method")

    # Try to patch common model classes in transformers
    try:
        # Try to patch PreTrainedModel first (this will affect all models)
        try:
            from transformers import PreTrainedModel

            # Store the original forward method
            original_pretrained_forward = PreTrainedModel.forward

            # Define a patched forward method
            def patched_pretrained_forward(self, *args, **kwargs):
                """
                Patched forward method that ensures outputs are always ModelOutput objects.
                """
                # Always set return_dict=True to avoid tuple outputs
                if "return_dict" not in kwargs:
                    kwargs["return_dict"] = True
                    logger.info("Setting return_dict=True to avoid tuple unpacking issues")

                # Call the original forward method
                outputs = original_pretrained_forward(self, *args, **kwargs)

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
            PreTrainedModel.forward = patched_pretrained_forward
            logger.info("✅ Successfully patched PreTrainedModel.forward")
            success = True
        except Exception as e:
            logger.warning(f"⚠️ Could not patch PreTrainedModel.forward: {e}")

        # Try to patch DeepSeek model specifically
        try:
            deepseek_module = importlib.import_module("transformers.models.deepseek.modeling_deepseek")
            DeepSeekModel = getattr(deepseek_module, "DeepSeekModel")

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
            success = True
        except Exception as e:
            logger.warning(f"⚠️ Could not patch DeepSeekModel.forward: {e}")
    except Exception as e:
        logger.error(f"❌ Failed to apply model class fix: {e}")

    return success

if __name__ == "__main__":
    logger.info("Applying fix for 'too many values to unpack (expected 2)' error")
    success = fix_tuple_unpacking_error()

    if success:
        logger.info("✅ Successfully applied tuple unpacking fix")
    else:
        logger.warning("⚠️ Could not apply tuple unpacking fix")
