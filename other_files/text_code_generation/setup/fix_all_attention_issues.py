#!/usr/bin/env python3
"""
Comprehensive fix for all attention mask and tuple unpacking issues in transformers models.

This script applies patches to fix:
1. Attention mask shape mismatches (2D vs 4D)
2. Attention mask dtype mismatches (BFloat16 vs Half)
3. Device mismatches between tensors
4. "too many values to unpack (expected 2)" error in model forward pass

Run this script before training to ensure all issues are fixed.
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

def fix_all_attention_issues():
    """
    Apply all fixes for attention mask and tuple unpacking issues.

    Returns:
        bool: True if all fixes were applied successfully, False otherwise.
    """
    success = True

    # Run all fix scripts
    try:
        # Add the setup directory to sys.path
        setup_dir = os.path.dirname(os.path.abspath(__file__))
        if setup_dir not in sys.path:
            sys.path.insert(0, setup_dir)

        # Import and run fix_transformers_attention_mask
        try:
            from fix_transformers_attention_mask import fix_transformers_attention_mask
            if fix_transformers_attention_mask():
                logger.info("✅ Successfully applied general attention mask fix")
            else:
                logger.warning("⚠️ Failed to apply general attention mask fix")
                success = False
        except ImportError:
            logger.warning("⚠️ Could not import fix_transformers_attention_mask")
            success = False

        # Import and run fix_attention_mask_params
        try:
            from fix_attention_mask_params import fix_attention_mask_params
            if fix_attention_mask_params():
                logger.info("✅ Successfully applied parameter-specific attention mask fix")
            else:
                logger.warning("⚠️ Failed to apply parameter-specific attention mask fix")
                success = False
        except ImportError:
            logger.warning("⚠️ Could not import fix_attention_mask_params")
            success = False

        # Import and run fix_tensor_size_mismatch
        try:
            from fix_tensor_size_mismatch import fix_tensor_size_mismatch
            if fix_tensor_size_mismatch():
                logger.info("✅ Successfully applied tensor size mismatch fix")
            else:
                logger.warning("⚠️ Failed to apply tensor size mismatch fix")
                success = False
        except ImportError:
            logger.warning("⚠️ Could not import fix_tensor_size_mismatch")
            success = False

        # Import and run fix_attention_dimension_mismatch
        try:
            from fix_attention_dimension_mismatch import fix_attention_dimension_mismatch
            if fix_attention_dimension_mismatch():
                logger.info("✅ Successfully applied attention dimension mismatch fix")
            else:
                logger.warning("⚠️ Failed to apply attention dimension mismatch fix")
                success = False
        except ImportError:
            logger.warning("⚠️ Could not import fix_attention_dimension_mismatch")
            success = False

        # Import and run fix_tuple_unpacking_error
        try:
            from fix_tuple_unpacking_error import fix_tuple_unpacking_error
            if fix_tuple_unpacking_error():
                logger.info("✅ Successfully applied tuple unpacking error fix")
            else:
                logger.warning("⚠️ Failed to apply tuple unpacking error fix")
                success = False
        except ImportError:
            logger.warning("⚠️ Could not import fix_tuple_unpacking_error")
            success = False

        # Import and run comprehensive_attention_mask_fix
        try:
            from comprehensive_attention_mask_fix import apply_comprehensive_fix
            if apply_comprehensive_fix():
                logger.info("✅ Successfully applied comprehensive attention mask fix")
            else:
                logger.warning("⚠️ Failed to apply comprehensive attention mask fix")
                success = False
        except ImportError:
            logger.warning("⚠️ Could not import comprehensive_attention_mask_fix")
            success = False
    except Exception as e:
        logger.error(f"❌ Error running fix scripts: {e}")
        success = False

    # Apply direct fixes to transformers library
    try:
        # Fix PreTrainedModel.forward to handle tuple outputs
        try:
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

                # Call the original forward method
                outputs = original_forward(self, *args, **kwargs)

                # Handle tuple outputs
                if isinstance(outputs, tuple):
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

                return outputs

            # Apply the patch
            PreTrainedModel.forward = patched_forward
            logger.info("✅ Successfully patched PreTrainedModel.forward")
        except Exception as e:
            logger.warning(f"⚠️ Could not patch PreTrainedModel.forward: {e}")
            success = False
    except Exception as e:
        logger.error(f"❌ Error applying direct fixes: {e}")
        success = False

    return success

if __name__ == "__main__":
    logger.info("Applying all attention mask and tuple unpacking fixes")
    if fix_all_attention_issues():
        logger.info("✅ Successfully applied all fixes")
    else:
        logger.warning("⚠️ Some fixes failed, but continuing anyway")
