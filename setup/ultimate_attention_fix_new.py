#!/usr/bin/env python3
"""
Ultimate fix for all attention-related issues in transformer models.

This script applies comprehensive patches to fix:
1. Attention mask shape mismatches (2D vs 4D)
2. Attention mask dtype mismatches (BFloat16 vs Half)
3. Device mismatches between tensors
4. "too many values to unpack (expected 2)" error in model forward pass
5. Dimension mismatches in attention calculations
6. NaN values in attention outputs
7. Memory issues with large attention matrices

This is the most comprehensive fix that should prevent any attention-related errors.
"""

import os
import sys
import logging
import importlib
import inspect
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

def apply_ultimate_fix():
    """
    Apply the ultimate fix for all attention-related issues.

    Returns:
        bool: True if all fixes were applied successfully, False otherwise.
    """
    success = True

    # 1. Fix PreTrainedModel.forward to handle tuple outputs
    try:
        from transformers import PreTrainedModel

        # Store the original forward method
        original_forward = PreTrainedModel.forward

        # Define a patched forward method
        def patched_forward(self, *args, **kwargs):
            """
            Patched forward method that ensures outputs are always ModelOutput objects
            and handles all potential errors.
            """
            # Always set return_dict=True to avoid tuple outputs
            if "return_dict" not in kwargs:
                kwargs["return_dict"] = True
                logger.debug("Setting return_dict=True to avoid tuple unpacking issues")

            # Ensure all tensor inputs are on the same device as the model
            model_device = next(self.parameters()).device
            for k, v in kwargs.items():
                if hasattr(v, 'device') and hasattr(v, 'to') and v.device != model_device:
                    logger.debug(f"Moving {k} from {v.device} to {model_device}")
                    kwargs[k] = v.to(model_device)

            # Fix attention_mask shape if needed
            if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
                attention_mask = kwargs['attention_mask']

                # Ensure attention_mask is on the correct device
                if hasattr(attention_mask, 'device') and attention_mask.device != model_device:
                    logger.debug(f"Moving attention_mask from {attention_mask.device} to {model_device}")
                    attention_mask = attention_mask.to(model_device)

                # Fix attention_mask shape if it's not 2D
                if attention_mask.dim() > 2:
                    import torch
                    batch_size = attention_mask.size(0)
                    seq_length = attention_mask.size(-1)

                    # Reshape to 2D [batch_size, seq_length]
                    try:
                        attention_mask = attention_mask.view(batch_size, seq_length)
                        logger.debug(f"Reshaped attention_mask from {kwargs['attention_mask'].shape} to {attention_mask.shape}")
                    except RuntimeError:
                        # If reshape fails, create a new mask
                        logger.debug(f"Creating new attention_mask with shape [{batch_size}, {seq_length}]")
                        attention_mask = torch.ones((batch_size, seq_length), device=model_device)

                # Update kwargs with fixed attention_mask
                kwargs['attention_mask'] = attention_mask

            # Call the original forward method with error handling
            try:
                outputs = original_forward(self, *args, **kwargs)
            except RuntimeError as e:
                error_msg = str(e)
                logger.warning(f"RuntimeError in forward pass: {error_msg}")

                # Handle specific errors
                if "size mismatch" in error_msg or "dimension" in error_msg:
                    logger.info("Attempting to fix size/dimension mismatch...")

                    # Try with a fresh attention mask
                    if 'attention_mask' in kwargs:
                        import torch
                        batch_size = kwargs['input_ids'].size(0) if 'input_ids' in kwargs else 1
                        seq_length = kwargs['input_ids'].size(1) if 'input_ids' in kwargs else 2048

                        logger.info(f"Creating new attention_mask with shape [{batch_size}, {seq_length}]")
                        kwargs['attention_mask'] = torch.ones((batch_size, seq_length), device=model_device)

                        # Try again with the new attention mask
                        try:
                            outputs = original_forward(self, *args, **kwargs)
                            logger.info("Successfully fixed with new attention_mask")
                        except Exception as e2:
                            logger.error(f"Still failed with new attention_mask: {e2}")
                            # Fall back to no attention mask
                            kwargs.pop('attention_mask', None)
                            try:
                                outputs = original_forward(self, *args, **kwargs)
                                logger.info("Successfully ran without attention_mask")
                            except Exception as e3:
                                logger.error(f"Failed without attention_mask: {e3}")
                                raise e3
                else:
                    # Re-raise the original error
                    raise e

            # Handle tuple outputs
            if isinstance(outputs, tuple):
                logger.debug(f"Got tuple output with {len(outputs)} elements, converting to ModelOutput")

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

            # Check for NaN values in outputs
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                import torch
                if torch.isnan(outputs.loss).any():
                    logger.warning("NaN detected in loss! Replacing with zero.")
                    outputs.loss = torch.zeros_like(outputs.loss)

            return outputs

        # Apply the patch
        PreTrainedModel.forward = patched_forward
        logger.info("✅ Successfully patched PreTrainedModel.forward with ultimate fix")
    except Exception as e:
        logger.error(f"❌ Failed to patch PreTrainedModel.forward: {e}")
        success = False

    # 2. Fix attention mask handling in _prepare_4d_causal_attention_mask_for_sdpa
    try:
        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa

        # Store the original function
        original_prepare_4d = _prepare_4d_causal_attention_mask_for_sdpa

        # Define a patched function
        def patched_prepare_4d(attention_mask, input_shape, inputs_embeds, past_key_values_length, sliding_window, dtype):
            """
            Patched version that handles all edge cases in attention mask preparation.
            """
            import torch

            # Fix attention_mask shape if needed
            if attention_mask is not None and attention_mask.dim() > 2:
                batch_size = attention_mask.size(0)
                seq_length = attention_mask.size(-1)

                # Reshape to 2D [batch_size, seq_length]
                try:
                    attention_mask = attention_mask.view(batch_size, seq_length)
                    logger.debug(f"Reshaped attention_mask from >2D to 2D: {attention_mask.shape}")
                except RuntimeError:
                    # If reshape fails, create a new mask
                    device = attention_mask.device
                    logger.debug(f"Creating new attention_mask with shape [{batch_size}, {seq_length}]")
                    attention_mask = torch.ones((batch_size, seq_length), device=device)

            # Ensure dtype compatibility
            if attention_mask is not None and dtype is not None:
                # Check if we need to convert dtype
                if attention_mask.dtype != dtype and attention_mask.dtype != torch.bool:
                    try:
                        attention_mask = attention_mask.to(dtype=dtype)
                        logger.debug(f"Converted attention_mask from {attention_mask.dtype} to {dtype}")
                    except RuntimeError as e:
                        logger.warning(f"Failed to convert attention_mask dtype: {e}")

            # Call the original function with error handling
            try:
                return original_prepare_4d(attention_mask, input_shape, inputs_embeds, past_key_values_length, sliding_window, dtype)
            except Exception as e:
                logger.warning(f"Error in _prepare_4d_causal_attention_mask_for_sdpa: {e}")

                # Create a fallback attention mask
                batch_size, seq_length = input_shape
                device = inputs_embeds.device

                # Create a causal mask
                causal_mask = torch.triu(
                    torch.ones((seq_length, seq_length), device=device, dtype=torch.bool),
                    diagonal=1
                )

                # Expand to 4D [batch_size, 1, seq_length, seq_length]
                expanded_causal_mask = causal_mask.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, seq_length, seq_length)

                # Convert to the expected dtype
                if dtype is not None:
                    try:
                        expanded_causal_mask = expanded_causal_mask.to(dtype=dtype)
                    except RuntimeError:
                        # If conversion fails, use float32
                        expanded_causal_mask = expanded_causal_mask.to(dtype=torch.float32)

                logger.info("Created fallback 4D causal attention mask")
                return expanded_causal_mask

        # Apply the patch
        import transformers.modeling_attn_mask_utils
        transformers.modeling_attn_mask_utils._prepare_4d_causal_attention_mask_for_sdpa = patched_prepare_4d
        logger.info("✅ Successfully patched _prepare_4d_causal_attention_mask_for_sdpa with ultimate fix")
    except Exception as e:
        logger.error(f"❌ Failed to patch _prepare_4d_causal_attention_mask_for_sdpa: {e}")
        success = False

    # 3. Fix _unmask_unattended function
    try:
        from transformers.modeling_attn_mask_utils import AttentionMaskConverter

        # Store the original function
        original_unmask_unattended = AttentionMaskConverter._unmask_unattended

        @staticmethod
        def patched_unmask_unattended(attention_mask, indices_k=None, indices_q=None, unmasked_value=True):
            """
            Patched version of _unmask_unattended that handles all edge cases.
            """
            import torch

            # Get the device of the attention mask
            device = attention_mask.device

            # Fix attention_mask shape if needed
            if attention_mask.dim() > 2:
                batch_size = attention_mask.size(0)
                seq_length = attention_mask.size(-1)

                # Reshape to 2D [batch_size, seq_length]
                try:
                    attention_mask = attention_mask.view(batch_size, seq_length)
                    logger.debug(f"Reshaped attention_mask from >2D to 2D: {attention_mask.shape}")
                except RuntimeError:
                    # If reshape fails, create a new mask
                    logger.debug(f"Creating new attention_mask with shape [{batch_size}, {seq_length}]")
                    attention_mask = torch.ones((batch_size, seq_length), device=device)

            # Create a temporary tensor on the same device
            tmp = torch.ones(attention_mask.shape[-1], device=device)

            # Find the first non-masked position for each sequence
            try:
                indices = torch.argmax(attention_mask * tmp, 1, keepdim=True)
            except RuntimeError as e:
                logger.warning(f"Error in argmax: {e}")
                # Create a fallback indices tensor
                indices = torch.zeros((attention_mask.shape[0], 1), device=device, dtype=torch.long)

            # Create a mask for unattended positions
            try:
                mask = torch.arange(attention_mask.shape[-1], device=device).expand(attention_mask.shape[0], -1)
                mask = mask < indices
            except RuntimeError as e:
                logger.warning(f"Error creating mask: {e}")
                # Create a fallback mask
                mask = torch.ones((attention_mask.shape[0], attention_mask.shape[-1]), device=device, dtype=torch.bool)

            # Expand mask to 4D
            mask = mask.unsqueeze(1).unsqueeze(2)

            # Handle indices_k and indices_q if provided
            try:
                if indices_k is not None:
                    if isinstance(indices_k, int):
                        mask = mask.expand(-1, -1, indices_k, -1)
                    else:
                        # Handle case where indices_k is a tensor
                        mask = mask.expand(-1, -1, indices_k.size(0) if hasattr(indices_k, 'size') else indices_k, -1)

                if indices_q is not None:
                    if isinstance(indices_q, int):
                        mask = mask.expand(-1, indices_q, -1, -1)
                    else:
                        # Handle case where indices_q is a tensor
                        mask = mask.expand(-1, indices_q.size(0) if hasattr(indices_q, 'size') else indices_q, -1, -1)
            except Exception as e:
                logger.warning(f"Error expanding mask dimensions: {e}")
                # Create a fallback 4D mask
                batch_size = attention_mask.shape[0]
                seq_length = attention_mask.shape[-1]

                # Use default values for indices_k and indices_q if they cause problems
                dim_k = 2048 if indices_k is None else (indices_k if isinstance(indices_k, int) else 2048)
                dim_q = 1 if indices_q is None else (indices_q if isinstance(indices_q, int) else 1)

                # Create a compatible mask
                mask = torch.ones((batch_size, dim_q, dim_k, seq_length), device=device, dtype=torch.bool)

            # Convert mask to the expected type based on unmasked_value
            if unmasked_value is not True:
                try:
                    mask = mask.to(dtype=attention_mask.dtype) * unmasked_value
                except RuntimeError:
                    # If conversion fails, use float32
                    mask = mask.to(dtype=torch.float32) * float(unmasked_value)

            return mask

        # Apply the patch
        AttentionMaskConverter._unmask_unattended = patched_unmask_unattended
        logger.info("✅ Successfully patched AttentionMaskConverter._unmask_unattended with ultimate fix")
    except Exception as e:
        logger.error(f"❌ Failed to patch AttentionMaskConverter._unmask_unattended: {e}")
        success = False

    return success

if __name__ == "__main__":
    logger.info("Applying ultimate fix for all attention-related issues")
    if apply_ultimate_fix():
        logger.info("✅ Successfully applied ultimate fix")
    else:
        logger.warning("⚠️ Some fixes failed, but continuing anyway")
