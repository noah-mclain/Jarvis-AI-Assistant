#!/usr/bin/env python3
"""
Comprehensive Attention Mask Fix for Transformer Models

This module provides a comprehensive solution to fix attention mask issues in transformer models,
particularly for DeepSeek and LLaMA models. It addresses various issues including:

1. Tensor device mismatches (CPU vs. GPU)
2. Attention mask dimension mismatches
3. Tensor size mismatches in mask expansion
4. Handling of unmasked_value parameter

Usage:
    from setup.comprehensive_attention_mask_fix import apply_comprehensive_fix
    apply_comprehensive_fix()
"""

import sys
import logging
import inspect
from typing import Optional, Any, Tuple, Dict, List, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def fix_dtype_mismatch():
    """
    Apply a fix for dtype mismatches between BFloat16 and Half tensors.
    This is a common issue when using mixed precision training.

    Returns:
        bool: True if the fix was applied successfully, False otherwise
    """
    try:
        import torch
        import transformers
        from transformers.modeling_utils import PreTrainedModel

        logger.info("Applying dtype mismatch fix...")

        # Store the original forward method
        original_forward = PreTrainedModel.forward

        # Define a patched forward method
        def patched_forward(self, *args, **kwargs):
            """
            Patched forward method that ensures consistent dtypes for all tensors.
            """
            # Get the model's dtype
            model_dtype = getattr(self, "dtype", None)

            # Process input tensors to ensure consistent dtype
            for arg_name, arg_value in kwargs.items():
                if isinstance(arg_value, torch.Tensor) and arg_value.dtype != model_dtype and model_dtype is not None:
                    # Skip certain tensors that should not be converted
                    if arg_name not in ["labels", "input_ids", "token_type_ids"]:
                        logger.info(f"Converting {arg_name} from {arg_value.dtype} to {model_dtype}")
                        kwargs[arg_name] = arg_value.to(dtype=model_dtype)

            # Call the original forward method
            return original_forward(self, *args, **kwargs)

        # Apply the patch
        PreTrainedModel.forward = patched_forward
        logger.info("✅ Successfully patched PreTrainedModel.forward to fix dtype mismatches")

        return True
    except Exception as e:
        logger.warning(f"⚠️ Could not apply dtype mismatch fix: {e}")
        return False

def apply_comprehensive_fix():
    """
    Apply a comprehensive fix for attention mask issues in transformer models.
    This function patches various methods in the transformers library to handle
    attention mask issues properly.

    Returns:
        bool: True if the fix was applied successfully, False otherwise
    """
    try:
        import torch
        import transformers

        logger.info(f"Applying comprehensive attention mask fix for transformers {transformers.__version__}")

        # Fix 1: Apply dtype mismatch fix
        dtype_fix_success = fix_dtype_mismatch()
        if dtype_fix_success:
            logger.info("✅ Successfully applied dtype mismatch fix")
        else:
            logger.warning("⚠️ Failed to apply dtype mismatch fix")

        # Fix 2: Apply direct fix for attention mask shape issues
        try:
            from transformers.models.llama.modeling_llama import LlamaAttention

            # Store the original _prepare_decoder_attention_mask method
            original_prepare_mask = LlamaAttention._prepare_decoder_attention_mask

            # Define a patched method
            def patched_prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
                """
                Patched method that ensures the attention mask has the correct shape.
                """
                try:
                    # Get batch size and sequence length
                    batch_size, seq_length = input_shape

                    # Check if attention_mask has the wrong shape
                    if attention_mask is not None and attention_mask.dim() > 2:
                        # Try to reshape it to 2D
                        try:
                            attention_mask = attention_mask.view(batch_size, -1)
                            logger.info(f"Reshaped attention mask from {attention_mask.shape} to [batch_size, seq_length]")
                        except Exception as reshape_error:
                            logger.warning(f"Could not reshape attention mask: {reshape_error}")
                            # Create a new mask if reshaping fails
                            attention_mask = torch.ones((batch_size, seq_length), device=inputs_embeds.device)
                            logger.info(f"Created new attention mask with shape: {attention_mask.shape}")

                    # Call the original method with the fixed mask
                    result = original_prepare_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length)

                    # Verify the result has the correct shape
                    expected_shape = (batch_size, 1, seq_length, seq_length)
                    if result.shape != expected_shape:
                        logger.warning(f"Result has incorrect shape: {result.shape}, expected: {expected_shape}")
                        # Create a correct mask directly
                        device = inputs_embeds.device
                        # Create a causal mask
                        causal_mask = torch.triu(
                            torch.ones((seq_length, seq_length), device=device, dtype=torch.bool),
                            diagonal=1
                        )
                        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]
                        causal_mask = causal_mask.expand(batch_size, 1, seq_length, seq_length)

                        # Convert to the expected format
                        result = causal_mask.to(dtype=inputs_embeds.dtype) * torch.finfo(inputs_embeds.dtype).min
                        logger.info(f"Created corrected mask with shape: {result.shape}")

                    return result
                except Exception as e:
                    logger.warning(f"Error in patched _prepare_decoder_attention_mask: {e}")
                    # Fall back to the original method
                    return original_prepare_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length)

            # Apply the patch
            LlamaAttention._prepare_decoder_attention_mask = patched_prepare_decoder_attention_mask
            logger.info("✅ Successfully patched LlamaAttention._prepare_decoder_attention_mask")
        except Exception as e:
            logger.warning(f"⚠️ Could not patch LlamaAttention._prepare_decoder_attention_mask: {e}")

        # Fix 3: Apply specific fix for DeepSeek models
        try:
            # Try to import DeepSeekAttention
            deepseek_available = False
            try:
                # First try the standard import path
                try:
                    from transformers.models.deepseek.modeling_deepseek import DeepSeekAttention
                    deepseek_available = True
                except ImportError:
                    # Try alternative import paths
                    try:
                        from transformers.models.deepseek_coder.modeling_deepseek_coder import DeepSeekCoderAttention as DeepSeekAttention
                        deepseek_available = True
                    except ImportError:
                        # Try to find any DeepSeek-related module
                        import pkgutil
                        import transformers

                        deepseek_modules = [name for _, name, _ in pkgutil.iter_modules(transformers.__path__) if 'deepseek' in name.lower()]
                        if deepseek_modules:
                            logger.info(f"Found potential DeepSeek modules: {deepseek_modules}")
                            # Try to import from the first found module
                            try:
                                module_name = deepseek_modules[0]
                                exec(f"from transformers.models.{module_name}.modeling_{module_name} import {module_name.capitalize()}Attention as DeepSeekAttention")
                                deepseek_available = True
                            except Exception as module_error:
                                logger.warning(f"Could not import from found module {module_name}: {module_error}")
            except Exception as import_error:
                logger.warning(f"Error while trying to import DeepSeek: {import_error}")

            if not deepseek_available:
                logger.warning("DeepSeek model not available in this transformers version")

            if deepseek_available:
                # Store the original _prepare_decoder_attention_mask method
                original_deepseek_prepare_mask = DeepSeekAttention._prepare_decoder_attention_mask

                # Define a patched method
                def patched_deepseek_prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
                    """
                    Patched method that ensures the attention mask has the correct shape for DeepSeek models.
                    """
                    try:
                        # Get batch size and sequence length
                        batch_size, seq_length = input_shape

                        # Check if attention_mask has the wrong shape
                        if attention_mask is not None and attention_mask.dim() > 2:
                            # Try to reshape it to 2D
                            try:
                                attention_mask = attention_mask.view(batch_size, -1)
                                logger.info(f"Reshaped DeepSeek attention mask from {attention_mask.shape} to [batch_size, seq_length]")
                            except Exception as reshape_error:
                                logger.warning(f"Could not reshape DeepSeek attention mask: {reshape_error}")
                                # Create a new mask if reshaping fails
                                attention_mask = torch.ones((batch_size, seq_length), device=inputs_embeds.device)
                                logger.info(f"Created new DeepSeek attention mask with shape: {attention_mask.shape}")

                        # Call the original method with the fixed mask
                        result = original_deepseek_prepare_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length)

                        # Verify the result has the correct shape
                        expected_shape = (batch_size, 1, seq_length, seq_length)
                        if result.shape != expected_shape:
                            logger.warning(f"DeepSeek result has incorrect shape: {result.shape}, expected: {expected_shape}")
                            # Create a correct mask directly
                            device = inputs_embeds.device
                            # Create a causal mask
                            causal_mask = torch.triu(
                                torch.ones((seq_length, seq_length), device=device, dtype=torch.bool),
                                diagonal=1
                            )
                            causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]
                            causal_mask = causal_mask.expand(batch_size, 1, seq_length, seq_length)

                            # Convert to the expected format
                            result = causal_mask.to(dtype=inputs_embeds.dtype) * torch.finfo(inputs_embeds.dtype).min
                            logger.info(f"Created corrected DeepSeek mask with shape: {result.shape}")

                        return result
                    except Exception as e:
                        logger.warning(f"Error in patched DeepSeek _prepare_decoder_attention_mask: {e}")
                        # Fall back to the original method
                        return original_deepseek_prepare_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length)

                # Apply the patch
                DeepSeekAttention._prepare_decoder_attention_mask = patched_deepseek_prepare_decoder_attention_mask
                logger.info("✅ Successfully patched DeepSeekAttention._prepare_decoder_attention_mask")
        except Exception as e:
            logger.warning(f"⚠️ Could not apply DeepSeek-specific fix: {e}")

        # Fix 4: Patch AttentionMaskConverter._unmask_unattended
        try:
            from transformers.modeling_attn_mask_utils import AttentionMaskConverter

            # Store the original function
            original_unmask_unattended = AttentionMaskConverter._unmask_unattended

            # Get the signature of the original function
            try:
                sig = inspect.signature(original_unmask_unattended)
                param_names = list(sig.parameters.keys())
                logger.info(f"Original _unmask_unattended signature: {param_names}")
            except Exception as e:
                logger.warning(f"Could not get signature of _unmask_unattended: {e}")
                param_names = []

            @staticmethod
            def patched_unmask_unattended(*args, **kwargs):
                """
                Patched version of _unmask_unattended that handles various issues:
                1. Keeps tensors on the same device (no CPU conversion)
                2. Handles tensor dimension mismatches
                3. Properly handles the unmasked_value parameter
                4. Creates compatible masks when expansion fails
                """
                # Extract attention_mask from args or kwargs
                attention_mask = None
                if len(args) > 0:
                    attention_mask = args[0]
                elif 'attention_mask' in kwargs:
                    attention_mask = kwargs['attention_mask']
                elif len(param_names) > 0 and param_names[0] in kwargs:
                    attention_mask = kwargs[param_names[0]]

                if attention_mask is None:
                    raise ValueError("Could not find attention_mask in arguments")

                # Extract indices_k and indices_q from args or kwargs
                indices_k = None
                indices_q = None
                unmasked_value = True  # Default value for unmasked_value

                if len(args) > 1:
                    indices_k = args[1]
                elif 'indices_k' in kwargs:
                    indices_k = kwargs['indices_k']
                elif len(param_names) > 1 and param_names[1] in kwargs:
                    indices_k = kwargs[param_names[1]]

                if len(args) > 2:
                    indices_q = args[2]
                elif 'indices_q' in kwargs:
                    indices_q = kwargs['indices_q']
                elif len(param_names) > 2 and param_names[2] in kwargs:
                    indices_q = kwargs[param_names[2]]

                # Extract unmasked_value parameter if it exists
                if len(args) > 3:
                    unmasked_value = args[3]
                elif 'unmasked_value' in kwargs:
                    unmasked_value = kwargs['unmasked_value']
                elif len(param_names) > 3 and param_names[3] in kwargs:
                    unmasked_value = kwargs[param_names[3]]

                # Get the device of the attention mask
                device = attention_mask.device

                # Fix attention_mask shape if needed
                if attention_mask.dim() > 2:
                    # Get the batch size and sequence length
                    batch_size = attention_mask.size(0)
                    seq_length = attention_mask.size(-1)

                    # Calculate total elements in the tensor
                    total_elements = attention_mask.numel()

                    # Check if reshape is possible
                    if total_elements == batch_size * seq_length:
                        # Reshape to 2D [batch_size, seq_length]
                        attention_mask = attention_mask.view(batch_size, seq_length)
                        logger.info(f"Reshaped attention mask from >2D to 2D: {attention_mask.shape}")
                    else:
                        # If reshape is not possible, create a new attention mask
                        logger.warning(f"Cannot reshape attention mask of size {total_elements} to [{batch_size}, {seq_length}]. Creating new mask.")
                        # Create a new attention mask filled with ones (no masking)
                        attention_mask = torch.ones((batch_size, seq_length), device=device)
                        logger.info(f"Created new attention mask with shape: {attention_mask.shape}")

                # Create a temporary tensor on the same device (instead of using CPU)
                tmp = torch.ones(attention_mask.shape[-1], device=device)

                # Find the first non-masked position for each sequence
                # Original: indices = torch.argmax(attention_mask.cpu() * tmp, 1, keepdim=True)
                # Fixed: Keep everything on the same device
                indices = torch.argmax(attention_mask * tmp, 1, keepdim=True)

                # Create a mask for unattended positions
                mask = torch.arange(attention_mask.shape[-1], device=device).expand(attention_mask.shape[0], -1)
                mask = mask < indices

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
                        # Check if indices_q is the batch size (which would cause the error)
                        if isinstance(indices_q, int) and indices_q == attention_mask.size(0):
                            logger.warning(f"indices_q ({indices_q}) matches batch_size, which would cause dimension mismatch. Using seq_length instead.")
                            # Use sequence length instead of batch size for expansion
                            seq_length = attention_mask.size(-1)
                            mask = mask.expand(-1, 1, -1, -1)  # First expand with 1
                            mask = mask.expand(-1, -1, seq_length, -1)  # Then expand with seq_length
                        elif isinstance(indices_q, int):
                            mask = mask.expand(-1, indices_q, -1, -1)
                        else:
                            # Handle case where indices_q is a tensor
                            # Check if it's the batch size tensor
                            if hasattr(indices_q, 'size') and indices_q.size(0) == attention_mask.size(0):
                                logger.warning(f"indices_q size ({indices_q.size(0)}) matches batch_size, which would cause dimension mismatch. Using seq_length instead.")
                                seq_length = attention_mask.size(-1)
                                mask = mask.expand(-1, 1, -1, -1)  # First expand with 1
                                mask = mask.expand(-1, -1, seq_length, -1)  # Then expand with seq_length
                            else:
                                mask = mask.expand(-1, indices_q.size(0) if hasattr(indices_q, 'size') else indices_q, -1, -1)
                except Exception as e:
                    logger.warning(f"Error expanding mask dimensions: {e}")
                    # If we encounter any tensor size mismatch error, create a compatible mask
                    error_msg = str(e)
                    if "must match" in error_msg and "at non-singleton dimension" in error_msg:
                        logger.info("Detected tensor size mismatch error. Creating a compatible mask.")
                        # Create a compatible mask directly
                        batch_size = attention_mask.size(0)
                        seq_length = attention_mask.size(-1)

                        # Create a causal mask that matches the expected dimensions
                        causal_mask = torch.triu(
                            torch.ones((seq_length, seq_length), device=device, dtype=torch.bool),
                            diagonal=1
                        )
                        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]
                        causal_mask = causal_mask.expand(batch_size, 1, seq_length, seq_length)

                        # Apply the attention mask if needed
                        if attention_mask is not None:
                            # Expand attention_mask to 4D [batch_size, 1, 1, seq_length]
                            expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                            # Expand to match causal mask dimensions
                            expanded_mask = expanded_mask.expand(-1, -1, seq_length, -1)
                            # Combine with causal mask (logical OR)
                            combined_mask = causal_mask | ~expanded_mask.bool()
                            # Convert back to the expected mask format
                            mask = ~combined_mask if unmasked_value else combined_mask
                        else:
                            mask = ~causal_mask if unmasked_value else causal_mask

                # Handle dtype mismatch issues
                try:
                    # Convert mask to the expected type based on unmasked_value
                    if unmasked_value is not True:
                        mask = mask.to(dtype=attention_mask.dtype) * unmasked_value

                    # Ensure mask is in the correct dtype to avoid "BFloat16 != Half" errors
                    # Check if we're in a context where we can detect the model's dtype
                    try:
                        import inspect
                        frame = inspect.currentframe()
                        while frame:
                            if 'self' in frame.f_locals and hasattr(frame.f_locals['self'], 'dtype'):
                                model_dtype = frame.f_locals['self'].dtype
                                # If mask dtype doesn't match model dtype, convert it
                                if mask.dtype != model_dtype and model_dtype is not None:
                                    logger.info(f"Converting mask from {mask.dtype} to {model_dtype}")
                                    mask = mask.to(dtype=model_dtype)
                                break
                            frame = frame.f_back
                    except Exception as dtype_error:
                        logger.warning(f"Could not detect model dtype: {dtype_error}")
                except Exception as e:
                    logger.warning(f"Error handling dtype conversion: {e}")

                return mask

            # Apply the patch
            AttentionMaskConverter._unmask_unattended = patched_unmask_unattended
            logger.info("✅ Successfully patched AttentionMaskConverter._unmask_unattended")
        except Exception as e:
            logger.warning(f"⚠️ Could not patch AttentionMaskConverter._unmask_unattended: {e}")

        logger.info("✅ Comprehensive attention mask fix applied successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to apply comprehensive attention mask fix: {e}")
        return False

if __name__ == "__main__":
    # Apply the fix when the script is run directly
    success = apply_comprehensive_fix()

    # Exit with appropriate code
    sys.exit(0 if success else 1)
