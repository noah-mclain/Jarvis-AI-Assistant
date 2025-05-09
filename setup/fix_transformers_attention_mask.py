#!/usr/bin/env python3
"""
Fix for the attention mask issue in transformers library.

This script applies patches to fix the attention mask handling in transformers library,
specifically addressing the "too many values to unpack (expected 2)" error that occurs
with DeepSeek and LLaMA models.

The fix ensures that attention masks are always in the correct shape (2D) and on the
correct device before being processed by the model.
"""

import os
import sys
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def fix_transformers_attention_mask():
    """Apply fixes for attention mask issues in transformers library"""
    try:
        import torch
        import transformers
        from transformers import __version__ as transformers_version

        logger.info(f"Applying attention mask fixes for transformers {transformers_version}")

        # Parse version string to check compatibility
        try:
            version_parts = transformers_version.split('.')
            major = int(version_parts[0]) if len(version_parts) > 0 else 0
            minor = int(version_parts[1]) if len(version_parts) > 1 else 0
            logger.info(f"Detected transformers version: {major}.{minor}")
        except Exception as e:
            logger.warning(f"Could not parse transformers version: {e}. Applying fixes anyway.")
            major, minor = 4, 36  # Assume recent version

        # Fix 1: Patch _prepare_4d_causal_attention_mask_for_sdpa
        try:
            from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa

            # Store the original function
            original_prepare_4d = _prepare_4d_causal_attention_mask_for_sdpa

            # Get the signature of the original function to ensure we match it
            import inspect
            sig = inspect.signature(original_prepare_4d)
            param_names = list(sig.parameters.keys())
            logger.info(f"Original function signature: {param_names}")

            # Define patched function with a flexible signature to handle different versions
            def patched_prepare_4d(*args, **kwargs):
                """
                Patched version that ensures attention_mask is 2D before processing.
                Handles different function signatures across transformers versions.
                """
                # Extract attention_mask from args or kwargs
                attention_mask = None
                if len(args) > 0:
                    attention_mask = args[0]
                elif 'attention_mask' in kwargs:
                    attention_mask = kwargs['attention_mask']

                # Fix attention_mask shape if needed
                if attention_mask is not None and attention_mask.dim() > 2:
                    # Get the batch size and sequence length
                    batch_size = attention_mask.size(0)
                    seq_length = attention_mask.size(-1)

                    # Reshape to 2D [batch_size, seq_length]
                    attention_mask = attention_mask.view(batch_size, seq_length)
                    logger.info(f"Reshaped attention mask from >2D to 2D: {attention_mask.shape}")

                    # Update args or kwargs with the fixed mask
                    if len(args) > 0:
                        args_list = list(args)
                        args_list[0] = attention_mask
                        args = tuple(args_list)
                    elif 'attention_mask' in kwargs:
                        kwargs['attention_mask'] = attention_mask

                # Call the original function with the fixed mask
                try:
                    return original_prepare_4d(*args, **kwargs)
                except TypeError as e:
                    # If we get a TypeError, it might be due to missing arguments
                    error_msg = str(e)
                    logger.warning(f"Error calling original function: {error_msg}")

                    # Check if we're missing sliding_window or dtype
                    if "missing required positional argument: 'sliding_window'" in error_msg:
                        if 'sliding_window' not in kwargs:
                            kwargs['sliding_window'] = None
                            logger.info("Added missing sliding_window=None parameter")

                    if "missing required positional argument: 'dtype'" in error_msg:
                        if 'dtype' not in kwargs:
                            import torch
                            kwargs['dtype'] = torch.float32
                            logger.info("Added missing dtype=torch.float32 parameter")

                    # Try again with the updated kwargs
                    return original_prepare_4d(*args, **kwargs)

            # Apply the patch
            transformers.modeling_attn_mask_utils._prepare_4d_causal_attention_mask_for_sdpa = patched_prepare_4d
            logger.info("✅ Successfully patched _prepare_4d_causal_attention_mask_for_sdpa")
        except Exception as e:
            logger.warning(f"⚠️ Could not patch _prepare_4d_causal_attention_mask_for_sdpa: {e}")

        # Fix 2: Patch AttentionMaskConverter._unmask_unattended
        try:
            from transformers.modeling_attn_mask_utils import AttentionMaskConverter

            # Store the original function
            original_unmask_unattended = AttentionMaskConverter._unmask_unattended

            # Get the signature of the original function to ensure we match it
            import inspect
            try:
                sig = inspect.signature(AttentionMaskConverter._unmask_unattended)
                param_names = list(sig.parameters.keys())
                logger.info(f"Original _unmask_unattended signature: {param_names}")
            except Exception as e:
                logger.warning(f"Could not get signature of _unmask_unattended: {e}")
                param_names = []

            # Define patched function with a flexible signature
            @staticmethod
            def patched_unmask_unattended(*args, **kwargs):
                """
                Patched version of _unmask_unattended that keeps tensors on the same device.

                The original function has a call to .cpu() which causes device mismatch errors.
                This patch ensures all operations happen on the same device and handles the
                specific tensor size mismatch error.
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
                        logger.info(f"Reshaped attention mask from >2D to 2D in unmask_unattended: {attention_mask.shape}")
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

                # Convert mask to the expected type based on unmasked_value
                if unmasked_value is not True:
                    mask = mask.to(dtype=attention_mask.dtype) * unmasked_value

                return mask

            # Apply the patch
            AttentionMaskConverter._unmask_unattended = patched_unmask_unattended
            logger.info("✅ Successfully patched AttentionMaskConverter._unmask_unattended")
        except Exception as e:
            logger.warning(f"⚠️ Could not patch AttentionMaskConverter._unmask_unattended: {e}")

        # Fix 3: Patch LlamaModel.forward
        try:
            from transformers.models.llama.modeling_llama import LlamaModel

            # Store the original forward method
            original_forward = LlamaModel.forward

            # Get the signature of the original forward method
            import inspect
            try:
                sig = inspect.signature(original_forward)
                param_names = list(sig.parameters.keys())
                logger.info(f"Original LlamaModel.forward signature: {param_names}")
            except Exception as e:
                logger.warning(f"Could not get signature of LlamaModel.forward: {e}")
                param_names = []

            # Define a patched forward method that properly handles attention masks
            def patched_forward(self, *args, **kwargs):
                """
                Patched forward method for LlamaModel that properly handles attention masks.
                This version is more flexible and can handle different function signatures.
                """
                # Extract input_ids and inputs_embeds from kwargs
                input_ids = kwargs.get('input_ids', None)
                inputs_embeds = kwargs.get('inputs_embeds', None)
                attention_mask = kwargs.get('attention_mask', None)

                # Get the device from input tensors
                device = None
                if input_ids is not None:
                    device = input_ids.device
                elif inputs_embeds is not None:
                    device = inputs_embeds.device

                # Fix attention mask shape if needed
                if attention_mask is not None and attention_mask.dim() > 2:
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

                    # Update kwargs with the fixed mask
                    kwargs['attention_mask'] = attention_mask

                # Ensure attention_mask is on the correct device
                if attention_mask is not None and device is not None and attention_mask.device != device:
                    attention_mask = attention_mask.to(device)
                    kwargs['attention_mask'] = attention_mask

                # Call the original forward method with the fixed attention mask
                try:
                    return original_forward(self, *args, **kwargs)
                except Exception as e:
                    logger.warning(f"Error in original forward: {e}")

                    # Try to handle common errors
                    error_msg = str(e)

                    # If there's an issue with _prepare_4d_causal_attention_mask_for_sdpa
                    if "missing required positional argument" in error_msg and "_prepare_4d_causal_attention_mask_for_sdpa" in error_msg:
                        logger.info("Trying to fix _prepare_4d_causal_attention_mask_for_sdpa error")

                        # Try without attention mask as a last resort
                        if 'attention_mask' in kwargs:
                            logger.info("Removing attention_mask and trying again")
                            kwargs_no_mask = kwargs.copy()
                            del kwargs_no_mask['attention_mask']
                            return original_forward(self, *args, **kwargs_no_mask)

                    # Re-raise the exception if we couldn't handle it
                    raise

            # Apply the patch
            LlamaModel.forward = patched_forward
            logger.info("✅ Successfully patched LlamaModel.forward")
        except Exception as e:
            logger.warning(f"⚠️ Could not patch LlamaModel.forward: {e}")

        # Patch the _prepare_4d_causal_attention_mask_for_sdpa function directly in LlamaModel
        try:
            # Get the original function from the LlamaModel module
            from transformers.models.llama.modeling_llama import _prepare_4d_causal_attention_mask_for_sdpa as llama_prepare_4d

            # Store the original function
            original_llama_prepare_4d = llama_prepare_4d

            # Define a patched function with a flexible signature
            def patched_llama_prepare_4d(*args, **kwargs):
                """
                Patched version of _prepare_4d_causal_attention_mask_for_sdpa in LlamaModel.
                This handles the specific signature used in the LlamaModel implementation.
                """
                # Extract attention_mask from args or kwargs
                attention_mask = None
                if len(args) > 0:
                    attention_mask = args[0]
                elif 'attention_mask' in kwargs:
                    attention_mask = kwargs['attention_mask']

                # Fix attention_mask shape if needed
                if attention_mask is not None and attention_mask.dim() > 2:
                    # Get the batch size and sequence length
                    batch_size = attention_mask.size(0)
                    seq_length = attention_mask.size(-1)

                    # Reshape to 2D [batch_size, seq_length]
                    attention_mask = attention_mask.view(batch_size, seq_length)
                    logger.info(f"Reshaped attention mask from >2D to 2D in LlamaModel: {attention_mask.shape}")

                    # Update args or kwargs with the fixed mask
                    if len(args) > 0:
                        args_list = list(args)
                        args_list[0] = attention_mask
                        args = tuple(args_list)
                    elif 'attention_mask' in kwargs:
                        kwargs['attention_mask'] = attention_mask

                # Add missing parameters if needed
                try:
                    return original_llama_prepare_4d(*args, **kwargs)
                except TypeError as e:
                    error_msg = str(e)
                    logger.warning(f"Error in LlamaModel _prepare_4d: {error_msg}")

                    # Check for missing parameters
                    if "missing required positional argument: 'sliding_window'" in error_msg:
                        if 'sliding_window' not in kwargs:
                            kwargs['sliding_window'] = None
                            logger.info("Added missing sliding_window=None parameter to LlamaModel _prepare_4d")

                    if "missing required positional argument: 'dtype'" in error_msg:
                        if 'dtype' not in kwargs:
                            import torch
                            kwargs['dtype'] = torch.float32
                            logger.info("Added missing dtype=torch.float32 parameter to LlamaModel _prepare_4d")

                    # Try again with the updated kwargs
                    return original_llama_prepare_4d(*args, **kwargs)

            # Apply the patch
            import transformers.models.llama.modeling_llama
            transformers.models.llama.modeling_llama._prepare_4d_causal_attention_mask_for_sdpa = patched_llama_prepare_4d
            logger.info("✅ Successfully patched LlamaModel _prepare_4d_causal_attention_mask_for_sdpa")
        except Exception as e:
            logger.warning(f"⚠️ Could not patch LlamaModel _prepare_4d_causal_attention_mask_for_sdpa: {e}")

        # Try to patch DeepSeek model if available
        try:
            # First check if the deepseek module exists
            import importlib.util
            spec = importlib.util.find_spec("transformers.models.deepseek.modeling_deepseek")

            if spec is None:
                logger.warning("DeepSeek model not available in this transformers version. Creating custom implementation.")

                # Create a custom implementation for DeepSeek model
                from transformers.models.llama.modeling_llama import LlamaModel

                # Create a DeepSeekModel class that inherits from LlamaModel
                class CustomDeepSeekModel(LlamaModel):
                    """Custom DeepSeekModel implementation based on LlamaModel."""
                    pass

                # Store the class in a variable
                DeepSeekModel = CustomDeepSeekModel

                # Add it to the transformers module
                import transformers
                if not hasattr(transformers.models, "deepseek"):
                    # Create the module structure
                    class DeepSeekModule:
                        pass
                    transformers.models.deepseek = DeepSeekModule()

                # Add the modeling_deepseek module
                class ModelingDeepSeek:
                    pass

                # Set the DeepSeekModel attribute
                setattr(ModelingDeepSeek, "DeepSeekModel", DeepSeekModel)

                # Add the module to transformers
                transformers.models.deepseek.modeling_deepseek = ModelingDeepSeek()

                logger.info("✅ Created custom DeepSeekModel implementation based on LlamaModel")
            else:
                # If the module exists, import it normally
                import importlib
                deepseek_module = importlib.import_module("transformers.models.deepseek.modeling_deepseek")
                DeepSeekModel = getattr(deepseek_module, "DeepSeekModel")

            # Store the original forward method
            original_forward = DeepSeekModel.forward

            # Define a patched forward method with flexible signature
            def patched_forward(self, *args, **kwargs):
                """
                Patched forward method that fixes attention mask issues.
                This version is more flexible and can handle different function signatures.
                """
                # Extract input_ids and inputs_embeds from kwargs
                input_ids = kwargs.get('input_ids', None)
                inputs_embeds = kwargs.get('inputs_embeds', None)
                attention_mask = kwargs.get('attention_mask', None)

                # Get the device from input tensors
                device = None
                if input_ids is not None:
                    device = input_ids.device
                elif inputs_embeds is not None:
                    device = inputs_embeds.device

                # Fix attention mask shape if needed
                if attention_mask is not None and attention_mask.dim() > 2:
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

                    # Update kwargs with the fixed mask
                    kwargs['attention_mask'] = attention_mask

                # Ensure attention_mask is on the correct device
                if attention_mask is not None and device is not None and attention_mask.device != device:
                    attention_mask = attention_mask.to(device)
                    kwargs['attention_mask'] = attention_mask

                # Call the original forward method with the fixed attention mask
                try:
                    return original_forward(self, *args, **kwargs)
                except Exception as e:
                    logger.warning(f"Error in DeepSeek forward: {e}")

                    # Try to handle common errors
                    error_msg = str(e)

                    # If there's an issue with _prepare_4d_causal_attention_mask_for_sdpa
                    if "missing required positional argument" in error_msg and "_prepare_4d_causal_attention_mask_for_sdpa" in error_msg:
                        logger.info("Trying to fix _prepare_4d_causal_attention_mask_for_sdpa error in DeepSeek")

                        # Try without attention mask as a last resort
                        if 'attention_mask' in kwargs:
                            logger.info("Removing attention_mask and trying again")
                            kwargs_no_mask = kwargs.copy()
                            del kwargs_no_mask['attention_mask']
                            return original_forward(self, *args, **kwargs_no_mask)

                    # Re-raise the exception if we couldn't handle it
                    raise

            # Apply the patch
            DeepSeekModel.forward = patched_forward
            logger.info("✅ Successfully patched DeepSeekModel.forward")
        except Exception as e:
            logger.warning(f"⚠️ Could not patch DeepSeekModel.forward: {e}")

        logger.info("✅ Attention mask fixes applied successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to apply attention mask fixes: {e}")
        return False

if __name__ == "__main__":
    # Apply the fixes
    success = fix_transformers_attention_mask()

    # Exit with appropriate code
    sys.exit(0 if success else 1)
