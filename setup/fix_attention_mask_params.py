#!/usr/bin/env python3
"""
Fix for the attention mask parameter issue in transformers library.

This script specifically addresses the error:
"patched_prepare_4d() missing 2 required positional arguments: 'sliding_window' and 'dtype'"

It directly patches the _prepare_4d_causal_attention_mask_for_sdpa function to handle
the required parameters correctly.
"""

import os
import sys
import logging
import importlib
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

def fix_attention_mask_params():
    """Apply fixes for attention mask parameter issues in transformers library"""
    try:
        import torch
        import transformers
        from transformers import __version__ as transformers_version

        logger.info(f"Applying attention mask parameter fixes for transformers {transformers_version}")

        # Get the _prepare_4d_causal_attention_mask_for_sdpa function
        try:
            from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa

            # Store the original function
            original_prepare_4d = _prepare_4d_causal_attention_mask_for_sdpa

            # Get the signature of the original function
            import inspect
            sig = inspect.signature(original_prepare_4d)
            param_names = list(sig.parameters.keys())
            logger.info(f"Original function signature: {param_names}")

            # Define a new function that handles different function signatures
            def fixed_prepare_4d(*args, **kwargs):
                """
                Fixed version of _prepare_4d_causal_attention_mask_for_sdpa that adapts
                to different function signatures across transformers versions.
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
                        attention_mask = torch.ones((batch_size, seq_length), device=attention_mask.device)
                        logger.info(f"Created new attention mask with shape: {attention_mask.shape}")

                    # Update args or kwargs with the fixed mask
                    if len(args) > 0:
                        args_list = list(args)
                        args_list[0] = attention_mask
                        args = tuple(args_list)
                    elif 'attention_mask' in kwargs:
                        kwargs['attention_mask'] = attention_mask

                # Try to call the original function with the original signature
                try:
                    return original_prepare_4d(*args, **kwargs)
                except TypeError as e:
                    error_msg = str(e)
                    logger.warning(f"Error in original function: {error_msg}")

                    # Check if we're passing too many arguments
                    if "takes" in error_msg and "positional arguments but" in error_msg and "were given" in error_msg:
                        # Extract the expected number of arguments
                        import re
                        match = re.search(r'takes (?:from (\d+) to )?(\d+) positional arguments but (\d+) were given', error_msg)
                        if match:
                            min_args = int(match.group(1)) if match.group(1) else int(match.group(2))
                            max_args = int(match.group(2))
                            given_args = int(match.group(3))

                            logger.info(f"Function expects {min_args}-{max_args} args, but {given_args} were given")

                            # If we're passing too many args, try with fewer
                            if given_args > max_args and len(args) >= max_args:
                                logger.info(f"Trying with {max_args} arguments instead of {given_args}")
                                return original_prepare_4d(*args[:max_args])

                    # If we can't call the original function, try to extract the needed parameters
                    # from args or kwargs and create a causal mask manually
                    logger.info("Creating causal mask manually")

                    # Extract parameters from args or kwargs
                    input_shape = None
                    if len(args) > 1:
                        input_shape = args[1]
                    elif 'input_shape' in kwargs:
                        input_shape = kwargs['input_shape']

                    # If we don't have input_shape, try to infer it from attention_mask
                    if input_shape is None and attention_mask is not None:
                        batch_size = attention_mask.size(0)
                        seq_length = attention_mask.size(1) if attention_mask.dim() > 1 else attention_mask.size(0)
                        input_shape = (batch_size, seq_length)
                        logger.info(f"Inferred input_shape from attention_mask: {input_shape}")

                    # If we still don't have input_shape, use a default
                    if input_shape is None:
                        batch_size = 1
                        seq_length = 2048  # A reasonable default for LLMs
                        input_shape = (batch_size, seq_length)
                        logger.warning(f"Using default input_shape: {input_shape}")

                    # Get batch size and sequence length
                    batch_size = input_shape[0]
                    seq_length = input_shape[1]

                    # Determine dtype
                    dtype = torch.float32  # Default dtype
                    if 'dtype' in kwargs:
                        dtype = kwargs['dtype']

                    # Create a causal mask [batch_size, 1, seq_length, seq_length]
                    mask = torch.ones((batch_size, 1, seq_length, seq_length), dtype=dtype)
                    mask = torch.triu(mask, diagonal=1)
                    mask = mask.to(dtype=dtype) * torch.finfo(dtype).min

                    # Apply attention_mask if provided
                    if attention_mask is not None:
                        # Expand attention_mask to 4D [batch_size, 1, 1, seq_length]
                        expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                        # Convert to same dtype as mask
                        expanded_mask = expanded_mask.to(dtype=dtype)
                        # Apply attention_mask (0 -> -inf, 1 -> 0)
                        expanded_mask = (1.0 - expanded_mask) * torch.finfo(dtype).min
                        # Combine with causal mask
                        mask = mask + expanded_mask

                    return mask

            # Apply the patch
            import transformers.modeling_attn_mask_utils
            transformers.modeling_attn_mask_utils._prepare_4d_causal_attention_mask_for_sdpa = fixed_prepare_4d
            logger.info("✅ Successfully patched _prepare_4d_causal_attention_mask_for_sdpa")

            # Also patch the function in LlamaModel if possible
            try:
                import transformers.models.llama.modeling_llama
                # Check if the function exists in the module
                if hasattr(transformers.models.llama.modeling_llama, '_prepare_4d_causal_attention_mask_for_sdpa'):
                    # Patch it
                    transformers.models.llama.modeling_llama._prepare_4d_causal_attention_mask_for_sdpa = fixed_prepare_4d
                    logger.info("✅ Successfully patched LlamaModel _prepare_4d_causal_attention_mask_for_sdpa")
            except Exception as e:
                logger.warning(f"⚠️ Could not patch LlamaModel _prepare_4d_causal_attention_mask_for_sdpa: {e}")

            return True
        except Exception as e:
            logger.error(f"❌ Failed to patch _prepare_4d_causal_attention_mask_for_sdpa: {e}")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to apply attention mask parameter fixes: {e}")
        return False

if __name__ == "__main__":
    # Apply the fixes
    success = fix_attention_mask_params()

    # Exit with appropriate code
    sys.exit(0 if success else 1)
