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
            
            # Define a new function that handles the missing parameters
            def fixed_prepare_4d(
                attention_mask,
                input_shape,
                inputs_embeds,
                past_key_values_length,
                sliding_window=None,  # Make this parameter optional with a default value
                dtype=None,           # Make this parameter optional with a default value
            ):
                """
                Fixed version of _prepare_4d_causal_attention_mask_for_sdpa that handles
                the sliding_window and dtype parameters correctly.
                """
                # If dtype is None, use float32 as a default
                if dtype is None:
                    dtype = torch.float32
                    logger.info("Using default dtype=torch.float32")
                
                # Fix attention_mask shape if needed
                if attention_mask is not None and attention_mask.dim() > 2:
                    # Get the batch size and sequence length
                    batch_size = attention_mask.size(0)
                    seq_length = attention_mask.size(-1)
                    
                    # Reshape to 2D [batch_size, seq_length]
                    attention_mask = attention_mask.view(batch_size, seq_length)
                    logger.info(f"Reshaped attention mask from >2D to 2D: {attention_mask.shape}")
                
                # Call the original function with all required parameters
                try:
                    return original_prepare_4d(
                        attention_mask,
                        input_shape,
                        inputs_embeds,
                        past_key_values_length,
                        sliding_window,
                        dtype,
                    )
                except TypeError as e:
                    error_msg = str(e)
                    logger.warning(f"Error in original function: {error_msg}")
                    
                    # If the error is about too many arguments, try with fewer arguments
                    if "takes" in error_msg and "positional arguments but" in error_msg:
                        logger.info("Trying with fewer arguments")
                        # Try without sliding_window and dtype
                        try:
                            return original_prepare_4d(
                                attention_mask,
                                input_shape,
                                inputs_embeds,
                                past_key_values_length,
                            )
                        except Exception as e2:
                            logger.warning(f"Error with fewer arguments: {e2}")
                    
                    # If we can't call the original function, create a causal mask manually
                    logger.info("Creating causal mask manually")
                    
                    # Get batch size and sequence length
                    batch_size = input_shape[0]
                    seq_length = input_shape[1]
                    
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
