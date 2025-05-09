#!/usr/bin/env python3
"""
Fix for the _prepare_4d_causal_attention_mask_for_sdpa function in transformers.

This script addresses the error:
"_prepare_4d_causal_attention_mask_for_sdpa() takes from 4 to 5 positional arguments but 6 were given"

It patches the function to handle different parameter signatures across transformers versions.
"""

import os
import sys
import logging
import inspect
from typing import Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def fix_prepare_4d_function():
    """
    Fix the _prepare_4d_causal_attention_mask_for_sdpa function to handle different parameter signatures.
    
    This function detects the transformers version and applies the appropriate fix.
    """
    try:
        import torch
        import transformers
        from transformers import __version__ as transformers_version
        
        logger.info(f"Fixing _prepare_4d_causal_attention_mask_for_sdpa for transformers {transformers_version}")
        
        # Parse version string to check compatibility
        try:
            version_parts = transformers_version.split('.')
            major = int(version_parts[0]) if len(version_parts) > 0 else 0
            minor = int(version_parts[1]) if len(version_parts) > 1 else 0
            patch = int(version_parts[2]) if len(version_parts) > 2 else 0
            logger.info(f"Detected transformers version: {major}.{minor}.{patch}")
        except Exception as e:
            logger.warning(f"Could not parse transformers version: {e}. Applying fixes anyway.")
            major, minor, patch = 4, 36, 0  # Assume recent version
        
        # Check if the function exists in modeling_attn_mask_utils
        try:
            from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
            
            # Get the signature of the original function
            sig = inspect.signature(_prepare_4d_causal_attention_mask_for_sdpa)
            param_names = list(sig.parameters.keys())
            logger.info(f"Original function signature: {param_names}")
            
            # Define a patched function that can handle different signatures
            def patched_prepare_4d(*args, **kwargs):
                """
                Patched version of _prepare_4d_causal_attention_mask_for_sdpa that handles different signatures.
                
                This function adapts to the different parameter signatures across transformers versions:
                - v4.36.0: (attention_mask, input_shape, inputs_embeds, past_key_values_length)
                - v4.37.0+: (attention_mask, input_shape, inputs_embeds, past_key_values_length, sliding_window)
                - v4.38.0+: (attention_mask, input_shape, inputs_embeds, past_key_values_length, sliding_window, dtype)
                """
                # Extract parameters from args or kwargs
                attention_mask = None
                input_shape = None
                inputs_embeds = None
                past_key_values_length = 0
                sliding_window = None
                dtype = None
                
                # Extract from args
                if len(args) > 0:
                    attention_mask = args[0]
                if len(args) > 1:
                    input_shape = args[1]
                if len(args) > 2:
                    inputs_embeds = args[2]
                if len(args) > 3:
                    past_key_values_length = args[3]
                if len(args) > 4:
                    sliding_window = args[4]
                if len(args) > 5:
                    dtype = args[5]
                
                # Extract from kwargs
                if 'attention_mask' in kwargs:
                    attention_mask = kwargs['attention_mask']
                if 'input_shape' in kwargs:
                    input_shape = kwargs['input_shape']
                if 'inputs_embeds' in kwargs:
                    inputs_embeds = kwargs['inputs_embeds']
                if 'past_key_values_length' in kwargs:
                    past_key_values_length = kwargs['past_key_values_length']
                if 'sliding_window' in kwargs:
                    sliding_window = kwargs['sliding_window']
                if 'dtype' in kwargs:
                    dtype = kwargs['dtype']
                
                # Check required parameters
                if input_shape is None:
                    raise ValueError("input_shape is required")
                if inputs_embeds is None:
                    raise ValueError("inputs_embeds is required")
                
                # Determine the expected signature based on the number of parameters
                if len(param_names) <= 4:
                    # v4.36.0 signature
                    logger.info("Using v4.36.0 signature (4 parameters)")
                    return _prepare_4d_causal_attention_mask_for_sdpa(
                        attention_mask, input_shape, inputs_embeds, past_key_values_length
                    )
                elif len(param_names) == 5:
                    # v4.37.0 signature
                    logger.info("Using v4.37.0 signature (5 parameters)")
                    return _prepare_4d_causal_attention_mask_for_sdpa(
                        attention_mask, input_shape, inputs_embeds, past_key_values_length, sliding_window
                    )
                else:
                    # v4.38.0+ signature
                    logger.info("Using v4.38.0+ signature (6 parameters)")
                    # If dtype is None, use inputs_embeds.dtype
                    if dtype is None:
                        dtype = inputs_embeds.dtype
                    
                    return _prepare_4d_causal_attention_mask_for_sdpa(
                        attention_mask, input_shape, inputs_embeds, past_key_values_length, sliding_window, dtype
                    )
            
            # Apply the patch
            transformers.modeling_attn_mask_utils._prepare_4d_causal_attention_mask_for_sdpa = patched_prepare_4d
            logger.info("✅ Successfully patched _prepare_4d_causal_attention_mask_for_sdpa in modeling_attn_mask_utils")
            
            # Also patch the function in specific model implementations
            try:
                # Try to patch in LlamaModel
                try:
                    from transformers.models.llama.modeling_llama import _prepare_4d_causal_attention_mask_for_sdpa as llama_prepare_4d
                    transformers.models.llama.modeling_llama._prepare_4d_causal_attention_mask_for_sdpa = patched_prepare_4d
                    logger.info("✅ Successfully patched _prepare_4d_causal_attention_mask_for_sdpa in LlamaModel")
                except (ImportError, AttributeError) as e:
                    logger.warning(f"⚠️ Could not patch in LlamaModel: {e}")
                
                # Try to patch in DeepSeekModel
                try:
                    # Try different possible import paths
                    deepseek_patched = False
                    
                    # Try standard path
                    try:
                        from transformers.models.deepseek.modeling_deepseek import _prepare_4d_causal_attention_mask_for_sdpa as deepseek_prepare_4d
                        transformers.models.deepseek.modeling_deepseek._prepare_4d_causal_attention_mask_for_sdpa = patched_prepare_4d
                        deepseek_patched = True
                    except (ImportError, AttributeError):
                        pass
                    
                    # Try deepseek_coder path
                    if not deepseek_patched:
                        try:
                            from transformers.models.deepseek_coder.modeling_deepseek_coder import _prepare_4d_causal_attention_mask_for_sdpa as deepseek_coder_prepare_4d
                            transformers.models.deepseek_coder.modeling_deepseek_coder._prepare_4d_causal_attention_mask_for_sdpa = patched_prepare_4d
                            deepseek_patched = True
                        except (ImportError, AttributeError):
                            pass
                    
                    if deepseek_patched:
                        logger.info("✅ Successfully patched _prepare_4d_causal_attention_mask_for_sdpa in DeepSeekModel")
                    else:
                        logger.warning("⚠️ Could not find DeepSeekModel to patch")
                except Exception as e:
                    logger.warning(f"⚠️ Could not patch in DeepSeekModel: {e}")
            except Exception as e:
                logger.warning(f"⚠️ Could not patch in specific models: {e}")
            
            return True
        except (ImportError, AttributeError) as e:
            logger.warning(f"⚠️ Could not find _prepare_4d_causal_attention_mask_for_sdpa in modeling_attn_mask_utils: {e}")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to fix _prepare_4d_causal_attention_mask_for_sdpa: {e}")
        return False

if __name__ == "__main__":
    # Apply the fix
    success = fix_prepare_4d_function()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
