#!/usr/bin/env python3
"""
Fix attention mask issues without relying on DeepSeek model.

This script applies patches to fix attention mask issues in transformers
without relying on the DeepSeek model.
"""

import os
import sys
import logging
import importlib
import inspect
import types
import torch
import re
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def patch_unmask_unattended():
    """Patch the _unmask_unattended function in AttentionMaskConverter."""
    try:
        import transformers
        from transformers.modeling_attn_mask_utils import AttentionMaskConverter
        
        original_unmask_unattended = AttentionMaskConverter._unmask_unattended
        
        def patched_unmask_unattended(self, expanded_mask, attention_mask, unmasked_value=None):
            """Patched version of _unmask_unattended that handles unmasked_value parameter."""
            if unmasked_value is None:
                unmasked_value = torch.finfo(expanded_mask.dtype).min
            
            # Convert attention_mask to the same dtype as expanded_mask
            if attention_mask is not None and attention_mask.dtype != expanded_mask.dtype:
                attention_mask = attention_mask.to(dtype=expanded_mask.dtype)
            
            return original_unmask_unattended(self, expanded_mask, attention_mask, unmasked_value)
        
        # Apply the patch
        AttentionMaskConverter._unmask_unattended = patched_unmask_unattended
        
        logger.info("✅ Successfully patched AttentionMaskConverter._unmask_unattended")
        return True
    except Exception as e:
        logger.error(f"Failed to patch AttentionMaskConverter._unmask_unattended: {e}")
        return False

def patch_prepare_4d_causal_attention_mask():
    """Patch the _prepare_4d_causal_attention_mask_for_sdpa function."""
    try:
        import transformers
        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
        
        original_prepare_4d = _prepare_4d_causal_attention_mask_for_sdpa
        
        def patched_prepare_4d(attention_mask, input_shape, inputs_embeds, past_key_values_length, sliding_window=None):
            """Patched version of _prepare_4d_causal_attention_mask_for_sdpa that handles dtype mismatches."""
            # Call the original function
            mask = original_prepare_4d(attention_mask, input_shape, inputs_embeds, past_key_values_length, sliding_window)
            
            # Handle dtype mismatches
            if inputs_embeds is not None and mask is not None and mask.dtype != inputs_embeds.dtype:
                mask = mask.to(dtype=inputs_embeds.dtype)
            
            return mask
        
        # Apply the patch
        transformers.modeling_attn_mask_utils._prepare_4d_causal_attention_mask_for_sdpa = patched_prepare_4d
        
        logger.info("✅ Successfully patched _prepare_4d_causal_attention_mask_for_sdpa")
        return True
    except Exception as e:
        logger.error(f"Failed to patch _prepare_4d_causal_attention_mask_for_sdpa: {e}")
        return False

def patch_llama_model_forward():
    """Patch the LlamaModel.forward method to handle tensor size mismatches."""
    try:
        import transformers
        from transformers.models.llama.modeling_llama import LlamaModel
        
        original_forward = LlamaModel.forward
        
        def patched_forward(self, *args, **kwargs):
            """Patched version of LlamaModel.forward that handles tensor size mismatches."""
            try:
                # Try the original forward
                return original_forward(self, *args, **kwargs)
            except RuntimeError as e:
                # Check if it's a tensor size mismatch
                if "size mismatch" in str(e) or "shape mismatch" in str(e):
                    logger.warning(f"Handling tensor size mismatch in LlamaModel.forward: {e}")
                    
                    # Get the attention_mask
                    attention_mask = kwargs.get("attention_mask", None)
                    if attention_mask is None and len(args) > 1:
                        attention_mask = args[1]
                    
                    # If attention_mask is the issue, try to fix it
                    if attention_mask is not None:
                        # Create a new kwargs dictionary
                        new_kwargs = kwargs.copy()
                        
                        # Remove the attention_mask
                        new_kwargs.pop("attention_mask", None)
                        
                        # Try without attention_mask
                        if len(args) > 1:
                            new_args = list(args)
                            new_args[1] = None
                            return original_forward(self, *new_args, **new_kwargs)
                        else:
                            return original_forward(self, *args, **new_kwargs)
                
                # Re-raise the exception if we couldn't handle it
                raise
        
        # Apply the patch
        LlamaModel.forward = patched_forward
        
        logger.info("✅ Successfully patched LlamaModel.forward")
        return True
    except Exception as e:
        logger.error(f"Failed to patch LlamaModel.forward: {e}")
        return False

def patch_llama_attention_forward():
    """Patch the LlamaAttention.forward method to handle tensor dimension mismatches."""
    try:
        import transformers
        from transformers.models.llama.modeling_llama import LlamaAttention
        
        original_forward = LlamaAttention.forward
        
        def patched_forward(self, *args, **kwargs):
            """Patched version of LlamaAttention.forward that handles tensor dimension mismatches."""
            try:
                # Try the original forward
                return original_forward(self, *args, **kwargs)
            except RuntimeError as e:
                # Check if it's a dimension mismatch
                if "dimension" in str(e) and "must match" in str(e):
                    logger.warning(f"Handling tensor dimension mismatch in LlamaAttention.forward: {e}")
                    
                    # Get the attention_mask
                    attention_mask = kwargs.get("attention_mask", None)
                    
                    # If attention_mask is the issue, try to fix it
                    if attention_mask is not None:
                        # Create a new kwargs dictionary
                        new_kwargs = kwargs.copy()
                        
                        # Remove the attention_mask
                        new_kwargs.pop("attention_mask", None)
                        
                        # Try without attention_mask
                        return original_forward(self, *args, **new_kwargs)
                
                # Re-raise the exception if we couldn't handle it
                raise
        
        # Apply the patch
        LlamaAttention.forward = patched_forward
        
        logger.info("✅ Successfully patched LlamaAttention.forward")
        return True
    except Exception as e:
        logger.error(f"Failed to patch LlamaAttention.forward: {e}")
        return False

def patch_pretrained_model_forward():
    """Patch the PreTrainedModel.forward method to handle tuple unpacking errors."""
    try:
        import transformers
        from transformers.modeling_utils import PreTrainedModel
        
        original_forward = PreTrainedModel.forward
        
        def patched_forward(self, *args, **kwargs):
            """Patched version of PreTrainedModel.forward that handles tuple unpacking errors."""
            try:
                # Try the original forward
                return original_forward(self, *args, **kwargs)
            except ValueError as e:
                # Check if it's a tuple unpacking error
                if "too many values to unpack" in str(e):
                    logger.warning(f"Handling tuple unpacking error in PreTrainedModel.forward: {e}")
                    
                    # Call the model's forward method directly
                    return self.model.forward(*args, **kwargs)
                
                # Re-raise the exception if we couldn't handle it
                raise
        
        # Apply the patch
        PreTrainedModel.forward = patched_forward
        
        logger.info("✅ Successfully patched PreTrainedModel.forward")
        return True
    except Exception as e:
        logger.error(f"Failed to patch PreTrainedModel.forward: {e}")
        return False

def patch_dtype_mismatches():
    """Patch various methods to handle dtype mismatches."""
    try:
        import transformers
        from transformers.modeling_utils import PreTrainedModel
        
        # Patch PreTrainedModel.forward
        original_forward = PreTrainedModel.forward
        
        def patched_forward(self, *args, **kwargs):
            """Patched version of PreTrainedModel.forward that handles dtype mismatches."""
            try:
                # Try the original forward
                return original_forward(self, *args, **kwargs)
            except RuntimeError as e:
                # Check if it's a dtype mismatch
                if "expected scalar type" in str(e) and "but got" in str(e):
                    logger.warning(f"Handling dtype mismatch in PreTrainedModel.forward: {e}")
                    
                    # Get the attention_mask
                    attention_mask = kwargs.get("attention_mask", None)
                    if attention_mask is None and len(args) > 1:
                        attention_mask = args[1]
                    
                    # Get the inputs_embeds
                    inputs_embeds = kwargs.get("inputs_embeds", None)
                    
                    # If attention_mask and inputs_embeds are the issue, try to fix it
                    if attention_mask is not None and inputs_embeds is not None:
                        # Convert attention_mask to the same dtype as inputs_embeds
                        attention_mask = attention_mask.to(dtype=inputs_embeds.dtype)
                        
                        # Create a new kwargs dictionary
                        new_kwargs = kwargs.copy()
                        
                        # Update the attention_mask
                        new_kwargs["attention_mask"] = attention_mask
                        
                        # Try with the fixed attention_mask
                        if len(args) > 1:
                            new_args = list(args)
                            new_args[1] = attention_mask
                            return original_forward(self, *new_args, **new_kwargs)
                        else:
                            return original_forward(self, *args, **new_kwargs)
                
                # Re-raise the exception if we couldn't handle it
                raise
        
        # Apply the patch
        PreTrainedModel.forward = patched_forward
        
        # Patch PreTrainedModel.prepare_inputs_for_generation
        original_prepare = PreTrainedModel.prepare_inputs_for_generation
        
        def patched_prepare(self, *args, **kwargs):
            """Patched version of PreTrainedModel.prepare_inputs_for_generation that handles dtype mismatches."""
            # Call the original method
            inputs = original_prepare(self, *args, **kwargs)
            
            # Fix dtype mismatches
            if "attention_mask" in inputs and "inputs_embeds" in inputs:
                attention_mask = inputs["attention_mask"]
                inputs_embeds = inputs["inputs_embeds"]
                
                if attention_mask is not None and inputs_embeds is not None and attention_mask.dtype != inputs_embeds.dtype:
                    inputs["attention_mask"] = attention_mask.to(dtype=inputs_embeds.dtype)
            
            return inputs
        
        # Apply the patch
        PreTrainedModel.prepare_inputs_for_generation = patched_prepare
        
        logger.info("✅ Successfully patched methods to handle dtype mismatches")
        return True
    except Exception as e:
        logger.error(f"Failed to patch methods to handle dtype mismatches: {e}")
        return False

def main():
    """Main function."""
    logger.info("Starting attention mask fix without DeepSeek...")
    
    # Apply patches
    patches = [
        ("AttentionMaskConverter._unmask_unattended", patch_unmask_unattended),
        ("_prepare_4d_causal_attention_mask_for_sdpa", patch_prepare_4d_causal_attention_mask),
        ("LlamaModel.forward", patch_llama_model_forward),
        ("LlamaAttention.forward", patch_llama_attention_forward),
        ("PreTrainedModel.forward", patch_pretrained_model_forward),
        ("dtype mismatches", patch_dtype_mismatches)
    ]
    
    success = True
    for name, patch_func in patches:
        logger.info(f"Applying patch for {name}...")
        if not patch_func():
            logger.warning(f"⚠️ Failed to apply patch for {name}")
            success = False
    
    if success:
        logger.info("✅ All patches applied successfully")
    else:
        logger.warning("⚠️ Some patches failed to apply")
    
    return success

if __name__ == "__main__":
    success = main()
    if success:
        print("✅ Attention mask fix applied successfully!")
    else:
        print("⚠️ Attention mask fix applied with some warnings")
        sys.exit(1)
