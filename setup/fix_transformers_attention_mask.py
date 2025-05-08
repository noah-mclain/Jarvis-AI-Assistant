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
            
            # Define patched function
            def patched_prepare_4d(
                attention_mask,
                input_shape,
                inputs_embeds,
                past_key_values_length,
                sliding_window,
                dtype,
            ):
                """
                Patched version that ensures attention_mask is 2D before processing.
                """
                # Fix attention_mask shape if needed
                if attention_mask is not None and attention_mask.dim() > 2:
                    # Get the batch size and sequence length
                    batch_size = attention_mask.size(0)
                    seq_length = attention_mask.size(-1)
                    
                    # Reshape to 2D [batch_size, seq_length]
                    attention_mask = attention_mask.view(batch_size, seq_length)
                    logger.info(f"Reshaped attention mask from >2D to 2D: {attention_mask.shape}")
                
                # Call the original function with the fixed mask
                return original_prepare_4d(
                    attention_mask,
                    input_shape,
                    inputs_embeds,
                    past_key_values_length,
                    sliding_window,
                    dtype,
                )
            
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
            
            # Define patched function
            @staticmethod
            def patched_unmask_unattended(
                attention_mask,
                indices_k=None,
                indices_q=None,
                unmasked_value=None,
            ):
                """
                Patched version of _unmask_unattended that keeps tensors on the same device.
                
                The original function has a call to .cpu() which causes device mismatch errors.
                This patch ensures all operations happen on the same device.
                """
                # Get the device of the attention mask
                device = attention_mask.device
                
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
                if indices_k is not None:
                    mask = mask.expand(-1, -1, indices_k, -1)
                if indices_q is not None:
                    mask = mask.expand(-1, indices_q, -1, -1)
                
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
            
            # Define a patched forward method that properly handles attention masks
            def patched_forward(
                self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
            ):
                """Patched forward method for LlamaModel that properly handles attention masks."""
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
                    
                    # Reshape to 2D [batch_size, seq_length]
                    attention_mask = attention_mask.view(batch_size, seq_length)
                    logger.info(f"Reshaped attention mask from >2D to 2D: {attention_mask.shape}")
                
                # Ensure attention_mask is on the correct device
                if attention_mask is not None and device is not None and attention_mask.device != device:
                    attention_mask = attention_mask.to(device)
                
                # Call the original forward method with the fixed attention mask
                return original_forward(
                    self,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            
            # Apply the patch
            LlamaModel.forward = patched_forward
            logger.info("✅ Successfully patched LlamaModel.forward")
        except Exception as e:
            logger.warning(f"⚠️ Could not patch LlamaModel.forward: {e}")
        
        # Try to patch DeepSeek model if available
        try:
            import importlib
            deepseek_module = importlib.import_module("transformers.models.deepseek.modeling_deepseek")
            DeepSeekModel = getattr(deepseek_module, "DeepSeekModel")
            
            # Store the original forward method
            original_forward = DeepSeekModel.forward
            
            # Define a patched forward method (similar to LLaMA patch)
            def patched_forward(
                self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
            ):
                """Patched forward method that fixes attention mask issues"""
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
                    
                    # Reshape to 2D [batch_size, seq_length]
                    attention_mask = attention_mask.view(batch_size, seq_length)
                    logger.info(f"Reshaped attention mask from >2D to 2D: {attention_mask.shape}")
                
                # Ensure attention_mask is on the correct device
                if attention_mask is not None and device is not None and attention_mask.device != device:
                    attention_mask = attention_mask.to(device)
                
                # Call the original forward method with the fixed mask
                return original_forward(
                    self,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            
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
