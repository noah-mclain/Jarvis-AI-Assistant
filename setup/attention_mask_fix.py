#!/usr/bin/env python3
"""
Attention Mask Fix for DeepSeek and LLaMA models

This script applies patches to fix the attention mask handling in transformers library,
specifically addressing the "too many values to unpack (expected 2)" error that occurs
with DeepSeek and LLaMA models.

The fix ensures that attention masks are always in the correct shape (2D) and on the
correct device before being processed by the model.
"""

import os
import sys
import logging
from typing import Optional, Union, List, Dict, Any, Tuple
import importlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def patch_attention_mask_converter():
    """Patch the AttentionMaskConverter class to fix device mismatch issues"""
    try:
        # Import the class
        from transformers.modeling_attn_mask_utils import AttentionMaskConverter
        
        # Store the original function
        original_unmask_unattended = AttentionMaskConverter._unmask_unattended
        
        # Define the patched function
        @staticmethod
        def patched_unmask_unattended(
            attention_mask: "torch.Tensor",
            indices_k: Optional["torch.LongTensor"] = None,
            indices_q: Optional["torch.LongTensor"] = None,
        ):
            """
            Patched version of _unmask_unattended that keeps tensors on the same device.
            
            The original function has a call to .cpu() which causes device mismatch errors.
            This patch ensures all operations happen on the same device.
            """
            import torch
            
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
        return True
    except Exception as e:
        logger.error(f"❌ Failed to patch AttentionMaskConverter: {e}")
        return False

def patch_prepare_4d_causal_attention_mask():
    """Patch the _prepare_4d_causal_attention_mask_for_sdpa function"""
    try:
        # Import the function
        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
        
        # Store the original function
        original_prepare_4d = _prepare_4d_causal_attention_mask_for_sdpa
        
        # Define the patched function
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
            import torch
            
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
        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
        _prepare_4d_causal_attention_mask_for_sdpa = patched_prepare_4d
        
        # Also patch the module's function
        import transformers.modeling_attn_mask_utils
        transformers.modeling_attn_mask_utils._prepare_4d_causal_attention_mask_for_sdpa = patched_prepare_4d
        
        logger.info("✅ Successfully patched _prepare_4d_causal_attention_mask_for_sdpa")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to patch _prepare_4d_causal_attention_mask_for_sdpa: {e}")
        return False

def patch_model_forward_methods():
    """Patch the forward methods of LLaMA and DeepSeek models"""
    success = False
    
    # Try to patch LLaMA model
    try:
        from transformers.models.llama.modeling_llama import LlamaModel
        
        # Store the original forward method
        original_forward = LlamaModel.forward
        
        # Define a patched forward method
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
            import torch
            
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
        LlamaModel.forward = patched_forward
        logger.info("✅ Successfully patched LlamaModel.forward")
        success = True
    except Exception as e:
        logger.warning(f"⚠️ Could not patch LlamaModel.forward: {e}")
    
    # Try to patch DeepSeek model (which may inherit from LLaMA)
    try:
        # Try to import DeepSeek model
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
            import torch
            
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
        success = True
    except Exception as e:
        logger.warning(f"⚠️ Could not patch DeepSeekModel.forward: {e}")
    
    return success

def apply_all_fixes():
    """Apply all attention mask fixes"""
    logger.info("Applying attention mask fixes...")
    
    # Apply all patches
    converter_patched = patch_attention_mask_converter()
    prepare_4d_patched = patch_prepare_4d_causal_attention_mask()
    models_patched = patch_model_forward_methods()
    
    # Check if any patch was successful
    if converter_patched or prepare_4d_patched or models_patched:
        logger.info("✅ Successfully applied attention mask fixes")
        return True
    else:
        logger.error("❌ Failed to apply any attention mask fixes")
        return False

if __name__ == "__main__":
    # Apply all fixes when run as a script
    success = apply_all_fixes()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
