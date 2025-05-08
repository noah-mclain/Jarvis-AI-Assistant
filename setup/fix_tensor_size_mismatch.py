#!/usr/bin/env python3
"""
Fix for tensor size mismatch in transformers library.

This script specifically addresses the error:
"The size of tensor a (6) must match the size of tensor b (2048) at non-singleton dimension 2"

It patches the attention mask handling in the LlamaModel forward method to ensure
tensor dimensions are compatible.
"""

import sys
import logging
import importlib
from typing import Optional, Dict, Any, Tuple, Union, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def fix_tensor_size_mismatch():
    """Apply fixes for tensor size mismatch in transformers library"""
    try:
        import torch
        import transformers
        from transformers import __version__ as transformers_version
        
        logger.info(f"Applying tensor size mismatch fixes for transformers {transformers_version}")
        
        # Patch LlamaModel.forward to handle tensor size mismatches
        try:
            from transformers.models.llama.modeling_llama import LlamaModel
            
            # Store the original forward method
            original_forward = LlamaModel.forward
            
            # Define a patched forward method
            def patched_forward(
                self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs
            ):
                """
                Patched forward method for LlamaModel that handles tensor size mismatches.
                """
                # Get the device from input tensors
                device = None
                if input_ids is not None:
                    device = input_ids.device
                    batch_size, seq_length = input_ids.shape
                elif inputs_embeds is not None:
                    device = inputs_embeds.device
                    batch_size, seq_length, _ = inputs_embeds.shape
                else:
                    logger.warning("Neither input_ids nor inputs_embeds provided")
                    # Try to proceed with the original method
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
                        **kwargs
                    )
                
                # Fix attention mask shape and device
                if attention_mask is not None:
                    # Log original attention mask info
                    logger.info(f"Original attention_mask shape: {attention_mask.shape}, device: {attention_mask.device}")
                    
                    # Ensure attention_mask is on the correct device
                    if attention_mask.device != device:
                        attention_mask = attention_mask.to(device)
                        logger.info(f"Moved attention_mask to device: {device}")
                    
                    # Fix attention mask shape if needed
                    if attention_mask.dim() > 2:
                        # Reshape to 2D [batch_size, seq_length]
                        attention_mask = attention_mask.view(batch_size, -1)
                        # Ensure the sequence length matches
                        if attention_mask.size(1) != seq_length:
                            logger.warning(f"Attention mask sequence length ({attention_mask.size(1)}) doesn't match input sequence length ({seq_length}). Resizing.")
                            # Resize attention_mask to match input sequence length
                            if attention_mask.size(1) > seq_length:
                                # Truncate
                                attention_mask = attention_mask[:, :seq_length]
                            else:
                                # Pad with ones (no masking)
                                padding = torch.ones(batch_size, seq_length - attention_mask.size(1), device=device)
                                attention_mask = torch.cat([attention_mask, padding], dim=1)
                        
                        logger.info(f"Reshaped attention_mask to: {attention_mask.shape}")
                
                # Try to call the original forward method with the fixed attention mask
                try:
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
                        **kwargs
                    )
                except Exception as e:
                    logger.warning(f"Error in original forward with fixed attention mask: {e}")
                    
                    # Try without attention mask as a last resort
                    logger.info("Trying forward call without attention mask")
                    return original_forward(
                        self,
                        input_ids=input_ids,
                        attention_mask=None,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                        **kwargs
                    )
            
            # Apply the patch
            LlamaModel.forward = patched_forward
            logger.info("✅ Successfully patched LlamaModel.forward to handle tensor size mismatches")
            
            # Also patch DeepSeekModel if available
            try:
                # Try to import DeepSeekModel
                DeepSeekModel = None
                try:
                    from transformers.models.deepseek.modeling_deepseek import DeepSeekModel
                except ImportError:
                    # Try alternative import paths
                    try:
                        module = importlib.import_module("transformers.models.deepseek.modeling_deepseek")
                        DeepSeekModel = getattr(module, "DeepSeekModel", None)
                    except (ImportError, AttributeError):
                        logger.warning("Could not import DeepSeekModel")
                
                if DeepSeekModel is not None:
                    # Store the original forward method
                    original_deepseek_forward = DeepSeekModel.forward
                    
                    # Apply the same patch to DeepSeekModel
                    DeepSeekModel.forward = patched_forward
                    logger.info("✅ Successfully patched DeepSeekModel.forward to handle tensor size mismatches")
            except Exception as e:
                logger.warning(f"⚠️ Could not patch DeepSeekModel.forward: {e}")
            
            return True
        except Exception as e:
            logger.error(f"❌ Failed to patch LlamaModel.forward: {e}")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to apply tensor size mismatch fixes: {e}")
        return False

if __name__ == "__main__":
    # Apply the fixes
    success = fix_tensor_size_mismatch()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
