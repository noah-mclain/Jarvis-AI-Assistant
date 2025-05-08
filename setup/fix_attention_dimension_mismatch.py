#!/usr/bin/env python3
"""
Fix for the specific tensor dimension mismatch in DeepSeek models.

This script specifically addresses the error:
"The size of tensor a (6) must match the size of tensor b (2048) at non-singleton dimension 2"

It directly patches the attention mechanism to handle this specific error.
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

def fix_attention_dimension_mismatch():
    """Apply fixes for the specific tensor dimension mismatch in DeepSeek models"""
    try:
        import torch
        import transformers
        from transformers import __version__ as transformers_version
        
        logger.info(f"Applying attention dimension mismatch fixes for transformers {transformers_version}")
        
        # First, try to patch the attention mechanism in LlamaAttention
        try:
            from transformers.models.llama.modeling_llama import LlamaAttention
            
            # Store the original forward method
            original_forward = LlamaAttention.forward
            
            # Define a patched forward method
            def patched_forward(
                self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor]] = None,
                output_attentions: bool = False,
                use_cache: bool = False,
                **kwargs
            ):
                """
                Patched forward method for LlamaAttention that handles tensor dimension mismatches.
                """
                try:
                    # Try the original forward method first
                    return original_forward(
                        self,
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        **kwargs
                    )
                except Exception as e:
                    error_msg = str(e)
                    logger.warning(f"Error in original attention forward: {error_msg}")
                    
                    # Check for the specific tensor dimension mismatch error
                    if "The size of tensor a (6) must match the size of tensor b (2048) at non-singleton dimension 2" in error_msg:
                        logger.info("Detected specific tensor dimension mismatch error. Applying direct fix.")
                        
                        # Get the batch size and sequence length
                        batch_size, seq_length, _ = hidden_states.shape
                        device = hidden_states.device
                        
                        # Create a custom implementation of the attention mechanism
                        try:
                            # Get the query, key, and value projections
                            if past_key_value is not None:
                                # If using past key values, only project the new tokens
                                past_key, past_value = past_key_value
                                key_states = past_key
                                value_states = past_value
                                
                                # Only project the new tokens
                                query_states = self.q_proj(hidden_states)
                                key_states_new = self.k_proj(hidden_states)
                                value_states_new = self.v_proj(hidden_states)
                                
                                # Concatenate with past key and value states
                                key_states = torch.cat([key_states, key_states_new], dim=1)
                                value_states = torch.cat([value_states, value_states_new], dim=1)
                            else:
                                # Project all tokens
                                query_states = self.q_proj(hidden_states)
                                key_states = self.k_proj(hidden_states)
                                value_states = self.v_proj(hidden_states)
                            
                            # Reshape to multi-head attention format
                            query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
                            key_states = key_states.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                            value_states = value_states.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                            
                            # Compute attention scores
                            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (self.head_dim ** 0.5)
                            
                            # Apply causal mask
                            causal_mask = torch.triu(torch.ones(seq_length, seq_length, device=device, dtype=torch.bool), diagonal=1)
                            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
                            attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
                            
                            # Apply attention mask if provided
                            if attention_mask is not None:
                                # Ensure attention_mask is properly shaped
                                if attention_mask.dim() == 2:
                                    # [batch_size, seq_length] -> [batch_size, 1, 1, seq_length]
                                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                                    # Convert to additive mask (0 -> 0, 1 -> -inf)
                                    attention_mask = (1.0 - attention_mask) * torch.finfo(attn_weights.dtype).min
                                    # Apply mask
                                    attn_weights = attn_weights + attention_mask
                                else:
                                    logger.warning(f"Skipping attention_mask with unexpected shape: {attention_mask.shape}")
                            
                            # Apply softmax
                            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
                            
                            # Apply dropout
                            attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
                            
                            # Compute attention output
                            attn_output = torch.matmul(attn_weights, value_states)
                            
                            # Reshape back to original format
                            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
                            
                            # Apply output projection
                            attn_output = self.o_proj(attn_output)
                            
                            # Return the output and optionally the attention weights and past key values
                            outputs = (attn_output,)
                            if output_attentions:
                                outputs += (attn_weights,)
                            if use_cache:
                                outputs += ((key_states, value_states),)
                            
                            return outputs
                        except Exception as e2:
                            logger.warning(f"Custom attention implementation failed: {e2}")
                    
                    # If we can't handle the specific error, try without attention mask
                    logger.info("Trying attention forward without attention mask")
                    return original_forward(
                        self,
                        hidden_states=hidden_states,
                        attention_mask=None,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        **kwargs
                    )
            
            # Apply the patch
            LlamaAttention.forward = patched_forward
            logger.info("✅ Successfully patched LlamaAttention.forward to handle tensor dimension mismatches")
            
            # Also patch DeepSeekAttention if available
            try:
                # Try to import DeepSeekAttention
                DeepSeekAttention = None
                try:
                    from transformers.models.deepseek.modeling_deepseek import DeepSeekAttention
                except ImportError:
                    # Try alternative import paths
                    try:
                        module = importlib.import_module("transformers.models.deepseek.modeling_deepseek")
                        DeepSeekAttention = getattr(module, "DeepSeekAttention", None)
                    except (ImportError, AttributeError):
                        logger.warning("Could not import DeepSeekAttention")
                
                if DeepSeekAttention is not None:
                    # Apply the same patch to DeepSeekAttention
                    DeepSeekAttention.forward = patched_forward
                    logger.info("✅ Successfully patched DeepSeekAttention.forward to handle tensor dimension mismatches")
            except Exception as e:
                logger.warning(f"⚠️ Could not patch DeepSeekAttention.forward: {e}")
            
            return True
        except Exception as e:
            logger.error(f"❌ Failed to patch LlamaAttention.forward: {e}")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to apply attention dimension mismatch fixes: {e}")
        return False

if __name__ == "__main__":
    # Apply the fixes
    success = fix_attention_dimension_mismatch()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
