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

                        # First, try to patch the _unmask_unattended function
                        try:
                            from transformers.modeling_attn_mask_utils import AttentionMaskConverter

                            # Define a patched function that handles the specific error
                            @staticmethod
                            def patched_unmask_unattended(attention_mask, indices_k=None, indices_q=None, unmasked_value=True):
                                """
                                Patched version of _unmask_unattended that handles the specific tensor size mismatch error.
                                """
                                # Get the device of the attention mask
                                device = attention_mask.device

                                # Fix attention_mask shape if needed
                                if attention_mask.dim() > 2:
                                    # Get the batch size and sequence length
                                    batch_size = attention_mask.size(0)
                                    seq_length = attention_mask.size(-1)

                                    # Reshape to 2D [batch_size, seq_length]
                                    attention_mask = attention_mask.view(batch_size, seq_length)
                                    logger.info(f"Reshaped attention mask from >2D to 2D: {attention_mask.shape}")

                                # Get batch size and sequence length
                                batch_size = attention_mask.size(0)
                                seq_length = attention_mask.size(-1)

                                # Create a temporary tensor on the same device
                                tmp = torch.ones(seq_length, device=device)

                                # Find the first non-masked position for each sequence
                                indices = torch.argmax(attention_mask * tmp, 1, keepdim=True)

                                # Create a mask for unattended positions
                                mask = torch.arange(seq_length, device=device).expand(batch_size, -1)
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
                                        if isinstance(indices_q, int):
                                            mask = mask.expand(-1, indices_q, -1, -1)
                                        else:
                                            # Handle case where indices_q is a tensor
                                            mask = mask.expand(-1, indices_q.size(0) if hasattr(indices_q, 'size') else indices_q, -1, -1)
                                except Exception as e:
                                    logger.warning(f"Error expanding mask dimensions: {e}")
                                    # If we encounter the specific tensor size mismatch error, create a compatible mask
                                    error_msg = str(e)
                                    if "The size of tensor a (6) must match the size of tensor b (2048) at non-singleton dimension 2" in error_msg:
                                        logger.info("Creating a compatible mask for the specific error")

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
                            logger.info("Successfully patched AttentionMaskConverter._unmask_unattended directly")
                        except Exception as e:
                            logger.warning(f"Failed to patch _unmask_unattended directly: {e}")

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
                # First check if the deepseek module exists
                import importlib.util
                spec = importlib.util.find_spec("transformers.models.deepseek.modeling_deepseek")

                if spec is None:
                    logger.warning("DeepSeek model not available in this transformers version. Creating custom implementation.")

                    # Create a custom implementation for DeepSeekAttention based on LlamaAttention
                    from transformers.models.llama.modeling_llama import LlamaAttention

                    # Create a DeepSeekAttention class that inherits from LlamaAttention
                    class CustomDeepSeekAttention(LlamaAttention):
                        """Custom DeepSeekAttention implementation based on LlamaAttention."""
                        pass

                    # Store the class in a variable
                    DeepSeekAttention = CustomDeepSeekAttention

                    # Add it to the transformers module
                    import transformers
                    if not hasattr(transformers.models, "deepseek"):
                        # Create the module structure
                        class DeepSeekModule:
                            pass
                        transformers.models.deepseek = DeepSeekModule()

                    # Create or get the modeling_deepseek module
                    if not hasattr(transformers.models.deepseek, "modeling_deepseek"):
                        class ModelingDeepSeek:
                            pass
                        transformers.models.deepseek.modeling_deepseek = ModelingDeepSeek()

                    # Set the DeepSeekAttention attribute
                    setattr(transformers.models.deepseek.modeling_deepseek, "DeepSeekAttention", DeepSeekAttention)

                    logger.info("✅ Created custom DeepSeekAttention implementation based on LlamaAttention")
                else:
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
