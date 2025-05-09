#!/usr/bin/env python3
"""
Fix for the attention mask size error in DeepSeek models.
This script directly patches the transformers library's attention mask handling.
"""

import os
import sys
import inspect
import importlib
import torch

def debug_function_signature(func):
    """Debug a function's signature to understand its parameters"""
    sig = inspect.signature(func)
    print(f"Function: {func.__name__}")
    print(f"Signature: {sig}")
    print(f"Parameters: {list(sig.parameters.keys())}")
    for name, param in sig.parameters.items():
        print(f"  {name}: {param.default if param.default is not param.empty else 'required'}")
    return sig

def patch_prepare_attention_mask():
    """
    Patch the prepare_attention_mask function to fix the attention mask size error.
    """
    try:
        from transformers.models.llama.modeling_llama import LlamaAttention
        
        # Store the original prepare_attention_mask method
        original_prepare_attention_mask = LlamaAttention._prepare_decoder_attention_mask
        
        # Define a fixed version of the method
        def fixed_prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
            """
            Fixed implementation of _prepare_decoder_attention_mask.
            This function prepares a 4D attention mask for decoder attention.
            
            Args:
                self: The LlamaAttention instance
                attention_mask: The attention mask to prepare
                input_shape: The shape of the input
                inputs_embeds: The input embeddings
                past_key_values_length: The length of past key values
                
            Returns:
                The prepared 4D attention mask
            """
            # Get the device and dtype from inputs_embeds
            device = inputs_embeds.device
            dtype = inputs_embeds.dtype
            
            # Check if attention_mask is already 4D
            if attention_mask.dim() == 4:
                return attention_mask
                
            # If attention_mask is 2D, expand it to 4D
            if attention_mask.dim() == 2:
                # [bsz, seq_len] -> [bsz, 1, 1, seq_len]
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                
            # Convert from [bsz, 1, 1, seq_len] to [bsz, 1, seq_len, seq_len]
            if attention_mask.size(2) != attention_mask.size(3):
                # Create a causal mask that matches the input shape
                seq_length = input_shape[1]
                causal_mask = torch.ones((seq_length, seq_length), device=device, dtype=dtype)
                causal_mask = torch.triu(causal_mask, diagonal=1).unsqueeze(0).unsqueeze(0)
                
                # Combine with the existing attention mask
                if attention_mask.size(3) < seq_length:
                    # Extend the attention mask
                    padding = torch.ones(
                        (attention_mask.size(0), attention_mask.size(1), attention_mask.size(2), seq_length - attention_mask.size(3)),
                        device=device,
                        dtype=dtype
                    )
                    attention_mask = torch.cat([attention_mask, padding], dim=3)
                
                # Create the final 4D mask
                attention_mask = attention_mask.expand(-1, -1, seq_length, -1)
                
            # Convert from 0/1 to -inf/0 for masked/unmasked positions
            attention_mask = (1.0 - attention_mask) * torch.finfo(dtype).min
            
            return attention_mask
        
        # Replace the original method with our fixed version
        LlamaAttention._prepare_decoder_attention_mask = fixed_prepare_decoder_attention_mask
        
        print("Successfully patched LlamaAttention._prepare_decoder_attention_mask")
        return True
        
    except ImportError as e:
        print(f"Error importing transformers: {e}")
        return False
    except Exception as e:
        print(f"Error patching prepare_attention_mask: {e}")
        return False

def main():
    """Main function to fix the attention mask size error"""
    print("=" * 50)
    print("FIXING ATTENTION MASK SIZE ERROR")
    print("=" * 50)
    
    # Patch the prepare_attention_mask function
    success = patch_prepare_attention_mask()
    
    if success:
        print("\n✅ Successfully patched attention mask handling!")
        print("The 'Attention mask should be of size (1, 1, 512, 512), but is torch.Size([1, 512])' error should be fixed.")
    else:
        print("\n❌ Failed to patch attention mask handling")
        print("The error will likely still occur")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
