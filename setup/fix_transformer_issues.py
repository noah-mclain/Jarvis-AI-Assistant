#!/usr/bin/env python3
"""
Comprehensive fix for transformer model issues:
1. Fixes the 'got multiple values for argument unmasked_value' error
2. Fixes attention mask size errors
3. Fixes tokenizer memory usage by forcing CPU
4. Clears GPU memory

This script combines functionality from:
- fix_attention_mask.py
- fix_attention_mask_error.py
- fix_attention_mask_size.py
- fix_tokenizer_memory.py
"""

import os
import sys
import inspect
import importlib
import gc
import torch
import time

def debug_function_signature(func):
    """Debug a function's signature to understand its parameters"""'
    sig = inspect.signature(func)
    print(f"Function: {func.__name__}")
    print(f"Signature: {sig}")
    print(f"Parameters: {list(sig.parameters.keys())}")
    for name, param in sig.parameters.items():
        print(f"  {name}: {param.default if param.default is not param.empty else 'required'}")
    return sig

def clear_gpu_memory():
    """Clear GPU memory by emptying cache and forcing garbage collection"""
    try:
        if torch.cuda.is_available():
            print("\n===== CLEARING GPU MEMORY =====")
            
            # Get initial memory usage
            initial_mem = torch.cuda.memory_allocated() / (1024**3)
            initial_reserved = torch.cuda.memory_reserved() / (1024**3)
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            print(f"Initial GPU memory: {initial_mem:.2f} GB allocated, {initial_reserved:.2f} GB reserved")
            print(f"Total GPU memory: {total_mem:.2f} GB")
            
            # Empty cache multiple times with pauses in between
            for i in range(3):
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(0.5)
            
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
            
            # Try to reset device
            try:
                torch.cuda.set_device(0)
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error resetting device: {e}")
            
            # Get memory usage after cleanup
            current_mem = torch.cuda.memory_allocated() / (1024**3)
            current_reserved = torch.cuda.memory_reserved() / (1024**3)
            
            print(f"After cleanup: {current_mem:.2f} GB allocated, {current_reserved:.2f} GB reserved")
            print(f"Freed: {initial_mem - current_mem:.2f} GB allocated, {initial_reserved - current_reserved:.2f} GB reserved")
            print(f"Free GPU memory: {total_mem - current_mem:.2f} GB")
            
            return True
        else:
            print("CUDA is not available. No GPU memory to clear.")
            return False
    except Exception as e:
        print(f"Error clearing GPU memory: {e}")
        return False

def patch_attention_mask_converter():
    """
    Patch the AttentionMaskConverter._unmask_unattended function to fix the
    'got multiple values for argument unmasked_value' error.
    """
    try:
        import transformers.modeling_attn_mask_utils as attn_utils

        # Debug the original function
        original_func = attn_utils.AttentionMaskConverter._unmask_unattended
        print("\n===== Original Function =====")
        debug_function_signature(original_func)

        # Define a completely new implementation that matches the original signature
        def fixed_unmask_unattended(self, attention_mask, unmasked_value=0.0):
            """
            Fixed implementation of _unmask_unattended that doesn't use CPU.'
            This function converts a causal attention mask to an unmasked attention mask.
            """
            # Get the device of the attention mask
            device = attention_mask.device
            
            # Create a temporary tensor on the same device
            tmp = torch.ones_like(attention_mask) * unmasked_value
            
            # Use argmax without forcing CPU
            indices = torch.argmax(attention_mask * tmp, 1, keepdim=True)
            
            # Create a range tensor on the same device
            range_tensor = torch.arange(attention_mask.shape[1], device=device).expand_as(attention_mask)
            
            # Create the expanded mask on the same device
            expanded_mask = (range_tensor <= indices).to(attention_mask.dtype)
            
            return expanded_mask

        # Replace the original function with our fixed version
        attn_utils.AttentionMaskConverter._unmask_unattended = fixed_unmask_unattended

        # Debug the new function
        new_func = attn_utils.AttentionMaskConverter._unmask_unattended
        print("\n===== Patched Function =====")
        debug_function_signature(new_func)

        # Test the patched function
        print("\n===== Testing Patched Function =====")
        test_mask = torch.ones((2, 10), dtype=torch.float32)
        try:
            # Check if AttentionMaskConverter requires is_causal parameter
            sig = inspect.signature(attn_utils.AttentionMaskConverter.__init__)
            if 'is_causal' in sig.parameters:
                converter = attn_utils.AttentionMaskConverter(is_causal=True)
            else:
                converter = attn_utils.AttentionMaskConverter()

            # Test the function directly without using the converter
            result = fixed_unmask_unattended(converter, test_mask)
            print("Test successful!")
            return True
        except Exception as e:
            print(f"Test failed: {e}")
            print("But we'll continue with the patch for _prepare_4d_causal_attention_mask_for_sdpa")'
            return False

    except ImportError as e:
        print(f"Error importing transformers: {e}")
        return False
    except Exception as e:
        print(f"Error patching attention mask converter: {e}")
        return False

def patch_prepare_4d_causal_attention_mask():
    """
    Patch the _prepare_4d_causal_attention_mask_for_sdpa function to fix the
    'got multiple values for argument unmasked_value' error.
    """
    try:
        import transformers.modeling_attn_mask_utils as attn_utils

        # Store the original function
        original_func = attn_utils._prepare_4d_causal_attention_mask_for_sdpa

        # Debug the original function
        print("\n===== Original _prepare_4d_causal_attention_mask_for_sdpa Function =====")
        debug_function_signature(original_func)

        # Define a fixed version of the function
        def fixed_prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        ):
            """
            Fixed implementation of _prepare_4d_causal_attention_mask_for_sdpa.
            This function prepares a 4D causal attention mask for scaled dot product attention.
            """
            # Get the device and dtype from inputs_embeds
            device = inputs_embeds.device
            dtype = inputs_embeds.dtype

            # Create causal mask
            batch_size, seq_length = input_shape

            # Create a causal mask that matches the input shape
            causal_mask = torch.ones((seq_length, seq_length), device=device, dtype=dtype)
            causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

            # If there are past key values, adjust the causal mask
            if past_key_values_length > 0:
                causal_mask = torch.cat(
                    [
                        torch.zeros(
                            (1, 1, seq_length, past_key_values_length),
                            device=device,
                            dtype=dtype,
                        ),
                        causal_mask,
                    ],
                    dim=-1,
                )

            # If attention_mask is provided, combine it with the causal mask
            if attention_mask is not None:
                # Convert attention_mask from [batch_size, seq_length] to [batch_size, 1, 1, seq_length]
                expanded_attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)

                # Convert from 0/1 to -inf/0 for masked/unmasked positions
                expanded_attn_mask = expanded_attn_mask.to(dtype=dtype)
                expanded_attn_mask = (1.0 - expanded_attn_mask) * torch.finfo(dtype).min

                # Combine the attention mask with the causal mask
                expanded_4d_mask = expanded_attn_mask + causal_mask
            else:
                expanded_4d_mask = causal_mask

            return expanded_4d_mask

        # Replace the original function with our fixed version
        attn_utils._prepare_4d_causal_attention_mask_for_sdpa = fixed_prepare_4d_causal_attention_mask_for_sdpa

        # Debug the new function
        new_func = attn_utils._prepare_4d_causal_attention_mask_for_sdpa
        print("\n===== Patched _prepare_4d_causal_attention_mask_for_sdpa Function =====")
        debug_function_signature(new_func)

        # Test the patched function
        print("\n===== Testing Patched _prepare_4d_causal_attention_mask_for_sdpa Function =====")
        try:
            batch_size, seq_length = 2, 10
            input_shape = (batch_size, seq_length)
            inputs_embeds = torch.rand((batch_size, seq_length, 32))
            attention_mask = torch.ones((batch_size, seq_length))
            past_key_values_length = 0

            result = fixed_prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )
            print("Test successful!")
            return True
        except Exception as e:
            print(f"Test failed: {e}")
            return False

    except ImportError as e:
        print(f"Error importing transformers: {e}")
        return False
    except Exception as e:
        print(f"Error patching _prepare_4d_causal_attention_mask_for_sdpa: {e}")
        return False

def patch_llama_model_forward():
    """
    Patch the LlamaModel.forward method to fix attention mask size errors and
    handle gradient checkpointing compatibility.
    """
    try:
        from transformers.models.llama.modeling_llama import LlamaModel

        # Store the original forward method
        original_forward = LlamaModel.forward

        # Debug the original forward method
        print("\n===== Original LlamaModel.forward Method =====")
        debug_function_signature(original_forward)

        # Define a patched forward method that handles attention mask issues
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
            """
            Patched forward method for LlamaModel that fixes attention mask issues.
            """
            # Force use_cache to False when using gradient checkpointing
            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                if use_cache:
                    print("use_cache=True is incompatible with gradient checkpointing. Setting use_cache=False...")
                use_cache = False
            
            # Call the original forward method but catch any errors related to attention mask
            try:
                # Fix attention mask shape if needed
                if attention_mask is not None and attention_mask.dim() == 2:
                    # Get the device and dtype
                    device = attention_mask.device
                    dtype = attention_mask.dtype
                    
                    # Get sequence length
                    seq_length = attention_mask.size(1)
                    
                    # Convert attention_mask from [batch_size, seq_length] to [batch_size, 1, 1, seq_length]
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                    
                    # Create a causal mask
                    causal_mask = torch.ones((seq_length, seq_length), device=device, dtype=dtype)
                    causal_mask = torch.triu(causal_mask, diagonal=1).unsqueeze(0).unsqueeze(0)
                    
                    # Convert to proper format for attention
                    attention_mask = attention_mask.expand(-1, -1, seq_length, -1)
                    attention_mask = (1.0 - attention_mask) * torch.finfo(dtype).min
                
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
            except (TypeError, ValueError) as e:
                error_msg = str(e)
                if "unmasked_value" in error_msg:
                    print("Caught 'unmasked_value' error, using fallback implementation")
                    # Handle unmasked_value error
                    device = self.device
                    dtype = self.dtype if hasattr(self, "dtype") else torch.float32

                    # Create a simple causal mask
                    if input_ids is not None:
                        batch_size, seq_length = input_ids.shape
                    else:
                        batch_size, seq_length = inputs_embeds.shape[:2]

                    # Create a causal mask
                    causal_mask = torch.triu(
                        torch.ones((seq_length, seq_length), device=device, dtype=dtype) * -10000.0,
                        diagonal=1,
                    )
                    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

                    # If attention_mask is provided, combine it with the causal mask
                    if attention_mask is not None:
                        # Convert attention_mask from [batch_size, seq_length] to [batch_size, 1, 1, seq_length]
                        expanded_attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                        expanded_attn_mask = expanded_attn_mask.to(dtype=dtype)
                        expanded_attn_mask = (1.0 - expanded_attn_mask) * -10000.0

                        # Combine the attention mask with the causal mask
                        attention_mask = expanded_attn_mask + causal_mask
                    else:
                        attention_mask = causal_mask
                        
                elif "Attention mask should be of size" in error_msg:
                    print("Caught attention mask size error, fixing mask dimensions")
                    
                    # Get the device and dtype
                    device = self.device
                    dtype = self.dtype if hasattr(self, "dtype") else torch.float32
                    
                    # Create a simple causal mask
                    if input_ids is not None:
                        batch_size, seq_length = input_ids.shape
                    else:
                        batch_size, seq_length = inputs_embeds.shape[:2]
                    
                    # If attention_mask is provided but has wrong dimensions
                    if attention_mask is not None:
                        # Ensure it's 4D: [batch_size, 1, seq_length, seq_length]
                        if attention_mask.dim() == 2:
                            # [batch_size, seq_length] -> [batch_size, 1, 1, seq_length]
                            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                            
                            # Expand to [batch_size, 1, seq_length, seq_length]
                            attention_mask = attention_mask.expand(-1, -1, seq_length, -1)
                            
                            # Convert from 0/1 to -inf/0
                            attention_mask = (1.0 - attention_mask) * torch.finfo(dtype).min
                    else:
                        # Create a causal mask if none provided
                        causal_mask = torch.triu(
                            torch.ones((seq_length, seq_length), device=device, dtype=dtype) * torch.finfo(dtype).min,
                            diagonal=1,
                        )
                        attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
                else:
                    # If it's not one of the known errors, re-raise the exception
                    raise

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

        # Replace the original forward method with our patched version
        LlamaModel.forward = patched_forward

        # Debug the new forward method
        new_forward = LlamaModel.forward
        print("\n===== Patched LlamaModel.forward Method =====")
        debug_function_signature(new_forward)

        print("Successfully patched LlamaModel.forward method")
        return True

    except ImportError as e:
        print(f"Error importing LlamaModel: {e}")
        return False
    except Exception as e:
        print(f"Error patching LlamaModel.forward: {e}")
        return False

def patch_tokenizer():
    """
    Patch the tokenizer to ensure it uses CPU memory only.
    This function patches the tokenizer's __call__ method to move tensors to CPU.'
    """
    try:
        from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
        
        print("\n===== Patching Tokenizer to Use CPU Memory =====")
        
        # Store original __call__ methods
        original_tokenizer_call = PreTrainedTokenizer.__call__
        original_fast_tokenizer_call = PreTrainedTokenizerFast.__call__
        
        # Define patched __call__ method for PreTrainedTokenizer
        def patched_tokenizer_call(self, *args, **kwargs):
            # Force return_tensors to be 'pt' for PyTorch tensors
            if 'return_tensors' in kwargs and kwargs['return_tensors'] is None:
                kwargs['return_tensors'] = 'pt'
                
            # Call original method
            result = original_tokenizer_call(self, *args, **kwargs)
            
            # If result is a dict of tensors, move them to CPU
            if isinstance(result, dict):
                for key, value in result.items():
                    if torch.is_tensor(value) and value.is_cuda:
                        result[key] = value.cpu()
            
            return result
        
        # Define patched __call__ method for PreTrainedTokenizerFast
        def patched_fast_tokenizer_call(self, *args, **kwargs):
            # Force return_tensors to be 'pt' for PyTorch tensors
            if 'return_tensors' in kwargs and kwargs['return_tensors'] is None:
                kwargs['return_tensors'] = 'pt'
                
            # Call original method
            result = original_fast_tokenizer_call(self, *args, **kwargs)
            
            # If result is a dict of tensors, move them to CPU
            if isinstance(result, dict):
                for key, value in result.items():
                    if torch.is_tensor(value) and value.is_cuda:
                        result[key] = value.cpu()
            
            return result
        
        # Apply patches
        PreTrainedTokenizer.__call__ = patched_tokenizer_call
        PreTrainedTokenizerFast.__call__ = patched_fast_tokenizer_call
        
        print("Successfully patched tokenizer to use CPU memory only")
        return True
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        return False
    except Exception as e:
        print(f"Error patching tokenizer: {e}")
        return False

def main():
    """Main function to fix transformer model issues"""
    print("=" * 50)
    print("TRANSFORMER MODEL ISSUES FIX")
    print("=" * 50)
    
    # Clear GPU memory
    cleared = clear_gpu_memory()
    
    # Patch the transformers library
    patched1 = patch_attention_mask_converter()
    patched2 = patch_prepare_4d_causal_attention_mask()
    patched3 = patch_llama_model_forward()
    patched4 = patch_tokenizer()
    
    if patched1 and patched2 and patched3 and patched4:
        print("\n✅ Successfully patched all functions!")
        print("All known transformer model issues should be fixed.")
    elif patched1 or patched2 or patched3 or patched4:
        print("\n⚠️ Partially successful: Fixed some functions")
        print("Some issues should be fixed, but others may still occur.")
    else:
        print("\n❌ Failed to patch any functions")
        print("Issues will likely still occur")
    
    if cleared:
        print("\nGPU memory has been cleared.")
    else:
        print("\nCould not clear GPU memory.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
