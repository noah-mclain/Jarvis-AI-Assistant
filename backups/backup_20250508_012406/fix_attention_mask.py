#!/usr/bin/env python3
"""
Fix for the transformers attention mask function that's causing device mismatch.
This script patches the _unmask_unattended function in the transformers library
to fix the 'got multiple values for argument unmasked_value' error.
"""

import torch
import sys
import os
import gc

def patch_transformers_attention_mask():
    """
    Patch the problematic function in transformers library that's causing device mismatch.
    This specifically targets the _unmask_unattended function that's using .cpu()
    """
    try:
        import transformers.modeling_attn_mask_utils as attn_utils
        import inspect
        
        # Store the original function
        original_unmask_unattended = attn_utils.AttentionMaskConverter._unmask_unattended
        
        # Check the original function signature
        sig = inspect.signature(original_unmask_unattended)
        print(f"Original function signature: {sig}")
        
        # Define our patched version with the exact same signature
        def patched_unmask_unattended(self, attention_mask, unmasked_value=0.0):
            """Patched version that doesn't force CPU conversion"""
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
        
        # Apply the patch
        attn_utils.AttentionMaskConverter._unmask_unattended = patched_unmask_unattended
        print("Successfully patched transformers attention mask function")
        return True
        
    except Exception as e:
        print(f"Error patching transformers attention mask function: {e}")
        print("Will continue without patching")
        return False

def clear_gpu_memory():
    """Clear GPU memory by emptying cache and forcing garbage collection"""
    try:
        if torch.cuda.is_available():
            print("CUDA is available. Clearing GPU memory...")
            
            # Get initial memory usage
            initial_mem = torch.cuda.memory_allocated() / (1024**3)
            initial_reserved = torch.cuda.memory_reserved() / (1024**3)
            print(f"Initial GPU memory: {initial_mem:.2f} GB allocated, {initial_reserved:.2f} GB reserved")
            
            # Empty cache
            torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            # Get memory usage after cleanup
            current_mem = torch.cuda.memory_allocated() / (1024**3)
            current_reserved = torch.cuda.memory_reserved() / (1024**3)
            print(f"After cleanup: {current_mem:.2f} GB allocated, {current_reserved:.2f} GB reserved")
            print(f"Freed: {initial_mem - current_mem:.2f} GB allocated, {initial_reserved - current_reserved:.2f} GB reserved")
            
            # Get total GPU memory
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            free_mem = total_mem - current_mem
            print(f"Total GPU memory: {total_mem:.2f} GB")
            print(f"Free GPU memory: {free_mem:.2f} GB")
            
            return True
        else:
            print("CUDA is not available. No GPU memory to clear.")
            return False
    except Exception as e:
        print(f"Error clearing GPU memory: {e}")
        return False

def main():
    """Main function to patch the transformers library and clear GPU memory"""
    print("=" * 50)
    print("TRANSFORMERS ATTENTION MASK PATCH")
    print("=" * 50)
    
    # Clear GPU memory
    cleared = clear_gpu_memory()
    
    # Patch the transformers library
    patched = patch_transformers_attention_mask()
    
    if patched:
        print("\nTransformers library has been patched successfully.")
        print("You can now run the training script with the fixed attention mask function.")
    else:
        print("\nFailed to patch the transformers library.")
    
    if cleared:
        print("\nGPU memory has been cleared.")
    else:
        print("\nCould not clear GPU memory.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
