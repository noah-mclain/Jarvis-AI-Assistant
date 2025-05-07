#!/usr/bin/env python3
"""
Script to fix tokenizer memory usage by forcing it to use CPU memory.
This script patches the tokenizer to ensure it doesn't use GPU memory.
"""

import os
import sys
import importlib
import inspect
import gc

def patch_tokenizer():
    """
    Patch the tokenizer to ensure it uses CPU memory only.
    This function patches the tokenizer's __call__ method to move tensors to CPU.
    """
    try:
        from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
        import torch
        
        print("Patching tokenizer to use CPU memory only...")
        
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

def patch_dataset_processor():
    """
    Patch the dataset processor to ensure it uses CPU memory only.
    This function patches the dataset processor to move tensors to CPU.
    """
    try:
        # Try to import the dataset processor from your codebase
        sys.path.append(os.getcwd())
        from src.generative_ai_module.data_processor import DataProcessor
        
        print("Patching dataset processor to use CPU memory only...")
        
        # Check if the class exists
        if 'DataProcessor' in locals():
            # Store original methods that might use GPU
            original_methods = {}
            for name, method in inspect.getmembers(DataProcessor, inspect.isfunction):
                if name.startswith('_') or name in ['__init__', '__call__']:
                    continue
                original_methods[name] = method
            
            # Define a decorator to ensure tensors are on CPU
            def ensure_cpu_tensors(func):
                def wrapper(*args, **kwargs):
                    # Call original method
                    result = func(*args, **kwargs)
                    
                    # Move result to CPU if it's a tensor
                    import torch
                    if torch.is_tensor(result) and result.is_cuda:
                        result = result.cpu()
                    
                    # If result is a dict of tensors, move them to CPU
                    if isinstance(result, dict):
                        for key, value in result.items():
                            if torch.is_tensor(value) and value.is_cuda:
                                result[key] = value.cpu()
                    
                    # If result is a list of tensors, move them to CPU
                    if isinstance(result, list):
                        for i, item in enumerate(result):
                            if torch.is_tensor(item) and item.is_cuda:
                                result[i] = item.cpu()
                    
                    return result
                return wrapper
            
            # Apply the decorator to all methods
            for name, method in original_methods.items():
                setattr(DataProcessor, name, ensure_cpu_tensors(method))
            
            print("Successfully patched dataset processor to use CPU memory only")
            return True
        else:
            print("DataProcessor class not found, skipping patch")
            return False
    except ImportError as e:
        print(f"Error importing DataProcessor: {e}")
        return False
    except Exception as e:
        print(f"Error patching dataset processor: {e}")
        return False

def clear_gpu_memory():
    """Clear GPU memory by emptying cache and forcing garbage collection"""
    try:
        import torch
        if torch.cuda.is_available():
            print("Clearing GPU memory...")
            
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
            
            return True
        else:
            print("CUDA is not available. No GPU memory to clear.")
            return False
    except ImportError:
        print("PyTorch is not installed. Cannot clear GPU memory.")
        return False
    except Exception as e:
        print(f"Error clearing GPU memory: {e}")
        return False

def main():
    """Main function to fix tokenizer memory usage"""
    print("=" * 50)
    print("TOKENIZER MEMORY USAGE FIX")
    print("=" * 50)
    
    # Clear GPU memory
    cleared = clear_gpu_memory()
    
    # Patch tokenizer
    patched_tokenizer = patch_tokenizer()
    
    # Patch dataset processor
    patched_processor = patch_dataset_processor()
    
    if patched_tokenizer or patched_processor:
        print("\nTokenizer and/or dataset processor have been patched to use CPU memory only.")
        print("This should prevent GPU memory from being used during tokenization.")
    else:
        print("\nFailed to patch tokenizer and dataset processor.")
    
    if cleared:
        print("\nGPU memory has been cleared.")
    else:
        print("\nCould not clear GPU memory.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
