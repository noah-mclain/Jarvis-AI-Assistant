#!/usr/bin/env python3
"""
Script to diagnose GPU memory usage and identify what's using it.
This script will:
1. Check what's currently using GPU memory
2. Try to clear GPU memory
3. Identify any cached tensors or models
"""

import os
import gc
import sys
import subprocess
import time

def check_gpu_memory():
    """Check GPU memory usage and print detailed information"""
    try:
        import torch
        if torch.cuda.is_available():
            print("\n===== GPU MEMORY INFORMATION =====")
            
            # Get device properties
            device_props = torch.cuda.get_device_properties(0)
            total_memory = device_props.total_memory / (1024**3)
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Total GPU memory: {total_memory:.2f} GB")
            
            # Get current memory usage
            allocated_memory = torch.cuda.memory_allocated() / (1024**3)
            reserved_memory = torch.cuda.memory_reserved() / (1024**3)
            free_memory = total_memory - allocated_memory
            
            print(f"Allocated memory: {allocated_memory:.2f} GB")
            print(f"Reserved memory: {reserved_memory:.2f} GB")
            print(f"Free memory: {free_memory:.2f} GB")
            
            # Get memory by allocation
            if hasattr(torch.cuda, 'memory_stats'):
                stats = torch.cuda.memory_stats()
                print("\n----- Memory Statistics -----")
                for key, value in stats.items():
                    if 'bytes' in key and value > 0:
                        print(f"{key}: {value / (1024**3):.4f} GB")
            
            return True
        else:
            print("CUDA is not available. No GPU memory to check.")
            return False
    except ImportError:
        print("PyTorch is not installed. Cannot check GPU memory.")
        return False
    except Exception as e:
        print(f"Error checking GPU memory: {e}")
        return False

def clear_gpu_memory():
    """Clear GPU memory by emptying cache and forcing garbage collection"""
    try:
        import torch
        if torch.cuda.is_available():
            print("\n===== CLEARING GPU MEMORY =====")
            
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

def check_cached_tensors():
    """Check for cached tensors in PyTorch's memory"""
    try:
        import torch
        if torch.cuda.is_available():
            print("\n===== CHECKING FOR CACHED TENSORS =====")
            
            # Get all tensors in memory
            tensors = []
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) and obj.is_cuda:
                        tensors.append((obj.shape, obj.dtype, obj.element_size() * obj.nelement() / (1024**3)))
                except:
                    pass
            
            if tensors:
                print(f"Found {len(tensors)} CUDA tensors in memory:")
                total_size = 0
                for shape, dtype, size in tensors:
                    print(f"  Shape: {shape}, Type: {dtype}, Size: {size:.4f} GB")
                    total_size += size
                print(f"Total tensor memory: {total_size:.4f} GB")
            else:
                print("No CUDA tensors found in memory.")
            
            return True
        else:
            print("CUDA is not available. Cannot check for cached tensors.")
            return False
    except ImportError:
        print("PyTorch is not installed. Cannot check for cached tensors.")
        return False
    except Exception as e:
        print(f"Error checking cached tensors: {e}")
        return False

def check_huggingface_cache():
    """Check for cached models in the Hugging Face cache"""
    try:
        from transformers.utils import TRANSFORMERS_CACHE
        print("\n===== CHECKING HUGGING FACE CACHE =====")
        
        cache_dir = os.environ.get("TRANSFORMERS_CACHE", TRANSFORMERS_CACHE)
        print(f"Hugging Face cache directory: {cache_dir}")
        
        if os.path.exists(cache_dir):
            # Get total size of cache
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(cache_dir):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
            
            print(f"Total cache size: {total_size / (1024**3):.2f} GB")
            
            # List largest files
            print("\nLargest files in cache:")
            file_sizes = []
            for dirpath, dirnames, filenames in os.walk(cache_dir):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    file_sizes.append((fp, os.path.getsize(fp)))
            
            file_sizes.sort(key=lambda x: x[1], reverse=True)
            for fp, size in file_sizes[:10]:  # Show top 10 largest files
                print(f"  {fp}: {size / (1024**3):.2f} GB")
        else:
            print(f"Cache directory {cache_dir} does not exist.")
        
        return True
    except ImportError:
        print("Transformers is not installed. Cannot check Hugging Face cache.")
        return False
    except Exception as e:
        print(f"Error checking Hugging Face cache: {e}")
        return False

def get_gpu_processes():
    """Try to identify processes using GPU memory"""
    try:
        # Try nvidia-smi first
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True
            )
            if result.stdout.strip():
                print("\n===== PROCESSES USING GPU MEMORY =====")
                print(result.stdout)
                return True
        except (subprocess.SubprocessError, FileNotFoundError):
            print("nvidia-smi not available or failed to run")
        
        # Try ps for Python processes
        try:
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True,
                text=True,
                check=True
            )
            
            python_processes = [line for line in result.stdout.split('\n') if 'python' in line.lower()]
            if python_processes:
                print("\n===== RUNNING PYTHON PROCESSES =====")
                for proc in python_processes:
                    print(proc)
                return True
        except subprocess.SubprocessError:
            print("Failed to list processes")
        
        return False
    except Exception as e:
        print(f"Error getting GPU processes: {e}")
        return False

def main():
    """Main function to diagnose GPU memory usage"""
    print("=" * 50)
    print("GPU MEMORY DIAGNOSTIC UTILITY")
    print("=" * 50)
    
    # Check GPU memory
    check_gpu_memory()
    
    # Check for cached tensors
    check_cached_tensors()
    
    # Check Hugging Face cache
    check_huggingface_cache()
    
    # Get processes using GPU
    get_gpu_processes()
    
    # Clear GPU memory
    clear_gpu_memory()
    
    # Check GPU memory again
    check_gpu_memory()
    
    print("\nTo kill a specific process, use: kill -9 <PID>")
    print("=" * 50)

if __name__ == "__main__":
    main()
