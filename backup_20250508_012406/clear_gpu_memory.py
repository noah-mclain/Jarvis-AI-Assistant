#!/usr/bin/env python3
"""
Script to clear GPU memory and provide information about GPU usage.
This script will:
1. Clear CUDA cache
2. Force garbage collection
3. Print GPU memory information
4. Attempt to identify processes using GPU memory
"""

import os
import gc
import sys
import subprocess
import time

def clear_gpu_memory():
    """Clear GPU memory by emptying cache and forcing garbage collection"""
    try:
        import torch
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
    except ImportError:
        print("PyTorch is not installed. Cannot clear GPU memory.")
        return False
    except Exception as e:
        print(f"Error clearing GPU memory: {e}")
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
                print("\nProcesses using GPU memory:")
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
                print("\nRunning Python processes (may or may not be using GPU):")
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
    """Main function to clear GPU memory and provide information"""
    print("=" * 50)
    print("GPU MEMORY CLEANUP UTILITY")
    print("=" * 50)
    
    # Clear GPU memory
    cleared = clear_gpu_memory()
    
    # Get processes using GPU
    got_processes = get_gpu_processes()
    
    if cleared:
        print("\nGPU memory has been cleared.")
        print("If you're still experiencing issues, you may need to:")
        print("1. Restart your Python environment")
        print("2. Use smaller batch sizes or model sizes")
        print("3. Enable memory-efficient training options like gradient checkpointing")
        print("4. Use 4-bit quantization (--use_4bit flag)")
    else:
        print("\nCould not clear GPU memory.")
    
    print("\nTo kill a specific process, use: kill -9 <PID>")
    print("=" * 50)

if __name__ == "__main__":
    main()
