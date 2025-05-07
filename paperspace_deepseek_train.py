#!/usr/bin/env python3
"""
Paperspace-compatible CPU-first loading script for DeepSeek Coder training.
This script is designed to work within Paperspace's permission constraints.

Usage:
    python paperspace_deepseek_train.py

This script will:
1. Free GPU memory as much as possible without requiring sudo privileges
2. Load the DeepSeek Coder model on CPU first
3. Train the model with optimal settings for Paperspace
"""

import os
import sys
import torch
import gc
import time
import subprocess
import argparse
from pathlib import Path

def clean_gpu_memory():
    """Clean GPU memory without requiring sudo privileges"""
    print("Cleaning GPU memory (Paperspace-compatible)...")
    
    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        # Get current GPU memory usage
        print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
        print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")
        
        # Clear cache
        torch.cuda.empty_cache()
        print('CUDA cache cleared')
        
        # Set memory fraction to avoid OOM
        torch.cuda.set_per_process_memory_fraction(0.8)
        print('GPU memory fraction set to 80%')
        
        # Create and delete a dummy tensor to trigger memory cleanup
        try:
            dummy = torch.ones(1, device='cuda')
            del dummy
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error creating dummy tensor: {e}")
        
        # Print memory stats after cleanup
        print(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / (1024**3):.2f} GB allocated")
        print(f"GPU memory reserved after cleanup: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")
        
        # Try to run nvidia-smi to get more detailed info
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            print("\nNVIDIA-SMI output:")
            print(result.stdout)
        except Exception as e:
            print(f"Error running nvidia-smi: {e}")

def setup_environment():
    """Set up environment variables for optimal memory usage"""
    print("Setting up environment for optimal memory usage...")
    
    # Force CPU for initial model loading
    os.environ["FORCE_CPU_ONLY_FOR_INITIAL_LOAD"] = "1"
    
    # Set PyTorch to use CPU as default device initially
    if hasattr(torch, 'set_default_device'):
        torch.set_default_device('cpu')
    torch.set_default_tensor_type('torch.FloatTensor')
    
    # Set environment variables for optimal memory usage
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,garbage_collection_threshold:0.9"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HOME"] = "/tmp/hf_cache"  # Use temporary directory for cache
    
    # Add the current directory to the path
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    
    print("Environment setup complete")

def start_gpu_monitoring():
    """Start GPU monitoring in a separate process"""
    print("Starting GPU monitoring...")
    try:
        # Create a log file for GPU monitoring
        log_file = open("gpu_memory_log.txt", "w")
        
        # Start nvidia-smi in a separate process
        process = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=timestamp,memory.used,memory.total,utilization.gpu", 
             "--format=csv", "-l", "5"],  # Log every 5 seconds
            stdout=log_file,
            stderr=subprocess.STDOUT
        )
        
        print(f"GPU monitoring started with PID {process.pid}")
        return process, log_file
    except Exception as e:
        print(f"Error starting GPU monitoring: {e}")
        return None, None

def main():
    """Main function to run the training with CPU-first loading"""
    print("\n===== DeepSeek Coder Training with CPU-First Loading =====")
    
    # Step 1: Clean GPU memory
    clean_gpu_memory()
    
    # Step 2: Set up environment
    setup_environment()
    
    # Step 3: Start GPU monitoring
    monitor_process, log_file = start_gpu_monitoring()
    
    try:
        # Step 4: Import the training module
        print("Importing training module...")
        from src.generative_ai_module.train_models import main as train_main
        
        # Step 5: Set up command-line arguments
        print("Setting up training arguments...")
        sys.argv = [
            sys.argv[0],
            "--model_type", "code",
            "--model_name_or_path", "deepseek-ai/deepseek-coder-5.7b-instruct",
            "--dataset", "codeparrot/github-code:0.7,code-search-net/code_search_net:0.3",
            "--batch_size", "1",
            "--max_length", "512",
            "--gradient_accumulation_steps", "64",
            "--use_4bit",
            "--use_qlora",
            "--use_flash_attention_2",
            "--gradient_checkpointing",
            "--optim", "adamw_bnb_8bit",
            "--learning_rate", "1.5e-5",
            "--weight_decay", "0.05",
            "--bf16",
            "--num_workers", "4",
            "--cache_dir", ".cache",
            "--force_gpu",
            "--pad_token_id", "50256",
            "--dataset_subset", "python,javascript",
            "--fim_rate", "0.6",
            "--evaluation_strategy", "steps",
            "--eval_steps", "500",
            "--save_steps", "1000",
            "--logging_steps", "50"
        ]
        
        # Step 6: Run the training
        print("\n===== Starting DeepSeek Coder Training =====")
        train_main()
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Step 7: Clean up
        if monitor_process:
            print("Stopping GPU monitoring...")
            monitor_process.terminate()
        
        if log_file:
            log_file.close()
        
        print("\n===== Training Complete =====")
        
        # Print peak GPU memory usage
        try:
            with open("gpu_memory_log.txt", "r") as f:
                lines = f.readlines()
                memory_values = []
                for line in lines:
                    if "MiB" in line:
                        try:
                            memory_part = line.split(",")[1].strip()
                            if memory_part.endswith("MiB"):
                                memory_values.append(float(memory_part[:-3]))
                        except:
                            pass
                
                if memory_values:
                    peak_memory = max(memory_values)
                    print(f"Peak GPU memory usage: {peak_memory} MiB")
        except Exception as e:
            print(f"Error analyzing GPU memory log: {e}")

if __name__ == "__main__":
    main()
