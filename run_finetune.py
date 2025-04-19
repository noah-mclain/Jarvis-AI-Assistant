#!/usr/bin/env python3
"""
DeepSeek-Coder Fine-tuning Runner

This script runs the fine-tuning process for the DeepSeek-Coder model
from the project root directory, avoiding any import issues.

Usage:
    python run_finetune.py 
    
Additional options:
    --max-samples=1000      # Limit the number of samples for memory constraints
    --batch-size=64         # Batch size for GPU training
    --load-in-4bit          # Use 4-bit quantization for extreme memory saving
    --subset=python         # Just use one language if memory is limited
    --all-subsets           # Only use specific subset if memory is limited
"""

import sys
import os
import torch

# Import the fine-tuning functionality
from src.generative_ai_module.finetune_deepseek import main, parse_args

def apply_memory_efficient_defaults():
    """Apply memory-efficient defaults for running on GPU"""
    # Detect if we're on Apple Silicon
    on_apple_silicon = hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

    # Set smaller batch size for Apple Silicon to avoid memory issues
    batch_size = "64"

    # Set appropriate command-line arguments based on the device
    args = [
        "--epochs", "50",                # Fewer epochs for full dataset
        "--batch-size", batch_size,      # Smaller batch size for Apple Silicon
        "--max-samples", "5000",         # More samples for better training quality
        "--all-subsets",                 # Use all language subsets (boolean flag)
        "--subset", "python",            # Python subset as fallback
        "--force-gpu"                    # Use GPU acceleration (boolean flag)
    ]

    # Add quantization flags - 4-bit works on CUDA but not on MPS (Apple Silicon)
    if not on_apple_silicon:
        args.append("--load-in-4bit")    # Use 4-bit quantization for NVIDIA GPUs

    # Extend sys.argv with the arguments
    sys.argv.extend(args)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments provided, use memory-efficient defaults
        print("\n=== Using memory-efficient defaults with GPU acceleration ===")
        print("Epochs: 50")
        
        # Detect if we're on Apple Silicon
        on_apple_silicon = hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        batch_size = 16 if on_apple_silicon else 64
        print(f"Batch size: {batch_size}")
        
        print("Max samples: 5000")
        if not on_apple_silicon:
            print("Using 4-bit quantization: Yes")
        print("Force GPU usage: Yes")
        print("Training only on python subset")
        print("=== To see all options, run with --help ===\n")
        apply_memory_efficient_defaults()
    
    # Parse command line arguments
    args = parse_args()
    
    # Run the fine-tuning main function with parsed arguments
    main(args)