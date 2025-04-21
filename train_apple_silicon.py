#!/usr/bin/env python3
"""
DeepSeek-Coder Apple Silicon GPU Training Script

This script runs the fine-tuning process for the DeepSeek-Coder model
with special memory-optimized settings for Apple Silicon Macs (M1/M2/M3).
It uses reduced settings to ensure GPU training can complete successfully.

Usage:
    python train_apple_silicon.py

Requirements:
    - Apple Silicon Mac with macOS 12.3+
    - PyTorch 2.0+ with MPS support
"""

import sys
import os
import torch
import argparse
import time

# Import the fine-tuning functionality
from src.generative_ai_module.finetune_deepseek import main, parse_args

def optimize_memory():
    """Set environment variables to optimize memory usage on Apple Silicon"""
    # Check if we're on Apple Silicon
    if not (hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        print("ERROR: This script is specifically for Apple Silicon Macs.")
        print("Your device does not have MPS (Metal Performance Shaders) support.")
        print("Please use the regular run_finetune.py script instead.")
        sys.exit(1)
        
    print("Optimizing memory settings for Apple Silicon...")
    
    # Optimize PyTorch MPS memory usage
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    os.environ["PYTORCH_MPS_ALLOCATOR_MEMPROFILE"] = "1"
    os.environ["PYTORCH_MPS_ACTIVE_MEMORY_MANAGER"] = "1"
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # Clear MPS cache if available
    if hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
        
    print("Memory optimization complete.")

def setup_apple_silicon_training():
    """Set up memory-efficient training parameters for Apple Silicon"""
    # Optimize memory first
    optimize_memory()
    # Use extremely minimal settings for training
    args = [
        "--epochs", "3",                 # Just 3 epochs for Apple Silicon
        "--batch-size", "1",             # Smallest possible batch size
        "--max-samples", "50",           # Very limited samples
        "--use-mini-dataset",            # Use tiny dataset
        "--subset", "python",            # Just Python code
        "--force-gpu",                   # Use GPU (MPS)
        "--load-in-8bit",                # Use 8-bit quantization
        "--sequence-length", "256",      # Shorter sequences to save memory
    ]
    
    # Set output directory to models/deepseek_finetuned in the root directory
    output_dir = os.path.join(os.path.dirname(__file__), "models", "deepseek_finetuned")
    args.extend(["--output-dir", output_dir])
    
    # Add verbose flag
    args.append("--verbose")
    
    # Extend sys.argv with the arguments
    sys.argv.extend(args)
    
    return output_dir

def parse_custom_args():
    """Parse command line arguments for Apple Silicon training script"""
    parser = argparse.ArgumentParser(description="DeepSeek-Coder Apple Silicon Training")
    parser.add_argument("--max-samples", type=int, default=50, 
                       help="Maximum number of samples (default: 50)")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs (default: 3)")
    
    args, remaining = parser.parse_known_args()
    
    # Remove processed args from sys.argv
    sys.argv = [sys.argv[0]] + remaining
    
    return args

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DeepSeek-Coder Apple Silicon GPU Training")
    print("=" * 60)
    print("\nThis script runs training with memory-optimized settings for")
    print("Apple Silicon Macs with MPS acceleration.")
    
    # Check for MPS support
    if not (hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        print("\nERROR: This script is specifically for Apple Silicon Macs.")
        print("Your device does not have MPS (Metal Performance Shaders) support.")
        print("Please use the regular run_finetune.py script instead.")
        sys.exit(1)
    
    # Get custom args first
    custom_args = parse_custom_args()
    
    # Set up Apple Silicon training with minimal parameters
    output_dir = setup_apple_silicon_training()
    
    # Override with user-provided values if specified
    if "--max-samples" not in sys.argv and custom_args.max_samples != 50:
        sys.argv.extend(["--max-samples", str(custom_args.max_samples)])
        
    if "--epochs" not in sys.argv and custom_args.epochs != 3:
        sys.argv.extend(["--epochs", str(custom_args.epochs)])
    
    print("\nTraining with the following settings:")
    print(f"- GPU: Yes (Apple Silicon MPS)")
    print(f"- Memory optimization: Yes (special settings for M1/M2/M3)")
    print(f"- Mini dataset: Yes")
    print(f"- Max samples: {custom_args.max_samples}")
    print(f"- Epochs: {custom_args.epochs}")
    print(f"- Batch size: 1 (enforced for memory efficiency)")
    print(f"- Output directory: {output_dir}")
    
    print("\nStarting training with memory-optimized settings...")
    start_time = time.time()
    
    # Final memory optimization before starting
    if hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
    
    # Parse args and run training
    try:
        args = parse_args()
        main(args)
        
        # Training success
        training_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("Apple Silicon GPU Training Completed Successfully!")
        print("=" * 60)
        print(f"\nTraining time: {training_time:.2f} seconds")
        print(f"Model saved to: {output_dir}")
        
    except Exception as e:
        # Training failed
        print("\n" + "=" * 60)
        print("Apple Silicon GPU Training Failed")
        print("=" * 60)
        print(f"\nError: {str(e)}")
        print("\nPossible solutions:")
        print("1. Try reducing max-samples further (--max-samples 25)")
        print("2. Try using CPU mode instead (run python train_cpu_only.py)")
        print("3. Make sure no other memory-intensive apps are running")
        print("4. Restart your computer to free up memory")
        import traceback
        traceback.print_exc() 