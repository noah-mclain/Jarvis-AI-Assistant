#!/usr/bin/env python3
"""
DeepSeek-Coder NVIDIA GPU Training Script

This script runs the fine-tuning process for the DeepSeek-Coder model
with special settings optimized for NVIDIA GPUs. It handles quantization
and dataset loading issues automatically.

Usage:
    python train_nvidia.py

Requirements:
    - NVIDIA GPU with CUDA support
    - PyTorch with CUDA
"""

import sys
import os
import torch
import argparse
import time

# Import the fine-tuning functionality
from src.generative_ai_module.finetune_deepseek import main, parse_args, create_mini_dataset

def check_requirements():
    """Check if the system has the required NVIDIA GPU and libraries"""
    if not torch.cuda.is_available():
        print("ERROR: This script requires an NVIDIA GPU with CUDA support.")
        print("No CUDA device was detected.")
        sys.exit(1)
        
    try:
        import transformers
        print(f"Using transformers version: {transformers.__version__}")
    except ImportError:
        print("ERROR: Transformers library not found. Please install it:")
        print("pip install transformers")
        sys.exit(1)
    
    print(f"CUDA device detected: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    
    # Try to import necessary libraries for quantization
    try:
        import bitsandbytes
        print(f"BitsAndBytes version: {bitsandbytes.__version__}")
        quantization_available = True
    except ImportError:
        print("WARNING: BitsAndBytes not found. Will use fallback precision.")
        quantization_available = False
    
    return quantization_available

def setup_nvidia_training(quantization_available=True):
    """Set up training parameters for NVIDIA GPUs"""
    print("Optimizing settings for NVIDIA GPU...")
    
    # Use CUDA specific environment variables
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid warnings
    
    # Force garbage collection before starting
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    # Set memory-efficient defaults
    args = [
        "--epochs", "50",               # Standard epochs
        "--batch-size", "512",            # Reasonable batch size for most GPUs
        "--max-samples", "50000",        # Number of samples
        "--subset", "python",           # Python subset as fallback
        "--force-gpu",                  # Use GPU acceleration
    ]
    
    # Try to use mini dataset first to validate the pipeline
    use_mini = input("Do you want to use the mini dataset first to test the pipeline? (y/n): ").strip().lower()
    if use_mini == 'y':
        args.append("--use-mini-dataset")
    else:
        args.append("--all-subsets")
    
    # Add quantization if available
    if quantization_available:
        try_4bit = input("Do you want to try 4-bit quantization? (y/n): ").strip().lower()
        if try_4bit == 'y':
            args.append("--load-in-4bit")
        else:
            args.append("--load-in-8bit")
    
    # Set output directory to models/deepseek_finetuned in the root directory
    output_dir = os.path.join(os.path.dirname(__file__), "models", "deepseek_finetuned")
    args.extend(["--output-dir", output_dir])
    
    # Add verbose flag
    args.append("--verbose")
    
    # Extend sys.argv with the arguments
    sys.argv.extend(args)
    
    return output_dir

def parse_custom_args():
    """Parse command line arguments for NVIDIA training script"""
    parser = argparse.ArgumentParser(description="DeepSeek-Coder NVIDIA Training")
    parser.add_argument("--max-samples", type=int, default=5000, 
                       help="Maximum number of samples (default: 5000)")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size (default: 8)")
    
    args, remaining = parser.parse_known_args()
    
    # Remove processed args from sys.argv
    sys.argv = [sys.argv[0]] + remaining
    
    return args

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DeepSeek-Coder NVIDIA GPU Training")
    print("=" * 60)
    print("\nThis script runs training with settings optimized for NVIDIA GPUs")
    
    # Check for CUDA and required libraries
    quantization_available = check_requirements()
    
    # Get custom args first
    custom_args = parse_custom_args()
    
    # Set up NVIDIA training
    output_dir = setup_nvidia_training(quantization_available)
    
    # Override with user-provided values if specified
    if "--max-samples" not in sys.argv and custom_args.max_samples != 5000:
        sys.argv.extend(["--max-samples", str(custom_args.max_samples)])
        
    if "--epochs" not in sys.argv and custom_args.epochs != 50:
        sys.argv.extend(["--epochs", str(custom_args.epochs)])
        
    if "--batch-size" not in sys.argv and custom_args.batch_size != 8:
        sys.argv.extend(["--batch-size", str(custom_args.batch_size)])
    
    # Display configuration
    print("\nTraining with the following settings:")
    print(f"- GPU: {torch.cuda.get_device_name(0)}")
    print(f"- Max samples: {custom_args.max_samples}")
    print(f"- Epochs: {custom_args.epochs}")
    print(f"- Batch size: {custom_args.batch_size}")
    print(f"- Output directory: {output_dir}")
    
    # Ready to start
    print("\nStarting training with NVIDIA-optimized settings...")
    print("(This may take several hours depending on your GPU)")
    start_time = time.time()
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    # Run training
    try:
        args = parse_args()
        main(args)
        
        # Training success
        training_time = time.time() - start_time
        
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        seconds = int(training_time % 60)
        
        print("\n" + "=" * 60)
        print("NVIDIA GPU Training Completed Successfully!")
        print("=" * 60)
        print(f"\nTraining time: {hours}h {minutes}m {seconds}s")
        print(f"Model saved to: {output_dir}")
        
    except Exception as e:
        # Training failed
        print("\n" + "=" * 60)
        print("NVIDIA GPU Training Failed")
        print("=" * 60)
        print(f"\nError: {str(e)}")
        print("\nPossible solutions:")
        print("1. Try using the mini dataset first with --use-mini-dataset")
        print("2. Disable 4-bit quantization and use 8-bit instead")
        print("3. Reduce batch size (--batch-size 4)")
        print("4. Reduce max samples (--max-samples 1000)")
        print("5. Check your CUDA and PyTorch installations")
        
        import traceback
        traceback.print_exc() 