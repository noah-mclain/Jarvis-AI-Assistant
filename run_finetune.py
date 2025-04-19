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
    
    # Set batch size based on hardware
    if on_apple_silicon:
        # Use smaller batch size for Apple Silicon
        batch_size = "2"
        max_samples = "100"  # Limit samples for Apple Silicon
        epochs = "3"         # Fewer epochs for quicker training
    else:
        # Larger batch size for NVIDIA GPUs
        batch_size = "8"
        max_samples = "5000" # More samples for NVIDIA GPUs
        epochs = "50"        # More epochs for better training
    
    print(f"Applying memory-efficient defaults for {'Apple Silicon' if on_apple_silicon else 'NVIDIA GPU'}")
    print(f"Batch size: {batch_size}, Max samples: {max_samples}, Epochs: {epochs}")
    
    # Set output directory to models/deepseek_finetuned in the root directory
    output_dir = os.path.join(os.path.dirname(__file__), "models", "deepseek_finetuned")
    
    # Set appropriate command-line arguments based on the device
    args = [
        "--epochs", epochs,              # Training epochs
        "--batch-size", batch_size,      # Batch size
        "--max-samples", max_samples,    # Number of samples for training
        "--output-dir", output_dir,      # Output directory in the root/models folder
    ]
    
    # Add boolean flags as standalone arguments
    if on_apple_silicon:
        # For Apple Silicon, use mini dataset for testing and special memory settings
        args.append("--use-mini-dataset")
        args.append("--force-gpu")       # Force GPU usage on Apple Silicon 
        print("Using memory-efficient GPU settings for Apple Silicon")
    else:
        # For NVIDIA GPUs, use all language subsets
        args.append("--all-subsets")
        args.append("--force-gpu")       # Use GPU acceleration for NVIDIA GPUs
        
    args.extend(["--subset", "python"])  # Python subset as fallback
    
    # Add quantization flags - 4-bit works on CUDA but not on MPS (Apple Silicon)
    if not on_apple_silicon:
        args.append("--load-in-4bit")    # Use 4-bit quantization for NVIDIA GPUs
    else:
        args.append("--load-in-8bit")    # Use 8-bit quantization for Apple Silicon
    
    # Extend sys.argv with the arguments
    sys.argv.extend(args)

def train_deepseek_and_text_models():
    """Run both DeepSeek fine-tuning and text model training"""
    # First run the DeepSeek fine-tuning
    apply_memory_efficient_defaults()
    args = parse_args()
    main(args)
    
    # Then run the unified pipeline to train both text models
    pipeline_script = os.path.join(os.path.dirname(__file__), 
                                  "src", "generative_ai_module", 
                                  "unified_generation_pipeline.py")
    
    if os.path.exists(pipeline_script):
        print("\n\n===== Now training text generation models =====")
        
        # Detect if we're on Apple Silicon
        on_apple_silicon = hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
        # Configure appropriate parameters based on hardware
        if on_apple_silicon:
            # Smaller settings for Apple Silicon
            max_samples = "100"
            epochs = "3"
            print(f"Configuring text model training for Apple Silicon (samples: {max_samples}, epochs: {epochs})")
            print("NOTE: Text models will still use GPU as they are small enough for Apple Silicon")
        else:
            # Larger settings for NVIDIA GPUs
            max_samples = "5000"
            epochs = "50"
        
        # Set output directory in the root models folder
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        
        # Construct command to run the unified pipeline
        cmd = [sys.executable, pipeline_script, 
               "--mode", "train",
               "--train-type", "both",
               "--dataset", "all",
               "--max-samples", max_samples,
               "--epochs", epochs,
               "--output-dir", model_dir,  # Specify output directory
               "--save-model",   # Boolean flag as standalone argument
               "--force-gpu"]    # Still use GPU for text models
        
        # Call the script using the same interpreter
        import subprocess
        subprocess.run(cmd)
    else:
        print(f"Error: Could not find unified pipeline script at {pipeline_script}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments provided, use memory-efficient defaults
        # Detect if we're on Apple Silicon
        on_apple_silicon = hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
        print("\n=== Using memory-efficient defaults with hardware-appropriate settings ===")
        if on_apple_silicon:
            print("Apple Silicon GPU detected")
            print("Epochs: 3 (reduced for Apple Silicon)")
            print("Batch size: 2 (small batch for memory constraints)")
            print("Max samples: 100 (limited samples for faster training)")
            print("Using mini dataset: Yes (reduced dataset for testing)")
            print("Using GPU: Yes (with memory-optimized settings)")
            print("NOTE: DeepSeek-Coder is a very large model. Training will use memory-optimized settings.")
        else:
            print("NVIDIA GPU detected")
            print("Epochs: 50")
            print("Batch size: 8")
            print("Max samples: 5000")
            if not on_apple_silicon:
                print("Using 4-bit quantization: Yes (NVIDIA GPU only)")
                
        print("Training on python subset")
        print("=== To see all options, run with --help ===\n")
        
        # Check if the user wants to train all models
        train_all = input("Do you want to train both DeepSeek code model and text models? (y/n): ").strip().lower()
        
        if train_all == 'y':
            train_deepseek_and_text_models()
        else:
            # Just train DeepSeek
            apply_memory_efficient_defaults()
            args = parse_args()
            main(args)
    else:
        # Parse command line arguments and run just DeepSeek fine-tuning
        args = parse_args()
        main(args)