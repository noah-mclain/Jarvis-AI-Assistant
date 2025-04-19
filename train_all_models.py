#!/usr/bin/env python3
"""
Unified Training Script for Jarvis AI Assistant

This script provides a convenient way to train all models:
1. DeepSeek-Coder model for code generation
2. Text generation models for persona_chat and writing_prompts datasets

It ensures all training uses the GPU efficiently and handles arguments correctly.
"""

import os
import sys
import subprocess
import torch
import argparse

def install_dependencies():
    """Install required dependencies for training"""
    print("Checking and installing required dependencies...")
    
    # Try to import tensorboard, install if not available
    try:
        import tensorboard
        print("TensorBoard is already installed.")
    except ImportError:
        print("Installing TensorBoard...")
        subprocess.run([sys.executable, "-m", "pip", "install", "tensorboard"])
        print("TensorBoard installed successfully.")

def train_deepseek_coder(use_mini_dataset=False, max_samples=5000, batch_size=64, epochs=50):
    """Train the DeepSeek-Coder model for code generation"""
    print("\n===== Training DeepSeek-Coder Model =====")
    
    # On Apple Silicon (M1/M2/M3)
    on_apple_silicon = hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    
    # Common arguments
    args = [
        sys.executable,  # Use the current Python interpreter
        "src/generative_ai_module/finetune_deepseek.py",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--max-samples", str(max_samples),
        "--all-subsets",  # Boolean flag (no value)
        "--force-gpu",    # Boolean flag (no value)
    ]
    
    # Use mini dataset for faster testing if requested
    if use_mini_dataset:
        args.append("--use-mini-dataset")
        print("Using mini dataset for quick testing")
    
    # Add quantization for NVIDIA GPUs only
    if not on_apple_silicon and torch.cuda.is_available():
        args.append("--load-in-4bit")

    # Run the training process
    print("Running DeepSeek-Coder training with command:")
    print(" ".join(args))
    result = subprocess.run(args)
    
    if result.returncode != 0:
        print("ERROR: DeepSeek-Coder training failed!")
        return False
    
    return True

def train_text_models(max_samples=100, batch_size=64, epochs=50):
    """Train text generation models for persona_chat and writing_prompts"""
    print("\n===== Training Text Generation Models =====")
    
    args = [
        sys.executable,  # Use the current Python interpreter
        "src/generative_ai_module/unified_generation_pipeline.py",
        "--mode", "train",
        "--train-type", "both",
        "--dataset", "both",
        "--save-model",
        "--force-gpu",
        "--max-samples", str(max_samples),
        "--epochs", str(epochs)
    ]
    
    # Run the training process
    print("Running text model training with command:")
    print(" ".join(args))
    result = subprocess.run(args)
    
    if result.returncode != 0:
        print("ERROR: Text model training failed!")
        return False
    
    return True

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train all models for Jarvis AI Assistant")
    parser.add_argument("--use-mini-dataset", action="store_true", help="Use mini dataset for quick testing")
    parser.add_argument("--max-samples", type=int, default=5000, help="Maximum number of samples to use")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training (for Apple Silicon)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--text-models-only", action="store_true", help="Train only the text models")
    parser.add_argument("--code-model-only", action="store_true", help="Train only the code model")
    return parser.parse_args()

def main():
    """Run the complete training pipeline"""
    # Parse command line arguments
    args = parse_args()
    
    print("==================================================")
    print("Jarvis AI Assistant - Training All Models")
    print("==================================================")
    
    # Customize training based on args
    if args.use_mini_dataset:
        print("\nUsing mini dataset for quick testing")
        max_samples = 50
        epochs = 3
    else:
        max_samples = args.max_samples
        epochs = args.epochs
    
    # Adjust batch size based on GPU
    batch_size = args.batch_size
    
    # Show what we're training
    if args.text_models_only:
        print("\nTraining text models only")
    elif args.code_model_only:
        print("\nTraining code model only")
    else:
        print("\nTraining both code and text models")
        
    print(f"- Max samples: {max_samples}")
    print(f"- Batch size: {batch_size}")
    print(f"- Epochs: {epochs}")
    
    print("\nThe training will use your GPU if available.")
    
    # Install dependencies
    install_dependencies()
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"\nNVIDIA GPU detected: {torch.cuda.get_device_name(0)}")
        print("Using CUDA for training")
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("\nApple Silicon GPU detected")
        print("Using MPS for training")
    else:
        print("\nWARNING: No GPU detected. Training will be very slow on CPU.")
        proceed = input("Do you want to continue anyway? (y/n): ").strip().lower()
        if proceed != 'y':
            print("Training cancelled.")
            return
    
    # Start the training process
    print("\nStarting training pipeline...")
    
    # Train code model if requested
    deepseek_success = True
    if not args.text_models_only:
        deepseek_success = train_deepseek_coder(
            use_mini_dataset=args.use_mini_dataset,
            max_samples=max_samples,
            batch_size=batch_size,
            epochs=epochs
        )
    
    # Train text models if requested
    text_success = True
    if not args.code_model_only:
        # Set smaller max_samples for text models (they're typically smaller)
        text_max_samples = max(100, max_samples // 10)
        text_success = train_text_models(
            max_samples=text_max_samples,
            batch_size=batch_size,
            epochs=epochs
        )
    
    # Final status
    print("\n==================================================")
    print("Training Pipeline Completed")
    print("==================================================")
    
    if not args.text_models_only:
        print(f"DeepSeek-Coder: {'SUCCESS' if deepseek_success else 'FAILED'}")
    
    if not args.code_model_only:
        print(f"Text Models: {'SUCCESS' if text_success else 'FAILED'}")
    
    print("\nYou can now use the trained models in your Jarvis AI Assistant!")

if __name__ == "__main__":
    main() 