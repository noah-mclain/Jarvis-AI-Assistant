#!/usr/bin/env python3
"""
Test script to verify GPU training with the fixed device handling.
This script will:
1. Load a small dataset
2. Initialize the model on CPU first
3. Move to GPU before training
4. Verify all tensors are on the same device during training
"""

import os
import sys
import torch
import gc
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Force CPU for initial model loading
os.environ["FORCE_CPU_ONLY_FOR_INITIAL_LOAD"] = "1"

# Import after setting environment variables
from src.generative_ai_module.code_generator import CodeGenerator
from src.generative_ai_module.code_preprocessing import load_and_preprocess_dataset

def test_gpu_training():
    """Test GPU training with the fixed device handling"""
    print("\n===== Testing GPU Training with Fixed Device Handling =====")
    
    # Check for GPU
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("No GPU available. This test requires a GPU.")
        return
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Create output directory
    output_dir = "models/test_gpu_training"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load a small dataset
    print("\nLoading a small dataset...")
    train_dataset, eval_dataset = load_and_preprocess_dataset(
        max_samples=100,  # Very small for testing
        sequence_length=128,  # Short sequences for testing
        subset="python",
        all_subsets=False
    )
    
    print(f"Loaded {len(train_dataset)} training samples and {len(eval_dataset)} evaluation samples")
    
    # Initialize the model on CPU first
    print("\nInitializing model on CPU first...")
    code_gen = CodeGenerator(
        use_deepseek=True,
        load_in_4bit=True,
        force_gpu=False  # Will be moved to GPU later
    )
    
    # Verify initial device
    initial_device = next(code_gen.model.parameters()).device
    print(f"Initial model device: {initial_device}")
    
    # Fine-tune with very short training
    print("\nStarting fine-tuning with GPU transition...")
    training_metrics = code_gen.fine_tune_deepseek(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=output_dir,
        epochs=1,  # Just one epoch for testing
        batch_size=2,
        sequence_length=128,
        learning_rate=2e-5,
        warmup_steps=10,
        skip_layer_freezing=True  # Skip for faster testing
    )
    
    # Check final device
    final_device = next(code_gen.model.parameters()).device
    print(f"Final model device: {final_device}")
    
    # Print training metrics
    print("\nTraining metrics:")
    for key, value in training_metrics.items():
        if not isinstance(value, (dict, list)):
            print(f"  {key}: {value}")
    
    print("\n===== Test completed =====")

if __name__ == "__main__":
    test_gpu_training()
