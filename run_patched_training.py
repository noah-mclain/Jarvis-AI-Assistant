#!/usr/bin/env python3
"""
Patched training script for DeepSeek Coder model.
This script applies the necessary patches to fix device mismatch errors
and runs a small training job to verify the fixes.
"""

import os
import sys
import torch
import gc
import argparse
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Force CPU for initial model loading
os.environ["FORCE_CPU_ONLY_FOR_INITIAL_LOAD"] = "1"

# Set PyTorch to use CPU as default device initially
if hasattr(torch, 'set_default_device'):
    torch.set_default_device('cpu')
torch.set_default_tensor_type('torch.FloatTensor')

# Clear any existing CUDA memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

# Set environment variables for optimal memory usage
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,garbage_collection_threshold:0.9"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Patch the problematic function in transformers library
def patch_transformers_attention_mask():
    """
    Patch the problematic function in transformers library that's causing device mismatch.
    This specifically targets the _unmask_unattended function that's using .cpu()
    """
    try:
        import transformers.modeling_attn_mask_utils as attn_utils
        
        # Store the original function
        original_unmask_unattended = attn_utils.AttentionMaskConverter._unmask_unattended
        
        # Define our patched version that doesn't use .cpu()
        def patched_unmask_unattended(attention_mask, unmasked_value=0.0):
            """Patched version that doesn't force CPU conversion"""
            # Get the device of the attention mask
            device = attention_mask.device
            
            # Create a temporary tensor on the same device
            tmp = torch.ones_like(attention_mask) * unmasked_value
            
            # Use argmax without forcing CPU
            indices = torch.argmax(attention_mask * tmp, 1, keepdim=True)
            
            # Create a range tensor on the same device
            range_tensor = torch.arange(attention_mask.shape[1], device=device).expand_as(attention_mask)
            
            # Create the expanded mask on the same device
            expanded_mask = (range_tensor <= indices).to(attention_mask.dtype)
            
            return expanded_mask
        
        # Apply the patch
        attn_utils.AttentionMaskConverter._unmask_unattended = patched_unmask_unattended
        print("Successfully patched transformers attention mask function")
        
    except Exception as e:
        print(f"Error patching transformers attention mask function: {e}")
        print("Will continue without patching")

# Apply the patch
patch_transformers_attention_mask()

def run_training():
    """Run a small training job to verify the fixes"""
    from src.generative_ai_module.code_generator import CodeGenerator
    from src.generative_ai_module.code_preprocessing import load_and_preprocess_dataset
    
    print("\n===== Running Patched Training =====")
    
    # Check for GPU
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("No GPU available. This test requires a GPU.")
        return
    
    # Create output directory
    output_dir = "models/patched_training"
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
    
    print("\n===== Training completed =====")

if __name__ == "__main__":
    run_training()
