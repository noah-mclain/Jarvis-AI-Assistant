#!/bin/bash
# Script to run memory-efficient training for DeepSeek Coder model

# Clear GPU memory
echo "Clearing GPU memory..."
python -c "import torch; torch.cuda.empty_cache(); print('GPU memory cleared')"

# Set environment variables for optimal memory usage
export FORCE_CPU_ONLY_FOR_INITIAL_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:32,garbage_collection_threshold:0.8"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1

# Create a Python script for memory-efficient training
cat > memory_efficient_training.py << 'EOF'
#!/usr/bin/env python3
"""
Memory-efficient training script for DeepSeek Coder model.
This script uses a CPU-first loading approach and optimizes memory usage
to work within limited GPU memory constraints.
"""

import os
import sys
import torch
import gc
import argparse
from pathlib import Path

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

def main():
    """Run memory-efficient training for DeepSeek Coder model"""
    # Import after patching
    from src.generative_ai_module.train_models import main as train_main
    
    # Create a simple argument parser
    parser = argparse.ArgumentParser(description="Memory-efficient training for DeepSeek Coder")
    parser.add_argument('--model_type', type=str, default='code', help='Model type (code)')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=64, 
                        help='Number of gradient accumulation steps')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to use')
    parser.add_argument('--learning_rate', type=float, default=1.5e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--use_4bit', action='store_true', help='Use 4-bit quantization')
    parser.add_argument('--use_qlora', action='store_true', help='Use QLoRA')
    parser.add_argument('--skip_layer_freezing', action='store_true', help='Skip layer freezing')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Force memory-efficient settings
    args.use_4bit = True
    args.use_qlora = True
    args.batch_size = 1
    args.gradient_accumulation_steps = 64
    args.skip_layer_freezing = True
    
    # Run the training
    train_main(args)

if __name__ == "__main__":
    main()
EOF

# Make the script executable
chmod +x memory_efficient_training.py

# Run the memory-efficient training script
echo "Running memory-efficient training..."
python memory_efficient_training.py --model_type code --epochs 3 --max_length 512 --use_4bit --use_qlora --skip_layer_freezing

# Check the result
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed!"
fi
