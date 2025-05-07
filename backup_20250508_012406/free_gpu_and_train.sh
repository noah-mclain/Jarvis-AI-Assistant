#!/bin/bash
# Script to completely free GPU memory and run DeepSeek Coder training
# This script addresses the GPU memory issues with DeepSeek Coder model

echo "===== GPU Memory Cleanup and DeepSeek Coder Training ====="

# Step 1: Kill any running Python processes that might be using GPU
echo "Killing any running Python processes..."
pkill -9 python
sleep 2

# Step 2: Clean GPU memory without requiring sudo privileges
echo "Cleaning GPU memory (Paperspace-compatible)..."
if command -v nvidia-smi &> /dev/null; then
    # Get current GPU memory usage
    echo "Current GPU memory usage:"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits

    # Force Python garbage collection
    python -c "
import gc
import torch
import time

# Force garbage collection
gc.collect()

# Clear CUDA cache if available
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('CUDA cache cleared')

    # Set memory fraction to avoid OOM
    torch.cuda.set_per_process_memory_fraction(0.8)
    print('GPU memory fraction set to 80%')

    # Print memory stats
    print(f'Memory allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB')
    print(f'Memory reserved: {torch.cuda.memory_reserved() / (1024**3):.2f} GB')

    # Create and delete a dummy tensor to trigger memory cleanup
    dummy = torch.ones(1, device='cuda')
    del dummy
    torch.cuda.empty_cache()
    print('Additional cleanup performed')
"

    # Show GPU memory after cleanup
    echo "GPU memory usage after cleanup:"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
fi

# Step 3: Clear CUDA cache
echo "Clearing CUDA cache..."
python -c "import torch; torch.cuda.empty_cache(); print('✓ CUDA cache cleared')"

# Step 4: Set environment variables for optimal memory usage
echo "Setting environment variables for optimal memory usage..."
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:32,garbage_collection_threshold:0.9"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_HOME="/tmp/hf_cache"  # Use temporary directory for cache

# Step 5: Clear Hugging Face cache
echo "Clearing Hugging Face cache..."
rm -rf ~/.cache/huggingface/datasets .cache
mkdir -p /tmp/hf_cache

# Step 6: Start GPU monitoring
echo "Starting GPU monitoring..."
(watch -n 1 "nvidia-smi --query-gpu=memory.used --format=csv" &> gpu_memory_log.txt) &
MONITOR_PID=$!

# Display GPU info
echo "===================================================================="
echo "GPU Information:"
nvidia-smi
echo "===================================================================="

# Step 7: Create a Paperspace-compatible CPU-first loading script
echo "Creating Paperspace-compatible CPU-first loading script..."
cat > paperspace_train.py << 'EOF'
#!/usr/bin/env python3
"""
Paperspace-compatible CPU-first loading script for DeepSeek Coder training.
This script is designed to work within Paperspace's permission constraints.
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
    print(f"Initial GPU memory: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")

# Set environment variables for optimal memory usage
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,garbage_collection_threshold:0.9"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/tmp/hf_cache"  # Use temporary directory for cache

# Add the current directory to the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import the actual training module
from src.generative_ai_module.train_models import main

if __name__ == "__main__":
    # Run the main function with hardcoded optimal parameters for Paperspace
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

    # Run the main function
    main()
EOF

# Make the script executable
chmod +x paperspace_train.py

# Step 8: Run the training with our Paperspace-compatible script
echo "Starting DeepSeek Coder training with Paperspace-compatible CPU-first approach..."
python paperspace_train.py

# Kill the background monitoring process
kill $MONITOR_PID

echo "Training complete!"
echo "Check gpu_memory_log.txt for memory usage during training."

# Memory usage analysis
echo "Peak GPU memory usage:"
grep -o "[0-9]\+\.[0-9]\+ MiB" gpu_memory_log.txt | sort -nr | head -1

echo "✓ Done"
