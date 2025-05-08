#!/bin/bash
# Script to run the patched training with CPU-first loading and GPU training

# Clear GPU memory
echo "Clearing GPU memory..."
python -c "import torch; torch.cuda.empty_cache(); print('GPU memory cleared')"

# Set environment variables
export FORCE_CPU_ONLY_FOR_INITIAL_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:32,garbage_collection_threshold:0.9"
export TOKENIZERS_PARALLELISM=false

# Run the patched training script
echo "Running patched training..."
python run_patched_training.py

# Check the result
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed!"
fi
