#!/bin/bash
# Script to test the CPU-first loading and GPU training transition

# Clear GPU memory
echo "Clearing GPU memory..."
python -c "import torch; torch.cuda.empty_cache(); print('GPU memory cleared')"

# Set environment variables
export FORCE_CPU_ONLY_FOR_INITIAL_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:32,garbage_collection_threshold:0.9"

# Run the test script
echo "Running GPU training test..."
python test_gpu_training.py

# Check the result
if [ $? -eq 0 ]; then
    echo "Test completed successfully!"
else
    echo "Test failed!"
fi
