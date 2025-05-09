#!/bin/bash
# Fixed training script for Jarvis AI Assistant on RTX 5000 GPU

echo "===== Jarvis AI Assistant RTX 5000 Optimized Training ====="
echo "Setting up environment for optimal GPU utilization..."

# Step 1: Clear CUDA cache
echo "Clearing CUDA cache..."
python -c "import torch; torch.cuda.empty_cache(); print('✓ CUDA cache cleared')"

# Step 2: Set environment variables to prevent memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
echo "✓ Set PYTORCH_CUDA_ALLOC_CONF to prevent memory fragmentation"

# Enable benchmarking for faster training
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDNN_V8_API_ENABLED=1
export TORCH_CUDNN_BENCHMARK=1
echo "✓ Enabled CUDNN benchmarking for faster training"

# Step 3: Monitor GPU usage in the background
echo "Starting GPU monitoring..."
(watch -n 1 "nvidia-smi --query-gpu=memory.used --format=csv" &> gpu_memory_log.txt) &
MONITOR_PID=$!

# Display GPU info
echo "===================================================================="
echo "GPU Information:"
nvidia-smi
echo "===================================================================="

# Clean GPU memory
echo "GPU memory has been cleared."
echo "=================================================="
python -c "import gc
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    print(f'Initial GPU memory: {torch.cuda.memory_allocated() / (1024**3):.2f} GB')"

# Set environment variables for optimal memory usage
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:32,garbage_collection_threshold:0.9"
export TOKENIZERS_PARALLELISM=false
export FORCE_CPU_ONLY_FOR_INITIAL_LOAD=1

# Run the fixed deepseek training script
echo "Starting optimized CODE model training with DeepSeek Coder 5.7B..."
chmod +x run_deepseek_training.sh
./run_deepseek_training.sh

# Kill the background monitoring process
kill $MONITOR_PID

echo "Training complete!"
echo "Check gpu_memory_log.txt for memory usage during training."

# Memory usage analysis
echo "Peak GPU memory usage:"
grep -o "[0-9]\+\.[0-9]\+ MiB" gpu_memory_log.txt | sort -nr | head -1

echo "✓ Done"
