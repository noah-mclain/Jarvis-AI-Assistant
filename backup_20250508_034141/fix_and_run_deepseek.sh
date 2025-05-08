#!/bin/bash
# Fix and run DeepSeek training script
# This script applies the fix for the max_seq_length parameter issue
# and then runs the unified_deepseek_training.py script

# Set default values
GPU_TYPE="A6000"  # Default to A6000 with 50 GiB VRAM
VRAM_SIZE=50      # Default to 50 GiB

# Print banner
echo "===== Jarvis AI Assistant - Fix and Run DeepSeek Training ====="
echo "GPU Type: $GPU_TYPE"
echo "VRAM Size: $VRAM_SIZE GiB"
echo "========================================"

# Step 1: Clear CUDA cache
echo "Clearing CUDA cache..."
python -c "
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print('✓ CUDA cache cleared')
    else:
        print('✓ No CUDA device available, skipping cache clearing')
except ImportError:
    print('⚠️ PyTorch not installed, skipping cache clearing')
"

# Step 2: Set environment variables to prevent memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.8"
echo "✓ Set PYTORCH_CUDA_ALLOC_CONF to prevent memory fragmentation"

# Enable benchmarking for faster training
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDNN_V8_API_ENABLED=1
export TORCH_CUDNN_BENCHMARK=1
echo "✓ Enabled CUDNN benchmarking for faster training"

# Step 3: Monitor GPU usage in the background
echo "Starting GPU monitoring..."
python gpu_utils.py monitor --interval 5 --log-file gpu_memory_log.txt &
MONITOR_PID=$!

# Display GPU info
echo "===================================================================="
echo "GPU Information:"
nvidia-smi
echo "===================================================================="

# Apply the fix for the max_seq_length parameter issue
echo "Applying fix for the max_seq_length parameter issue..."
python fix_unsloth_deepseek.py

# Check if the fix was successful
if [ $? -ne 0 ]; then
    echo "❌ Failed to apply the fix. Exiting."
    kill $MONITOR_PID
    exit 1
fi

# Set environment variables for optimal memory usage
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:32,garbage_collection_threshold:0.8"
export TOKENIZERS_PARALLELISM=false
export FORCE_CPU_ONLY_FOR_INITIAL_LOAD=1
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1
export FORCE_CPU_ONLY_FOR_TOKENIZATION=1
export FORCE_CPU_ONLY_FOR_DATASET_PROCESSING=1
export TRANSFORMERS_OFFLINE=0
export TOKENIZERS_FORCE_CPU=1
export HF_DATASETS_CPU_ONLY=1
export JARVIS_FORCE_CPU_TOKENIZER=1

# Set BF16 capability - always enabled for high-VRAM setups
echo "High-VRAM setup detected ($VRAM_SIZE GiB) - enabling BF16 mixed precision"
BF16_FLAG="--bf16"

# Ensure directories exist
echo "Ensuring directories exist..."
# Check if we're in Paperspace environment
if [ -d "/notebooks" ]; then
    # Paperspace environment
    BASE_DIR="/notebooks/Jarvis_AI_Assistant"
else
    # Local environment
    BASE_DIR="./Jarvis_AI_Assistant"
fi

mkdir -p "$BASE_DIR/models"
mkdir -p "$BASE_DIR/metrics"
mkdir -p "$BASE_DIR/checkpoints"
mkdir -p "$BASE_DIR/logs"
mkdir -p "$BASE_DIR/evaluation_metrics"
mkdir -p "$BASE_DIR/preprocessed_data"
mkdir -p "$BASE_DIR/visualization"

# Set output directory variable for later use
OUTPUT_DIR="$BASE_DIR"

# Adjust batch size for unified training
BATCH_SIZE=4
MAX_LENGTH=1024
GRAD_ACCUM=16
NUM_WORKERS=4

# Run the unified training script
echo "Starting unified training..."
python unified_deepseek_training.py \
    --model_name "deepseek-ai/deepseek-coder-6.7b-instruct" \
    --output_dir "$OUTPUT_DIR/models/deepseek-coder-6.7b-finetuned" \
    --batch_size $BATCH_SIZE \
    --max_length $MAX_LENGTH \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate 2e-5 \
    --epochs 3 \
    --max_samples 5000 \
    --load_in_4bit \
    --use_flash_attn \
    --dataset_subset "python" \
    --all_subsets \
    --warmup_steps 100 \
    --num_workers $NUM_WORKERS \
    $BF16_FLAG

# Kill the background monitoring process
kill $MONITOR_PID

echo "Training complete!"
echo "Check gpu_memory_log.txt for memory usage during training."

# Memory usage analysis
echo "Peak GPU memory usage:"
grep -o "[0-9]\+\.[0-9]\+" gpu_memory_log.txt | sort -nr | head -1

echo "✓ Done"
