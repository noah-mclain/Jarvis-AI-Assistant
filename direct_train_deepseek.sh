#!/bin/bash
# Direct training script for DeepSeek Coder without using run_deepseek_training.sh

echo "===== Direct DeepSeek Coder Training ====="
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
(nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu --format=csv -l 10 > gpu_memory_log.txt) &
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
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1
export FORCE_CPU_ONLY_FOR_TOKENIZATION=1
export FORCE_CPU_ONLY_FOR_DATASET_PROCESSING=1
export TRANSFORMERS_OFFLINE=0
export TOKENIZERS_FORCE_CPU=1
export HF_DATASETS_CPU_ONLY=1
export JARVIS_FORCE_CPU_TOKENIZER=1

# Apply the attention mask error fix
echo "Applying a more robust patch for the attention mask error..."
python fix_attention_mask_error.py

# Ensure directories exist
echo "Ensuring directories exist..."
mkdir -p /notebooks/Jarvis_AI_Assistant/models
mkdir -p /notebooks/Jarvis_AI_Assistant/metrics
mkdir -p /notebooks/Jarvis_AI_Assistant/checkpoints
mkdir -p /notebooks/Jarvis_AI_Assistant/logs
mkdir -p /notebooks/Jarvis_AI_Assistant/evaluation_metrics
mkdir -p /notebooks/Jarvis_AI_Assistant/preprocessed_data
mkdir -p /notebooks/Jarvis_AI_Assistant/visualization

# Check if we're running on a CUDA device and set BF16 flag accordingly
if python -c "import torch; exit(0 if torch.cuda.is_available() and (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / (1024**3) > 2.0 else 1)"; then
    echo "GPU detected with sufficient memory, enabling BF16 mixed precision"
    BF16_FLAG="--bf16"
else
    echo "GPU with limited memory or no GPU detected, disabling BF16 mixed precision"
    BF16_FLAG=""
fi

# Run the training with memory-efficient settings
echo "Starting DeepSeek Coder training with memory optimizations..."
echo "Executing the following command:"
echo "python -m src.generative_ai_module.train_models \
    --model_type code \
    --model_name_or_path deepseek-ai/deepseek-coder-5.7b-instruct \
    --dataset \"code-search-net/code_search_net\" \
    --batch_size 1 \
    --max_length 512 \
    --gradient_accumulation_steps 64 \
    --use_4bit \
    --use_qlora \
    --gradient_checkpointing \
    --optim adamw_bnb_8bit \
    --learning_rate 1.5e-5 \
    --weight_decay 0.05 \
    $BF16_FLAG \
    --num_workers 1 \
    --cache_dir .cache \
    --force_gpu \
    --pad_token_id 50256 \
    --dataset_subset \"python\" \
    --skip_layer_freezing \
    --fim_rate 0.6 \
    --evaluation_strategy \"steps\" \
    --eval_steps 500 \
    --save_steps 1000 \
    --logging_steps 50 \
    --output_dir \"/notebooks/Jarvis_AI_Assistant/models/deepseek-coder-finetuned\""

# Run the training command
python -m src.generative_ai_module.train_models \
    --model_type code \
    --model_name_or_path deepseek-ai/deepseek-coder-5.7b-instruct \
    --dataset "code-search-net/code_search_net" \
    --batch_size 1 \
    --max_length 512 \
    --gradient_accumulation_steps 64 \
    --use_4bit \
    --use_qlora \
    --gradient_checkpointing \
    --optim adamw_bnb_8bit \
    --learning_rate 1.5e-5 \
    --weight_decay 0.05 \
    $BF16_FLAG \
    --num_workers 1 \
    --cache_dir .cache \
    --force_gpu \
    --pad_token_id 50256 \
    --dataset_subset "python" \
    --skip_layer_freezing \
    --fim_rate 0.6 \
    --evaluation_strategy "steps" \
    --eval_steps 500 \
    --save_steps 1000 \
    --logging_steps 50 \
    --output_dir "/notebooks/Jarvis_AI_Assistant/models/deepseek-coder-finetuned"

# Kill the background monitoring process
kill $MONITOR_PID

echo "Training complete!"
echo "Check gpu_memory_log.txt for memory usage during training."

# Memory usage analysis
echo "Peak GPU memory usage:"
grep -o "[0-9]\+\.[0-9]\+" gpu_memory_log.txt | sort -nr | head -1

echo "✓ Done"
