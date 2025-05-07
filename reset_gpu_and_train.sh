#!/bin/bash
# Aggressive GPU memory clearing and training script

echo "===== AGGRESSIVE GPU MEMORY CLEARING AND TRAINING ====="
echo "This script will attempt to completely clear GPU memory before training"

# Kill all Python processes except this one
echo "Killing all other Python processes..."
THIS_PID=$$
for pid in $(ps aux | grep python | grep -v grep | awk '{print $2}'); do
    if [ "$pid" != "$THIS_PID" ]; then
        echo "Killing Python process $pid"
        kill -9 $pid 2>/dev/null
    fi
done

# Clear CUDA cache with a dedicated Python script
echo "Clearing CUDA cache aggressively..."
python -c "
import torch
import gc
import os
import time

# Force garbage collection
gc.collect()

# Clear CUDA cache if available
if torch.cuda.is_available():
    print(f'Initial GPU memory: {torch.cuda.memory_allocated() / (1024**3):.2f} GB')
    
    # Empty cache multiple times
    for i in range(5):
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(0.5)
    
    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    
    # Try to reset device
    try:
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
    except Exception as e:
        print(f'Error resetting device: {e}')
    
    print(f'After cleanup: {torch.cuda.memory_allocated() / (1024**3):.2f} GB')
    print(f'Free GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3) - torch.cuda.memory_allocated() / (1024**3):.2f} GB')
"

# Display GPU info
echo "===================================================================="
echo "GPU Information after cleanup:"
nvidia-smi
echo "===================================================================="

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

# Apply the attention mask error fix
echo "Applying attention mask error fix..."
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

# Start GPU monitoring
echo "Starting GPU monitoring..."
(nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu --format=csv -l 10 > gpu_memory_log.txt) &
MONITOR_PID=$!

# Set BF16 capability based on available memory
if python -c "import torch; exit(0 if torch.cuda.is_available() and (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / (1024**3) > 2.0 else 1)"; then
    echo "Setting BF16_CAPABLE=true"
    BF16_FLAG="--bf16"
else
    echo "Setting BF16_CAPABLE=false"
    BF16_FLAG=""
fi

# Run the training with memory-efficient settings and reduced batch size
echo "Starting DeepSeek Coder training with memory optimizations..."
echo "Executing the following command:"
echo "python -m src.generative_ai_module.train_models \
    --model_type code \
    --model_name_or_path deepseek-ai/deepseek-coder-5.7b-instruct \
    --dataset \"code-search-net/code_search_net\" \
    --batch_size 1 \
    --max_length 256 \
    --gradient_accumulation_steps 32 \
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
    --max_samples 10000 \
    --output_dir \"/notebooks/Jarvis_AI_Assistant/models/deepseek-coder-finetuned\""

# Run the training command with reduced parameters
python -m src.generative_ai_module.train_models \
    --model_type code \
    --model_name_or_path deepseek-ai/deepseek-coder-5.7b-instruct \
    --dataset "code-search-net/code_search_net" \
    --batch_size 1 \
    --max_length 256 \
    --gradient_accumulation_steps 32 \
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
    --max_samples 10000 \
    --output_dir "/notebooks/Jarvis_AI_Assistant/models/deepseek-coder-finetuned"

# Kill the background monitoring process
kill $MONITOR_PID

echo "Training complete!"
echo "Check gpu_memory_log.txt for memory usage during training."

# Memory usage analysis
echo "Peak GPU memory usage:"
grep -o "[0-9]\+\.[0-9]\+" gpu_memory_log.txt | sort -nr | head -1

echo "✓ Done"
