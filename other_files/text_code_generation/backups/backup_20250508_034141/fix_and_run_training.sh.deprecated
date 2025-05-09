#!/bin/bash
# Optimized training script for Jarvis AI Assistant on RTX 5000 GPU

echo "===== Jarvis AI Assistant RTX 5000 Optimized Training ====="
echo "Setting up environment for optimal GPU utilization..."

# Step 1: Clear CUDA cache
echo "Clearing CUDA cache..."
python -c "import torch; torch.cuda.empty_cache(); print('✓ CUDA cache cleared')"

# Step 2: Skip installing packages to avoid conflicts
echo "Skipping package installation to preserve your working environment..."
# Comment out all installation commands
# pip install "huggingface_hub[hf_xet]" --no-dependencies
# pip install "hf_xet"
# pip install matplotlib pandas numpy --no-dependencies
# pip install flash-attn --no-build-isolation
# pip install bitsandbytes

# Step 3: Set environment variables to prevent memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
echo "✓ Set PYTORCH_CUDA_ALLOC_CONF to prevent memory fragmentation"

# Enable benchmarking for faster training
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDNN_V8_API_ENABLED=1
export TORCH_CUDNN_BENCHMARK=1
echo "✓ Enabled CUDNN benchmarking for faster training"

# Step 4: Monitor GPU usage in the background
echo "Starting GPU monitoring..."
(watch -n 1 "nvidia-smi --query-gpu=memory.used --format=csv" &> gpu_memory_log.txt) &
MONITOR_PID=$!

# Display GPU info
echo "===================================================================="
echo "GPU Information:"
nvidia-smi
echo "===================================================================="

# Ask which model to train
echo "Which model would you like to train?"
echo "1) Text model (optimized for story/writing generation)"
echo "2) Code model (optimized for code generation)"
echo "3) CNN-enhanced text model"
read -p "Enter choice (1-3): " MODEL_CHOICE

case $MODEL_CHOICE in
    1)
        echo "Starting optimized TEXT model training with FLAN-UL2 20B..."
        python -m src.generative_ai_module.train_models \
            --model_type text \
            --dataset "euclaise/writingprompts,google/Synthetic-Persona-Chat,EleutherAI/pile,teknium/GPTeacher-General-Instruct,agie-ai/OpenAssistant-oasst1" \
            --model_name_or_path google/flan-ul2 \
            --batch_size 1 \
            --max_length 512 \
            --gradient_accumulation_steps 128 \
            --max_samples 10000 \
            --learning_rate 1e-5 \
            --weight_decay 0.01 \
            --use_4bit \
            --use_flash_attention_2 \
            --gradient_checkpointing \
            --optim adamw_bnb_8bit \
            --eval_steps 500 \
            --save_steps 1000 \
            --epochs 4 \
            --evaluation_strategy steps \
            --save_strategy steps \
            --logging_steps 100 \
            --output_dir /notebooks/Jarvis_AI_Assistant/models/flan-ul2-finetuned \
            --visualize_metrics \
            --num_workers 2 \
            --bf16 \
            --force_gpu \
            --cache_dir .cache \
            --use_qlora \
            --skip_layer_freezing \
            --lora_r 32 \
            --lora_alpha 64 \
            --lora_dropout 0.05 \
            --pad_token_id 0
        ;;
    2)
        echo "Starting optimized CODE model training with DeepSeek Coder 5.7B..."
        # Clean GPU memory
        python -c "import gc
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    print(f'Initial GPU memory: {torch.cuda.memory_allocated() / (1024**3):.2f} GB')"

        # Kill all Python processes except this one to free GPU memory
        echo "Killing all other Python processes to free GPU memory..."
        THIS_PID=$$
        for pid in $(ps aux | grep python | grep -v grep | awk '{print $2}'); do
            if [ "$pid" != "$THIS_PID" ]; then
                echo "Killing Python process $pid"
                kill -9 $pid 2>/dev/null
            fi
        done
        sleep 2

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

        echo "===================================================="
        echo "Clearing GPU memory again..."
        python -c "
import torch
import gc
import os
import time

# Force garbage collection
gc.collect()

# Clear CUDA cache if available
if torch.cuda.is_available():
    # Get initial memory usage
    initial_mem = torch.cuda.memory_allocated() / (1024**3)
    initial_reserved = torch.cuda.memory_reserved() / (1024**3)
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    print(f'Initial GPU memory: {initial_mem:.2f} GB allocated, {initial_reserved:.2f} GB reserved')
    print(f'Total GPU memory: {total_mem:.2f} GB')

    # Empty cache multiple times with pauses in between
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

    # Check memory again
    current_mem = torch.cuda.memory_allocated() / (1024**3)
    current_reserved = torch.cuda.memory_reserved() / (1024**3)

    print(f'After cleanup: {current_mem:.2f} GB allocated, {current_reserved:.2f} GB reserved')
    print(f'Freed: {initial_mem - current_mem:.2f} GB allocated, {initial_reserved - current_reserved:.2f} GB reserved')
    print(f'Free GPU memory: {total_mem - current_mem:.2f} GB')

    # Set memory fraction for A6000 with 50 GiB VRAM
    try:
        torch.cuda.set_per_process_memory_fraction(0.97)
        print('Set GPU memory fraction to 97% for A6000 with 50 GiB VRAM')
    except:
        print('Could not set memory fraction, continuing anyway')
else:
    print('No GPU available, will use CPU only')
"

        # Check GPU availability and memory
        echo "Checking GPU availability and memory..."
        python -c "
import torch
if torch.cuda.is_available():
    device = torch.device('cuda')
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    free_mem = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / (1024**3)
    print(f'GPU available: {torch.cuda.get_device_name(0)}')
    print(f'Total memory: {total_mem:.2f} GB')
    print(f'Free memory: {free_mem:.2f} GB')

    # Check if there's sufficient memory for BF16 (A6000 with 50 GiB VRAM)
    if free_mem > 15.0:
        print('Abundant memory for BF16 mixed precision (A6000 with 50 GiB VRAM)')
    else:
        print('Warning: Less than 15GB free VRAM, but continuing with BF16 mixed precision')
else:
    print('No GPU available, will use CPU only')
"

        # Set BF16 capability - always enabled for A6000 with 50 GiB VRAM
        echo "A6000 with 50 GiB VRAM detected - enabling BF16 mixed precision"
        BF16_FLAG="--bf16"

        # Apply patches for attention mask errors
        echo "Applying patches for attention mask errors..."
        python fix_attention_mask_error.py

        # Apply additional fix for attention mask size error
        echo "Applying additional fix for attention mask size error..."
        python fix_attention_mask_size.py

        # Ensure directories exist
        echo "Ensuring directories exist..."
        mkdir -p /notebooks/Jarvis_AI_Assistant/models
        mkdir -p /notebooks/Jarvis_AI_Assistant/metrics
        mkdir -p /notebooks/Jarvis_AI_Assistant/checkpoints
        mkdir -p /notebooks/Jarvis_AI_Assistant/logs
        mkdir -p /notebooks/Jarvis_AI_Assistant/evaluation_metrics
        mkdir -p /notebooks/Jarvis_AI_Assistant/preprocessed_data
        mkdir -p /notebooks/Jarvis_AI_Assistant/visualization

        # Run the training with optimized settings for A6000 with 50 GiB VRAM
        echo "Starting DeepSeek Coder training with A6000 optimizations (50 GiB VRAM)..."
        echo "Executing the following command:"
        echo "python -m src.generative_ai_module.train_models \
    --model_type code \
    --model_name_or_path deepseek-ai/deepseek-coder-6.7b-instruct \
    --dataset \"code-search-net/code_search_net\" \
    --batch_size 8 \
    --max_length 2048 \
    --gradient_accumulation_steps 8 \
    --use_8bit \
    --use_qlora \
    --gradient_checkpointing \
    --optim adamw_bnb_8bit \
    --learning_rate 2e-5 \
    --weight_decay 0.05 \
    $BF16_FLAG \
    --num_workers 8 \
    --cache_dir .cache \
    --force_gpu \
    --pad_token_id 50256 \
    --dataset_subset \"python\" \
    --skip_layer_freezing \
    --fim_rate 0.7 \
    --evaluation_strategy \"steps\" \
    --eval_steps 250 \
    --save_steps 500 \
    --logging_steps 50 \
    --epochs 5 \
    --output_dir \"/notebooks/Jarvis_AI_Assistant/models/deepseek-coder-6.7b-finetuned\""

        # Run the training command with optimized parameters for A6000 with 50 GiB VRAM
        python -m src.generative_ai_module.train_models \
            --model_type code \
            --model_name_or_path deepseek-ai/deepseek-coder-6.7b-instruct \
            --dataset "code-search-net/code_search_net" \
            --batch_size 8 \
            --max_length 2048 \
            --gradient_accumulation_steps 8 \
            --use_8bit \
            --use_qlora \
            --gradient_checkpointing \
            --optim adamw_bnb_8bit \
            --learning_rate 2e-5 \
            --weight_decay 0.05 \
            $BF16_FLAG \
            --num_workers 8 \
            --cache_dir .cache \
            --force_gpu \
            --pad_token_id 50256 \
            --dataset_subset "python" \
            --skip_layer_freezing \
            --fim_rate 0.7 \
            --evaluation_strategy "steps" \
            --eval_steps 250 \
            --save_steps 500 \
            --logging_steps 50 \
            --epochs 5 \
            --output_dir "/notebooks/Jarvis_AI_Assistant/models/deepseek-coder-6.7b-finetuned"

        unset FORCE_CPU_DATA_PIPELINE
        ;;
    3)
        echo "Starting CNN-enhanced TEXT model training with A6000 optimizations (50 GiB VRAM)..."
        python -m src.generative_ai_module.train_models \
            --model_type text \
            --use_cnn \
            --cnn_layers 6 \
            --dataset "agie-ai/OpenAssistant-oasst1,teknium/GPTeacher-General-Instruct,google/Synthetic-Persona-Chat,euclaise/writingprompts" \
            --model_name_or_path "google/flan-ul2" \
            --batch_size 12 \
            --gradient_accumulation_steps 4 \
            --max_length 2048 \
            --learning_rate 3e-5 \
            --weight_decay 0.05 \
            --use_8bit \
            --use_qlora \
            --gradient_checkpointing \
            --optim adamw_bnb_8bit \
            --eval_steps 250 \
            --save_steps 500 \
            --epochs 5 \
            --save_strategy steps \
            --logging_steps 50 \
            --evaluation_strategy steps \
            --sequence_packing \
            --output_dir "/notebooks/Jarvis_AI_Assistant/models/flan-ul2-finetuned" \
            --visualize_metrics \
            --use_unsloth \
            --num_workers 8 \
            --cache_dir .cache \
            --force_gpu
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

# Kill the background monitoring process
kill $MONITOR_PID

echo "Training complete!"
echo "Check gpu_memory_log.txt for memory usage during training."

# Memory usage analysis
echo "Peak GPU memory usage:"
grep -o "[0-9]\+\.[0-9]\+ MiB" gpu_memory_log.txt | sort -nr | head -1

echo "✓ Done"