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

        # Set environment variables for optimal memory usage
        export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:32,garbage_collection_threshold:0.9"
        export TOKENIZERS_PARALLELISM=false
        export FORCE_CPU_ONLY_FOR_INITIAL_LOAD=1

        ./run_deepseek_training.sh
        # # Run the training with optimal parameters for DeepSeek Coder 5.7B
        # python -m src.generative_ai_module.train_models \
        #     --model_type code \
        #     --model_name_or_path deepseek-ai/deepseek-coder-5.7b-instruct \
        #     --dataset "code-search-net/code_search_net" \
        #     --batch_size 1 \
        #     --max_length 512 \
        #     --gradient_accumulation_steps 64 \
        #     --use_4bit \
        #     --use_qlora \
        #     --use_flash_attention_2 \
        #     --gradient_checkpointing \
        #     --optim adamw_bnb_8bit \
        #     --learning_rate 1.5e-5 \
        #     --weight_decay 0.05 \
        #     --bf16 \
        #     --num_workers 4 \
        #     --cache_dir .cache \
        #     --force_gpu \
        #     --pad_token_id 50256 \
        #     --dataset_subset "python" \
        #     --skip_layer_freezing \
        #     --fim_rate 0.6 \
        #     --evaluation_strategy "steps" \
        #     --eval_steps 500 \
        #     --save_steps 1000 \
        #     --logging_steps 50 \
        #     --output_dir "/notebooks/Jarvis_AI_Assistant/models/deepseek-coder-finetuned"

        unset FORCE_CPU_DATA_PIPELINE
        ;;
    3)
        echo "Starting CNN-enhanced TEXT model training with 4 CNN layers..."
        python -m src.generative_ai_module.train_models \
            --model_type text \
            --use_cnn \
            --cnn_layers 4 \
            --dataset "agie-ai/OpenAssistant-oasst1,teknium/GPTeacher-General-Instruct,google/Synthetic-Persona-Chat,euclaise/writingprompts" \
            --model_name_or_path distilgpt2 \
            --batch_size 6 \
            --gradient_accumulation_steps 5 \
            --max_length 1024 \
            --max_samples 50000 \
            --learning_rate 3e-5 \
            --weight_decay 0.05 \
            --use_4bit \
            --use_flash_attention_2 \
            --gradient_checkpointing \
            --optim adamw_bnb_8bit \
            --eval_steps 500 \
            --save_steps 1000 \
            --epochs 3 \
            --save_strategy steps \
            --logging_steps 50 \
            --evaluation_strategy steps \
            --sequence_packing \
            --output_dir Jarvis_AI_Assistant/models \
            --visualize_metrics \
            --use_unsloth \
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