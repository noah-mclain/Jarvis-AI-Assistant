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
(watch -n 5 "nvidia-smi --query-gpu=memory.used --format=csv" &> gpu_memory_log.txt) &
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
        echo "Starting optimized TEXT model training..."
        python -m src.generative_ai_module.train_models \
            --model_type text \
            --dataset "agie-ai/OpenAssistant-oasst1,teknium/GPTeacher-General-Instruct" \
            --model_name_or_path mistralai/Mistral-7B-v0.1 \
            --batch_size 12 \
            --max_length 4096 \
            --gradient_accumulation_steps 2 \
            --max_samples 120000 \
            --learning_rate 2e-5 \
            --weight_decay 0.01 \
            --use_4bit \
            --use_flash_attention_2 \
            --gradient_checkpointing \
            --optim adamw_bnb_8bit \
            --eval_steps 250 \
            --save_steps 500 \
            --epochs 4 \
            --evaluation_strategy steps \
            --save_strategy steps \
            --logging_steps 50 \
            --output_dir models/text \
            --visualize_metrics \
            --warmup_steps 150 \
            --use_unsloth \
            --sequence_packing \
            --num_workers 6 \
            --fp16 \
            --force_gpu \
            --cache_dir .cache \
            --use_qlora
        ;;
    2)
        echo "Starting optimized CODE model training with DeepSeek + Unsloth..."
        # First approach - Direct finetune_deepseek.py script with maximum optimizations
        export CUDA_DEVICE_ORDER=PCI_BUS_ID
        export CUDA_VISIBLE_DEVICES=0
        export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:24,garbage_collection_threshold:0.8"
        export FORCE_CPU_DATA_PIPELINE=1  # Critical for Paperspace GPU constraints
        # python -m src.generative_ai_module.finetune_deepseek \
        #     --epochs 4 \
        #     --batch-size 2 \
        #     --max-samples 100000 \
        #     --all-subsets \
        #     --sequence-length 4096 \
        #     --learning-rate 3e-5 \
        #     --warmup-steps 200 \
        #     --load-in-4bit \
        #     --save-steps 1000 \
        #     --save-total-limit 3 \
        #     --use-unsloth \
        #     --force-gpu \
        #     --output-dir Jarvis_AI_Assistant/models/deepseek \
        #     --eval-split 0.15 \
        #     --verbose
        
        # Second approach (uncomment if the first one fails)
        # This uses train_models.py with all optimizations
        # echo "Alternatively trying code model training with train_models.py..."
        python -m src.generative_ai_module.train_models \
            --model_type code \
            --dataset "codeparrot/github-code,code-search-net/code_search_net" \  # Combined datasets
            --model_name_or_path deepseek-ai/deepseek-coder-6.7b-instruct \
            --batch_size 1 \
            --max_length 768 \                  # Optimal for 16GB VRAM
            --gradient_accumulation_steps 64 \   # Increased from 32 → 64
            --max_samples 80000 \
            --learning_rate 2e-5 \
            --weight_decay 0.1 \
            --use_4bit \
            --use_qlora \                       # QLoRA for parameter efficiency
            --lora_r 64 \                       # Added LoRA configuration
            --lora_alpha 16 \                   # LoRA alpha scaling
            --use_flash_attention_2 \
            --gradient_checkpointing \
            --optim adamw_bnb_8bit \
            --eval_steps 1000 \
            --save_steps 2000 \
            --epochs 3 \
            --evaluation_strategy steps \
            --save_strategy steps \
            --logging_steps 100 \
            --output_dir models/code \
            --visualize_metrics \
            --warmup_steps 100 \
            --force_gpu \
            --pad_token_id 50256 \
            --dataset_subset python \
            --fim_rate 0.5 \
            --use_unsloth \
            --cache_dir .cache \
            --num_workers 1 \                   # Reduced to 1 for stability
            --fp16 \                            # Added mixed precision
            --sequence_packing \                # Better batch utilization
            --use_cpu_data_loader               # Prevent GPU memory spikes

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
            --warmup_steps 100 \
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