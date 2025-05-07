#!/bin/bash
# Script to run DeepSeek Coder training with the fixed code

echo "===== Running FLAN-UL2 20B Training with Fixed Code ====="

# Step 1: Clean GPU memory
echo "Cleaning GPU memory..."
python -c "import gc
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    print(f'Initial GPU memory: {torch.cuda.memory_allocated() / (1024**3):.2f} GB')"

# Step 2: Set environment variables for optimal memory usage
echo "Setting environment variables for optimal memory usage..."
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:32,garbage_collection_threshold:0.9"
export TOKENIZERS_PARALLELISM=false
export FORCE_CPU_ONLY_FOR_INITIAL_LOAD=1

# Step 3: Ensure directories exist
echo "Ensuring directories exist..."
mkdir -p /notebooks/Jarvis_AI_Assistant/models
mkdir -p /notebooks/Jarvis_AI_Assistant/metrics
mkdir -p /notebooks/Jarvis_AI_Assistant/checkpoints
mkdir -p /notebooks/Jarvis_AI_Assistant/logs
mkdir -p /notebooks/Jarvis_AI_Assistant/evaluation_metrics
mkdir -p /notebooks/Jarvis_AI_Assistant/preprocessed_data
mkdir -p /notebooks/Jarvis_AI_Assistant/visualization

# Step 4: Run the training with optimal parameters
echo "Starting FLAN-UL2 20B training..."
python -m src.generative_ai_module.train_models \
    --model_type text \
    --model_name_or_path google/flan-ul2 \
    --dataset "euclaise/writingprompts,google/Synthetic-Persona-Chat,EleutherAI/pile,teknium/GPTeacher-General-Instruct,agie-ai/OpenAssistant-oasst1" \
    --batch_size 1 \
    --max_length 512 \
    --gradient_accumulation_steps 128 \
    --use_4bit \
    --use_qlora \
    --use_flash_attention_2 \
    --gradient_checkpointing \
    --optim adamw_bnb_8bit \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --bf16 \
    --num_workers 2 \
    --max_samples 10000 \
    --cache_dir .cache \
    --force_gpu \
    --pad_token_id 0 \
    --dataset_subset "all" \
    --skip_layer_freezing \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --evaluation_strategy "steps" \
    --eval_steps 500 \
    --save_steps 1000 \
    --logging_steps 50 \
    --output_dir "/notebooks/Jarvis_AI_Assistant/models/flan-ul2-finetuned"

echo "Training complete!"
