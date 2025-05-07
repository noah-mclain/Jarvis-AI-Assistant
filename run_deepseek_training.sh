#!/bin/bash
# Script to run DeepSeek Coder 5.7B training with the fixed code

echo "===== Running DeepSeek Coder 5.7B Training with Fixed Code ====="

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

# Check if we're running on a CUDA device and set USE_BF16 accordingly
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "CUDA device detected, enabling BF16 mixed precision"
    export USE_BF16=true
else
    echo "No CUDA device detected, disabling BF16 mixed precision"
    export USE_BF16=false
fi

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
echo "Starting DeepSeek Coder 5.7B training..."

# Check if we're running on a CUDA device and set BF16 flag accordingly
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "CUDA device detected, enabling BF16 mixed precision"
    BF16_FLAG="--bf16"
else
    echo "No CUDA device detected, disabling BF16 mixed precision"
    BF16_FLAG=""
fi

python -m src.generative_ai_module.train_models \
    --model_type code \
    --model_name_or_path deepseek-ai/deepseek-coder-5.7b-instruct \
    --dataset "code-search-net/code_search_net" \
    --batch_size 1 \
    --max_length 512 \
    --gradient_accumulation_steps 64 \
    --use_4bit \
    --use_qlora \
    --use_flash_attention_2 \
    --gradient_checkpointing \
    --optim adamw_bnb_8bit \
    --learning_rate 1.5e-5 \
    --weight_decay 0.05 \
    $BF16_FLAG \
    --num_workers 4 \
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

echo "Training complete!"
