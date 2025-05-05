#!/bin/bash
# Fix script for Jarvis AI Assistant CNN training

echo "===== Jarvis AI Assistant Training Fix Script ====="
echo "Fixing issues and running training properly..."

# Step 1: Clear CUDA cache
echo "Clearing CUDA cache..."
python -c "import torch; torch.cuda.empty_cache(); print('✓ CUDA cache cleared')"

# Step 2: Install required packages without conflicts
echo "Installing required packages..."
pip install "huggingface_hub[hf_xet]" --no-dependencies
pip install "hf_xet"
pip install matplotlib pandas numpy --no-dependencies

# Step 3: Set environment variables to prevent memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
echo "✓ Set PYTORCH_CUDA_ALLOC_CONF to prevent memory fragmentation"

# Step 4: Run with optimized parameters for RTX5000
echo "Starting CNN training with fixed configuration..."
python -m src.generative_ai_module.train_models \
    --model_type text \
    --use_cnn \
    --cnn_layers 2 \
    --dataset "agie-ai/OpenAssistant-oasst1,teknium/GPTeacher-General-Instruct,google/Synthetic-Persona-Chat,euclaise/writingprompts" \
    --model_name_or_path distilgpt2 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_length 512 \
    --max_samples 10000 \
    --epochs 3 \
    --learning_rate 2e-5 \
    --save_strategy steps \
    --logging_steps 50 \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --output_dir Jarvis_AI_Assistant/models \
    --visualize_metrics

echo "Training complete!" 