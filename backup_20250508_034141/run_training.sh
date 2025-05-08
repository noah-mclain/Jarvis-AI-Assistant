#!/bin/bash
# Script to run DeepSeek-Coder fine-tuning with attention mask fix

# Set up environment
echo "Setting up environment..."
export PYTHONPATH=$PYTHONPATH:/notebooks

# Clear GPU cache
echo "Clearing GPU cache..."
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else print('No CUDA available')"

# Apply the attention mask fix
echo "Applying attention mask fix..."
python fix_attention_mask.py

# Run the training with optimized parameters
echo "Starting DeepSeek-Coder fine-tuning..."
python -m src.generative_ai_module.finetune_deepseek \
    --epochs 3 \
    --batch-size 8 \
    --learning-rate 1e-5 \
    --sequence-length 2048 \
    --max-samples 3000 \
    --all-subsets \
    --load-in-4bit \
    --warmup-steps 100 \
    --output-dir /notebooks/Jarvis_AI_Assistant/models/deepseek-coder-6.7b-finetuned

echo "Training complete!"
