#!/bin/bash

echo "======================================================================"
echo "🚀 Running Jarvis AI Training with Optimized Parameters"
echo "======================================================================"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export CODE_SUBSET="jarvis_code_instructions"

# Change to the notebooks directory in Paperspace
if [ -d "/notebooks" ]; then
  cd /notebooks
  echo "Working in /notebooks directory"
else
  echo "Not in Paperspace environment, using current directory"
fi

# Run the training with optimized parameters
python src/generative_ai_module/train_models.py \
    --model-type code \
    --use-deepseek \
    --subset $CODE_SUBSET \
    --batch-size 4 \
    --epochs 3 \
    --learning-rate 3e-5 \
    --warmup-steps 150 \
    --load-in-4bit \
    --max-samples 1000 \
    --validation-split 0.2 \
    --test-split 0.1 \
    --sequence-length 1024

echo "======================================================================"
echo "Training complete!"
echo "======================================================================" 