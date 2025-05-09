#!/bin/bash

echo "======================================================================"
echo "🔧 Installing Dependencies for Jarvis AI Training"
echo "======================================================================"

# Function to check if a command succeeded
check_success() {
  if [ $? -ne 0 ]; then
    echo "❌ Error: $1 failed"
    echo "Continuing with setup process..."
  else
    echo "✅ $1 successful"
  fi
}

# Make sure we're in the notebooks directory in Paperspace
if [ -d "/notebooks" ]; then
  cd /notebooks
  echo "Working in /notebooks directory"
else
  echo "Not in Paperspace environment, using current directory"
fi

# Install core dependencies
echo ""
echo "Step 1: Installing core dependencies..."
echo ""

pip install -U pip
pip install torch transformers datasets accelerate unsloth
check_success "Core ML dependencies"

# Install evaluation dependencies
echo ""
echo "Step 2: Installing evaluation dependencies..."
echo ""

pip install bert-score rouge-score nltk tensorboard
check_success "Evaluation dependencies"

# Install NLTK data
echo ""
echo "Step 3: Installing NLTK data..."
echo ""

python -c "
import nltk
try:
    nltk.download('punkt', quiet=True)
    print('✅ NLTK punkt downloaded successfully')
except Exception as e:
    print(f'❌ Error downloading NLTK data: {e}')
"
check_success "NLTK data"

# Install Google Drive integration
echo ""
echo "Step 4: Installing Google Drive integration..."
echo ""

pip install gdown google-auth google-auth-oauthlib google-auth-httplib2
check_success "Google Drive integration"

# Set up environment for training
echo ""
echo "Step 5: Setting up environment..."
echo ""

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export CODE_SUBSET="jarvis_code_instructions"

# Add to bashrc for persistence across sessions
echo "export CUDA_VISIBLE_DEVICES=0" >> ~/.bashrc
echo "export CODE_SUBSET=\"jarvis_code_instructions\"" >> ~/.bashrc

check_success "Environment setup"

# Create directories
echo ""
echo "Step 6: Creating directories..."
echo ""

mkdir -p models metrics logs checkpoints evaluation_metrics visualizations datasets
check_success "Directory creation"

# Final instructions
echo ""
echo "======================================================================"
echo "✅ All dependencies installed successfully!"
echo ""
echo "Run your training with:"
echo ""
echo "cd /notebooks"
echo "python src/generative_ai_module/train_models.py \\"
echo "    --model-type code \\"
echo "    --use-deepseek \\"
echo "    --code-subset \$CODE_SUBSET \\"
echo "    --batch-size 6 \\"
echo "    --epochs 3 \\"
echo "    --learning-rate 3e-5 \\"
echo "    --warmup-steps 150 \\"
echo "    --load-in-4bit"
echo ""
echo "======================================================================" 