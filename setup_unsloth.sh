#!/bin/bash

# Setup script for Unsloth with DeepSeek integration
echo "Setting up Unsloth for DeepSeek integration..."

# Install required packages
echo "Installing dependencies..."
pip install -U pip
pip install -U torch numpy
pip install -U bitsandbytes
pip install -U accelerate
pip install -U transformers
pip install -U peft
pip install -U trl
pip install -U datasets
pip install -U unsloth 
pip install -U unsloth_zoo

# Fix CUDA library linking issues (especially on Kaggle)
echo "Fixing CUDA library linking..."
if [ -d "/usr/lib64-nvidia" ]; then
    echo "Found NVIDIA libraries at /usr/lib64-nvidia"
    # First try with regular user
    ldconfig /usr/lib64-nvidia 2>/dev/null || sudo ldconfig /usr/lib64-nvidia
fi

# Look for CUDA installations
for cuda_version in 11.0 11.1 11.2 11.3 11.4 11.5 11.6 11.7 11.8 12.0 12.1 12.2 12.3; do
    if [ -d "/usr/local/cuda-$cuda_version" ]; then
        echo "Found CUDA $cuda_version at /usr/local/cuda-$cuda_version"
        ldconfig /usr/local/cuda-$cuda_version/lib64 2>/dev/null || sudo ldconfig /usr/local/cuda-$cuda_version/lib64
    fi
done

# Check xformers
echo "Checking xformers installation..."
python -c "import xformers" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing xformers..."
    pip install -U xformers
fi

# Check if bitsandbytes is installed correctly
echo "Testing bitsandbytes installation..."
python -m bitsandbytes

# Test unsloth import
echo "Testing unsloth import..."
python -c "
import unsloth
from unsloth import FastLanguageModel
print('Unsloth imported successfully!')
"

echo "Setup complete!"
echo "Now you can run the DeepSeek fine-tuning with: python src/generative_ai_module/run_finetune.py" 