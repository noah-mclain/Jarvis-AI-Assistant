#!/bin/bash

# Setup script for Unsloth with DeepSeek integration
echo "Setting up Unsloth for DeepSeek integration..."

# Install required packages
echo "Installing dependencies..."
pip install -U pip

# Install PyTorch with CUDA 12.1
echo "Installing PyTorch with CUDA 12.1..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu121

# Install GPU optimization libraries
echo "Installing GPU optimization libraries..."
pip install bitsandbytes==0.41.1
pip install triton==2.1.0
pip install flash-attn==2.3.4
pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121

# Install Hugging Face ecosystem
echo "Installing Hugging Face ecosystem..."
pip install transformers==4.36.2
pip install accelerate==0.25.0
pip install peft==0.6.2
pip install trl==0.7.10
pip install datasets==2.19.0
pip install huggingface-hub==0.19.4
pip install safetensors==0.4.1

# Install Unsloth
echo "Installing Unsloth..."
pip install unsloth==2025.4.4

# Fix CUDA library linking issues (especially on Kaggle and Colab)
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

# Check if bitsandbytes is installed correctly
echo "Testing bitsandbytes installation..."
python -c "
import torch
import bitsandbytes as bnb
print(f'PyTorch version: {torch.__version__}')
print(f'bitsandbytes version: {bnb.__version__}')
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU:', torch.cuda.get_device_name(0))
"

# Test xformers import
echo "Testing xformers import..."
python -c "
import xformers
print(f'xformers version: {xformers.__version__}')
"

# Test unsloth import
echo "Testing unsloth import..."
python -c "
import unsloth
print(f'unsloth version: {unsloth.__version__}')
"

echo "Setup complete!"
echo "Now you can run the DeepSeek fine-tuning with: python src/generative_ai_module/run_finetune.py" 