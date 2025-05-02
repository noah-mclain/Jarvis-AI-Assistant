#!/bin/bash

# Google Colab Setup Script for Jarvis AI Assistant
echo "Setting up Jarvis AI Assistant on Google Colab..."

# Check for NVIDIA GPU
if [ -f "/usr/local/cuda/version.txt" ]; then
    echo "NVIDIA GPU detected with CUDA:"
    cat /usr/local/cuda/version.txt
    CUDA_AVAILABLE=true
    nvidia-smi
else
    echo "No NVIDIA GPU detected. This setup requires GPU acceleration."
    exit 1
fi

# Install dependencies using pip
echo "Installing dependencies..."

# PyTorch with CUDA
echo "Installing PyTorch with CUDA..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu121

# GPU optimization libraries
echo "Installing GPU optimization libraries..."
pip install bitsandbytes==0.41.1
pip install triton==2.1.0
pip install flash-attn==2.3.4
pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121

# Hugging Face ecosystem
echo "Installing Hugging Face ecosystem..."
pip install transformers==4.36.2
pip install accelerate==0.25.0
pip install peft==0.6.2
pip install trl==0.7.10
pip install datasets==2.19.0
pip install huggingface-hub==0.19.4
pip install safetensors==0.4.1

# Unsloth
echo "Installing Unsloth..."
pip install unsloth==2025.4.4

# NLP & Utilities
echo "Installing NLP & Utilities..."
pip install spacy==3.7.4
pip install nltk==3.8.1
pip install tqdm==4.66.2
pip install pandas==2.2.2
pip install scikit-learn==1.4.2
pip install numpy==1.26.4
pip install pydantic==1.10.13

# Storage & Cloud Integration
echo "Installing Storage & Cloud Integration..."
pip install boto3==1.34.86
pip install gdown==5.1.0
pip install fsspec==2024.3.1
pip install psutil==5.9.8

# Development Tools
echo "Installing Development Tools..."
pip install jupyterlab==4.4.1
pip install tensorboard==2.16.2

# Spacy language model
python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0.tar.gz

# Mount Google Drive
echo "Mounting Google Drive..."
python -c "
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print('Google Drive mounted successfully')
except ImportError:
    print('Not running in Google Colab')
"

# Create directories in Google Drive
mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant
mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant/models
mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant/datasets

# Clone the repository if not already present
if [ ! -d "Jarvis-AI-Assistant" ]; then
    echo "Cloning repository..."
    git clone https://github.com/your-username/Jarvis-AI-Assistant.git
    cd Jarvis-AI-Assistant
else
    echo "Repository already exists, changing to directory"
    cd Jarvis-AI-Assistant
fi

# Test installations
echo "Testing installations..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU:', torch.cuda.get_device_name(0))

import bitsandbytes as bnb
print(f'bitsandbytes version: {bnb.__version__}')

import xformers
print(f'xformers version: {xformers.__version__}')

import unsloth
print(f'unsloth version: {unsloth.__version__}')
"

echo "Setup complete! You can now run your notebooks or Python scripts." 