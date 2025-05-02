#!/bin/bash

# Setup script for storage optimization requirements

echo "Setting up storage optimization for DeepSeek fine-tuning..."

# Detect environment (Gradient, Colab, or standard)
if [ -n "$GRADIENT_NODE_ID" ]; then
    ENVIRONMENT="gradient"
    echo "Detected Gradient environment"
elif [ -n "$COLAB_GPU" ]; then
    ENVIRONMENT="colab"
    echo "Detected Google Colab environment"
else
    ENVIRONMENT="standard"
    echo "Standard environment detected"
fi

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected"
    nvidia-smi
    CUDA_AVAILABLE=true
else
    echo "No NVIDIA GPU detected"
    CUDA_AVAILABLE=false
fi

# Install base requirements
echo "Installing base requirements..."

# Install dependencies based on environment
if [ "$CUDA_AVAILABLE" = true ]; then
    echo "Installing CUDA-optimized packages..."
    # Install PyTorch with CUDA 12.1
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu121
    
    # Install GPU optimization libraries
    pip install bitsandbytes==0.41.1
    pip install triton==2.1.0
    pip install flash-attn==2.3.4
    pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121
fi

# Install Hugging Face ecosystem
echo "Installing Hugging Face ecosystem..."
pip install transformers==4.36.2
pip install accelerate==0.25.0
pip install peft==0.6.2
pip install trl==0.7.10
pip install datasets==2.19.0
pip install huggingface-hub==0.19.4
pip install safetensors==0.4.1

# Install unsloth for efficient fine-tuning
echo "Installing Unsloth..."
pip install unsloth==2025.4.4

# Install additional packages for storage optimization
echo "Installing storage optimization requirements..."
pip install boto3==1.34.86 gdown==5.1.0 psutil==5.9.8 fsspec==2024.3.1

# Setup environment-specific optimizations
if [ "$ENVIRONMENT" = "gradient" ]; then
    echo "Setting up Gradient-specific optimizations..."
    
    # Set environment variables for better CUDA memory management
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
    
    # Create persistent storage directories
    mkdir -p /storage/models
    mkdir -p /storage/datasets
    mkdir -p /storage/checkpoints
    
    # Add environment variables to ~/.bashrc for persistence
    echo 'export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128' >> ~/.bashrc
    
    echo "Created persistent storage directories in /storage/"
    
elif [ "$ENVIRONMENT" = "colab" ]; then
    echo "Setting up Colab-specific optimizations..."
    
    # Mount Google Drive for external storage
    python -c "
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print('Google Drive mounted successfully')
except ImportError:
    print('Not running in Google Colab')
"
    
    # Create directories in Google Drive
    mkdir -p /content/drive/MyDrive/DeepSeek_Models
    mkdir -p /content/drive/MyDrive/DeepSeek_Datasets
    
    echo "Created directories in Google Drive"
fi

# Clone the repository if not already present
if [ ! -d "Jarvis-AI-Assistant" ] && [ "$PWD" != *"Jarvis-AI-Assistant"* ]; then
    echo "Cloning repository..."
    git clone https://github.com/your-username/Jarvis-AI-Assistant.git
    cd Jarvis-AI-Assistant
else
    if [ "$PWD" != *"Jarvis-AI-Assistant"* ]; then
        echo "Repository already exists, changing to directory"
        cd Jarvis-AI-Assistant
    else
        echo "Already in repository directory"
    fi
fi

# Install project requirements
if [ -f "pyproject.toml" ]; then
    echo "Installing project with poetry (if available)..."
    if command -v poetry &> /dev/null; then
        poetry install
    else
        echo "Poetry not found, installing with pip..."
        pip install -e .
    fi
elif [ -f "requirements.txt" ]; then
    echo "Installing requirements with pip..."
    pip install -r requirements.txt
else
    echo "No pyproject.toml or requirements.txt found."
fi

# Test installations
echo "Testing installations..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print('CUDA available:', torch.cuda.is_available())

import bitsandbytes as bnb
print(f'bitsandbytes version: {bnb.__version__}')

import xformers
print(f'xformers version: {xformers.__version__}')

import unsloth
print(f'unsloth version: {unsloth.__version__}')
"

echo "Storage optimization setup complete!"

# Print usage information
echo ""
echo "To fine-tune DeepSeek with storage optimization:"
echo "================================================"
echo ""

if [ "$ENVIRONMENT" = "gradient" ]; then
    echo "# On Gradient with NVIDIA GPU:"
    echo "python src/generative_ai_module/optimize_deepseek_storage.py \\"
    echo "    --storage-type local \\"
    echo "    --output-dir /storage/models/deepseek_optimized \\"
    echo "    --quantize 4 \\"
    echo "    --max-steps 500 \\"
    echo "    --batch-size 4 \\"
    echo "    --sequence-length 1024 \\"
    echo "    --checkpoint-strategy improvement \\"
    echo "    --max-checkpoints 2"
elif [ "$ENVIRONMENT" = "colab" ]; then
    echo "# On Google Colab:"
    echo "python src/generative_ai_module/optimize_deepseek_storage.py \\"
    echo "    --storage-type gdrive \\"
    echo "    --remote-path DeepSeek_Models \\"
    echo "    --quantize 4 \\"
    echo "    --max-steps 200 \\"
    echo "    --batch-size 2 \\"
    echo "    --use-mini-dataset"
else
    echo "# Standard usage:"
    echo "python src/generative_ai_module/optimize_deepseek_storage.py \\"
    echo "    --storage-type local \\"
    echo "    --quantize 4 \\"
    echo "    --max-steps 200 \\"
    echo "    --batch-size 2 \\"
    echo "    --use-mini-dataset"
fi

echo ""
echo "For AWS S3 storage:"
echo "python src/generative_ai_module/optimize_deepseek_storage.py \\"
echo "    --storage-type s3 \\"
echo "    --s3-bucket your-bucket-name \\"
echo "    --aws-access-key-id YOUR_ACCESS_KEY \\"
echo "    --aws-secret-access-key YOUR_SECRET_KEY \\"
echo "    --quantize 4 \\"
echo "    --output-dir /storage/models/deepseek_optimized"

echo ""
echo "For more options, run:"
echo "python src/generative_ai_module/optimize_deepseek_storage.py --help" 