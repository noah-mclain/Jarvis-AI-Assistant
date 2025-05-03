#!/bin/bash

echo "===================================================================="
echo "Fixing dependency issues for Jarvis AI Assistant on Paperspace"
echo "===================================================================="

# First, clear out incompatible packages
echo "Removing conflicting packages..."
pip uninstall -y numpy bitsandbytes peft xformers unsloth unsloth_zoo flash-attn

# Fix NumPy version first (downgrade to compatible version)
echo "Installing NumPy 1.26.4 (compatible with PyTorch 2.1.x and SciPy)..."
pip install numpy==1.26.4

# Fix protobuf to remain compatible
echo "Installing compatible protobuf version..."
pip install protobuf==3.20.3

# Install PyTorch with matched components
echo "Installing matched PyTorch ecosystem..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu121

# Install scientific packages
echo "Installing scientific packages..."
pip install scipy==1.12.0 matplotlib==3.8.3

# Install dependencies in specific order
echo "Installing core dependencies in correct order..."

# HuggingFace ecosystem
pip install huggingface-hub==0.19.4 --no-deps
pip install filelock requests tqdm pyyaml typing-extensions packaging==23.2 fsspec

# Install transformers ecosystem
pip install tokenizers==0.14.1
pip install transformers==4.36.2 --no-deps
pip install transformers==4.36.2 
pip install peft==0.6.0 --no-deps
pip install peft==0.6.0
pip install accelerate==0.27.0 --no-deps
pip install accelerate==0.27.0
pip install safetensors==0.4.1
pip install datasets==2.19.0 --ignore-installed
pip install trl==0.7.10 --no-deps
pip install trl==0.7.10
pip install einops==0.7.0

# Install bitsandbytes
echo "Installing bitsandbytes..."
pip install --no-cache-dir bitsandbytes==0.41.0

# Install xformers compatible with PyTorch 2.1.2
echo "Installing xformers..."
pip install -U xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121 --no-deps
pip install -U xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121

# Try newer version of unsloth
echo "Installing newer version of unsloth..."
pip install unsloth==2025.4.4 --no-deps

# Install utility packages
echo "Installing utility packages..."
pip install ninja==1.11.1 packaging==23.2 psutil==5.9.8
pip install gdown==5.1.0 fsspec==2024.3.1 boto3==1.28.51
pip install jupyterlab tensorboard

# Configure GPU-specific optimizations for RTX5000
mkdir -p ~/.config/accelerate
cat > ~/.config/accelerate/default_config.yaml << EOF
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'no'
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
mixed_precision: 'fp16'
num_machines: 1
num_processes: 1
use_cpu: false
EOF

echo "===================================================================="
echo "Dependency fixes complete!"
echo ""
echo "IMPORTANT NOTES:"
echo "1. NumPy has been downgraded to 1.26.4 to be compatible with PyTorch 2.1.2"
echo "2. Using a more recent unsloth version (2025.4.4)"
echo "3. Configured for RTX 5000 with FP16 precision"
echo ""
echo "Start Jupyter with this compatible command:"
echo "jupyter lab --allow-root --ip=0.0.0.0 --no-browser"
echo "====================================================================" 