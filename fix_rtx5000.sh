#!/bin/bash

echo "===================================================================="
echo "RTX 5000 Optimized Fix for Jarvis AI Assistant on Paperspace"
echo "===================================================================="

# Aggressively clean the environment
echo "Performing deep cleanup of conflicting packages..."
pip uninstall -y numpy scipy matplotlib pywavelets cupy-cuda12x bitsandbytes peft accelerate xformers unsloth unsloth_zoo flash-attn protobuf tensorflow tensorflow-estimator tensorflow-io-gcs-filesystem tensorboard huggingface-hub transformers tokenizers

# Make sure PATH is set to avoid using system packages
echo "Ensuring we're using the correct environment..."
which python
which pip

# Fix NumPy version first (critical foundation)
echo "Installing NumPy 1.26.4 (compatible with PyTorch 2.1.x and SciPy)..."
pip install numpy==1.26.4 --no-deps
pip install numpy==1.26.4 # Second install to verify

# Test NumPy installation
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

# Install PyTorch matching components
echo "Installing PyTorch with matched components..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu121

# Install Google Protobuf (fix for TensorFlow issues)
echo "Installing compatible protobuf version..."
pip install protobuf==3.20.3 --no-deps
pip install protobuf==3.20.3

# Install core scientific packages with fixed versions
echo "Installing scientific packages..."
pip install scipy==1.12.0 --no-deps
pip install scipy==1.12.0
pip install matplotlib==3.8.3 --no-deps
pip install matplotlib==3.8.3

# Install in very specific order - avoid pulling in incompatible dependencies
echo "Installing core dependencies in correct order..."

# Install base requirements
pip install packaging==23.2 wheel>=0.38.0 ninja==1.11.1 

# HuggingFace ecosystem core - install with compatible versions
pip install filelock requests tqdm pyyaml typing-extensions fsspec scipy==1.12.0
pip install huggingface-hub==0.17.3 --no-deps
pip install tokenizers==0.14.1

# Transformers ecosystem (fixed versions)
pip install transformers==4.36.2 --no-deps
pip install transformers==4.36.2 --ignore-installed
pip install peft==0.6.0 --no-deps
pip install peft==0.6.0
pip install accelerate==0.27.0 --no-deps
pip install accelerate==0.27.0
pip install safetensors==0.4.1
pip install datasets==2.19.0
pip install trl==0.7.10 --no-deps
pip install trl==0.7.10
pip install einops==0.7.0

# Install bitsandbytes with specific version
echo "Installing bitsandbytes..."
pip install --no-cache-dir bitsandbytes==0.41.0

# Install xformers compatible with PyTorch 2.1.2
echo "Installing xformers..."
pip install xformers==0.0.23.post1 --extra-index-url https://download.pytorch.org/whl/cu121

# Install unsloth with compatible version
echo "Installing compatible unsloth version..."
pip install sentencepiece==0.2.0
# Install a specific version that works with our setup
pip install unsloth==2025.3.3 --no-deps

# Install utility packages
echo "Installing utility packages..."
pip install psutil==5.9.8 gdown==5.1.0 boto3==1.28.51 
pip install jupyterlab tensorboard

# Setup Google Drive integration
echo "Setting up Google Drive integration for Paperspace..."
pip install -q pydrive2

# Create Google Drive mount script
cat > ~/mount_google_drive.py << 'EOF'
from google.colab import drive
import os

print("Mounting Google Drive...")
try:
    drive.mount('/content/drive')
    print("Google Drive mounted successfully at /content/drive")
    
    # Create directories for Jarvis AI Assistant
    os.makedirs('/content/drive/MyDrive/Jarvis_AI_Assistant/models', exist_ok=True)
    os.makedirs('/content/drive/MyDrive/Jarvis_AI_Assistant/datasets', exist_ok=True)
    os.makedirs('/content/drive/MyDrive/Jarvis_AI_Assistant/checkpoints', exist_ok=True)
    os.makedirs('/content/drive/MyDrive/Jarvis_AI_Assistant/metrics', exist_ok=True)
    
    print("Created Jarvis AI Assistant directories in Google Drive")
    
    # Create symlink for easier access
    os.system('ln -sf /content/drive/MyDrive/Jarvis_AI_Assistant /notebooks/google_drive_jarvis')
    print("Created symlink to Google Drive at /notebooks/google_drive_jarvis")
    
    # Set environment variables for the project
    with open(os.path.expanduser("~/.bashrc"), "a") as bashrc:
        bashrc.write('\n# Jarvis AI Assistant paths\n')
        bashrc.write('export JARVIS_STORAGE_PATH="/content/drive/MyDrive/Jarvis_AI_Assistant"\n')
        bashrc.write('export JARVIS_MODELS_PATH="/content/drive/MyDrive/Jarvis_AI_Assistant/models"\n')
        bashrc.write('export JARVIS_DATA_PATH="/content/drive/MyDrive/Jarvis_AI_Assistant/datasets"\n')
    
    print("Environment variables added to .bashrc")
    
except Exception as e:
    print(f"Error mounting Google Drive: {e}")
    print("You may need to run this script manually and authorize access.")
EOF

# Try to mount Google Drive
echo "Attempting to mount Google Drive..."
python ~/mount_google_drive.py

# Configure GPU-specific optimizations for RTX5000
echo "Setting up RTX 5000 optimizations..."
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

# Optimize CUDA memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=true
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/lib64-nvidia

# Add to bashrc for persistence
echo 'export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256' >> ~/.bashrc
echo 'export CUDA_LAUNCH_BLOCKING=0' >> ~/.bashrc
echo 'export TOKENIZERS_PARALLELISM=true' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/lib64-nvidia' >> ~/.bashrc

# Verify installation
echo "Verifying key installations..."
python -c "
import sys
print(f'Python version: {sys.version}')

import numpy
print(f'NumPy version: {numpy.__version__}')

try:
    from google import protobuf
    print(f'Protobuf successfully imported')
except Exception as e:
    print(f'Protobuf error: {e}')

import torch
print(f'PyTorch version: {torch.__version__}')
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU:', torch.cuda.get_device_name(0))
"

echo "===================================================================="
echo "RTX 5000 optimization complete!"
echo ""
echo "IMPORTANT NOTES FOR RTX 5000:"
echo "1. Use batch size 2-4 for optimal performance"
echo "2. Sequence length 1024-2048 recommended"
echo "3. Gradient accumulation steps 4+ for larger models"
echo "4. Always use FP16 mix-precision (--fp16 flag)"
echo "5. For 6.7B models, always use 4-bit quantization (--load-in-4bit flag)"
echo ""
echo "Example command:"
echo "python src/generative_ai_module/jarvis_unified.py \\"
echo "  --mode train \\"
echo "  --model deepseek-ai/deepseek-coder-6.7b-base \\"
echo "  --datasets pile \\"
echo "  --max-samples 500 \\"
echo "  --epochs 1 \\"
echo "  --batch-size 4 \\"
echo "  --gradient-accumulation-steps 4 \\"
echo "  --load-in-4bit \\"
echo "  --fp16 \\"
echo "  --sequence-length 1024"
echo "====================================================================" 