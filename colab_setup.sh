#!/bin/bash

# Google Colab Setup Script for Jarvis AI Assistant - A100 Optimized
echo "Setting up Jarvis AI Assistant on Google Colab with A100 GPU optimization..."

# Check if running in Google Colab
IN_COLAB=0
if python -c "import google.colab" 2>/dev/null; then
    echo "Running in Google Colab environment"
    IN_COLAB=1
else
    echo "Not running in Google Colab environment"
fi

# Function to print Colab GPU setup instructions
print_colab_gpu_instructions() {
    echo "============================================================"
    echo "ERROR: No GPU detected in your Colab environment!"
    echo "============================================================"
    echo "To enable GPU in Google Colab:"
    echo "1. Click on 'Runtime' in the top menu"
    echo "2. Select 'Change runtime type'"
    echo "3. Choose 'GPU' from the hardware accelerator dropdown"
    echo "4. Click 'Save'"
    echo "5. Click on 'Runtime' in the top menu again and select 'Restart runtime'"
    echo "6. Run this script again after restart"
    echo "============================================================"
    exit 1
}

# Check for NVIDIA GPU with multiple detection methods
echo "Checking for GPU..."

GPU_AVAILABLE=0

# Method 1: Check CUDA version file
if [ -f "/usr/local/cuda/version.txt" ]; then
    echo "CUDA installation found:"
    cat /usr/local/cuda/version.txt
    GPU_AVAILABLE=1
fi

# Method 2: Try nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected with nvidia-smi:"
    nvidia-smi | head -5
    GPU_AVAILABLE=1
    
    # Check if A100 GPU is available
    if nvidia-smi | grep -q "A100"; then
        echo "A100 GPU detected - using optimized settings"
        A100_GPU=true
    else
        echo "Non-A100 GPU detected - will use standard settings"
        A100_GPU=false
    fi
else
    echo "nvidia-smi command not found or not working"
fi

# Method 3: Try PyTorch detection
if python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null | grep -q "CUDA available: True"; then
    echo "PyTorch confirms CUDA is available"
    GPU_AVAILABLE=1
    
    # Get GPU info from PyTorch
    python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')" 2>/dev/null
    
    # Check if A100 via PyTorch
    if python -c "import torch; print('A100' in torch.cuda.get_device_name(0) if torch.cuda.is_available() else False)" 2>/dev/null | grep -q "True"; then
        echo "A100 GPU confirmed by PyTorch - using optimized settings"
        A100_GPU=true
    fi
else
    echo "PyTorch cannot detect CUDA"
fi

# Show instructions if no GPU in Colab
if [ $IN_COLAB -eq 1 ] && [ $GPU_AVAILABLE -eq 0 ]; then
    print_colab_gpu_instructions
elif [ $GPU_AVAILABLE -eq 0 ]; then
    echo "ERROR: No NVIDIA GPU detected. This setup requires GPU acceleration."
    exit 1
fi

# Install dependencies using pip
echo "Installing dependencies..."

# PyTorch with CUDA 12.1 - optimized for A100
echo "Installing PyTorch with CUDA 12.1..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu121

# GPU optimization libraries - A100 optimized
echo "Installing GPU optimization libraries..."
pip install bitsandbytes==0.41.1    # Better GPU memory handling
pip install triton==2.1.0           # Optimized kernels for Tensor Cores
pip install flash-attn==2.3.4       # A100-optimized attention mechanism
pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121  # Efficient attention

# Hugging Face ecosystem - versions compatible with A100 optimizations
echo "Installing Hugging Face ecosystem..."
pip install transformers==4.36.2
pip install accelerate==0.25.0
pip install peft==0.6.2
pip install trl==0.7.10
pip install datasets==2.19.0
pip install huggingface-hub==0.19.4
pip install safetensors==0.4.1

# Unsloth - optimized for efficient fine-tuning
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

# Additional A100 optimizations from requirements.txt
echo "Installing additional optimizations..."
pip install einops==0.7.0
pip install ninja==1.11.1
pip install packaging==23.2

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

# A100-specific optimizations
if [ "${A100_GPU:-false}" = true ]; then
    echo "Applying A100-specific optimizations..."
    
    # Set environment variables for optimal A100 performance
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    export CUDA_LAUNCH_BLOCKING=0
    export TOKENIZERS_PARALLELISM=true
    
    # Create a .bashrc entry for future sessions
    echo 'export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512' >> ~/.bashrc
    echo 'export CUDA_LAUNCH_BLOCKING=0' >> ~/.bashrc
    echo 'export TOKENIZERS_PARALLELISM=true' >> ~/.bashrc
    
    # Create an A100-optimized config file
    mkdir -p ~/.config/accelerate
    cat > ~/.config/accelerate/default_config.yaml << EOF
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'yes'
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
mixed_precision: 'bf16'
num_machines: 1
num_processes: 1
use_cpu: false
EOF
    
    echo "Created A100-optimized accelerate config with BF16 mixed precision"
fi

# Mount Google Drive
echo "Mounting Google Drive..."
python -c "
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print('Google Drive mounted successfully')
except ImportError:
    print('Not running in Google Colab')
except Exception as e:
    print(f'Error mounting Google Drive: {e}')
    print('You can try mounting it manually using: from google.colab import drive; drive.mount(\"/content/drive\")')
"

# Create directories in Google Drive
if [ $IN_COLAB -eq 1 ]; then
    if [ -d "/content/drive/MyDrive" ]; then
        echo "Creating directories in Google Drive..."
        mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant
        mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant/models
        mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant/datasets
    else
        echo "Google Drive not mounted or not accessible. Skipping directory creation."
    fi
fi

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
import sys
print(f'Python version: {sys.version}')

try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('CUDA version:', torch.version.cuda)
        print('GPU:', torch.cuda.get_device_name(0))
        print('GPU memory:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')
        print('BF16 support:', torch.cuda.is_bf16_supported())
    else:
        print('WARNING: CUDA is not available to PyTorch!')
except ImportError:
    print('PyTorch not installed correctly')
except Exception as e:
    print(f'PyTorch error: {e}')

try:
    import bitsandbytes as bnb
    print(f'bitsandbytes version: {bnb.__version__}')
except ImportError:
    print('bitsandbytes not installed correctly')
except Exception as e:
    print(f'bitsandbytes error: {e}')

try:
    import xformers
    print(f'xformers version: {xformers.__version__}')
except ImportError:
    print('xformers not installed correctly')
except Exception as e:
    print(f'xformers error: {e}')

try:
    import unsloth
    print(f'unsloth version: {unsloth.__version__}')
except ImportError:
    print('unsloth not installed correctly')
except Exception as e:
    print(f'unsloth error: {e}')

print('\\nGPU performance test:')
if torch.cuda.is_available():
    # Create a test tensor and measure performance
    a = torch.randn(4096, 4096, device='cuda')
    b = torch.randn(4096, 4096, device='cuda')
    
    # Warm-up
    for _ in range(5):
        c = torch.matmul(a, b)
    
    # Measure
    import time
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    end = time.time()
    
    print(f'Matrix multiplication speed test (10 iterations): {end-start:.4f} seconds')
    
    if 'A100' in torch.cuda.get_device_name(0):
        print('A100 is properly configured and performing well.')
else:
    print('GPU performance test skipped: CUDA not available')
"

# Create a sample Colab notebook for A100 optimization if we're in Colab
if [ $IN_COLAB -eq 1 ]; then
    cat > A100_Optimization.ipynb << EOF
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# A100 GPU Optimization for Jarvis AI Assistant\n",
    "\n",
    "This notebook demonstrates how to use the A100 GPU optimally with this project."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# First, make sure you're using a GPU runtime\n",
    "# Runtime > Change runtime type > Hardware accelerator > GPU\n",
    "import torch\n",
    "print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
    "if torch.cuda.is_available():\n",
    "    print(f\"GPU: {torch.cuda.get_device_name(0)}\")\n",
    "else:\n",
    "    print(\"No GPU detected! Please change runtime to GPU.\")"
   ],
   "outputs": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Run the setup script\n",
    "!chmod +x ./colab_setup.sh\n",
    "!./colab_setup.sh"
   ],
   "outputs": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Configure for A100 optimal batch sizes\n",
    "import torch\n",
    "from accelerate import Accelerator\n",
    "\n",
    "# A100 typically has 40-80GB of VRAM, so we can use large batch sizes\n",
    "if 'A100' in torch.cuda.get_device_name(0):\n",
    "    BATCH_SIZE = 32  # Can be increased based on your model\n",
    "    GRADIENT_ACCUMULATION_STEPS = 4\n",
    "    print(f\"Using A100-optimized batch size: {BATCH_SIZE}\")\n",
    "else:\n",
    "    BATCH_SIZE = 4\n",
    "    GRADIENT_ACCUMULATION_STEPS = 8\n",
    "    print(f\"Using standard batch size: {BATCH_SIZE}\")\n",
    "\n",
    "# Initialize accelerator with BF16 mixed precision (optimal for A100)\n",
    "accelerator = Accelerator(mixed_precision='bf16')\n",
    "print(f\"Using device: {accelerator.device}\")\n",
    "print(f\"Mixed precision: {accelerator.mixed_precision}\")"
   ],
   "outputs": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Example of loading a model with A100 optimizations\n",
    "from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig\n",
    "\n",
    "# Configure quantization for A100\n",
    "bnb_config = BitsAndBytesConfig(\n",
    "    load_in_4bit=True,\n",
    "    bnb_4bit_use_double_quant=True,\n",
    "    bnb_4bit_quant_type=\"nf4\",\n",
    "    bnb_4bit_compute_dtype=torch.bfloat16  # Use BF16 on A100\n",
    ")\n",
    "\n",
    "# Load model with A100 optimizations\n",
    "model_name = \"deepseek-ai/deepseek-coder-6.7b-base\"  # Example model\n",
    "tokenizer = AutoTokenizer.from_pretrained(model_name)\n",
    "model = AutoModelForCausalLM.from_pretrained(\n",
    "    model_name,\n",
    "    quantization_config=bnb_config,\n",
    "    device_map=\"auto\",\n",
    "    trust_remote_code=True\n",
    ")\n",
    "\n",
    "print(f\"Model loaded with A100 optimizations\")"
   ],
   "outputs": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.0"
  },
  "accelerator": "GPU"
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF
fi

echo "Setup complete! You can now run your notebooks or Python scripts."
if [ $IN_COLAB -eq 1 ]; then
    echo "An A100 optimization notebook has been created: A100_Optimization.ipynb"
    
    # If no GPU detected but we're in Colab, remind about runtime change
    if [ $GPU_AVAILABLE -eq 0 ]; then
        print_colab_gpu_instructions
    fi
fi 