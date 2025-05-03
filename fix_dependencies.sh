#!/bin/bash

echo "===================================================================="
echo "Fixing dependency issues for Jarvis AI Assistant on Paperspace..."
echo "===================================================================="

# Clean environment first 
echo "Uninstalling conflicting packages..."
pip uninstall -y flash-attn bitsandbytes unsloth peft accelerate xformers unsloth_zoo protobuf tokenizers huggingface-hub

# Fix LD_LIBRARY_PATH for CUDA compatibility
echo "Setting up CUDA library paths..."
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/lib64-nvidia' >> ~/.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/lib64-nvidia

# Fix torch version to ensure compatibility
echo "Installing PyTorch 2.1.2 with CUDA 12.1..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu121

# Installing packages in a very specific order to avoid version conflicts
echo "Installing compatible core dependencies..."

# First install huggingface hub at a version that works with everything
echo "Installing huggingface-hub..."
pip install huggingface-hub==0.19.4 --no-deps
pip install filelock requests tqdm pyyaml typing-extensions packaging fsspec

# Next install tokenizers at a compatible version
echo "Installing tokenizers..."
pip install tokenizers==0.14.1

# Then install core components in order
echo "Installing transformers ecosystem..."
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
pip install -U "xformers==0.0.23.post1" --index-url https://download.pytorch.org/whl/cu121 --no-deps
pip install -U "xformers==0.0.23.post1" --index-url https://download.pytorch.org/whl/cu121

# Install dependencies for unsloth
echo "Installing unsloth dependencies..."
pip install protobuf==3.20.3
pip install sentencepiece==0.2.0

# Install unsloth with appropriate flags to avoid compilation issues
echo "Installing Unsloth..."
pip install "unsloth>=2025.3.0,<2025.4.5" --no-deps
pip install unsloth_zoo==2025.4.4 --no-deps
pip install hf_transfer --no-deps

# Install remaining packages
echo "Installing utility packages..."
pip install ninja==1.11.1 packaging==23.2 psutil==5.9.8
pip install gdown==5.1.0 fsspec==2024.3.1 boto3==1.28.51

# Skip flash-attention installation in this script to avoid build issues
echo "Note: Flash-attention installation is skipped to avoid build errors."
echo "If you need flash-attention, please run fix_flash_attn.sh separately."

# Verify installations
echo "Verifying installations..."
python -c "
import sys
print(f'Python version: {sys.version}')

try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
except Exception as e:
    print(f'PyTorch error: {e}')

try:
    import bitsandbytes as bnb
    print(f'bitsandbytes version: {bnb.__version__ if hasattr(bnb, \"__version__\") else \"(version unknown)\"}'  )
    try:
        lin8bit = bnb.nn.Linear8bitLt(10, 10, has_fp16_weights=False)
        print('bitsandbytes working correctly!')
    except Exception as e:
        print(f'bitsandbytes layer creation error: {e}')
except Exception as e:
    print(f'bitsandbytes import error: {e}')

try:
    import unsloth
    print(f'unsloth version: {unsloth.__version__ if hasattr(unsloth, \"__version__\") else \"(version unknown)\"}')
    
    import transformers
    print(f'transformers version: {transformers.__version__}')
except Exception as e:
    print(f'transformers/unsloth error: {e}')

try:
    import peft
    print(f'peft version: {peft.__version__}')
except Exception as e:
    print(f'peft error: {e}')

try:
    import accelerate
    print(f'accelerate version: {accelerate.__version__}')
except Exception as e:
    print(f'accelerate error: {e}')

try:
    import unsloth_zoo
    print('unsloth_zoo installed successfully')
except Exception as e:
    print(f'unsloth_zoo error: {e}')
"

echo "===================================================================="
echo "Dependency fixes complete!"
echo ""
echo "If you still have issues with unsloth, try running:"
echo "pip install unsloth==2025.3.0 --no-deps"
echo "pip install unsloth_zoo==2025.4.4 --no-deps"
echo ""
echo "You can now run Jarvis AI Assistant with lower-demand parameters:"
echo "python src/generative_ai_module/jarvis_unified.py \\"
echo "  --mode train \\"
echo "  --model deepseek-ai/deepseek-coder-1.3b-base \\"
echo "  --datasets pile \\"
echo "  --max-samples 100 \\"
echo "  --epochs 1 \\"
echo "  --batch-size 1 \\"
echo "  --gradient-accumulation-steps 8 \\"
echo "  --load-in-4bit \\"
echo "  --sequence-length 512"
echo "====================================================================" 