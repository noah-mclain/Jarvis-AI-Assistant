#!/bin/bash

echo "===================================================================="
echo "Fixing dependency issues for Jarvis AI Assistant on Paperspace..."
echo "===================================================================="

# Clean environment first
pip uninstall -y flash-attn bitsandbytes unsloth peft accelerate xformers

# Fix LD_LIBRARY_PATH for CUDA compatibility
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/lib64-nvidia' >> ~/.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/lib64-nvidia

# Fix torch version to ensure compatibility
echo "Installing PyTorch 2.1.2 with CUDA 12.1..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu121

# Set up compatible core libraries - using a specific order to avoid conflicts
echo "Installing compatible core dependencies..."
pip install -U huggingface-hub==0.19.4
pip install -U bitsandbytes==0.41.0
pip install -U accelerate==0.27.0
pip install -U peft==0.6.0
pip install -U tokenizers==0.14.1
pip install -U transformers==4.36.2
pip install -U trl==0.7.10
pip install -U datasets==2.19.0
pip install -U einops==0.7.0

# Install xformers compatible with PyTorch 2.1.2
echo "Installing xformers..."
pip install -U xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121

# Skip flash-attention for RTX4000, install for RTX5000 - without building from source
echo "Checking GPU for flash-attention compatibility..."
if python -c "import torch; print('RTX 5000' in torch.cuda.get_device_name(0))" | grep -q "True"; then
  echo "RTX 5000 GPU detected - installing pre-built flash-attention wheel..."
  pip install "flash-attn<2.3.5" --no-build-isolation --prefer-binary
else
  echo "Not installing flash-attention (not needed for RTX4000 or smaller GPUs)"
fi

# Install unsloth with appropriate flags to avoid compilation issues
echo "Installing Unsloth..."
pip install "unsloth>=2025.3.0,<2025.4.5" --no-deps
pip install safetensors==0.4.1

# Install other compatible packages
echo "Installing utility packages..."
pip install ninja==1.11.1 packaging==23.2 psutil==5.9.8
pip install gdown==5.1.0 fsspec==2024.3.1 boto3==1.28.51

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
    print(f'bitsandbytes version: {bnb.__version__}')
    try:
        lin8bit = bnb.nn.Linear8bitLt(10, 10, has_fp16_weights=False)
        print('bitsandbytes working correctly!')
    except Exception as e:
        print(f'bitsandbytes layer creation error: {e}')
except Exception as e:
    print(f'bitsandbytes import error: {e}')

try:
    import transformers
    print(f'transformers version: {transformers.__version__}')
except Exception as e:
    print(f'transformers error: {e}')

try:
    import unsloth
    print(f'unsloth version: {unsloth.__version__}')
except Exception as e:
    print(f'unsloth error: {e}')

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
"

echo "===================================================================="
echo "Dependency fixes complete!"
echo ""
echo "If you still have issues with unsloth, try running:"
echo "pip install unsloth==2025.3.0 --no-deps"
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