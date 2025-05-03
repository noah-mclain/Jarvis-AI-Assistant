#!/bin/bash

echo "===================================================================="
echo "COMPLETE FIX: NumPy, PyTorch, and CUDA Compatibility for Paperspace"
echo "===================================================================="

# Stop if any command fails
set -e

# Fix LD_LIBRARY_PATH for CUDA first
echo "Setting up CUDA library paths..."
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/lib64-nvidia
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/lib64-nvidia' >> ~/.bashrc

# 1. CRITICAL: Remove all potentially conflicting packages
echo "Aggressively removing ALL conflicting packages..."
sudo pip uninstall -y numpy scipy matplotlib pandas transformers tokenizers peft
sudo pip uninstall -y huggingface-hub accelerate bitsandbytes flash-attn xformers unsloth
sudo pip uninstall -y torch torchvision torchaudio triton

# 2. Remove ALL numpy directories with forced sudo
echo "Forcefully removing NumPy directories..."
sudo rm -rf /usr/local/lib/python3.11/dist-packages/numpy*
sudo rm -rf /usr/local/lib/python3.11/dist-packages/numpy-*
sudo rm -rf /usr/local/lib/python3.11/site-packages/numpy*
sudo rm -rf /usr/local/lib/python3.11/site-packages/numpy-*

# Clean pip cache
echo "Cleaning pip cache..."
pip cache purge
rm -rf ~/.cache/pip

# 3. Install NumPy 1.26.4 with maximum force using multiple methods
echo "Installing NumPy 1.26.4 with maximum force..."
sudo pip install numpy==1.26.4 --no-deps --force-reinstall --no-cache-dir
pip install numpy==1.26.4 --force-reinstall
python -m pip install numpy==1.26.4 --force-reinstall

# Verify NumPy installation
echo "Verifying NumPy installation..."
if ! python -c "import numpy; print(f'NumPy version: {numpy.__version__}'); exit(0 if numpy.__version__.startswith('1.') else 1)"; then
    echo "❌ CRITICAL ERROR: NumPy installation failed. Cannot continue."
    exit 1
else
    echo "✅ NumPy 1.26.4 successfully installed!"
fi

# 4. Install PyTorch 2.1.2 with CUDA 12.1 support (specific older version)
echo "Installing PyTorch 2.1.2 with CUDA 12.1 support..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA version: {torch.version.cuda}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# 5. Install scientific packages
echo "Installing scientific packages..."
pip install scipy==1.12.0 matplotlib==3.8.3

# 6. Install Hugging Face ecosystem in the correct order
echo "Installing HuggingFace ecosystem in the correct order..."
pip install packaging==23.2 filelock requests tqdm pyyaml typing-extensions fsspec

# First install huggingface hub
pip install huggingface-hub==0.19.4 --no-deps
pip install huggingface-hub==0.19.4 --force-reinstall

# Install tokenizers
pip install tokenizers==0.14.1 --no-deps
pip install tokenizers==0.14.1 --force-reinstall

# Install core components in order
pip install transformers==4.36.2 --no-deps
pip install transformers==4.36.2 --force-reinstall
pip install peft==0.6.0 --no-deps
pip install peft==0.6.0 --force-reinstall
pip install accelerate==0.27.0 --no-deps
pip install accelerate==0.27.0 --force-reinstall
pip install safetensors==0.4.1
pip install datasets==2.19.0 --ignore-installed
pip install trl==0.7.10 --no-deps
pip install trl==0.7.10 --force-reinstall
pip install einops==0.7.0

# 7. Install bitsandbytes with CUDA 12.1 compatibility
echo "Installing bitsandbytes compatible with CUDA 12.1..."
pip install bitsandbytes==0.41.0 --no-deps
pip install bitsandbytes==0.41.0 --force-reinstall

# 8. Install xformers compatible with PyTorch 2.1.2
echo "Installing xformers compatible with PyTorch 2.1.2..."
pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall

# 9. Install unsloth with a version that exists (use 2025.3.3)
echo "Installing unsloth dependencies..."
pip install sentencepiece==0.2.0
pip install hf-transfer wheel>=0.38.0

echo "Installing unsloth 2025.3.3..."
pip install unsloth==2025.3.3 --no-deps

# 10. Configure GPU-specific optimizations
echo "Configuring GPU optimizations..."
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

# Set environment variables for CUDA
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=true

# Add to bashrc
echo 'export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256' >> ~/.bashrc
echo 'export CUDA_LAUNCH_BLOCKING=0' >> ~/.bashrc
echo 'export TOKENIZERS_PARALLELISM=true' >> ~/.bashrc

# 11. Verify all installations
echo "Verifying all installations..."
python -c "
import sys
print(f'Python version: {sys.version}')

try:
    import numpy
    print(f'NumPy version: {numpy.__version__}')
    if numpy.__version__.startswith('1.'):
        print('✅ NumPy 1.x confirmed')
    else:
        print('❌ NumPy 2.x detected')
except Exception as e:
    print(f'❌ NumPy error: {e}')

try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA version: {torch.version.cuda}')
        print(f'GPU: {torch.cuda.get_device_name(0)}')
except Exception as e:
    print(f'❌ PyTorch error: {e}')

try:
    import transformers
    print(f'transformers version: {transformers.__version__}')
except Exception as e:
    print(f'❌ transformers error: {e}')

try:
    import peft
    print(f'peft version: {peft.__version__}')
except Exception as e:
    print(f'❌ peft error: {e}')

try:
    import accelerate
    print(f'accelerate version: {accelerate.__version__}')
except Exception as e:
    print(f'❌ accelerate error: {e}')

try:
    import bitsandbytes as bnb
    print(f'bitsandbytes loaded')
    try:
        lin8bit = bnb.nn.Linear8bitLt(10, 10, has_fp16_weights=False)
        print('✅ bitsandbytes working correctly!')
    except Exception as e:
        print(f'❌ bitsandbytes layer creation error: {e}')
except Exception as e:
    print(f'❌ bitsandbytes import error: {e}')

try:
    import unsloth
    print(f'unsloth version: {unsloth.__version__ if hasattr(unsloth, \"__version__\") else \"unknown\"}')
except Exception as e:
    print(f'❌ unsloth error: {e}')
"

echo "===================================================================="
echo "Fix complete! Your environment should now have:"
echo "- NumPy 1.26.4 (compatible with all ML libraries)"
echo "- PyTorch 2.1.2 with CUDA 12.1"
echo "- Transformers 4.36.2, PEFT 0.6.0, Accelerate 0.27.0"
echo "- Unsloth 2025.3.3 (compatible with this setup)"
echo ""
echo "This should resolve all compatibility issues for Paperspace RTX5000."
echo "====================================================================" 