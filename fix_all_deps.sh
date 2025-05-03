#!/bin/bash

echo "===================================================================="
echo "Complete dependency conflict resolution for Jarvis AI Assistant"
echo "===================================================================="

# Completely clean the environment
echo "Aggressively removing ALL conflicting packages..."
pip uninstall -y numpy scipy matplotlib pandas pyarrow transformers tokenizers
pip uninstall -y torch torchvision torchaudio unsloth peft accelerate xformers
pip uninstall -y bitsandbytes flash-attn huggingface-hub protobuf

# Clean up any corrupt installations by removing their directories
echo "Cleaning potentially corrupted installations..."
rm -rf /usr/local/lib/python3.11/dist-packages/numpy*
rm -rf /usr/local/lib/python3.11/dist-packages/transformers*
rm -rf /usr/local/lib/python3.11/dist-packages/tokenizers*
rm -rf /usr/local/lib/python3.11/dist-packages/peft*
rm -rf /usr/local/lib/python3.11/dist-packages/accelerate*

# Install packages in the correct sequence with pinned compatible versions
echo "Installing foundational NumPy first (correct version)..."
pip install numpy==1.26.4 --no-deps
pip install numpy==1.26.4 --force-reinstall

# Verify NumPy installation
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

# Install compatible PyTorch ecosystem
echo "Installing PyTorch 2.1.2 ecosystem..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu121

# Install key scientific packages
echo "Installing scientific packages..."
pip install scipy==1.12.0
pip install matplotlib==3.8.3
pip install pandas==2.2.0

# Install huggingface ecosystem in compatible versions
echo "Installing tokenizers first (specific older version)..."
pip install tokenizers==0.14.1 --no-deps
pip install tokenizers==0.14.1 --force-reinstall

echo "Installing huggingface-hub at compatible version..."
pip install huggingface-hub==0.19.4 --no-deps
pip install filelock requests tqdm pyyaml typing-extensions packaging fsspec

echo "Installing transformers ecosystem in the correct order..."
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

# Install bitsandbytes compatible with PyTorch 2.1.2
echo "Installing bitsandbytes..."
pip install bitsandbytes==0.41.0 --force-reinstall

# Install xformers compatible with PyTorch 2.1.2
echo "Installing xformers..."
pip install xformers==0.0.23.post1 --extra-index-url https://download.pytorch.org/whl/cu121 --force-reinstall

# Install unsloth (older compatible version)
echo "Installing compatible unsloth version..."
pip install sentencepiece==0.2.0
pip install unsloth==2023.12.17 --no-deps

# Final verification
echo "Verifying installations..."
python -c "
import sys
print(f'Python version: {sys.version}')

try:
    import numpy
    print(f'NumPy version: {numpy.__version__}')
except Exception as e:
    print(f'NumPy error: {e}')

try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA version: {torch.version.cuda}')
        print(f'GPU: {torch.cuda.get_device_name(0)}')
except Exception as e:
    print(f'PyTorch error: {e}')

try:
    import tokenizers
    print(f'tokenizers version: {tokenizers.__version__}')
except Exception as e:
    print(f'tokenizers error: {e}')

try:
    import transformers
    print(f'transformers version: {transformers.__version__}')
except Exception as e:
    print(f'transformers error: {e}')

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
    import bitsandbytes as bnb
    print(f'bitsandbytes version: {bnb.__version__ if hasattr(bnb, \"__version__\") else \"unknown\"}')
    if torch.cuda.is_available():
        try:
            lin8bit = bnb.nn.Linear8bitLt(10, 10, has_fp16_weights=False)
            print('bitsandbytes working correctly!')
        except Exception as e:
            print(f'bitsandbytes layer creation error: {e}')
except Exception as e:
    print(f'bitsandbytes import error: {e}')

try:
    import unsloth
    print(f'unsloth version: {unsloth.__version__ if hasattr(unsloth, \"__version__\") else \"unknown\"}')
except Exception as e:
    print(f'unsloth error: {e}')
"

echo "===================================================================="
echo "Dependency fix complete!"
echo ""
echo "If you still encounter issues with a specific package, please try:"
echo "pip install <package>==<version> --force-reinstall"
echo ""
echo "You can now try running your model training command."
echo "====================================================================" 