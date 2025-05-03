#!/bin/bash

echo "===================================================================="
echo "Flash-Attention Fix for Paperspace"
echo "===================================================================="

# Ensure we have numpy installed
pip install numpy==1.26.4

# Uninstall any existing flash-attn
echo "Removing any existing flash-attn installations..."
pip uninstall -y flash-attn

# Check GPU type
echo "Checking GPU type..."
if python -c "import torch; print('RTX 5000' in torch.cuda.get_device_name(0))" | grep -q "True"; then
  echo "RTX 5000 GPU detected - attempting to install pre-built flash-attention wheel..."
  
  # First, try downloading a pre-built wheel to avoid compilation
  echo "Attempting to download pre-built wheel..."
  mkdir -p ~/flash_attn_wheels
  cd ~/flash_attn_wheels
  
  # Try to download for CUDA 12.1, Python 3.11
  echo "Downloading pre-built wheel for CUDA 12.1 + PyTorch 2.1.x..."
  wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.3.3/flash_attn-2.3.3+cu121torch2.1cxx11abiFALSE-cp311-cp311-linux_x86_64.whl -O flash_attn-2.3.3-cp311-cp311-linux_x86_64.whl || \
  wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.3.2/flash_attn-2.3.2+cu121torch2.1cxx11abiFALSE-cp311-cp311-linux_x86_64.whl -O flash_attn-2.3.2-cp311-cp311-linux_x86_64.whl || \
  echo "Failed to download pre-built wheel directly"
  
  # Try to install the downloaded wheel
  if ls flash_attn*.whl 1> /dev/null 2>&1; then
    echo "Installing downloaded wheel..."
    pip install flash_attn*.whl
    if python -c "import flash_attn" 2>/dev/null; then
      echo "Successfully installed flash-attention from pre-built wheel!"
      cd - > /dev/null
      echo "Flash-attention is now available!"
      exit 0
    fi
  fi
  
  cd - > /dev/null
  
  # Extremely simplified attempt
  echo "Attempting simplified installation method..."
  pip install 'flash-attn<2.1.0' --prefer-binary || echo "Flash-attention installation failed"
else
  echo "RTX 4000 or other GPU detected - flash-attention is not recommended for this GPU"
  echo "Skipping flash-attention installation"
fi

# Install alternative optimizations
echo "Installing alternative memory optimizations..."
pip install -U xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121

# Verify installation
echo "Verifying flash-attention installation..."
if python -c "import flash_attn" 2>/dev/null; then
  echo "flash-attention is successfully installed and importable!"
  python -c "import flash_attn; print(f'flash-attention version: {flash_attn.__version__}' if hasattr(flash_attn, '__version__') else 'version unknown')"
else
  echo "flash-attention is not installed or not importable."
  echo "This is expected - the system will use xformers optimizations instead."
fi

echo "===================================================================="
echo "Flash-attention setup complete!"
echo "NOTE: If flash-attention installation failed, that's completely fine."
echo "Jarvis will automatically fall back to using xformers optimizations."
echo "====================================================================" 