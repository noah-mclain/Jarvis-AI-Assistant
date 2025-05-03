#!/bin/bash

echo "===================================================================="
echo "Flash-Attention Fix for Paperspace"
echo "===================================================================="

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
  wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.3.3/flash_attn-2.3.3+cu121torch2.1cxx11abiFALSE-cp311-cp311-linux_x86_64.whl || \
  wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.3.2/flash_attn-2.3.2+cu121torch2.1cxx11abiFALSE-cp311-cp311-linux_x86_64.whl || \
  echo "Failed to download pre-built wheel directly"
  
  # Try to install the downloaded wheel
  if ls flash_attn*.whl 1> /dev/null 2>&1; then
    echo "Installing downloaded wheel..."
    pip install flash_attn*.whl
    if python -c "import flash_attn" 2>/dev/null; then
      echo "Successfully installed flash-attention from pre-built wheel!"
      cd - > /dev/null
      exit 0
    fi
  fi
  
  cd - > /dev/null
  
  # If direct wheel download failed, try different installation methods
  echo "Attempting alternative installation methods..."
  
  echo "Method 1: Using pre-built wheel with prefer-binary flag..."
  pip install "flash-attn==2.3.3" --prefer-binary --no-build-isolation || \
  
  echo "Method 2: Using specific version 2.3.2..."
  pip install "flash-attn==2.3.2" --prefer-binary --no-build-isolation || \
  
  echo "Method 3: Trying another version..."
  pip install "flash-attn==2.3.0" --prefer-binary --no-build-isolation || \
  
  echo "Method 4: Trying with explicit CUDA installation..."
  pip install "flash-attn<2.3.5" --prefer-binary --no-build-isolation --extra-index-url https://download.pytorch.org/whl/cu121 || \
  
  echo "All installation attempts failed. This is okay - system will use xformers optimizations instead."
  echo "The flash-attention package requires specific CUDA configurations that might not be available in your environment."
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
  python -c "import flash_attn; print(f'flash-attention version: {flash_attn.__version__}')"
else
  echo "flash-attention is not installed or not importable - this is okay!"
  echo "The system will use xformers optimizations instead."
fi

echo "===================================================================="
echo "Flash-attention setup complete!"
echo "NOTE: If flash-attention installation failed, that's completely fine."
echo "Jarvis will automatically fall back to using xformers optimizations."
echo "====================================================================" 