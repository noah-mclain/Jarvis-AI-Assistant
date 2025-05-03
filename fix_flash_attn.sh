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
  echo "RTX 5000 GPU detected - installing pre-built flash-attention wheel..."
  
  # Try multiple approaches to install flash-attn
  echo "Attempt 1: Using pre-built wheel with prefer-binary..."
  pip install "flash-attn<2.3.5" --prefer-binary --no-build-isolation || \
  
  echo "Attempt 2: Using specific version 2.3.3..."
  pip install flash-attn==2.3.3 --prefer-binary --no-build-isolation || \
  
  echo "Attempt 3: Trying explicit CUDA installation..."
  pip install "flash-attn<2.3.5" --prefer-binary --extra-index-url https://download.pytorch.org/whl/cu121 || \
  
  echo "Skipping flash-attention installation - this is fine, other optimizations will still work"
else
  echo "RTX 4000 or other GPU detected - flash-attention not recommended for this GPU"
  echo "Skipping flash-attention installation"
fi

# Install alternative optimizations
echo "Installing alternative memory optimizations..."
pip install -U xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121

echo "===================================================================="
echo "Flash-attention fix complete! If installation failed, this is okay,"
echo "as the system will fall back to using xformers optimizations instead."
echo "====================================================================" 