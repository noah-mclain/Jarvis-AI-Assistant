#!/bin/bash

echo "===================================================================="
echo "Minimal dependency fix for Jarvis AI Assistant"
echo "===================================================================="

# Completely uninstall NumPy, SciPy and essential ML packages
echo "Removing all conflicting packages..."
pip uninstall -y numpy torch torchvision torchaudio transformers tokenizers huggingface-hub peft accelerate xformers scipy scikit-learn pandas

# Force system to forget about NumPy
echo "Removing NumPy directories..."
rm -rf /usr/local/lib/python3.11/dist-packages/numpy*
rm -rf /usr/local/lib/python3.11/dist-packages/scipy*

# Install NumPy 1.x with maximum force
echo "Installing NumPy 1.26.4 (with extreme prejudice)..."
pip install numpy==1.26.4 --no-deps --force-reinstall --ignore-installed

# Verify NumPy version
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

# Install PyTorch ecosystem with CUDA
echo "Installing PyTorch ecosystem..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu121 --force-reinstall

# Install SciPy
echo "Installing SciPy..."
pip install scipy==1.12.0 --force-reinstall

# Install core HuggingFace packages (minimal set)
echo "Installing HuggingFace core packages..."
pip install huggingface-hub==0.19.4 tokenizers==0.14.1 transformers==4.36.2 --force-reinstall

# Install other essentials
echo "Installing other essentials..."
pip install peft==0.6.0 accelerate==0.27.0 --force-reinstall

# Install a newer version of unsloth that should be compatible
echo "Installing unsloth (newer version)..."
pip install unsloth==2025.3.3 --no-deps

echo "===================================================================="
echo "Minimal dependency fix complete!"
echo "Try running your code now. If you need more packages, install them"
echo "one at a time using 'pip install package==version'"
echo "=====================================================================" 