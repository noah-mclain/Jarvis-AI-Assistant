#!/bin/bash

echo "================================================================"
echo "Critical NumPy Fix for Paperspace RTX5000"
echo "================================================================"

# Step 1: Aggressive cleanup - remove problematic packages
echo "Removing conflicting packages..."
pip uninstall -y numpy torch torchvision torchaudio transformers tokenizers huggingface-hub

# Step 2: Clean directories to ensure no corrupted files remain
echo "Cleaning NumPy directories..."
rm -rf /usr/local/lib/python3.11/dist-packages/numpy*
rm -rf /usr/local/lib/python3.11/dist-packages/numpy-*.dist-info

# Step 3: Install NumPy 1.26.4 with maximum constraints
echo "Installing NumPy 1.26.4 with strong constraints..."
pip install numpy==1.26.4 --no-deps --ignore-installed --force-reinstall

# Step 4: Create a barrier file to prevent NumPy from upgrading
echo "Creating NumPy version barrier..."
mkdir -p /usr/local/lib/python3.11/dist-packages/numpy-2.0.0.dist-info
echo "Metadata-Version: 2.1" > /usr/local/lib/python3.11/dist-packages/numpy-2.0.0.dist-info/METADATA
echo "Name: numpy" >> /usr/local/lib/python3.11/dist-packages/numpy-2.0.0.dist-info/METADATA
echo "Version: 2.0.0" >> /usr/local/lib/python3.11/dist-packages/numpy-2.0.0.dist-info/METADATA
echo "This is a barrier file to prevent unwanted numpy upgrades." >> /usr/local/lib/python3.11/dist-packages/numpy-2.0.0.dist-info/METADATA

# Step 5: Verify NumPy installation
echo "Verifying NumPy installation..."
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

# Step 6: Install core components with fixed versions
echo "Installing core PyTorch components..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --no-deps --extra-index-url https://download.pytorch.org/whl/cu121
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu121

# Step 7: Install scipy with explicit NumPy dependency
echo "Installing SciPy with explicit NumPy dependency..."
pip install scipy==1.12.0 --no-deps --ignore-installed
pip install scipy==1.12.0 

# Step 8: Install transformers ecosystem components
echo "Installing transformers ecosystem..."
pip install huggingface-hub==0.17.3 --no-deps
pip install tokenizers==0.14.1 --no-deps
pip install transformers==4.36.2 --no-deps
pip install peft==0.6.0 accelerate==0.27.0 --no-deps

# Step 9: Install dependencies with exact versions
echo "Installing dependencies with exact versions..."
pip install filelock==3.12.2 requests==2.31.0 tqdm==4.66.1 pyyaml==6.0.1 typing-extensions==4.8.0 packaging==23.1 fsspec==2023.6.0

# Step 10: Final verification
echo "Final verification..."
python -c "
import sys
print(f'Python version: {sys.version}')

try:
    import numpy
    print(f'NumPy version: {numpy.__version__}')
    if numpy.__version__.startswith('1.'):
        print('NumPy 1.x confirmed ✅')
    else:
        print('WARNING: NumPy 2.x detected ❌')
except Exception as e:
    print(f'NumPy error: {e}')

try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    if torch.__version__.startswith('2.1.'):
        print('PyTorch 2.1.x confirmed ✅')
    else:
        print(f'Incorrect PyTorch version: {torch.__version__} ❌')
except Exception as e:
    print(f'PyTorch error: {e}')
"

echo "================================================================"
echo "NumPy fix complete!"
echo "If NumPy is showing version 1.26.4, the fix was successful."
echo "You should now be able to run your training without conflicts."
echo "================================================================" 