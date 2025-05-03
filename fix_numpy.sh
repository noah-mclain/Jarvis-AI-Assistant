#!/bin/bash

echo "================================================================"
echo "CRITICAL: NumPy/PyTorch Dependency Fix for Paperspace RTX5000"
echo "================================================================"

# Step 1: Extremely aggressive cleanup - forcefully remove corrupted installations
echo "Forcefully removing corrupted package installations..."
rm -rf /usr/local/lib/python3.11/dist-packages/numpy*
rm -rf /usr/local/lib/python3.11/dist-packages/torch*
rm -rf /usr/local/lib/python3.11/dist-packages/transformers*
rm -rf /usr/local/lib/python3.11/dist-packages/accelerate*
rm -rf /usr/local/lib/python3.11/dist-packages/peft*
rm -rf /usr/local/lib/python3.11/dist-packages/unsloth*

# Step 2: Now use pip to remove any remaining dependencies
echo "Uninstalling remaining dependencies..."
pip uninstall -y numpy torch torchvision torchaudio transformers huggingface-hub tokenizers accelerate peft unsloth scipy triton

# Step 3: Install NumPy 1.26.4 with maximum force
echo "Installing NumPy 1.26.4..."
pip install numpy==1.26.4 --no-deps --force-reinstall

# Step 4: Install PyTorch ecosystem with proper CUDA support
echo "Installing PyTorch 2.1.2 with CUDA support..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu121 --no-deps
pip install triton==2.1.0 --no-deps 

# Step 5: Install core dependencies in specific compatible versions
echo "Installing core ML dependencies with exact versions..."
pip install scipy==1.12.0 --no-deps
pip install filelock==3.12.2 requests==2.31.0 tqdm==4.66.1 \
            pyyaml==6.0.1 typing-extensions==4.8.0 packaging==23.1 \
            fsspec==2023.6.0 --no-deps

# Step 6: HuggingFace ecosystem - install older versions known to be compatible
echo "Installing HuggingFace ecosystem with compatible versions..."
pip install huggingface-hub==0.17.3 --no-deps
pip install tokenizers==0.14.1 --no-deps
pip install transformers==4.36.2 --no-deps
pip install peft==0.6.0 accelerate==0.27.0 --no-deps

# Step 7: Install unsloth with compatible version
echo "Installing Unsloth with compatible version..."
pip install sentencepiece==0.2.0 --no-deps
pip install unsloth==2023.12.17 --no-deps

# Step 8: Verify the critical package versions
echo "Verifying critical package installations..."
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

try:
    import transformers
    print(f'Transformers version: {transformers.__version__}')
except Exception as e:
    print(f'Transformers error: {e}')
"

echo "================================================================"
echo "NumPy and dependency fix complete!"
echo ""
echo "IMPORTANT: If NumPy shows version 1.26.4, the fix was successful."
echo "You may now need to run your original setup script again to install"
echo "additional packages, but NumPy will remain at the correct version."
echo "================================================================" 