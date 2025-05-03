#!/bin/bash

echo "===================================================================="
echo "NumPy + Unsloth Compatibility Fix for Paperspace RTX5000"
echo "===================================================================="

# Step 1: Fix NumPy first (most critical)
echo "CRITICAL: Fixing NumPy version..."
sudo rm -rf /usr/local/lib/python3.11/dist-packages/numpy*
sudo rm -rf /usr/local/lib/python3.11/dist-packages/numpy-*
pip uninstall -y numpy
pip cache purge

# Install NumPy 1.26.4 with maximum force
echo "Installing NumPy 1.26.4..."
sudo pip install numpy==1.26.4 --no-deps --force-reinstall --no-cache-dir
pip install numpy==1.26.4 --force-reinstall

# Verify NumPy installation
if ! python -c "import numpy; print(f'NumPy version: {numpy.__version__}'); exit(0 if numpy.__version__.startswith('1.') else 1)"; then
    echo "❌ CRITICAL: NumPy is still not at version 1.x. Cannot continue."
    exit 1
else
    echo "✅ NumPy 1.26.4 successfully installed!"
fi

# Step 2: Fix unsloth and dependencies
echo "Fixing unsloth and dependencies..."

# Remove unsloth and its conflicts
pip uninstall -y unsloth unsloth_zoo bitsandbytes peft accelerate transformers

# Install the correct versions in the right order
echo "Installing transformers ecosystem (specific versions)..."
pip install transformers==4.36.2 --no-deps
pip install transformers==4.36.2 --force-reinstall
pip install peft==0.6.0 --no-deps
pip install peft==0.6.0 --force-reinstall
pip install accelerate==0.27.0 --no-deps
pip install accelerate==0.27.0 --force-reinstall

# Install compatible bitsandbytes
echo "Installing bitsandbytes..."
pip install --no-cache-dir bitsandbytes==0.41.0

# Install an older compatible version of unsloth
echo "Installing older compatible version of unsloth..."
pip install sentencepiece==0.2.0
pip install unsloth==2023.12.17 --no-deps

# Step 3: Verify the installation
echo "Verifying installation..."
python -c "
import sys
print(f'Python version: {sys.version}')

try:
    import numpy
    print(f'NumPy version: {numpy.__version__}')
    if numpy.__version__.startswith('1.'):
        print('✅ NumPy 1.x confirmed')
    else:
        print('❌ NumPy 2.x detected - this will cause problems!')
except Exception as e:
    print(f'❌ NumPy error: {e}')

try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
except Exception as e:
    print(f'❌ PyTorch error: {e}')

try:
    import transformers
    print(f'✅ transformers version: {transformers.__version__}')
except Exception as e:
    print(f'❌ transformers error: {e}')

try:
    import peft
    print(f'✅ peft version: {peft.__version__}')
except Exception as e:
    print(f'❌ peft error: {e}')

try:
    import bitsandbytes as bnb
    print(f'✅ bitsandbytes working')
except Exception as e:
    print(f'❌ bitsandbytes error: {e}')

try:
    import unsloth
    print(f'✅ unsloth working')
except Exception as e:
    print(f'❌ unsloth error: {e}')
"

echo "===================================================================="
echo "Fix complete! Your environment should now have:"
echo "- NumPy 1.26.4 (compatible with all ML libraries)"
echo "- unsloth 2023.12.17 (older but compatible version)"
echo "- Compatible versions of transformers, peft and accelerate"
echo ""
echo "If you're still having issues, please run:"
echo "./fix_all_deps.sh"
echo "====================================================================" 