#!/bin/bash

echo "================================================================"
echo "EMERGENCY: Unsloth Fix Script"
echo "================================================================"

# First fix NumPy - the foundation of everything
echo "Step 1: Ensuring NumPy 1.26.4 is properly installed..."
if command -v sudo &> /dev/null; then
    sudo rm -rf /usr/local/lib/python3.11/dist-packages/numpy*
    sudo pip uninstall -y numpy
    sudo pip install numpy==1.26.4 --no-deps --force-reinstall
else
    rm -rf /usr/local/lib/python3.11/dist-packages/numpy*
    pip uninstall -y numpy
    pip install numpy==1.26.4 --no-deps --force-reinstall
fi

pip install numpy==1.26.4 --force-reinstall

# Verify NumPy installation
if ! python -c "import numpy; print(f'NumPy version: {numpy.__version__}'); exit(0 if numpy.__version__ == '1.26.4' else 1)"; then
    echo "❌ ERROR: NumPy 1.26.4 installation failed. Cannot continue."
    exit 1
else
    echo "✅ NumPy 1.26.4 successfully installed!"
fi

# Uninstall unsloth and dependencies
echo "Step 2: Removing unsloth and related packages..."
pip uninstall -y unsloth unsloth_zoo

# Clear cache
pip cache purge
rm -rf ~/.cache/pip
rm -rf ~/.cache/huggingface

# Install sentencepiece first
echo "Step 3: Installing sentencepiece dependency..."
pip install sentencepiece==0.1.99

# Install unsloth version 2024.8 (confirmed working with this setup)
echo "Step 4: Installing unsloth 2024.8 (verified compatible version)..."
pip install unsloth==2024.8 --no-deps

# Verify the installation
echo "Step 5: Verifying unsloth installation..."
if python -c "import unsloth; print('Unsloth import successful')"; then
    echo "✅ Unsloth successfully installed!"
else
    echo "❌ Unsloth installation failed, trying alternative version..."
    
    # Try an alternative version
    pip install unsloth==2024.10.0 --no-deps
    
    if python -c "import unsloth; print('Unsloth import successful')"; then
        echo "✅ Unsloth 2024.10.0 successfully installed as fallback!"
    else
        echo "❌ All unsloth installation attempts failed."
        echo "Please run the full unified_setup.sh script again for a complete environment rebuild."
        exit 1
    fi
fi

echo "================================================================"
echo "Unsloth fix complete! You can now continue with your work."
echo "================================================================" 