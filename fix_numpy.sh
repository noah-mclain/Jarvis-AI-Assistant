#!/bin/bash

echo "========================================================"
echo "Aggressive NumPy/SciPy Fix Script for Jarvis AI Assistant"
echo "========================================================"

# First, aggressively remove all related packages
echo "Removing conflicting packages..."
pip uninstall -y numpy scipy matplotlib scikit-image pandas pyarrow transformers

# Clean any corrupt NumPy installation
echo "Cleaning potentially corrupted NumPy installation..."
rm -rf /usr/local/lib/python3.11/dist-packages/numpy*
rm -rf /usr/local/lib/python3.11/dist-packages/numpy-*.dist-info

# Install NumPy with specific version
echo "Installing NumPy 1.26.4..."
pip install numpy==1.26.4 --no-deps
pip install numpy==1.26.4 --no-deps  # Double install to be sure

# Verify NumPy installation
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

# Install SciPy with the correct NumPy
echo "Installing SciPy 1.12.0..."
pip install scipy==1.12.0 --no-deps
pip install scipy==1.12.0  # Get dependencies while keeping NumPy

# Install core dependencies in correct order
echo "Installing core ML dependencies..."
pip install transformers==4.36.2 --no-deps
pip install accelerate==0.27.0 --no-deps
pip install peft==0.6.0 --no-deps

# Install with dependencies but ignore the installed ones
pip install transformers==4.36.2 --ignore-installed
pip install accelerate==0.27.0 --ignore-installed
pip install peft==0.6.0 --ignore-installed

# Final verification
echo "Verifying installations..."
python -c "
import sys
print(f'Python version: {sys.version}')
import numpy
print(f'NumPy version: {numpy.__version__}')
import scipy
print(f'SciPy version: {scipy.__version__}')
import transformers
print(f'Transformers version: {transformers.__version__}')
try:
    import peft
    print(f'PEFT version: {peft.__version__}')
except ImportError:
    print('PEFT not installed properly')
try:
    import accelerate
    print(f'Accelerate version: {accelerate.__version__}')
except ImportError:
    print('Accelerate not installed properly')
"

echo "========================================================"
echo "NumPy/SciPy fix complete!"
echo "You should now be able to run jarvis_unified.py properly."
echo "========================================================" 