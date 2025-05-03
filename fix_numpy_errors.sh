#!/bin/bash

echo "================================================================"
echo "CRITICAL: Emergency NumPy Fix for Paperspace"
echo "================================================================"

# EMERGENCY CLEANUP: Force remove corrupted NumPy
echo "EMERGENCY: Forcefully removing corrupted NumPy installation..."
rm -rf /usr/local/lib/python3.11/dist-packages/numpy*
rm -rf /usr/local/lib/python3.11/dist-packages/numpy-*

# Also remove any possible remaining trace of NumPy 2.x
rm -rf /tmp/pip-*
pip cache purge

# Clear the environment
echo "Clearing Python environment cache..."
python -c "import sys; sys.path = [p for p in sys.path if not 'numpy' in p]; print(sys.path)"

# Install NumPy 1.26.4 with maximum force
echo "Installing NumPy 1.26.4 with maximum force..."
pip install numpy==1.26.4 --no-deps --force-reinstall --no-cache-dir

# Verify the install
echo "Verifying NumPy installation..."
if python -c "import numpy; print(f'NumPy version: {numpy.__version__}'); exit(0 if numpy.__version__.startswith('1.') else 1)"; then
    echo "✅ NumPy 1.26.4 successfully installed!"
else
    echo "❌ CRITICAL ERROR: NumPy is still not correctly installed."
    echo "Please manually install with:"
    echo "sudo rm -rf /usr/local/lib/python3.11/dist-packages/numpy*"
    echo "sudo pip install numpy==1.26.4 --force-reinstall --no-deps"
    exit 1
fi

echo "================================================================"
echo "NumPy fix complete! You can now continue with your setup."
echo "================================================================" 