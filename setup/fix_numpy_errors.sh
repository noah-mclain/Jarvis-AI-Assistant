#!/bin/bash

echo "================================================================"
echo "CRITICAL: Emergency NumPy Fix for Paperspace"
echo "================================================================"

# EMERGENCY CLEANUP: Force remove corrupted NumPy with sudo permissions
echo "EMERGENCY: Forcefully removing corrupted NumPy installation..."
sudo rm -rf /usr/local/lib/python3.11/dist-packages/numpy*
sudo rm -rf /usr/local/lib/python3.11/dist-packages/numpy-*

# Also remove any possible remaining trace of NumPy 2.x
rm -rf /tmp/pip-*
pip cache purge

# Uninstall any remaining NumPy via pip
echo "Removing all NumPy via pip..."
pip uninstall -y numpy

# Clear the environment
echo "Clearing Python environment cache..."
python -c "import sys; sys.path = [p for p in sys.path if not 'numpy' in p]; print(sys.path)"

# Install NumPy 1.26.4 with maximum force
echo "Installing NumPy 1.26.4 with maximum force..."
sudo pip install numpy==1.26.4 --no-deps --force-reinstall --no-cache-dir

# Second install with user permissions to ensure proper setup
echo "Installing again with normal permissions..."
pip install numpy==1.26.4 --force-reinstall

# Verify the install
echo "Verifying NumPy installation..."
if python -c "import numpy; print(f'NumPy version: {numpy.__version__}'); exit(0 if numpy.__version__.startswith('1.') else 1)"; then
    echo "✅ NumPy 1.26.4 successfully installed!"
else
    echo "❌ CRITICAL ERROR: NumPy is still not correctly installed."
    echo "Attempting one last approach with system Python..."
    sudo $(which python3) -m pip install numpy==1.26.4 --force-reinstall --no-deps
    
    # Final check
    if python -c "import numpy; print(f'NumPy version: {numpy.__version__}'); exit(0 if numpy.__version__.startswith('1.') else 1)"; then
        echo "✅ NumPy 1.26.4 successfully installed on final attempt!"
    else
        echo "❌ CRITICAL ERROR: NumPy installation failed after multiple attempts."
        echo "Please try the following manual commands:"
        echo "sudo rm -rf /usr/local/lib/python3.11/dist-packages/numpy*"
        echo "sudo $(which python3) -m pip install numpy==1.26.4 --no-deps"
        exit 1
    fi
fi

echo "================================================================"
echo "NumPy fix complete! You should now be able to install other packages."
echo "If you still encounter issues, run fix_all_deps.sh for a complete fix."
echo "================================================================" 