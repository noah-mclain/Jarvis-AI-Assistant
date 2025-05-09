#!/bin/bash

echo "===================================================================="
echo "Jarvis AI Assistant - Consolidated Dependencies Installation"
echo "===================================================================="

# Stop on errors
set -e

# Detect environment
IN_COLAB=0
IN_PAPERSPACE=0
if python -c "import google.colab" 2>/dev/null; then
    echo "Running in Google Colab environment"
    IN_COLAB=1
elif [ -d "/notebooks" ] || [ -d "/storage" ]; then
    echo "Running in Paperspace environment"
    IN_PAPERSPACE=1
else
    echo "Running in standard environment"
fi

# Function to install a package with version constraint
install_package() {
    package=$1
    version=$2
    options=$3

    echo "Installing $package $version..."

    if [ -z "$version" ]; then
        pip install "$package" --no-deps $options
        pip install "$package" $options
    else
        pip install "$package$version" --no-deps $options
        pip install "$package$version" $options
    fi

    # Check if installation was successful
    if python -c "import $package" 2>/dev/null; then
        echo "✅ $package installed successfully"
    else
        echo "⚠️ $package installation may have issues"
    fi
}

# Skip CUDA check initially since PyTorch isn't installed yet
echo "Checking for CUDA availability will be done after PyTorch installation..."

# Set a flag to check CUDA later
CHECK_CUDA_LATER=1

# Set LD_LIBRARY_PATH for CUDA libraries
echo "Setting up CUDA library paths..."
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/lib64-nvidia
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/lib64-nvidia' >> ~/.bashrc

# 1. Install core dependencies
echo "Installing core dependencies with specific compatible versions..."

# 1.1 NumPy 1.26.4 (crucial for compatibility)
echo "Installing NumPy 1.26.4 (foundation package)..."
pip install numpy==1.26.4 --no-deps --force-reinstall
pip install numpy==1.26.4 --force-reinstall

# Verify NumPy installation
if ! python -c "import numpy; print(f'NumPy version: {numpy.__version__}'); exit(0 if numpy.__version__ == '1.26.4' else 1)"; then
    echo "ERROR: NumPy 1.26.4 installation failed. Cannot continue."
    exit 1
else
    echo "✅ NumPy 1.26.4 successfully installed!"
fi

# 1.2 Install PyTorch 2.1.2 with CUDA 12.1 (verified compatible version)
echo "Installing PyTorch 2.1.2 with CUDA 12.1 support..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu121

# Now check if CUDA is available after PyTorch installation
echo "Checking for CUDA availability..."
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo "✅ CUDA is available!"
    # Get CUDA version
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
    echo "CUDA Version: $CUDA_VERSION"
    # Check PyTorch version
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    echo "PyTorch Version: $TORCH_VERSION"
else
    echo "⚠️ CUDA is not available. Continuing anyway, but GPU acceleration won't work."
fi

# 1.3 Install core scientific packages
echo "Installing core scientific packages..."
pip install scipy==1.12.0 matplotlib==3.8.3 pandas==2.2.0

# 1.4 Install utility packages required by transformers ecosystem
pip install filelock==3.12.2 requests==2.31.0 tqdm==4.66.1
pip install pyyaml==6.0.1 typing-extensions==4.13.2 packaging==23.1
pip install fsspec==2023.6.0 psutil==5.9.5 ninja==1.11.1 wheel
pip install markdown protobuf\<4.24 werkzeug

# 1.5 Hugging Face ecosystem - in exact order for compatibility
echo "Installing Hugging Face ecosystem..."
pip install safetensors==0.4.0
pip install huggingface-hub==0.19.4 --no-deps
pip install huggingface-hub==0.19.4  # Second install pulls in compatible dependencies

pip install tokenizers==0.14.0 --no-deps
pip install tokenizers==0.14.0

# Install transformers with all dependencies to ensure transformers.utils is available
echo "Installing transformers with all dependencies..."
pip uninstall -y transformers  # Remove any existing installation
pip install transformers==4.36.2  # Install with all dependencies

# Verify transformers.utils is available
python -c "
try:
    import transformers.utils
    print('✅ transformers.utils is available')
except ImportError as e:
    print(f'❌ transformers.utils is NOT available: {e}')
    print('Trying alternative installation method...')
    import os
    os.system('pip install -U pip')
    os.system('pip install transformers==4.36.2 --force-reinstall')
"

pip install peft==0.6.0 --no-deps
pip install peft==0.6.0

pip install accelerate==0.25.0 --no-deps
pip install accelerate==0.25.0

pip install datasets==2.14.5 --no-deps
pip install datasets==2.14.5

pip install trl==0.7.4 --no-deps
pip install trl==0.7.4

pip install einops==0.7.0

# 1.6 Install optimization libraries with exact versions
echo "Installing optimization libraries..."
pip install bitsandbytes==0.41.0
pip install triton==2.1.0

# 2. Install xFormers with enhanced attention support
echo "Installing xFormers with enhanced attention support..."
pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121 --no-deps
pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121

# 3. Install additional dependencies for enhanced attention mechanisms
echo "Installing additional dependencies for enhanced attention mechanisms..."
pip install einops==0.7.0 --no-deps  # Required for attention operations
pip install opt_einsum==3.3.0 --no-deps  # Optimized einsum operations for attention

# 4. Install unsloth (version 2024.8 is confirmed to work with this setup)
echo "Installing unsloth dependencies..."
pip install sentencepiece==0.1.99
pip install unsloth==2024.8 --no-deps

# 5. Install Flash Attention 2.5.5 with proper dependency handling
echo "Installing Flash Attention 2.5.5..."

# Ensure all build dependencies are available
pip install packaging==23.1 ninja==1.11.1 wheel setuptools --no-deps
pip install packaging==23.1 ninja==1.11.1 wheel setuptools

# Try multiple installation methods for Flash Attention
echo "Trying Flash Attention installation method 1..."
pip install flash-attn==2.5.5 --no-build-isolation --no-deps || true

# Verify Flash Attention installation
if python -c "import flash_attn; print(f'Flash Attention version: {flash_attn.__version__}')" 2>/dev/null; then
    echo "✅ Flash Attention 2.5.5 successfully installed!"
else
    echo "⚠️ Flash Attention installation method 1 failed. Trying method 2..."
    pip install flash-attn==2.5.5 --no-build-isolation || true

    # Verify again
    if python -c "import flash_attn; print(f'Flash Attention version: {flash_attn.__version__}')" 2>/dev/null; then
        echo "✅ Flash Attention successfully installed with method 2!"
    else
        echo "⚠️ Flash Attention installation method 2 failed. Trying method 3 (pre-built wheel)..."
        # Try to find a pre-built wheel
        pip install flash-attn==2.5.5 --find-links https://github.com/Dao-AILab/flash-attention/releases/tag/v2.5.5 || true

        # Final verification
        if python -c "import flash_attn; print(f'Flash Attention version: {flash_attn.__version__}')" 2>/dev/null; then
            echo "✅ Flash Attention successfully installed with method 3!"
        else
            echo "⚠️ Flash Attention installation failed after multiple attempts."
            echo "Training will continue without Flash Attention."
        fi
    fi
fi

# 6. Install additional dependencies
echo "Installing additional dependencies..."
install_package "markdown" ""
pip install "protobuf<4.24" --no-deps
pip install "protobuf<4.24"
echo "✅ protobuf<4.24 installed"
install_package "werkzeug" ""
install_package "pandas" "==2.2.0"
install_package "huggingface_hub" "==0.19.4"

# 7. Install TensorBoard for training visualization
echo "Installing TensorBoard..."
pip install tensorboard==2.15.2 --no-deps
pip install tensorboard==2.15.2
python -c "
try:
    import tensorboard
    print(f'✅ TensorBoard version: {tensorboard.__version__}')
except ImportError as e:
    print(f'❌ TensorBoard error: {e}')
    print('Installing TensorBoard with pip...')
    import os
    os.system('pip install tensorboard==2.15.2')
"

# Verify installations
echo "Verifying installations..."

# Check xFormers
python -c "
try:
    import xformers
    import xformers.ops
    print(f'xFormers version: {xformers.__version__ if hasattr(xformers, \"__version__\") else \"installed\"}')
    print('✅ xFormers successfully imported')
except Exception as e:
    print(f'❌ xFormers error: {e}')
"

# Check einops
python -c "
try:
    import einops
    print(f'einops version: {einops.__version__ if hasattr(einops, \"__version__\") else \"installed\"}')
    print('✅ einops successfully imported')
except Exception as e:
    print(f'❌ einops error: {e}')
"

# Check opt_einsum
python -c "
try:
    import opt_einsum
    print(f'opt_einsum version: {opt_einsum.__version__ if hasattr(opt_einsum, \"__version__\") else \"installed\"}')
    print('✅ opt_einsum successfully imported')
except Exception as e:
    print(f'❌ opt_einsum error: {e}')
"

echo "===================================================================="
echo "Performing final verification of critical dependencies..."
echo "===================================================================="

# Verify critical dependencies
python -c "
import sys
print(f'Python version: {sys.version}')

try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA device: {torch.cuda.get_device_name(0)}')
except Exception as e:
    print(f'❌ PyTorch error: {e}')
    # Install PyTorch again as a fallback
    import os
    os.system('pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu121')

try:
    import transformers
    print(f'Transformers version: {transformers.__version__}')

    # Specifically check for transformers.utils
    try:
        import transformers.utils
        print(f'✅ transformers.utils is available')
    except ImportError as e:
        print(f'❌ transformers.utils is NOT available: {e}')
        print('Reinstalling transformers with all dependencies...')
        import os
        os.system('pip uninstall -y transformers')
        os.system('pip install transformers==4.36.2')

        # Verify again after reinstall
        try:
            import transformers.utils
            print(f'✅ transformers.utils is now available after reinstall')
        except ImportError as e:
            print(f'❌ transformers.utils is STILL NOT available after reinstall: {e}')
            print('This may cause issues with attention mask fixes and model training')
except Exception as e:
    print(f'❌ Transformers error: {e}')
    # Install Transformers again as a fallback
    import os
    os.system('pip uninstall -y transformers')
    os.system('pip install transformers==4.36.2')

try:
    import peft
    print(f'PEFT version: {peft.__version__}')
except Exception as e:
    print(f'❌ PEFT error: {e}')
    # Install PEFT again as a fallback
    import os
    os.system('pip install peft==0.6.0')

try:
    import accelerate
    print(f'Accelerate version: {accelerate.__version__}')
except Exception as e:
    print(f'❌ Accelerate error: {e}')
    # Install Accelerate again as a fallback
    import os
    os.system('pip install accelerate==0.25.0')

try:
    import bitsandbytes
    print(f'BitsAndBytes version: {bitsandbytes.__version__}')
except Exception as e:
    print(f'❌ BitsAndBytes error: {e}')
    # Install BitsAndBytes again as a fallback
    import os
    os.system('pip install bitsandbytes==0.41.0')
"

echo "===================================================================="
echo "All dependencies installed successfully!"
echo "===================================================================="
