#!/bin/bash

# Unified Setup Script for Jarvis AI Assistant
# Works on both Google Colab and Paperspace environments
# Includes dependency fixes and Google Drive integration

echo "===================================================================="
echo "Jarvis AI Assistant - Unified Setup Script"
echo "This script works for both Google Colab and Paperspace environments"
echo "===================================================================="

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

# Create a helper function for GPU detection
detect_gpu() {
    echo "Checking for GPU..."
    GPU_AVAILABLE=0
    A100_GPU=false
    RTX4000_GPU=false
    RTX5000_GPU=false
    T4_GPU=false

    # Install minimal PyTorch to check GPU
    pip install torch==2.1.2 --quiet --extra-index-url https://download.pytorch.org/whl/cu121

    if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
        echo "PyTorch confirms CUDA is available"
        GPU_AVAILABLE=1
        
        # Get CUDA and GPU info
        CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
        GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')")
        
        echo "CUDA Version: $CUDA_VERSION"
        echo "GPU: $GPU_NAME"
        
        # Detect specific GPU types
        if echo "$GPU_NAME" | grep -q "A100"; then
            echo "A100 GPU detected - using optimized settings for high VRAM"
            A100_GPU=true
            export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
        elif echo "$GPU_NAME" | grep -q "RTX 4000"; then
            echo "RTX 4000 GPU detected - using memory-optimized settings for 8GB VRAM"
            RTX4000_GPU=true
            export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
        elif echo "$GPU_NAME" | grep -q "RTX 5000"; then
            echo "RTX 5000 GPU detected - using optimized settings for 16GB VRAM"
            RTX5000_GPU=true
            export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
        elif echo "$GPU_NAME" | grep -q "T4"; then
            echo "T4 GPU detected - using memory-optimized settings for 16GB VRAM"
            T4_GPU=true
            export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
        else
            echo "Unknown GPU: $GPU_NAME - using generic settings"
        fi
        
        # Common optimizations for all GPUs
        export CUDA_LAUNCH_BLOCKING=0
        export TOKENIZERS_PARALLELISM=true
    else
        echo "PyTorch cannot detect CUDA"
        echo "ERROR: No GPU detected. This setup requires GPU acceleration."
        exit 1
    fi
}

# Setup virtual environment for Paperspace
setup_virtualenv() {
    if [ $IN_PAPERSPACE -eq 1 ]; then
        echo "Creating Python 3.11 virtual environment for Paperspace..."
        pip install --upgrade pip
        pip install virtualenv

        # Create virtual environment in persistent storage if available
        if [ -d "/storage" ]; then
            VENV_PATH="/storage/jarvis_venv"
        else
            VENV_PATH="$HOME/jarvis_venv"
        fi

        # Remove existing environment if it exists
        if [ -d "$VENV_PATH" ]; then
            echo "Removing existing virtual environment..."
            rm -rf "$VENV_PATH"
        fi

        # Create a new virtual environment
        echo "Creating new virtual environment at $VENV_PATH..."
        virtualenv --python=python3.11 "$VENV_PATH"

        # Activate the virtual environment
        echo "Activating virtual environment..."
        source "$VENV_PATH/bin/activate"

        # Setup auto-activation
        echo "Setting up .bashrc to auto-activate environment on login..."
        if ! grep -q "source $VENV_PATH/bin/activate" ~/.bashrc; then
            echo "# Auto-activate Jarvis AI virtual environment" >> ~/.bashrc
            echo "source $VENV_PATH/bin/activate" >> ~/.bashrc
        fi

        # Add optimized settings to virtualenv activation
        cat >> "$VENV_PATH/bin/activate" << EOF
# Optimized GPU settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=true
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/lib64-nvidia
EOF
    fi
}

# Setup Google Drive integration
setup_gdrive() {
    if [ $IN_COLAB -eq 1 ]; then
        echo "Setting up Google Drive integration for Colab..."
        echo "from google.colab import drive; drive.mount('/content/drive')" > mount_drive.py
        python mount_drive.py || echo "Failed to mount Google Drive. You can try manually using: from google.colab import drive; drive.mount('/content/drive')"
        
        # Create directories in Google Drive if mounted
        if [ -d "/content/drive/MyDrive" ]; then
            echo "Creating directories in Google Drive..."
            mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant
            mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant/models
            mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant/datasets
            mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant/checkpoints
        else
            echo "Google Drive not mounted or not accessible. Skipping directory creation."
        fi
    elif [ $IN_PAPERSPACE -eq 1 ]; then
        echo "Setting up headless Google Drive integration for Paperspace..."
        
        # Install rclone for Google Drive integration
        sudo apt-get update && sudo apt-get install -y rclone
        
        # Create necessary directories
        mkdir -p ~/.config/rclone
        mkdir -p /content/drive/MyDrive
        
        # Create headless auth script
        cat > headless_gdrive_auth.py << EOF
#!/usr/bin/env python3
"""
Headless Google Drive authentication for Paperspace environments.
"""
import os
import subprocess
from urllib.parse import urlparse, parse_qs

print("Generating Google Drive authentication URL...")
result = subprocess.run(
    ["rclone", "config", "reconnect", "gdrive:", "--no-browser"],
    capture_output=True,
    text=True
)

# Extract the auth URL from rclone output
auth_lines = result.stderr.split("\\n")
auth_url = None
for line in auth_lines:
    if "https://accounts.google.com/o/oauth2/auth" in line:
        auth_url = line.strip()
        break

if not auth_url:
    print("ERROR: Could not generate authentication URL")
    print("rclone output:", result.stderr)
    exit(1)

print("\\n" + "="*80)
print("GOOGLE DRIVE AUTHENTICATION REQUIRED")
print("="*80)
print("\\n1. Open this URL in a browser on your local machine:")
print("\\n" + auth_url + "\\n")
print("2. Log in with your Google account and grant access")
print("3. Copy the authorization code from the browser")
print("4. Paste the authorization code below:")
print("="*80 + "\\n")

auth_code = input("Enter the authorization code: ")

print("Completing authentication...")
subprocess.run(
    ["rclone", "config", "reconnect", "gdrive:", "--code", auth_code],
    check=True
)

print("\\nGoogle Drive authentication completed successfully!")
EOF

        chmod +x headless_gdrive_auth.py
        
        # Create mount script
        cat > mount_gdrive.sh << 'EOF'
#!/bin/bash

# Check if already mounted
if mountpoint -q /content/drive/MyDrive; then
    echo "Google Drive is already mounted at /content/drive/MyDrive"
    exit 0
fi

# Configure rclone if needed
if ! grep -q "\[gdrive\]" ~/.config/rclone/rclone.conf; then
    echo "Setting up rclone configuration..."
    cat > ~/.config/rclone/rclone.conf << CONF
[gdrive]
type = drive
client_id = 202264815644.apps.googleusercontent.com
client_secret = X4Z3ca8xfWDb1Voo-F9a7ZxJ
scope = drive
CONF
    echo "Please run 'python headless_gdrive_auth.py' to authenticate"
    exit 1
fi

# Mount Google Drive in the background
echo "Mounting Google Drive at /content/drive/MyDrive..."
mkdir -p /content/drive/MyDrive
rclone mount gdrive: /content/drive/MyDrive --daemon --vfs-cache-mode writes

# Wait for mount to be ready
echo "Waiting for mount to be ready..."
sleep 3

if mountpoint -q /content/drive/MyDrive; then
    echo "Google Drive successfully mounted at /content/drive/MyDrive"
    
    # Create Jarvis directories
    mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant
    mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant/models
    mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant/datasets
    mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant/checkpoints
    
    # Create symlink for compatibility
    ln -sf /content/drive/MyDrive/Jarvis_AI_Assistant /notebooks/google_drive_storage
    echo "Created symlink to Google Drive storage in /notebooks/google_drive_storage"
else
    echo "Failed to mount Google Drive. Run 'python headless_gdrive_auth.py' first."
fi
EOF
        chmod +x mount_gdrive.sh
    fi
}

# Setup storage directories
setup_storage_dirs() {
    if [ $IN_PAPERSPACE -eq 1 ]; then
        # Set up persistent storage in Paperspace
        if [ -d "/storage" ]; then
            echo "Creating directories in Paperspace persistent storage..."
            mkdir -p /storage/Jarvis_AI_Assistant
            mkdir -p /storage/Jarvis_AI_Assistant/models
            mkdir -p /storage/Jarvis_AI_Assistant/datasets
            mkdir -p /storage/Jarvis_AI_Assistant/checkpoints
            
            # Create symlinks for easier access
            ln -sf /storage/Jarvis_AI_Assistant /notebooks/Jarvis_AI_Assistant_storage
            echo "Created symlink to persistent storage in /notebooks/Jarvis_AI_Assistant_storage"
            
            # Set storage path
            STORAGE_PATH="/storage/Jarvis_AI_Assistant"
        else
            echo "Paperspace /storage not found. Using local storage instead."
            mkdir -p ~/Jarvis_AI_Assistant
            mkdir -p ~/Jarvis_AI_Assistant/models
            mkdir -p ~/Jarvis_AI_Assistant/datasets
            
            # Set storage path
            STORAGE_PATH="$HOME/Jarvis_AI_Assistant"
        fi
    else
        # Standard environment storage
        if [ $IN_COLAB -eq 1 ] && [ -d "/content/drive/MyDrive" ]; then
            STORAGE_PATH="/content/drive/MyDrive/Jarvis_AI_Assistant"
        else
            mkdir -p ./Jarvis_AI_Assistant
            STORAGE_PATH="./Jarvis_AI_Assistant"
        fi
    fi
    
    echo "Using storage path: $STORAGE_PATH"
}

# Install core dependencies in the correct order
install_core_deps() {
    echo "Installing core dependencies with correct versioning..."
    
    # EMERGENCY FIX: Extremely aggressive cleanup - forcefully remove corrupted installations
    echo "Performing emergency cleanup of potentially corrupted installations..."
    rm -rf /usr/local/lib/python3.11/dist-packages/numpy*
    rm -rf /usr/local/lib/python3.11/dist-packages/numpy-*
    rm -rf /usr/local/lib/python3.11/dist-packages/torch*
    rm -rf /usr/local/lib/python3.11/dist-packages/transformers*
    rm -rf /usr/local/lib/python3.11/dist-packages/huggingface_hub*
    rm -rf /usr/local/lib/python3.11/dist-packages/tokenizers*
    rm -rf /usr/local/lib/python3.11/dist-packages/accelerate*
    rm -rf /usr/local/lib/python3.11/dist-packages/peft*
    rm -rf /usr/local/lib/python3.11/dist-packages/unsloth*
    rm -rf /tmp/pip-*
    pip cache purge
    
    # Then use pip to clean out any remaining package references
    echo "Removing all potentially conflicting packages..."
    pip uninstall -y flash-attn bitsandbytes unsloth peft accelerate xformers unsloth_zoo 
    pip uninstall -y protobuf tokenizers huggingface-hub numpy tensorflow tensorflow-estimator 
    pip uninstall -y tensorflow-io-gcs-filesystem tensorboard triton tqdm scipy

    # Fix LD_LIBRARY_PATH for CUDA
    echo "Setting up CUDA library paths..."
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/lib64-nvidia
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/lib64-nvidia' >> ~/.bashrc
    
    # Clear the Python path to avoid any cached NumPy references
    echo "Clearing Python environment cache..."
    python -c "import sys; sys.path = [p for p in sys.path if not 'numpy' in p]; print(sys.path)"
    
    # Install NumPy 1.26.4 with maximum force to ensure correct version
    echo "Installing NumPy 1.26.4 (compatible with PyTorch 2.1.2)..."
    pip install numpy==1.26.4 --no-deps --force-reinstall --no-cache-dir
    # Second install to confirm NumPy 1.x is properly installed
    pip install numpy==1.26.4
    
    # Verify NumPy installation is 1.x
    if ! python -c "import numpy; print(f'NumPy version: {numpy.__version__}'); exit(0 if numpy.__version__.startswith('1.') else 1)"; then
        echo "⚠️ CRITICAL ERROR: NumPy is still not correctly installed at version 1.x."
        echo "This is a common issue with Paperspace environments."
        echo "Will attempt one more aggressive fix..."
        
        # Try one more aggressive approach
        sudo rm -rf /usr/local/lib/python3.11/dist-packages/numpy*
        sudo pip install numpy==1.26.4 --force-reinstall --no-deps
        
        # Check again
        if ! python -c "import numpy; print(f'NumPy version: {numpy.__version__}'); exit(0 if numpy.__version__.startswith('1.') else 1)"; then
            echo "❌ CRITICAL ERROR: NumPy is still not correctly installed."
            echo "Please run fix_numpy_errors.sh after this script completes or manually fix:"
            echo "sudo rm -rf /usr/local/lib/python3.11/dist-packages/numpy*"
            echo "sudo pip install numpy==1.26.4 --force-reinstall --no-deps"
        else
            echo "✅ NumPy 1.26.4 successfully installed on second attempt!"
        fi
    else
        echo "✅ NumPy 1.26.4 successfully installed!"
    fi
    
    echo "Installing protobuf 3.20.3 (compatible with TensorFlow and transformers)..."
    pip install protobuf==3.20.3 --no-deps
    pip install protobuf==3.20.3
    
    echo "Installing PyTorch 2.1.2 with CUDA 12.1 support..."
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu121
    
    echo "Installing scientific packages..."
    pip install scipy==1.12.0 matplotlib==3.8.3
    
    echo "Installing HuggingFace ecosystem in the correct order..."
    # First install huggingface hub at a version that works with everything
    pip install huggingface-hub==0.19.4 --no-deps
    pip install filelock requests tqdm pyyaml typing-extensions packaging fsspec
    
    # Install tokenizers at a compatible version
    pip install tokenizers==0.14.1
    
    # Install core components in order
    pip install transformers==4.36.2 --no-deps
    pip install transformers==4.36.2
    pip install peft==0.6.0 --no-deps
    pip install peft==0.6.0
    pip install accelerate==0.27.0 --no-deps
    pip install accelerate==0.27.0
    pip install safetensors==0.4.1
    pip install datasets==2.19.0 --ignore-installed
    pip install trl==0.7.10 --no-deps
    pip install trl==0.7.10
    pip install einops==0.7.0
    
    # Install bitsandbytes
    pip install bitsandbytes==0.41.0
    
    # Install xformers compatible with PyTorch 2.1.2
    pip install -U xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121 --no-deps
    pip install -U xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121
    
    # Install unsloth dependencies
    pip install sentencepiece==0.2.0 wheel>=0.38.0
    
    # Install unsloth with a compatible version that's available
    # Updated to use the newer version that works with current environments
    pip install unsloth==2025.3.3 --no-deps
    
    # Install utility packages
    pip install ninja==1.11.1 packaging==23.2 psutil==5.9.8
    pip install gdown==5.1.0 fsspec==2024.3.1 boto3==1.28.51
    pip install jupyterlab tensorboard
}

# Configure GPU-specific optimizations
configure_gpu_optimizations() {
    mkdir -p ~/.config/accelerate
    
    if [ "$A100_GPU" = true ]; then
        echo "Applying A100-specific optimizations..."
        cat > ~/.config/accelerate/default_config.yaml << EOF
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'yes'
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
mixed_precision: 'bf16'
num_machines: 1
num_processes: 1
use_cpu: false
EOF
    else
        echo "Applying RTX-optimized settings..."
        cat > ~/.config/accelerate/default_config.yaml << EOF
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'no'
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
mixed_precision: 'fp16'
num_machines: 1
num_processes: 1
use_cpu: false
EOF
    fi
}

# Create utility scripts
create_utility_scripts() {
    # Create fix_numpy_errors.sh script for emergency NumPy fixes
    cat > fix_numpy_errors.sh << 'EOF'
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
EOF
    chmod +x fix_numpy_errors.sh

    # Create fix_numpy.sh for full-blown dependency cleanup and NumPy reinstall
    cat > fix_numpy.sh << 'EOF'
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
# Using a verified working version of unsloth for this specific environment
pip install unsloth==2025.3.3 --no-deps

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
EOF
    chmod +x fix_numpy.sh

    # Create fix_flash_attn.sh script
    cat > fix_flash_attn.sh << 'EOF'
#!/bin/bash

echo "===================================================================="
echo "Flash-Attention Fix for GPU Training"
echo "===================================================================="

# Ensure we have numpy installed
pip install numpy==1.26.4

# Uninstall any existing flash-attn
echo "Removing any existing flash-attn installations..."
pip uninstall -y flash-attn

# Check GPU type
echo "Checking GPU type..."
GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")

if echo "$GPU_NAME" | grep -q "A100\|RTX 5000"; then
    echo "$GPU_NAME GPU detected - attempting to install pre-built flash-attention wheel..."
    
    # Try downloading a pre-built wheel to avoid compilation
    mkdir -p ~/flash_attn_wheels
    cd ~/flash_attn_wheels
    
    # Try to download wheel for current CUDA version and Python
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")
    PY_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')" 2>/dev/null || echo "unknown")
    
    echo "Downloading pre-built wheel for CUDA $CUDA_VERSION and Python $PY_VERSION..."
    wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.3.3/flash_attn-2.3.3+cu${CUDA_VERSION}torch2.1cxx11abiFALSE-cp${PY_VERSION}-cp${PY_VERSION}-linux_x86_64.whl -O flash_attn.whl || \
    wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.3.2/flash_attn-2.3.2+cu${CUDA_VERSION}torch2.1cxx11abiFALSE-cp${PY_VERSION}-cp${PY_VERSION}-linux_x86_64.whl -O flash_attn.whl || \
    echo "Failed to download exact wheel, trying generic download..."
    
    # Try simpler installation if specific wheel download failed
    if [ ! -f flash_attn.whl ]; then
        echo "Attempting simplified installation method..."
        pip install 'flash-attn<2.3.5' --prefer-binary --no-build-isolation || echo "Flash-attention installation failed"
    else
        echo "Installing downloaded wheel..."
        pip install flash_attn.whl
    fi
    
    cd - > /dev/null
else
    echo "$GPU_NAME detected - flash-attention may not be optimal for this GPU"
    echo "Using alternative optimizations instead..."
    pip install -U xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121
fi

# Verify installation
echo "Verifying flash-attention installation..."
if python -c "import flash_attn" 2>/dev/null; then
    echo "flash-attention is successfully installed!"
    python -c "import flash_attn; print(f'flash-attention version: {flash_attn.__version__}' if hasattr(flash_attn, '__version__') else 'version unknown')"
else
    echo "flash-attention is not installed - using xformers optimizations instead."
fi

echo "===================================================================="
echo "Setup complete! Your system will use the best available optimizations."
echo "===================================================================="
EOF
    chmod +x fix_flash_attn.sh

    # Create fix_protobuf.sh script
    cat > fix_protobuf.sh << 'EOF'
#!/bin/bash

echo "===================================================================="
echo "Fixing protobuf dependency issue for transformers+tensorflow"
echo "===================================================================="

# Uninstall potentially conflicting packages
pip uninstall -y protobuf tensorflow tensorboard tensorflow-estimator

# Install protobuf first at a compatible version
echo "Installing protobuf 3.20.3 (compatible with both transformers and TensorFlow)..."
pip install protobuf==3.20.3

# Verify installation
python -c "
try:
    from google import protobuf
    print('Google protobuf successfully imported!')
    print(f'Protobuf version: {protobuf.__version__}')
except Exception as e:
    print(f'Error importing protobuf: {e}')
"

echo "===================================================================="
echo "Protobuf dependency fix complete!"
echo "===================================================================="
EOF
    chmod +x fix_protobuf.sh
}

# Verify installations
verify_installation() {
    echo "Testing installations..."
    python -c "
import sys
print(f'Python version: {sys.version}')

try:
    import numpy
    print(f'NumPy version: {numpy.__version__}')
    if numpy.__version__.startswith('1.'):
        print('NumPy 1.x confirmed ✅')
    else:
        print('WARNING: NumPy 2.x detected ❌ - Run ./fix_numpy_errors.sh to fix this')
except Exception as e:
    print(f'NumPy error: {e}')

try:
    from google import protobuf
    print(f'Protobuf successfully imported - version {protobuf.__version__ if hasattr(protobuf, \"__version__\") else \"unknown\"}')
except Exception as e:
    print(f'Protobuf error: {e}')

try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('CUDA version:', torch.version.cuda)
        print('GPU:', torch.cuda.get_device_name(0))
        print('GPU memory:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')
        print('BF16 support:', torch.cuda.is_bf16_supported())
except Exception as e:
    print(f'PyTorch error: {e}')

try:
    import scipy
    print(f'SciPy version: {scipy.__version__}')
except Exception as e:
    print(f'SciPy error: {e}')

try:
    import bitsandbytes as bnb
    print(f'bitsandbytes version: {bnb.__version__ if hasattr(bnb, \"__version__\") else \"(version unknown)\"}')
    if torch.cuda.is_available():
        try:
            lin8bit = bnb.nn.Linear8bitLt(10, 10, has_fp16_weights=False)
            print('bitsandbytes working correctly!')
        except Exception as e:
            print(f'bitsandbytes layer creation error: {e}')
except Exception as e:
    print(f'bitsandbytes import error: {e}')

try:
    import transformers
    print(f'transformers version: {transformers.__version__}')
except Exception as e:
    print(f'transformers error: {e}')

try:
    import peft
    print(f'peft version: {peft.__version__}')
except Exception as e:
    print(f'peft error: {e}')

try:
    import unsloth
    print(f'unsloth version: {unsloth.__version__ if hasattr(unsloth, \"__version__\") else \"(version unknown)\"}')
except Exception as e:
    print(f'unsloth error: {e}')

try:
    import xformers
    print(f'xformers version: {xformers.__version__}')
except Exception as e:
    print(f'xformers error: {e}')
"
}

# Print summary and usage instructions
print_summary() {
    echo "===================================================================="
    echo "Setup complete! Jarvis AI Assistant is ready to use."
    echo ""
    
    if [ $IN_COLAB -eq 1 ]; then
        echo "Google Colab Environment Detected"
        echo "- Storage: /content/drive/MyDrive/Jarvis_AI_Assistant (Google Drive)"
    elif [ $IN_PAPERSPACE -eq 1 ]; then
        echo "Paperspace Environment Detected"
        if [ "$RTX4000_GPU" = true ]; then
            echo "- GPU: RTX 4000 (8GB VRAM)"
            echo "- Recommended batch size: 1-2"
            echo "- Recommended sequence length: 512 (max 1024)"
            echo "- Always use gradient accumulation (8+ steps)"
        elif [ "$RTX5000_GPU" = true ]; then
            echo "- GPU: RTX 5000 (16GB VRAM)"
            echo "- Recommended batch size: 2-4"
            echo "- Recommended sequence length: 1024 (max 2048)"
            echo "- Always use gradient accumulation (4+ steps)"
        fi
        echo "- Storage: $STORAGE_PATH"
    fi
    
    echo ""
    echo "Troubleshooting:"
    echo "- If you encounter NumPy version conflicts: run ./fix_numpy_errors.sh"
    echo "- For more severe dependency issues: run ./fix_numpy.sh"
    echo "- For flash-attention issues: run ./fix_flash_attn.sh"
    echo "- For protobuf issues: run ./fix_protobuf.sh"
    echo ""
    echo "Example usage:"
    echo "python src/generative_ai_module/jarvis_unified.py \\"
    echo "  --mode train \\"
    echo "  --model deepseek-ai/deepseek-coder-1.3b-base \\"
    echo "  --datasets pile \\"
    echo "  --max-samples 100 \\"
    echo "  --epochs 1 \\"
    echo "  --batch-size 2 \\"
    echo "  --gradient-accumulation-steps 8 \\"
    echo "  --load-in-4bit \\"
    echo "  --sequence-length 512"
    echo "===================================================================="
}

# Run all setup steps
detect_gpu
setup_virtualenv
setup_gdrive
setup_storage_dirs
install_core_deps
configure_gpu_optimizations
create_utility_scripts
verify_installation
print_summary 