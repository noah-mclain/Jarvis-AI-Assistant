#!/bin/bash

# Paperspace Setup Script for RTX4000/5000 GPUs with Google Drive Integration
echo "Setting up Paperspace environment with RTX GPU optimizations and Google Drive integration..."

# Check if running in Paperspace
if [ ! -d "/notebooks" ] && [ ! -d "/storage" ]; then
    echo "ERROR: This script is designed for Paperspace environments."
    echo "Please run this on a Paperspace machine with RTX4000 or RTX5000 GPU."
    exit 1
fi

# Create a virtual environment first
echo "Creating a clean Python 3.11 virtual environment..."
pip install --upgrade pip
pip install virtualenv

# Create virtual environment in a persistent location
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

# Make sure we're using the virtual environment
which python
python --version
which pip
pip --version

# Setup .bashrc to auto-activate the environment
echo "Setting up .bashrc to auto-activate environment on login..."
if ! grep -q "source $VENV_PATH/bin/activate" ~/.bashrc; then
    echo "# Auto-activate Jarvis AI virtual environment" >> ~/.bashrc
    echo "source $VENV_PATH/bin/activate" >> ~/.bashrc
fi

# Install core Python packages first
echo "Installing core Python dependencies..."
pip install wheel setuptools
pip install numpy==1.26.4  # Install NumPy 1.x explicitly before torch

# Check for GPU using PyTorch
echo "Checking for GPU..."
GPU_AVAILABLE=0
RTX4000_GPU=false
RTX5000_GPU=false

# Install minimal PyTorch to check GPU
pip install torch==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu121

if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo "PyTorch confirms CUDA is available"
    GPU_AVAILABLE=1
    
    # Get CUDA version and GPU info from PyTorch
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')")
    GPU_MEMORY=$(python -c "import torch; print(torch.cuda.get_device_properties(0).total_memory / 1e9)")
    
    echo "CUDA Version: $CUDA_VERSION"
    echo "GPU: $GPU_NAME"
    echo "GPU Memory: $GPU_MEMORY GB"
    
    # Check GPU type via PyTorch
    if echo "$GPU_NAME" | grep -q "RTX 4000"; then
        echo "RTX 4000 GPU confirmed - applying memory-optimized settings for 8GB VRAM"
        RTX4000_GPU=true
        # Export environment variables for optimal RTX4000 performance
        export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
        export CUDA_LAUNCH_BLOCKING=0
        export TOKENIZERS_PARALLELISM=true
    elif echo "$GPU_NAME" | grep -q "RTX 5000"; then
        echo "RTX 5000 GPU confirmed - applying optimized settings for 16GB VRAM"
        RTX5000_GPU=true
        # Export environment variables for optimal RTX5000 performance
        export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
        export CUDA_LAUNCH_BLOCKING=0
        export TOKENIZERS_PARALLELISM=true
    else
        echo "Unknown GPU: $GPU_NAME - using generic settings"
    fi
else
    echo "PyTorch cannot detect CUDA"
    echo "ERROR: No NVIDIA GPU detected. This setup requires GPU acceleration."
    exit 1
fi

# Add environment variables to the virtualenv activation script for persistence
cat >> "$VENV_PATH/bin/activate" << EOF
# Optimized GPU settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=true
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/lib64-nvidia
EOF

# Install scientific dependencies
echo "Installing scientific dependencies..."
pip install scipy==1.12.0
pip install matplotlib==3.8.3

# Install Google Drive integration tools
echo "Installing Google Drive integration tools..."
pip install -q gdown pydrive2 google-auth google-auth-oauthlib google-auth-httplib2

# Set up Google Drive integration using rclone
echo "Setting up rclone for Google Drive integration..."
sudo apt-get update && sudo apt-get install -y rclone

# Create necessary directories
mkdir -p ~/.config/rclone
mkdir -p /content/drive/MyDrive

# Create rclone config for Google Drive
cat > ~/.config/rclone/rclone.conf << EOF
[gdrive]
type = drive
scope = drive
EOF

# Create Google Drive mount script
cat > ~/mount_google_drive.sh << 'EOF'
#!/bin/bash

# Configure rclone if needed
if ! grep -q "\[gdrive\]" ~/.config/rclone/rclone.conf; then
    echo "Setting up rclone configuration..."
    rclone config create gdrive drive scope=drive
fi

# Check if already mounted
if mountpoint -q /content/drive/MyDrive; then
    echo "Google Drive is already mounted at /content/drive/MyDrive"
else
    # Mount Google Drive in the background
    echo "Mounting Google Drive at /content/drive/MyDrive..."
    mkdir -p /content/drive/MyDrive
    rclone mount gdrive: /content/drive/MyDrive --daemon --vfs-cache-mode writes

    # Wait for mount to be available
    echo "Waiting for mount to be ready..."
    sleep 3

    if mountpoint -q /content/drive/MyDrive; then
        echo "Google Drive successfully mounted at /content/drive/MyDrive"
        
        # Create Jarvis directories (same as Colab would)
        mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant
        mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant/models
        mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant/datasets
        mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant/checkpoints
        
        # Create symlink for compatibility
        ln -sf /content/drive/MyDrive/Jarvis_AI_Assistant /notebooks/google_drive_storage
    else
        echo "Failed to mount Google Drive. Please run 'rclone config' manually to set up Google Drive."
    fi
fi

# Set environment variables
export JARVIS_STORAGE_PATH="/content/drive/MyDrive/Jarvis_AI_Assistant"
export JARVIS_MODELS_PATH="/content/drive/MyDrive/Jarvis_AI_Assistant/models"
export JARVIS_DATASETS_PATH="/content/drive/MyDrive/Jarvis_AI_Assistant/datasets"
export JARVIS_CHECKPOINTS_PATH="/content/drive/MyDrive/Jarvis_AI_Assistant/checkpoints"
EOF

chmod +x ~/mount_google_drive.sh

# Ask user if they want to configure Google Drive now
echo ""
echo "Do you want to configure Google Drive now? This will let you use the same directory structure as Google Colab. [y/N]"
read -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Setting up rclone configuration. Please follow the prompts in your browser..."
    rclone config create gdrive drive scope=drive
    ~/mount_google_drive.sh
    if mountpoint -q /content/drive/MyDrive; then
        echo "Google Drive successfully mounted. Using Google Drive for storage."
        DRIVE_MOUNTED=true
        # Create directories in Google Drive (same structure as Colab)
        mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant
        mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant/models
        mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant/datasets
        mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant/checkpoints
        
        # Create symlinks from Paperspace storage to Google Drive
        ln -sf /content/drive/MyDrive/Jarvis_AI_Assistant /notebooks/google_drive_storage
        echo "Created symlink to Google Drive storage in /notebooks/google_drive_storage"
        
        # Set primary storage path to Google Drive
        STORAGE_BASE_PATH="/content/drive/MyDrive/Jarvis_AI_Assistant"
    else
        echo "Google Drive mount failed. Using Paperspace persistent storage instead."
        DRIVE_MOUNTED=false
        # Continue with Paperspace storage setup
    fi
else
    DRIVE_MOUNTED=false
    echo "Skipping Google Drive setup. Using Paperspace persistent storage instead."
fi

if [ "$DRIVE_MOUNTED" != true ]; then
    # Set up persistent storage directory structure in Paperspace
    echo "Setting up persistent storage in /storage..."
    if [ -d "/storage" ]; then
        mkdir -p /storage/Jarvis_AI_Assistant
        mkdir -p /storage/Jarvis_AI_Assistant/models
        mkdir -p /storage/Jarvis_AI_Assistant/datasets
        mkdir -p /storage/Jarvis_AI_Assistant/checkpoints
        
        # Create symlinks for easier access
        ln -sf /storage/Jarvis_AI_Assistant /notebooks/Jarvis_AI_Assistant_storage
        echo "Created symlink to persistent storage in /notebooks/Jarvis_AI_Assistant_storage"
        
        # Set primary storage path to Paperspace storage
        STORAGE_BASE_PATH="/storage/Jarvis_AI_Assistant"
    else
        echo "WARNING: /storage directory not found. Using local storage instead."
        mkdir -p ~/Jarvis_AI_Assistant
        mkdir -p ~/Jarvis_AI_Assistant/models
        mkdir -p ~/Jarvis_AI_Assistant/datasets
        
        # Set primary storage path to local storage
        STORAGE_BASE_PATH="$HOME/Jarvis_AI_Assistant"
    fi
fi

# Clean up any conflicting packages
echo "Cleaning up potential conflicts..."
pip uninstall -y bitsandbytes unsloth peft accelerate xformers flash-attn

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA 12.1 support..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu121

python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

# Fix LD_LIBRARY_PATH for CUDA compatibility
echo "Setting up CUDA library path..."
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/lib64-nvidia
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/lib64-nvidia' >> ~/.bashrc

# Install with care to avoid conflicting dependencies
echo "Installing core dependencies in correct order to avoid conflicts..."

# First, install HuggingFace Hub to a version compatible with multiple libraries
pip install -U huggingface-hub==0.19.4 --no-deps
pip install filelock requests tqdm pyyaml typing-extensions packaging fsspec

# Install bitsandbytes with direct installation
echo "Installing bitsandbytes..."
pip install bitsandbytes==0.41.0

# Install accelerate 
echo "Installing accelerate..."
pip install accelerate==0.27.0 --no-deps
pip install accelerate==0.27.0  # Second install to get dependencies

# Install tokenizers to match transformers version
pip install tokenizers==0.14.1

# Install transformers (which will respect our hub version)
echo "Installing transformers..."
pip install transformers==4.36.2 --no-deps
pip install transformers==4.36.2  # Second install to get dependencies

# Install TRL after transformers
pip install trl==0.7.10 --no-deps
pip install trl==0.7.10  # Second install to get dependencies

# Install peft after transformers
pip install peft==0.6.0 --no-deps
pip install peft==0.6.0  # Second install to get dependencies

# Install other core dependencies
pip install safetensors==0.4.1
pip install datasets==2.19.0 --ignore-installed
pip install einops==0.7.0

# Install xformers compatible with PyTorch 2.1.2
pip install -U xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121 --no-deps
pip install -U xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121

# Install an older unsloth to avoid version conflicts
echo "Installing Unsloth (older compatible version)..."
pip install "unsloth==2023.12.17" --no-deps

# Skip installing flash-attn in the setup script to avoid compilation errors
# The fix_flash_attn.sh script will handle this later if needed

# Create RTX-optimized config file
mkdir -p ~/.config/accelerate
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

echo "Created RTX-optimized accelerate config with FP16 mixed precision (BF16 not used)"

# Install other dependencies
echo "Installing additional dependencies..."
pip install ninja==1.11.1 packaging==23.2 psutil==5.9.8
pip install jupyterlab==3.6.5 tensorboard==2.15.1
pip install gdown==5.1.0 fsspec==2024.3.1 boto3==1.28.51
pip install sentencepiece==0.2.0 protobuf==3.20.3

# Create the fix_dependencies.sh script to repair any future issues
cat > fix_dependencies.sh << 'EOF'
#!/bin/bash

echo "===================================================================="
echo "Fixing dependency issues for Jarvis AI Assistant on Paperspace..."
echo "===================================================================="

# Clean environment first 
echo "Uninstalling conflicting packages..."
pip uninstall -y flash-attn bitsandbytes unsloth peft accelerate xformers unsloth_zoo protobuf tokenizers huggingface-hub numpy

# Fix LD_LIBRARY_PATH for CUDA compatibility
echo "Setting up CUDA library paths..."
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/lib64-nvidia' >> ~/.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/lib64-nvidia

# Install NumPy 1.x first (PyTorch 2.1.2 is not compatible with NumPy 2.x)
echo "Installing NumPy 1.x (compatible with PyTorch 2.1.2)..."
pip install numpy==1.26.4

# Fix torch version to ensure compatibility
echo "Installing PyTorch 2.1.2 with CUDA 12.1..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu121

# Installing core scientific packages
echo "Installing scientific packages..."
pip install scipy==1.12.0
pip install matplotlib==3.8.3

# Installing packages in a very specific order to avoid version conflicts
echo "Installing compatible core dependencies..."

# First install huggingface hub at a version that works with everything
echo "Installing huggingface-hub..."
pip install huggingface-hub==0.19.4 --no-deps
pip install filelock requests tqdm pyyaml typing-extensions packaging fsspec

# Next install tokenizers at a compatible version
echo "Installing tokenizers..."
pip install tokenizers==0.14.1

# Then install core components in order
echo "Installing transformers ecosystem..."
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
echo "Installing bitsandbytes..."
pip install --no-cache-dir bitsandbytes==0.41.0

# Install xformers compatible with PyTorch 2.1.2
echo "Installing xformers..."
pip install -U "xformers==0.0.23.post1" --index-url https://download.pytorch.org/whl/cu121 --no-deps
pip install -U "xformers==0.0.23.post1" --index-url https://download.pytorch.org/whl/cu121

# Install dependencies for unsloth
echo "Installing unsloth dependencies..."
pip install protobuf==3.20.3
pip install sentencepiece==0.2.0
pip install wheel>=0.38.0

# Install an older unsloth version compatible with our dependencies
echo "Installing Unsloth with compatible version..."
pip install "unsloth==2023.12.17" --no-deps

# We'll skip unsloth_zoo since it's not compatible with older unsloth versions
echo "Skipping unsloth_zoo installation to avoid compatibility issues"

# Install remaining packages
echo "Installing utility packages..."
pip install ninja==1.11.1 packaging==23.2 psutil==5.9.8
pip install gdown==5.1.0 fsspec==2024.3.1 boto3==1.28.51
pip install jupyter jupyterlab

# Skip flash-attention installation in this script to avoid build issues
echo "Note: Flash-attention installation is skipped to avoid build errors."
echo "If you need flash-attention, please run fix_flash_attn.sh separately."

# Verify installations
echo "Verifying installations..."
python -c "
import sys
print(f'Python version: {sys.version}')

try:
    import numpy
    print(f'NumPy version: {numpy.__version__}')
except Exception as e:
    print(f'NumPy error: {e}')

try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
except Exception as e:
    print(f'PyTorch error: {e}')

try:
    import scipy
    print(f'SciPy version: {scipy.__version__}')
except Exception as e:
    print(f'SciPy error: {e}')

try:
    import bitsandbytes as bnb
    print(f'bitsandbytes version: {bnb.__version__ if hasattr(bnb, \"__version__\") else \"(version unknown)\"}'  )
    try:
        lin8bit = bnb.nn.Linear8bitLt(10, 10, has_fp16_weights=False)
        print('bitsandbytes working correctly!')
    except Exception as e:
        print(f'bitsandbytes layer creation error: {e}')
except Exception as e:
    print(f'bitsandbytes import error: {e}')

try:
    import unsloth
    print(f'unsloth version: {unsloth.__version__ if hasattr(unsloth, \"__version__\") else \"(version unknown)\"}')
    
    import transformers
    print(f'transformers version: {transformers.__version__}')
except Exception as e:
    print(f'transformers/unsloth error: {e}')

try:
    import peft
    print(f'peft version: {peft.__version__}')
except Exception as e:
    print(f'peft error: {e}')

try:
    import accelerate
    print(f'accelerate version: {accelerate.__version__}')
except Exception as e:
    print(f'accelerate error: {e}')
"

echo "===================================================================="
echo "Dependency fixes complete!"
echo ""
echo "Important notes:"
echo "1. We've installed an older but compatible version of unsloth"
echo "2. unsloth_zoo is not installed to prevent conflicts"
echo "3. NumPy has been downgraded to 1.26.x to be compatible with PyTorch 2.1.2"
echo ""
echo "You can now run Jarvis AI Assistant with lower-demand parameters:"
echo "python src/generative_ai_module/jarvis_unified.py \\"
echo "  --mode train \\"
echo "  --model deepseek-ai/deepseek-coder-1.3b-base \\"
echo "  --datasets pile \\"
echo "  --max-samples 100 \\"
echo "  --epochs 1 \\"
echo "  --batch-size 1 \\"
echo "  --gradient-accumulation-steps 8 \\"
echo "  --load-in-4bit \\"
echo "  --sequence-length 512"
echo "===================================================================="
EOF

# Create a flash-attention specific fix script
cat > fix_flash_attn.sh << 'EOF'
#!/bin/bash

echo "===================================================================="
echo "Flash-Attention Fix for Paperspace"
echo "===================================================================="

# Ensure we have numpy installed
pip install numpy==1.26.4

# Uninstall any existing flash-attn
echo "Removing any existing flash-attn installations..."
pip uninstall -y flash-attn

# Check GPU type
echo "Checking GPU type..."
if python -c "import torch; print('RTX 5000' in torch.cuda.get_device_name(0))" | grep -q "True"; then
  echo "RTX 5000 GPU detected - attempting to install pre-built flash-attention wheel..."
  
  # First, try downloading a pre-built wheel to avoid compilation
  echo "Attempting to download pre-built wheel..."
  mkdir -p ~/flash_attn_wheels
  cd ~/flash_attn_wheels
  
  # Try to download for CUDA 12.1, Python 3.11
  echo "Downloading pre-built wheel for CUDA 12.1 + PyTorch 2.1.x..."
  wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.3.3/flash_attn-2.3.3+cu121torch2.1cxx11abiFALSE-cp311-cp311-linux_x86_64.whl -O flash_attn-2.3.3-cp311-cp311-linux_x86_64.whl || \
  wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.3.2/flash_attn-2.3.2+cu121torch2.1cxx11abiFALSE-cp311-cp311-linux_x86_64.whl -O flash_attn-2.3.2-cp311-cp311-linux_x86_64.whl || \
  echo "Failed to download pre-built wheel directly"
  
  # Try to install the downloaded wheel
  if ls flash_attn*.whl 1> /dev/null 2>&1; then
    echo "Installing downloaded wheel..."
    pip install flash_attn*.whl
    if python -c "import flash_attn" 2>/dev/null; then
      echo "Successfully installed flash-attention from pre-built wheel!"
      cd - > /dev/null
      echo "Flash-attention is now available!"
      exit 0
    fi
  fi
  
  cd - > /dev/null
  
  # Extremely simplified attempt
  echo "Attempting simplified installation method..."
  pip install 'flash-attn<2.1.0' --prefer-binary || echo "Flash-attention installation failed"
else
  echo "RTX 4000 or other GPU detected - flash-attention is not recommended for this GPU"
  echo "Skipping flash-attention installation"
fi

# Install alternative optimizations
echo "Installing alternative memory optimizations..."
pip install -U xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121

# Verify installation
echo "Verifying flash-attention installation..."
if python -c "import flash_attn" 2>/dev/null; then
  echo "flash-attention is successfully installed and importable!"
  python -c "import flash_attn; print(f'flash-attention version: {flash_attn.__version__}' if hasattr(flash_attn, '__version__') else 'version unknown')"
else
  echo "flash-attention is not installed or not importable."
  echo "This is expected - the system will use xformers optimizations instead."
fi

echo "===================================================================="
echo "Flash-attention setup complete!"
echo "NOTE: If flash-attention installation failed, that's completely fine."
echo "Jarvis will automatically fall back to using xformers optimizations."
echo "===================================================================="
EOF

chmod +x fix_dependencies.sh
chmod +x fix_flash_attn.sh

# Test installations
echo "Testing fine-tuning components..."
python -c "
import sys
print(f'Python version: {sys.version}')

try:
    import numpy
    print(f'NumPy version: {numpy.__version__}')
except Exception as e:
    print(f'NumPy error: {e}')

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
    # Test if bitsandbytes can create a quantized linear layer
    if torch.cuda.is_available():
        cuda_setup_success = bnb.cuda_setup.get_compute_capability() is not None
        print(f'bitsandbytes CUDA setup: {cuda_setup_success}')
        lin8bit = bnb.nn.Linear8bitLt(10, 10, has_fp16_weights=False)
        print('Successfully created 8-bit linear layer')
except Exception as e:
    print(f'bitsandbytes error: {e}')

try:
    # Import unsloth first
    import unsloth
    print(f'unsloth version: {unsloth.__version__ if hasattr(unsloth, \"__version__\") else \"(version unknown)\"}')
    
    import transformers
    print(f'transformers version: {transformers.__version__}')
except Exception as e:
    print(f'transformers/unsloth error: {e}')

try:
    import peft
    print(f'peft version: {peft.__version__}')
except Exception as e:
    print(f'peft error: {e}')

try:
    import xformers
    print(f'xformers version: {xformers.__version__}')
except Exception as e:
    print(f'xformers error: {e}')
"

# Use the correct storage path for the output summary
if [ "$DRIVE_MOUNTED" = true ]; then
    STORAGE_PATH="/content/drive/MyDrive/Jarvis_AI_Assistant"
    echo "Google Drive integration active. Using: $STORAGE_PATH"
else
    STORAGE_PATH="$STORAGE_BASE_PATH"
    echo "Using local storage path: $STORAGE_PATH"
fi

# Output summary
if [ "$RTX4000_GPU" = true ]; then
    echo "==================================================================="
    echo "Setup complete for RTX 4000 GPU (8GB VRAM)!"
    echo ""
    echo "RECOMMENDED SETTINGS FOR RTX 4000:"
    echo "- Use 4-bit quantization (enabled by default)"
    echo "- Use smaller models like deepseek-ai/deepseek-coder-1.3b"
    echo "- Batch size: 1-2"
    echo "- Gradient accumulation steps: 8+"
    echo "- Sequence length: 512 (max 1024)"
    echo "==================================================================="
elif [ "$RTX5000_GPU" = true ]; then
    echo "==================================================================="
    echo "Setup complete for RTX 5000 GPU (16GB VRAM)!"
    echo ""
    echo "RECOMMENDED SETTINGS FOR RTX 5000:"
    echo "- Use 4-bit quantization (enabled by default)"
    echo "- Can use deepseek-ai/deepseek-coder-6.7b models"
    echo "- Batch size: 2-4"
    echo "- Gradient accumulation steps: 4+"
    echo "- Sequence length: 1024 (max 2048)"
    echo "==================================================================="
else
    echo "==================================================================="
    echo "Setup complete for GPU!"
    echo "==================================================================="
fi

if [ "$DRIVE_MOUNTED" = true ]; then
    echo "Google Drive mounted path: /content/drive/MyDrive/Jarvis_AI_Assistant"
    echo "Google Drive symlink: /notebooks/google_drive_storage"
    echo ""
    echo "Use Google Drive commands for consistent storage with Colab:"
    echo "See paperspace_gdrive_commands.md for details."
else
    echo "Google Drive was not mounted. Fallback storage paths:"
    echo "Persistent storage path: $STORAGE_BASE_PATH"
    echo "Storage symlink: /notebooks/Jarvis_AI_Assistant_storage"
    echo ""
    echo "To try mounting Google Drive manually, run:"
    echo "from google.colab import drive; drive.mount('/content/drive')"
fi

echo ""
echo "Virtual environment created at: $VENV_PATH"
echo "This environment will be auto-activated when you log in."
echo ""
echo "If you encounter dependency issues, run:"
echo "bash fix_dependencies.sh"
echo ""
echo "If you encounter flash-attention issues, run:"
echo "bash fix_flash_attn.sh"
echo ""
echo "For details on running the assistant, see PAPERSPACE_COMMANDS.md" 