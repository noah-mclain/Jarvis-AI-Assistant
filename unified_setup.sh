#!/bin/bash

echo "===================================================================="
echo "Jarvis AI Assistant - Unified Setup Script (No Dependency Conflicts)"
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

detect_gpu() {
    echo "Checking for GPU..."
    GPU_AVAILABLE=0
    A100_GPU=false
    RTX4000_GPU=false
    RTX5000_GPU=false
    T4_GPU=false

    # Install minimal NumPy first to avoid any dependency issues
    echo "Installing NumPy 1.26.4 (required foundation package)..."
    pip install numpy==1.26.4 --no-deps --quiet

    # Verify NumPy installation
    if ! python -c "import numpy; assert numpy.__version__ == '1.26.4', f'Wrong NumPy version: {numpy.__version__}'"; then
        echo "ERROR: NumPy 1.26.4 installation failed. Cannot continue."
        exit 1
    fi
    
    echo "✅ NumPy 1.26.4 successfully installed"

    # Install minimal PyTorch to check GPU
    echo "Installing minimal PyTorch to verify GPU..."
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

cleanup_environment() {
    echo "Performing complete environment cleanup..."
    
    # Uninstall all relevant packages
    pip uninstall -y torch torchvision torchaudio
    pip uninstall -y numpy scipy matplotlib pandas
    pip uninstall -y transformers tokenizers huggingface-hub
    pip uninstall -y peft accelerate trl
    pip uninstall -y bitsandbytes xformers triton
    pip uninstall -y unsloth
    pip uninstall -y flash-attn
    
    # Remove package directories with sudo if available
    echo "Removing package directories..."
    if command -v sudo &> /dev/null; then
        sudo rm -rf /usr/local/lib/python3.11/dist-packages/numpy*
        sudo rm -rf /usr/local/lib/python3.11/dist-packages/torch*
        sudo rm -rf /usr/local/lib/python3.11/dist-packages/transformers*
        sudo rm -rf /usr/local/lib/python3.11/dist-packages/unsloth*
    else
        rm -rf /usr/local/lib/python3.11/dist-packages/numpy*
        rm -rf /usr/local/lib/python3.11/dist-packages/torch*
        rm -rf /usr/local/lib/python3.11/dist-packages/transformers*
        rm -rf /usr/local/lib/python3.11/dist-packages/unsloth*
    fi
    
    # Clear cache
    pip cache purge
    rm -rf ~/.cache/pip
    rm -rf ~/.cache/huggingface
}

install_core_dependencies() {
    echo "Installing core dependencies with specific compatible versions..."
    
    # Set LD_LIBRARY_PATH for CUDA libraries
    echo "Setting up CUDA library paths..."
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/lib64-nvidia
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/lib64-nvidia' >> ~/.bashrc
    
    # 1. NumPy 1.26.4 (crucial for compatibility)
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
    
    # 2. Install PyTorch 2.1.2 with CUDA 12.1 (verified compatible version)
    echo "Installing PyTorch 2.1.2 with CUDA 12.1 support..."
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu121
    
    # 3. Install core dependencies 
    echo "Installing core scientific packages..."
    pip install scipy==1.12.0 matplotlib==3.8.3 pandas==2.2.0
    
    # 4. Install utility packages required by transformers ecosystem
    pip install filelock==3.12.2 requests==2.31.0 tqdm==4.66.1
    pip install pyyaml==6.0.1 typing-extensions==4.8.0 packaging==23.1
    pip install fsspec==2023.6.0 psutil==5.9.5 ninja==1.11.1
    
    # 5. Hugging Face ecosystem - in exact order for compatibility
    echo "Installing Hugging Face ecosystem..."
    pip install safetensors==0.4.0
    pip install huggingface-hub==0.19.4 --no-deps
    pip install huggingface-hub==0.19.4  # Second install pulls in compatible dependencies
    
    pip install tokenizers==0.14.0 --no-deps
    pip install tokenizers==0.14.0
    
    pip install transformers==4.36.2 --no-deps
    pip install transformers==4.36.2  # Second install pulls in compatible dependencies
    
    pip install peft==0.6.0 --no-deps
    pip install peft==0.6.0
    
    pip install accelerate==0.25.0 --no-deps
    pip install accelerate==0.25.0
    
    pip install datasets==2.14.5 --no-deps
    pip install datasets==2.14.5
    
    pip install trl==0.7.4 --no-deps
    pip install trl==0.7.4
    
    pip install einops==0.7.0
    
    # 6. Install optimization libraries with exact versions
    echo "Installing optimization libraries..."
    pip install bitsandbytes==0.41.0
    pip install triton==2.1.0
    pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121
    
    # 7. Install unsloth (version 2024.8 is confirmed to work with this setup)
    echo "Installing unsloth dependencies..."
    pip install sentencepiece==0.1.99
    pip install unsloth==2024.8 --no-deps
    
    echo "✅ All core dependencies installed!"
}

configure_gpu_optimizations() {
    echo "Configuring GPU optimizations..."
    mkdir -p ~/.config/accelerate
    
    if [ "$A100_GPU" = true ]; then
        echo "Applying A100-specific optimizations (BF16)..."
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
        echo "Applying RTX-optimized settings (FP16)..."
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

    # Set and save environment variables
    export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:256"}
    export CUDA_LAUNCH_BLOCKING=0
    export TOKENIZERS_PARALLELISM=true
    
    echo "export PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF" >> ~/.bashrc
    echo "export CUDA_LAUNCH_BLOCKING=0" >> ~/.bashrc
    echo "export TOKENIZERS_PARALLELISM=true" >> ~/.bashrc
}

verify_installation() {
    echo "Verifying installations..."
    python -c "
import sys
print(f'Python version: {sys.version}')

try:
    import numpy
    print(f'NumPy version: {numpy.__version__}')
    if numpy.__version__ == '1.26.4':
        print('✅ NumPy 1.26.4 confirmed')
    else:
        print(f'❌ Wrong NumPy version: {numpy.__version__}')
except Exception as e:
    print(f'❌ NumPy error: {e}')

try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA version: {torch.version.cuda}')
        print(f'GPU: {torch.cuda.get_device_name(0)}')
except Exception as e:
    print(f'❌ PyTorch error: {e}')

try:
    import transformers
    print(f'transformers version: {transformers.__version__}')
except Exception as e:
    print(f'❌ transformers error: {e}')

try:
    import peft
    print(f'peft version: {peft.__version__}')
except Exception as e:
    print(f'❌ peft error: {e}')

try:
    import accelerate
    print(f'accelerate version: {accelerate.__version__}')
except Exception as e:
    print(f'❌ accelerate error: {e}')

try:
    import bitsandbytes as bnb
    print(f'bitsandbytes version: {bnb.__version__ if hasattr(bnb, \"__version__\") else \"installed\"}')
    try:
        if torch.cuda.is_available():
            lin8bit = bnb.nn.Linear8bitLt(10, 10, has_fp16_weights=False)
            print('✅ bitsandbytes 8-bit layers working!')
    except Exception as e:
        print(f'❌ bitsandbytes layer creation error: {e}')
except Exception as e:
    print(f'❌ bitsandbytes import error: {e}')

try:
    import unsloth
    print(f'unsloth version: {unsloth.__version__ if hasattr(unsloth, \"__version__\") else \"installed\"}')
    print('✅ unsloth successfully imported')
except Exception as e:
    print(f'❌ unsloth error: {e}')
"
}

create_emergency_fix_script() {
    echo "Creating emergency fix script..."
    cat > fix_numpy_emergency.sh << 'EOF'
#!/bin/bash

echo "================================================================"
echo "EMERGENCY: NumPy 1.26.4 Installation Fix"
echo "================================================================"

# Uninstall NumPy
pip uninstall -y numpy

# Force remove all NumPy directories (with sudo if available)
if command -v sudo &> /dev/null; then
    sudo rm -rf /usr/local/lib/python3.11/dist-packages/numpy*
    sudo rm -rf /usr/local/lib/python3.11/site-packages/numpy*
else
    rm -rf /usr/local/lib/python3.11/dist-packages/numpy*
    rm -rf /usr/local/lib/python3.11/site-packages/numpy*
fi

# Clear pip cache
pip cache purge

# Install NumPy 1.26.4 with maximum force
pip install numpy==1.26.4 --no-deps --force-reinstall
pip install numpy==1.26.4 --force-reinstall

# Verify installation
if python -c "import numpy; print(f'NumPy version: {numpy.__version__}'); exit(0 if numpy.__version__ == '1.26.4' else 1)"; then
    echo "✅ NumPy 1.26.4 successfully installed!"
else
    echo "❌ NumPy installation failed after multiple attempts."
    echo "Please try running this script with sudo: sudo ./fix_numpy_emergency.sh"
    exit 1
fi

echo "================================================================"
echo "NumPy 1.26.4 emergency fix complete!"
echo "================================================================"
EOF

    chmod +x fix_numpy_emergency.sh
}

# Main execution flow
echo "Starting unified setup for Jarvis AI Assistant..."

# Clean environment first
cleanup_environment

# Detect GPU and install minimal dependencies
detect_gpu

# Install dependencies in the correct order
install_core_dependencies

# Configure GPU optimizations
configure_gpu_optimizations

# Create emergency fix script
create_emergency_fix_script

# Verify installation
verify_installation

echo "===================================================================="
echo "Setup complete! Jarvis AI Assistant environment is ready."
echo ""
echo "Environment has:"
echo "- NumPy 1.26.4 (compatible foundation)"
echo "- PyTorch 2.1.2 with CUDA 12.1"
echo "- Transformers 4.36.2, PEFT 0.6.0"
echo "- Unsloth 2024.8 (compatible version)"
echo ""
echo "If NumPy or other packages get corrupted, run:"
echo "./fix_numpy_emergency.sh"
echo "====================================================================" 