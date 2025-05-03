#!/bin/bash

# Paperspace Setup Script for RTX4000/5000 GPUs with Google Drive Integration
echo "Setting up Paperspace environment with RTX GPU optimizations and Google Drive integration..."

# Check if running in Paperspace
if [ ! -d "/notebooks" ] && [ ! -d "/storage" ]; then
    echo "ERROR: This script is designed for Paperspace environments."
    echo "Please run this on a Paperspace machine with RTX4000 or RTX5000 GPU."
    exit 1
fi

# Check for GPU using PyTorch
echo "Checking for GPU..."
GPU_AVAILABLE=0
RTX4000_GPU=false
RTX5000_GPU=false

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

# Add mount script to bashrc 
if ! grep -q "mount_google_drive.sh" ~/.bashrc; then
    echo '
# Google Drive integration
echo "Would you like to mount Google Drive? [y/N]"
read -n 1 -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ~/mount_google_drive.sh
fi
' >> ~/.bashrc
    echo "Added Google Drive mount prompt to ~/.bashrc"
fi

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

# Install bitsandbytes with direct installation (not using the failing URL)
echo "Installing bitsandbytes..."
pip install bitsandbytes==0.41.0

# Install accelerate
echo "Installing accelerate..."
pip install accelerate==0.27.0

# Install core dependencies required for fine-tuning with compatible versions
echo "Installing fine-tuning dependencies..."
pip install transformers==4.36.2
pip install peft==0.6.0
pip install safetensors==0.4.1
pip install datasets==2.19.0
pip install trl==0.7.10
pip install einops==0.7.0
pip install tokenizers==0.14.1  # Compatible with transformers 4.36.2

# Install xformers compatible with PyTorch 2.1.2
pip install -U xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121

# Install flash-attention conditionally based on GPU type
if [ "$RTX5000_GPU" = true ]; then
    echo "Installing flash-attention for RTX5000..."
    pip install -U "flash-attn<2.3.5" --no-build-isolation --prefer-binary || \
    echo "Flash-attention installation failed, but this is okay - we'll proceed without it"
else
    echo "Skipping flash-attention for RTX4000 due to memory constraints"
fi

# Install unsloth without dependencies to avoid conflicts
echo "Installing unsloth..."
pip install "unsloth>=2025.3.0" --no-deps

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

# Test installations
echo "Testing fine-tuning components..."
python -c "
import sys
print(f'Python version: {sys.version}')

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
    print(f'unsloth version: {unsloth.__version__}')
except Exception as e:
    print(f'unsloth error: {e}')

try:
    import xformers
    print(f'xformers version: {xformers.__version__}')
except Exception as e:
    print(f'xformers error: {e}')
"

# Create a bitsandbytes and CUDA fixer script
cat > fix_bitsandbytes.py << EOF
import os
import torch
import subprocess
from pathlib import Path

print("BitsAndBytes and CUDA Compatibility Fixer")
print("========================================")

# Fix CUDA library path
os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + ':/usr/local/cuda/lib64:/usr/lib64-nvidia'
print(f"Set LD_LIBRARY_PATH to: {os.environ['LD_LIBRARY_PATH']}")

# Check CUDA version
if torch.cuda.is_available():
    cuda_version = torch.version.cuda
    print(f"CUDA version: {cuda_version}")
    
    # Find bitsandbytes installation
    try:
        import bitsandbytes as bnb
        bnb_path = Path(bnb.__file__).parent
        print(f"bitsandbytes installation found at: {bnb_path}")
        
        # Attempt to create a quantized layer to test functionality
        try:
            lin8bit = bnb.nn.Linear8bitLt(10, 10, has_fp16_weights=False)
            print("Successfully created 8-bit linear layer - bitsandbytes is working!")
        except Exception as e:
            print(f"Error testing bitsandbytes: {e}")
            print("Reinstalling bitsandbytes...")
            subprocess.run(["pip", "uninstall", "-y", "bitsandbytes"], check=False)
            subprocess.run(["pip", "install", "bitsandbytes==0.41.0"], check=False)
    except ImportError:
        print("bitsandbytes not installed, installing version 0.41.0...")
        subprocess.run(["pip", "install", "bitsandbytes==0.41.0"], check=False)
    
    print("Testing bitsandbytes installation...")
    try:
        import bitsandbytes as bnb
        if bnb.cuda_setup.get_compute_capability() is not None:
            print("bitsandbytes CUDA setup successful!")
            # Try to create a quantized layer as final test
            lin8bit = bnb.nn.Linear8bitLt(10, 10, has_fp16_weights=False)
            print("Successfully created 8-bit linear layer - bitsandbytes is working correctly!")
        else:
            print("bitsandbytes CUDA setup failed.")
    except Exception as e:
        print(f"Error testing bitsandbytes: {e}")
else:
    print("CUDA not available. Cannot fix bitsandbytes without CUDA.")

print("\nChecking torch and dependencies...")
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    import transformers
    print(f"Transformers version: {transformers.__version__}")
    
    try:
        import unsloth
        print(f"Unsloth version: {unsloth.__version__}")
    except ImportError:
        print("Unsloth not installed properly. Consider reinstalling with:")
        print("pip install unsloth --no-deps")
        
    print("\nIf you continue to have issues, try the fix_dependencies.sh script.")
except Exception as e:
    print(f"Error checking dependencies: {e}")
EOF

# Create a memory analyzer script
cat > analyze_gpu_memory.py << EOF
import torch
import argparse
import time
import os

def print_gpu_memory_usage():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        memory_allocated = torch.cuda.memory_allocated(0) / 1e9
        memory_reserved = torch.cuda.memory_reserved(0) / 1e9
        max_memory = torch.cuda.max_memory_allocated(0) / 1e9
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"GPU: {gpu_name}")
        print(f"Memory Allocated: {memory_allocated:.2f} GB")
        print(f"Memory Reserved: {memory_reserved:.2f} GB")
        print(f"Max Memory Allocated: {max_memory:.2f} GB")
        print(f"Total GPU Memory: {total_memory:.2f} GB")
        print(f"Available Memory: {total_memory - memory_reserved:.2f} GB")
    else:
        print("CUDA not available")

def test_model_load(model_name, load_in_4bit=True):
    """Test loading a model and measure memory usage"""
    print(f"Testing load of {model_name}...")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    # Starting memory
    print_gpu_memory_usage()
    
    if load_in_4bit:
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
    else:
        bnb_config = None
    
    # Load model
    print("Loading model...")
    start_time = time.time()
    
    try:
        # Try with unsloth first if available
        try:
            from unsloth import FastLanguageModel
            print("Using Unsloth for optimized loading")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                quantization_config=bnb_config,
                max_seq_length=1024,
                device_map="auto",
                trust_remote_code=True
            )
            print("Successfully loaded with Unsloth")
        except ImportError:
            # Fallback to normal loading
            print("Unsloth not available, using standard HuggingFace loading")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print("Successfully loaded with standard HuggingFace")
            
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
        
        # Memory after loading
        print("\nMemory after loading:")
        print_gpu_memory_usage()
        
        # Test inference
        print("\nTesting inference...")
        input_text = "def fibonacci(n):"
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            start_time = time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.95
            )
            inference_time = time.time() - start_time
            
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Inference time: {inference_time:.2f} seconds")
        
        # Memory after inference
        print("\nMemory after inference:")
        print_gpu_memory_usage()
        
        # Output sample
        print("\nOutput sample:")
        print(output_text[:500])
        
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def find_suitable_models():
    """Find suitable models based on GPU memory"""
    if not torch.cuda.is_available():
        return
        
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Total GPU memory: {total_memory:.2f} GB")
    
    # Recommendations based on memory
    if total_memory < 10:  # RTX4000 (8GB)
        print("Recommended models for your GPU (< 10GB):")
        print("  - deepseek-ai/deepseek-coder-1.3b-base")
        print("  - deepseek-ai/deepseek-coder-1.3b-instruct")
        print("  - CodeLlama-7b-hf (4-bit only)")
    elif total_memory < 18:  # RTX5000 (16GB)
        print("Recommended models for your GPU (10-18GB):")
        print("  - deepseek-ai/deepseek-coder-6.7b-base (4-bit)")
        print("  - deepseek-ai/deepseek-coder-6.7b-instruct (4-bit)")
        print("  - CodeLlama-7b-hf")
    else:  # More than 16GB
        print("Recommended models for your GPU (>18GB):")
        print("  - deepseek-ai/deepseek-coder-6.7b-base")
        print("  - deepseek-ai/deepseek-coder-6.7b-instruct")
        print("  - CodeLlama-13b-hf (4-bit)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Memory Analyzer for LLMs")
    parser.add_argument("--model", type=str, help="Test loading a specific model")
    parser.add_argument("--no-4bit", action="store_true", help="Don't use 4-bit quantization")
    
    args = parser.parse_args()
    
    print("GPU Memory Analyzer")
    print("==================")
    print_gpu_memory_usage()
    
    print("\nRecommended Models:")
    find_suitable_models()
    
    if args.model:
        print("\nTesting specific model loading:")
        test_model_load(args.model, not args.no_4bit)
    else:
        print("\nTo test a specific model, run:")
        print("python analyze_gpu_memory.py --model MODEL_NAME")
EOF

# Create comprehensive dependency fix script
cat > fix_dependencies.sh << 'EOF'
#!/bin/bash

echo "===================================================================="
echo "Fixing dependency issues for Jarvis AI Assistant on Paperspace..."
echo "===================================================================="

# Clean environment first
pip uninstall -y flash-attn bitsandbytes unsloth peft accelerate xformers

# Fix LD_LIBRARY_PATH for CUDA compatibility
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/lib64-nvidia' >> ~/.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/lib64-nvidia

# Fix torch version to ensure compatibility
echo "Installing PyTorch 2.1.2 with CUDA 12.1..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu121

# Set up compatible core libraries - using a specific order to avoid conflicts
echo "Installing compatible core dependencies..."
pip install -U huggingface-hub==0.19.4
pip install -U bitsandbytes==0.41.0
pip install -U accelerate==0.27.0
pip install -U peft==0.6.0
pip install -U tokenizers==0.14.1
pip install -U transformers==4.36.2
pip install -U trl==0.7.10
pip install -U datasets==2.19.0
pip install -U einops==0.7.0

# Install xformers compatible with PyTorch 2.1.2
echo "Installing xformers..."
pip install -U xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121

# Skip flash-attention for RTX4000, install for RTX5000 - without building from source
echo "Checking GPU for flash-attention compatibility..."
if python -c "import torch; print('RTX 5000' in torch.cuda.get_device_name(0))" | grep -q "True"; then
  echo "RTX 5000 GPU detected - installing pre-built flash-attention wheel..."
  pip install "flash-attn<2.3.5" --no-build-isolation --prefer-binary
else
  echo "Not installing flash-attention (not needed for RTX4000 or smaller GPUs)"
fi

# Install unsloth with appropriate flags to avoid compilation issues
echo "Installing Unsloth..."
pip install "unsloth>=2025.3.0,<2025.4.5" --no-deps
pip install safetensors==0.4.1

# Install other compatible packages
echo "Installing utility packages..."
pip install ninja==1.11.1 packaging==23.2 psutil==5.9.8
pip install gdown==5.1.0 fsspec==2024.3.1 boto3==1.28.51

# Verify installations
echo "Verifying installations..."
python -c "
import sys
print(f'Python version: {sys.version}')

try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
except Exception as e:
    print(f'PyTorch error: {e}')

try:
    import bitsandbytes as bnb
    print(f'bitsandbytes version: {bnb.__version__}')
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
    import unsloth
    print(f'unsloth version: {unsloth.__version__}')
except Exception as e:
    print(f'unsloth error: {e}')

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
echo "If you still have issues with unsloth, try running:"
echo "pip install unsloth==2025.3.0 --no-deps"
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

chmod +x fix_dependencies.sh

# Create Google Drive file sync script
cat > sync_to_drive.py << EOF
#!/usr/bin/env python
"""
Sync utility for synchronizing model files between Paperspace storage and Google Drive.
This ensures consistency between Paperspace and Google Colab workflows.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
import time

def check_drive_mounted():
    """Check if Google Drive is mounted"""
    return os.path.exists('/content/drive/MyDrive')

def sync_directory(source_dir, target_dir, dry_run=False):
    """Sync files from source to target directory recursively"""
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if not source_path.exists():
        print(f"Source directory does not exist: {source_path}")
        return 0
    
    # Create target dir if it doesn't exist
    if not target_path.exists() and not dry_run:
        print(f"Creating target directory: {target_path}")
        os.makedirs(target_path, exist_ok=True)
    
    # Count of files copied
    copied_files = 0
    
    # Walk through source directory
    for root, dirs, files in os.walk(source_path):
        # Create corresponding subdirectories in target
        rel_path = os.path.relpath(root, source_path)
        target_subdir = target_path / rel_path
        
        if not target_subdir.exists() and not dry_run:
            print(f"Creating directory: {target_subdir}")
            os.makedirs(target_subdir, exist_ok=True)
        
        # Copy each file
        for file in files:
            source_file = Path(root) / file
            target_file = target_subdir / file
            
            # Check if target file exists and is newer
            if target_file.exists() and target_file.stat().st_mtime >= source_file.stat().st_mtime:
                # Skip if target is same or newer
                continue
                
            # Copy the file
            if not dry_run:
                print(f"Copying: {source_file} -> {target_file}")
                shutil.copy2(source_file, target_file)
            else:
                print(f"Would copy: {source_file} -> {target_file}")
                
            copied_files += 1
    
    return copied_files

def main():
    parser = argparse.ArgumentParser(description="Sync files between Paperspace storage and Google Drive")
    parser.add_argument("--source", help="Source directory", default="/storage/Jarvis_AI_Assistant")
    parser.add_argument("--target", help="Target directory", default="/content/drive/MyDrive/Jarvis_AI_Assistant")
    parser.add_argument("--direction", choices=["to_drive", "from_drive"], default="to_drive", 
                        help="Sync direction: to Google Drive or from Google Drive")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be copied without making changes")
    
    args = parser.parse_args()
    
    if not check_drive_mounted():
        print("Error: Google Drive is not mounted. Please mount it first.")
        print("Run: from google.colab import drive; drive.mount('/content/drive')")
        return 1
    
    # Set source and target based on direction
    if args.direction == "to_drive":
        source_dir = args.source
        target_dir = args.target
        print(f"Syncing from Paperspace storage to Google Drive")
    else:
        source_dir = args.target
        target_dir = args.source
        print(f"Syncing from Google Drive to Paperspace storage")
    
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    
    if args.dry_run:
        print("Dry run mode: No files will be copied")
    
    start_time = time.time()
    copied_files = sync_directory(source_dir, target_dir, args.dry_run)
    elapsed_time = time.time() - start_time
    
    print(f"Sync completed in {elapsed_time:.2f} seconds")
    print(f"Files copied: {copied_files}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
EOF

chmod +x sync_to_drive.py

# Update the COLAB_COMMANDS.md file with Google Drive sync commands for Paperspace
cat > paperspace_gdrive_commands.md << EOF
# Google Drive Integration for Paperspace

This document provides commands for integrating Google Drive with Paperspace to maintain a consistent workflow between Colab and Paperspace.

## Mount Google Drive in Paperspace

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Verify Drive is mounted and list contents
!ls -la /content/drive/MyDrive
```

## Sync Models Between Paperspace and Google Drive

```python
# Sync models from Paperspace storage to Google Drive
!python sync_to_drive.py --direction to_drive

# Sync models from Google Drive to Paperspace storage
!python sync_to_drive.py --direction from_drive

# Dry run to see what would be synced without making changes
!python sync_to_drive.py --dry-run
```

## Using Google Drive Path in Commands

Replace `/storage/Jarvis_AI_Assistant` with `/content/drive/MyDrive/Jarvis_AI_Assistant` in all commands to use Google Drive directly.

### Example Training Command with Google Drive

```python
# Train using Google Drive storage
!python src/generative_ai_module/jarvis_unified.py \\
    --mode train \\
    --model deepseek-ai/deepseek-coder-1.3b-base \\
    --datasets pile \\
    --max-samples 200 \\
    --epochs 1 \\
    --batch-size 2 \\
    --gradient-accumulation-steps 8 \\
    --sequence-length 512 \\
    --load-in-4bit \\
    --use-unsloth \\
    --memory-file /content/drive/MyDrive/Jarvis_AI_Assistant/memory.json
```

### Example Interactive Mode with Google Drive

```python
# Interactive mode using Google Drive for memory file
!python src/generative_ai_module/jarvis_unified.py \\
    --mode interactive \\
    --model deepseek-ai/deepseek-coder-1.3b-instruct \\
    --load-in-4bit \\
    --use-unsloth \\
    --memory-file /content/drive/MyDrive/Jarvis_AI_Assistant/chat_history.json
```
EOF

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
    echo ""
    echo "Run the memory analyzer to find suitable models:"
    echo "python analyze_gpu_memory.py"
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
    echo ""
    echo "Run the memory analyzer to find suitable models:"
    echo "python analyze_gpu_memory.py"
    echo "==================================================================="
else
    echo "==================================================================="
    echo "Setup complete for GPU!"
    echo ""
    echo "Run the memory analyzer to find suitable models:"
    echo "python analyze_gpu_memory.py"
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
echo "If you encounter dependency issues, run:"
echo "bash fix_dependencies.sh"
echo ""
echo "If you encounter bitsandbytes or CUDA issues, run:"
echo "python fix_bitsandbytes.py"
echo ""
echo "For details on running the assistant, see PAPERSPACE_COMMANDS.md" 