#!/bin/bash

# Paperspace Setup Script for RTX4000/5000 GPUs
echo "Setting up Paperspace environment with RTX GPU optimizations..."

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

# Set up persistent storage directory structure
echo "Setting up persistent storage in /storage..."
if [ -d "/storage" ]; then
    mkdir -p /storage/Jarvis_AI_Assistant
    mkdir -p /storage/Jarvis_AI_Assistant/models
    mkdir -p /storage/Jarvis_AI_Assistant/datasets
    mkdir -p /storage/Jarvis_AI_Assistant/checkpoints
    
    # Create symlinks for easier access
    ln -sf /storage/Jarvis_AI_Assistant /notebooks/Jarvis_AI_Assistant_storage
    echo "Created symlink to persistent storage in /notebooks/Jarvis_AI_Assistant_storage"
else
    echo "WARNING: /storage directory not found. Using local storage instead."
    mkdir -p ~/Jarvis_AI_Assistant
    mkdir -p ~/Jarvis_AI_Assistant/models
    mkdir -p ~/Jarvis_AI_Assistant/datasets
fi

# Clean up any conflicting packages
echo "Cleaning up potential conflicts..."
pip uninstall -y bitsandbytes
pip uninstall -y unsloth
pip uninstall -y peft
pip uninstall -y accelerate

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA 12.1 support..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu121

python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

# Fix bitsandbytes CUDA compatibility issues
echo "Setting up bitsandbytes for CUDA $CUDA_VERSION..."
pip install --no-cache-dir --upgrade --no-deps https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-any.whl

# Install accelerate
echo "Installing accelerate..."
pip install accelerate==0.30.0

# Install core dependencies required for fine-tuning
echo "Installing fine-tuning dependencies..."
pip install transformers==4.36.2
pip install peft==0.7.0
pip install safetensors==0.4.1
pip install datasets==2.19.0
pip install trl==0.7.10
pip install einops==0.7.0
pip install tokenizers==0.19.1

# Install flash-attention conditionally based on GPU type
if [ "$RTX5000_GPU" = true ]; then
    echo "Installing flash-attention for RTX5000..."
    pip install flash-attn==2.3.4
else
    echo "Skipping flash-attention for RTX4000 due to memory constraints"
fi

# Install unsloth
echo "Installing unsloth..."
pip install unsloth==2025.4.4

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
pip install xformers==0.0.23.post1
pip install triton==2.1.0
pip install boto3==1.34.86 gdown==5.1.0 fsspec==2024.3.1 psutil==5.9.8
pip install jupyterlab==4.4.1 tensorboard==2.16.2
pip install ninja==1.11.1 packaging==23.2

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

# Create a bitsandbytes fixer script
cat > fix_bitsandbytes.py << EOF
import os
import torch
import subprocess
from pathlib import Path

print("BitsAndBytes CUDA Compatibility Fixer")
print("=====================================")

# Check CUDA version
if torch.cuda.is_available():
    cuda_version = torch.version.cuda
    print(f"CUDA version: {cuda_version}")
    
    # Find bitsandbytes installation
    try:
        import bitsandbytes as bnb
        bnb_path = Path(bnb.__file__).parent
        print(f"bitsandbytes installation found at: {bnb_path}")
        
        # Attempt to run the diagnostic
        print("Running bitsandbytes diagnostic...")
        try:
            subprocess.run(["python", "-m", "bitsandbytes"], check=True)
        except subprocess.CalledProcessError:
            print("Diagnostic failed, trying to fix...")
        
        # Check if we're on CUDA 12.x which might need special handling
        if cuda_version.startswith("12"):
            print(f"Detected CUDA 12.x - attempting special fix for this version")
            # Special case for CUDA 12.x
            print("Re-installing bitsandbytes with a CUDA 12.x compatible wheel...")
            subprocess.run(["pip", "install", "--force-reinstall", "--no-deps", 
                          "https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-any.whl"], 
                          check=True)
            print("Fixed bitsandbytes for CUDA 12.x")
    except ImportError:
        print("bitsandbytes not installed, installing compatible version...")
        subprocess.run(["pip", "install", "--no-deps", 
                      "https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-any.whl"], 
                      check=True)
    
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

echo "Persistent storage path: /storage/Jarvis_AI_Assistant"
echo "Storage symlink: /notebooks/Jarvis_AI_Assistant_storage"
echo ""
echo "If you encounter issues with bitsandbytes, run:"
echo "python fix_bitsandbytes.py"
echo ""
echo "For details on running the assistant, see COLAB_COMMANDS.md" 