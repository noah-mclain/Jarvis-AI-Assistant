#!/bin/bash

# Google Colab / Paperspace Fine-Tuning Setup Script for GPU
echo "Setting up GPU-optimized environment for model fine-tuning..."

# Check if running in Google Colab or Paperspace
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

# Function to print GPU setup instructions
print_gpu_instructions() {
    echo "============================================================"
    echo "ERROR: No GPU detected in your environment!"
    echo "============================================================"
    if [ $IN_COLAB -eq 1 ]; then
        echo "To enable GPU in Google Colab:"
        echo "1. Click on 'Runtime' in the top menu"
        echo "2. Select 'Change runtime type'"
        echo "3. Choose 'GPU' from the hardware accelerator dropdown"
        echo "4. Click 'Save'"
        echo "5. Click on 'Runtime' in the top menu again and select 'Restart runtime'"
    elif [ $IN_PAPERSPACE -eq 1 ]; then
        echo "Please ensure you selected a GPU machine in Paperspace."
        echo "You should select a machine with RTX4000, RTX5000, or A100 GPU."
    else
        echo "Please ensure you have a CUDA-capable GPU and NVIDIA drivers installed."
    fi
    echo "6. Run this script again after restart"
    echo "============================================================"
    exit 1
}

# Check for GPU using PyTorch
echo "Checking for GPU..."
GPU_AVAILABLE=0
A100_GPU=false
RTX4000_GPU=false
RTX5000_GPU=false

if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo "PyTorch confirms CUDA is available"
    GPU_AVAILABLE=1
    
    # Get CUDA version and GPU info from PyTorch
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')")
    
    echo "CUDA Version: $CUDA_VERSION"
    echo "GPU: $GPU_NAME"
    
    # Check GPU type via PyTorch
    if echo "$GPU_NAME" | grep -q "A100"; then
        echo "A100 GPU confirmed by PyTorch - using optimized settings"
        A100_GPU=true
    elif echo "$GPU_NAME" | grep -q "RTX 4000"; then
        echo "RTX 4000 GPU confirmed by PyTorch - using optimized settings"
        RTX4000_GPU=true
    elif echo "$GPU_NAME" | grep -q "RTX 5000"; then
        echo "RTX 5000 GPU confirmed by PyTorch - using optimized settings"
        RTX5000_GPU=true
    else
        echo "Using generic GPU settings for $GPU_NAME"
    fi
else
    echo "PyTorch cannot detect CUDA"
fi

# Show instructions if no GPU 
if [ $GPU_AVAILABLE -eq 0 ]; then
    print_gpu_instructions
fi

# First, clean up any conflicting packages with a clean environment
echo "First, let's clean up potential conflicts..."
if [ $IN_COLAB -eq 1 ] || [ $IN_PAPERSPACE -eq 1 ]; then
    # Remove problematic packages that cause conflicts
    pip uninstall -y bitsandbytes
    pip uninstall -y unsloth
    pip uninstall -y peft
    pip uninstall -y accelerate
fi

# Use the built-in PyTorch in Colab/Paperspace or install if not available
if [ $IN_COLAB -eq 1 ]; then
    echo "Using pre-installed PyTorch in Google Colab"
else
    echo "Installing PyTorch with CUDA support"
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu121
fi

python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

# IMPORTANT: Fix the bitsandbytes CUDA compatibility issue
echo "Setting up bitsandbytes for CUDA $CUDA_VERSION..."
# For CUDA 12.x, we need to build bitsandbytes from source or use a special installation method
pip install --no-cache-dir --upgrade --no-deps https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-any.whl

# Install accelerate first since many packages depend on it
echo "Installing accelerate..."
pip install -q "accelerate>=0.30.0" --no-deps
pip install -q "accelerate>=0.30.0" --upgrade

# Install core dependencies required for fine-tuning
echo "Installing fine-tuning dependencies..."
pip install -q "transformers>=4.36.0,<4.40.0" --no-deps
pip install -q "transformers>=4.36.0,<4.40.0"
pip install -q "peft>=0.7.0" --no-deps
pip install -q "peft>=0.7.0"
pip install -q "safetensors>=0.4.0"
pip install -q "datasets>=2.14.0"
pip install -q "trl>=0.7.1"
pip install -q "einops>=0.7.0"
pip install -q "tokenizers>=0.19.0"
pip install -q "flash-attn>=2.3.0" --no-deps
pip install -q "flash-attn>=2.3.0"

# Install unsloth safely
echo "Installing unsloth..."
pip install -q "unsloth>=2025.3.0"

# GPU-specific optimizations based on detected hardware
if [ "$A100_GPU" = true ]; then
    echo "Applying A100-specific optimizations..."
    
    # Set environment variables for optimal A100 performance
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    export CUDA_LAUNCH_BLOCKING=0
    export TOKENIZERS_PARALLELISM=true
    
    # Create an A100-optimized config file
    mkdir -p ~/.config/accelerate
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
    
    echo "Created A100-optimized accelerate config with BF16 mixed precision"
elif [ "$RTX4000_GPU" = true ] || [ "$RTX5000_GPU" = true ]; then
    echo "Applying RTX 4000/5000-specific optimizations..."
    
    # Set environment variables for optimal RTX performance
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
    export CUDA_LAUNCH_BLOCKING=0
    export TOKENIZERS_PARALLELISM=true
    
    # Create an RTX-optimized config file for reduced memory usage
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
fi

# Install other dependencies
echo "Installing additional dependencies..."
pip install -q "xformers>=0.0.23.post1" --no-deps
pip install -q "xformers>=0.0.23.post1"
pip install -q "triton>=2.1.0"
pip install -q "boto3>=1.34.0" "gdown>=5.1.0" "fsspec>=2024.3.0" "psutil>=5.9.0"
pip install -q "jupyterlab>=4.4.0" "tensorboard>=2.16.0"
pip install -q "ninja>=1.11.0" "packaging>=23.2"

# Mount Google Drive in Colab or create Paperspace storage links
if [ $IN_COLAB -eq 1 ]; then
    echo "Mounting Google Drive (if in Colab)..."
    echo "from google.colab import drive; drive.mount('/content/drive')" > mount_drive.py
    python mount_drive.py || echo "Failed to mount Google Drive. You can try manually using: from google.colab import drive; drive.mount('/content/drive')"
    
    # Create directories in Google Drive if mounted
    if [ -d "/content/drive/MyDrive" ]; then
        echo "Creating directories in Google Drive..."
        mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant
        mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant/models
        mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant/datasets
    else
        echo "Google Drive not mounted or not accessible. Skipping directory creation."
    fi
elif [ $IN_PAPERSPACE -eq 1 ]; then
    echo "Setting up Paperspace storage..."
    
    # Create directories in Paperspace persistent storage
    if [ -d "/storage" ]; then
        echo "Creating directories in Paperspace persistent storage..."
        mkdir -p /storage/Jarvis_AI_Assistant
        mkdir -p /storage/Jarvis_AI_Assistant/models
        mkdir -p /storage/Jarvis_AI_Assistant/datasets
        
        # Create symlinks for easier access
        ln -sf /storage/Jarvis_AI_Assistant /notebooks/Jarvis_AI_Assistant_storage
        echo "Created symlink to persistent storage in /notebooks/Jarvis_AI_Assistant_storage"
    else
        echo "Paperspace /storage not found. Using local storage instead."
        mkdir -p ~/Jarvis_AI_Assistant
        mkdir -p ~/Jarvis_AI_Assistant/models
        mkdir -p ~/Jarvis_AI_Assistant/datasets
    fi
fi

# Create project directory
echo "Creating project directory..."
mkdir -p Jarvis-AI-Assistant
cd Jarvis-AI-Assistant || echo "Failed to change to project directory"

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

# Create a fine-tuning example notebook
echo "Creating GPU fine-tuning example notebook..."
cat > GPU_Fine_Tuning.ipynb << EOF
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# GPU Fine-Tuning with Unsloth\n",
    "\n",
    "This notebook demonstrates how to fine-tune a model using GPU hardware in Google Colab or Paperspace."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Verify GPU setup\n",
    "import torch\n",
    "print(f\"PyTorch version: {torch.__version__}\")\n",
    "print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
    "if torch.cuda.is_available():\n",
    "    print(f\"GPU: {torch.cuda.get_device_name(0)}\")\n",
    "    print(f\"CUDA version: {torch.version.cuda}\")\n",
    "    print(f\"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\")\n",
    "    print(f\"BF16 support: {torch.cuda.is_bf16_supported()}\")"
   ],
   "outputs": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Import required libraries\n",
    "import os\n",
    "from datasets import load_dataset\n",
    "from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig\n",
    "from peft import LoraConfig\n",
    "\n",
    "# Unsloth for efficient fine-tuning\n",
    "from unsloth import FastLanguageModel"
   ],
   "outputs": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Detect environment and adjust settings accordingly\n",
    "IN_COLAB = 'google.colab' in str(get_ipython())\n",
    "IN_PAPERSPACE = os.path.exists('/storage') or os.path.exists('/notebooks')\n",
    "\n",
    "# Detect GPU type and adjust settings\n",
    "gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"\"\n",
    "IS_A100 = \"A100\" in gpu_name\n",
    "IS_RTX4000 = \"RTX 4000\" in gpu_name \n",
    "IS_RTX5000 = \"RTX 5000\" in gpu_name\n",
    "\n",
    "# Adjust batch size and sequence length based on GPU type\n",
    "if IS_A100:\n",
    "    # A100 has plenty of memory\n",
    "    BATCH_SIZE = 8\n",
    "    SEQ_LENGTH = 2048\n",
    "    USE_BF16 = torch.cuda.is_bf16_supported()\n",
    "    LOAD_IN_4BIT = True\n",
    "elif IS_RTX4000 or IS_RTX5000:\n",
    "    # RTX4000/5000 have less memory\n",
    "    BATCH_SIZE = 4\n",
    "    SEQ_LENGTH = 1024\n",
    "    USE_BF16 = False  # Use FP16 instead\n",
    "    LOAD_IN_4BIT = True\n",
    "else:\n",
    "    # Conservative defaults\n",
    "    BATCH_SIZE = 2\n",
    "    SEQ_LENGTH = 512\n",
    "    USE_BF16 = False\n",
    "    LOAD_IN_4BIT = True\n",
    "\n",
    "print(f\"Environment: {'Colab' if IN_COLAB else 'Paperspace' if IN_PAPERSPACE else 'Other'}\")\n",
    "print(f\"GPU Type: {gpu_name}\")\n",
    "print(f\"Batch Size: {BATCH_SIZE}\")\n",
    "print(f\"Sequence Length: {SEQ_LENGTH}\")\n",
    "print(f\"Using BF16: {USE_BF16}\")\n",
    "print(f\"Load in 4-bit: {LOAD_IN_4BIT}\")\n",
    "\n",
    "# Set up storage path based on environment\n",
    "if IN_COLAB and os.path.exists('/content/drive/MyDrive'):\n",
    "    STORAGE_PATH = \"/content/drive/MyDrive/Jarvis_AI_Assistant\"\n",
    "elif IN_PAPERSPACE and os.path.exists('/storage'):\n",
    "    STORAGE_PATH = \"/storage/Jarvis_AI_Assistant\"\n",
    "else:\n",
    "    STORAGE_PATH = \"./jarvis_models\"\n",
    "    \n",
    "print(f\"Storage Path: {STORAGE_PATH}\")\n",
    "os.makedirs(STORAGE_PATH, exist_ok=True)"
   ],
   "outputs": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Configure model and quantization\n",
    "model_id = \"deepseek-ai/deepseek-coder-6.7b-base\"  # Replace with your preferred model\n",
    "\n",
    "# Create BitsAndBytesConfig for 4-bit quantization\n",
    "bnb_config = BitsAndBytesConfig(\n",
    "    load_in_4bit=LOAD_IN_4BIT,\n",
    "    bnb_4bit_quant_type=\"nf4\",\n",
    "    bnb_4bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16,\n",
    "    bnb_4bit_use_double_quant=True,\n",
    ")"
   ],
   "outputs": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Load and prepare the model with Unsloth\n",
    "# This loads faster and uses less memory than the standard HF way\n",
    "model, tokenizer = FastLanguageModel.from_pretrained(\n",
    "    model_name=model_id,\n",
    "    quantization_config=bnb_config,\n",
    "    max_seq_length=SEQ_LENGTH,\n",
    "    device_map=\"auto\",\n",
    "    trust_remote_code=True\n",
    ")\n",
    "\n",
    "# Set up LoRA for efficient fine-tuning\n",
    "lora_config = LoraConfig(\n",
    "    r=64,                     # Rank\n",
    "    lora_alpha=16,            # Alpha parameter for LoRA scaling\n",
    "    target_modules=[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\", \"up_proj\", \"down_proj\"],\n",
    "    lora_dropout=0.05,        # Dropout\n",
    "    bias=\"none\",              # No bias parameters\n",
    "    task_type=\"CAUSAL_LM\"     # Task type\n",
    ")\n",
    "\n",
    "# Apply LoRA to the model\n",
    "model = FastLanguageModel.get_peft_model(\n",
    "    model, \n",
    "    lora_config,\n",
    "    use_gradient_checkpointing=True,  # Save GPU memory with gradient checkpointing\n",
    ")"
   ],
   "outputs": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Example: Prepare a small dataset\n",
    "# For a real task, replace this with your actual dataset\n",
    "data = [\n",
    "    {\"text\": \"def fibonacci(n):\\n    if n <= 1:\\n        return n\\n    else:\\n        return fibonacci(n-1) + fibonacci(n-2)\\n\"},\n",
    "    {\"text\": \"def quicksort(arr):\\n    if len(arr) <= 1:\\n        return arr\\n    pivot = arr[len(arr) // 2]\\n    left = [x for x in arr if x < pivot]\\n    middle = [x for x in arr if x == pivot]\\n    right = [x for x in arr if x > pivot]\\n    return quicksort(left) + middle + quicksort(right)\\n\"},\n",
    "]\n",
    "\n",
    "from datasets import Dataset\n",
    "dataset = Dataset.from_list(data)\n",
    "\n",
    "# GPU-optimized training parameters\n",
    "training_args = {\n",
    "    \"num_train_epochs\": 1,\n",
    "    \"per_device_train_batch_size\": BATCH_SIZE,\n",
    "    \"gradient_accumulation_steps\": 8 // BATCH_SIZE,  # Adjust based on batch size\n",
    "    \"optim\": \"adamw_torch_fused\",      # Optimized for NVIDIA GPUs\n",
    "    \"learning_rate\": 2e-4,\n",
    "    \"max_grad_norm\": 0.3,\n",
    "    \"warmup_ratio\": 0.03,\n",
    "    \"lr_scheduler_type\": \"constant\",\n",
    "    \"save_strategy\": \"steps\",\n",
    "    \"save_steps\": 10,\n",
    "    \"save_total_limit\": 2,\n",
    "    \"bf16\": USE_BF16,                 # Use BF16 for A100, FP16 for others\n",
    "    \"fp16\": not USE_BF16,             # Use FP16 when not using BF16\n",
    "    \"logging_steps\": 1,\n",
    "    \"report_to\": [\"tensorboard\"],\n",
    "    \"output_dir\": os.path.join(STORAGE_PATH, \"models\", \"fine_tuned\")\n",
    "}\n",
    "\n",
    "# This is just a setup example - no actual training happens here to save notebook space\n",
    "print(\"Model and dataset prepared for fine-tuning\")"
   ],
   "outputs": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Fine-tune the model\n",
    "# Uncomment the following to run actual training:\n",
    "\n",
    "'''\n",
    "from transformers import TrainingArguments, Trainer\n",
    "\n",
    "# Tokenize the dataset\n",
    "def tokenize_function(examples):\n",
    "    return tokenizer(examples[\"text\"], truncation=True, max_length=SEQ_LENGTH)\n",
    "\n",
    "tokenized_dataset = dataset.map(tokenize_function, batched=True)\n",
    "\n",
    "# Create training arguments\n",
    "args = TrainingArguments(**training_args)\n",
    "\n",
    "# Create Trainer\n",
    "trainer = Trainer(\n",
    "    model=model,\n",
    "    args=args,\n",
    "    train_dataset=tokenized_dataset,\n",
    ")\n",
    "\n",
    "# Start training\n",
    "trainer.train()\n",
    "\n",
    "# Save the model\n",
    "model.save_pretrained(training_args[\"output_dir\"])\n",
    "tokenizer.save_pretrained(training_args[\"output_dir\"])\n",
    "'''"
   ],
   "outputs": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Example of text generation with the model\n",
    "prompt = \"def calculate_fibonacci_sequence(n):\\n\"\n",
    "\n",
    "# Set the generation parameters\n",
    "generation_config = {\n",
    "    \"max_new_tokens\": 500,\n",
    "    \"temperature\": 0.7,\n",
    "    \"top_p\": 0.95,\n",
    "    \"top_k\": 50,\n",
    "    \"repetition_penalty\": 1.1,\n",
    "    \"do_sample\": True,\n",
    "    \"use_cache\": True\n",
    "}\n",
    "\n",
    "# Generate text\n",
    "input_ids = tokenizer(prompt, return_tensors=\"pt\").input_ids.to(model.device)\n",
    "generated_ids = model.generate(input_ids, **generation_config)\n",
    "\n",
    "# Decode the generated text\n",
    "generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)\n",
    "print(generated_text)"
   ],
   "outputs": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.0"
  },
  "accelerator": "GPU"
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

# Create a quick-fix script for bitsandbytes issues
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

# Create a Paperspace-specific instructions file
echo "Creating Paperspace-specific instructions..."
cat > PAPERSPACE_INSTRUCTIONS.md << EOF
# Running Jarvis AI Assistant on Paperspace

This guide provides instructions for running Jarvis AI Assistant on Paperspace with RTX4000 or RTX5000 GPUs.

## Creating a Paperspace Machine

1. Sign up or log in to [Paperspace](https://www.paperspace.com/)
2. Create a new Gradient Notebook with:
   - Runtime: PyTorch 2.1.0
   - Machine: Choose one of:
     - RTX4000 (8GB VRAM): Good for smaller models or tight budgets
     - RTX5000 (16GB VRAM): Better for mid-size models
   - Disk Size: At least 50GB recommended

## Initial Setup

After your Paperspace notebook is running:

1. Open a terminal and clone the repository:
   \`\`\`bash
   git clone https://github.com/your-username/Jarvis-AI-Assistant.git
   cd Jarvis-AI-Assistant
   bash colab_setup.sh
   \`\`\`

2. The setup script will automatically detect that you're running in Paperspace and apply RTX4000/5000 appropriate optimizations.

## Using Persistent Storage

Paperspace notebooks have:
- **/notebooks**: Non-persistent storage (deleted when machine is off)
- **/storage**: Persistent storage (maintained between sessions)

The setup script creates these directories in persistent storage:
- **/storage/Jarvis_AI_Assistant**
- **/storage/Jarvis_AI_Assistant/models**
- **/storage/Jarvis_AI_Assistant/datasets**

It also creates a symlink in your notebook directory:
- **/notebooks/Jarvis_AI_Assistant_storage** → /storage/Jarvis_AI_Assistant

## RTX4000/5000 Optimizations

RTX4000 (8GB VRAM) and RTX5000 (16GB VRAM) have less memory than A100 GPUs, so:

1. Use smaller batch sizes:
   - RTX4000: batch size 2-4
   - RTX5000: batch size 4-6

2. Use shorter sequence lengths:
   - RTX4000: sequence length 512-1024
   - RTX5000: sequence length 1024-2048

3. Always use gradient accumulation:
   - Use \`--gradient-accumulation-steps 4\` or higher

4. Use FP16 instead of BF16 (RTX GPUs don't support BF16 natively)

## Example Commands

### Fine-tuning on RTX4000:

\`\`\`bash
python src/generative_ai_module/jarvis_unified.py \\
    --mode train \\
    --model deepseek-ai/deepseek-coder-1.3b-base \\
    --datasets pile \\
    --max-samples 200 \\
    --epochs 1 \\
    --batch-size 2 \\
    --load-in-4bit \\
    --use-unsloth \\
    --memory-file /storage/Jarvis_AI_Assistant/memory.json
\`\`\`

### Fine-tuning on RTX5000:

\`\`\`bash
python src/generative_ai_module/jarvis_unified.py \\
    --mode train \\
    --model deepseek-ai/deepseek-coder-6.7b-base \\
    --datasets pile \\
    --max-samples 500 \\
    --epochs 1 \\
    --batch-size 4 \\
    --load-in-4bit \\
    --use-unsloth \\
    --memory-file /storage/Jarvis_AI_Assistant/memory.json
\`\`\`

### Interactive mode:

\`\`\`bash
python src/generative_ai_module/jarvis_unified.py \\
    --mode interactive \\
    --model deepseek-ai/deepseek-coder-1.3b-instruct \\
    --load-in-4bit \\
    --use-unsloth \\
    --memory-file /storage/Jarvis_AI_Assistant/chat_history.json
\`\`\`

## Troubleshooting

If you encounter out-of-memory errors:
1. Reduce the batch size
2. Use a smaller model (e.g., 1.3B instead of 6.7B)
3. Reduce sequence length
4. Increase gradient accumulation steps
5. Run \`python fix_bitsandbytes.py\` to fix CUDA compatibility issues
EOF

echo "==================================================================="
echo "Setup complete! You can now run fine-tuning on your GPU."
echo ""
echo "IMPORTANT: If you encounter any issues with bitsandbytes, run:"
echo "python fix_bitsandbytes.py"
echo ""
echo "A fine-tuning notebook has been created: GPU_Fine_Tuning.ipynb"
echo ""
if [ $IN_PAPERSPACE -eq 1 ]; then
    echo "For Paperspace-specific instructions, see: PAPERSPACE_INSTRUCTIONS.md"
fi
echo "===================================================================" 