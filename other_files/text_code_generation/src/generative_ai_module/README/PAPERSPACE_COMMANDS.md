# Jarvis AI Assistant - Paperspace Setup Guide

This guide provides detailed instructions for setting up and running Jarvis AI Assistant on Paperspace with RTX4000/5000 GPUs, including Google Drive integration for persistent storage.

## Table of Contents

- [Initial Environment Setup](#initial-environment-setup)
- [Fix Dependency Issues](#fix-dependency-issues)
- [Google Drive Integration](#google-drive-integration)
- [Model Training](#model-training)
- [Interactive Mode](#interactive-mode)
- [Complete Pipeline](#complete-pipeline)
- [Troubleshooting](#troubleshooting)

## Initial Environment Setup

### Create a Paperspace Machine

1. Sign up or log in to [Paperspace](https://www.paperspace.com/)
2. Create a new Gradient Notebook with:
   - Runtime: PyTorch 2.1.0
   - Machine: Choose one of:
     - RTX4000 (8GB VRAM): Good for smaller models or tight budgets
     - RTX5000 (16GB VRAM): Better for mid-size models
   - Disk Size: At least 50GB recommended

### Clone Repository and Set Up Environment

```bash
# Clone the repository
git clone https://github.com/your-username/Jarvis-AI-Assistant.git
cd Jarvis-AI-Assistant

# Run the Paperspace-specific setup script
bash setup_paperspace.sh
```

## Fix Dependency Issues

For dependency issues encountered on Paperspace, use our comprehensive fix scripts:

### Quick Fix: All Dependencies

Run the included fix_dependencies.sh script:

```bash
# Run the existing dependency fix script
chmod +x fix_dependencies.sh
./fix_dependencies.sh
```

### Fix Flash-Attention Issues

If you're specifically having issues with flash-attn installation:

```bash
# Run the flash-attention specific fix script
chmod +x fix_flash_attn.sh
./fix_flash_attn.sh
```

### Manual Fix

If the scripts don't exist or you need to create them manually:

```bash
# Create the comprehensive fix script
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

# Create the flash-attention specific fix script
cat > fix_flash_attn.sh << 'EOF'
#!/bin/bash

echo "===================================================================="
echo "Flash-Attention Fix for Paperspace"
echo "===================================================================="

# Uninstall any existing flash-attn
echo "Removing any existing flash-attn installations..."
pip uninstall -y flash-attn

# Check GPU type
echo "Checking GPU type..."
if python -c "import torch; print('RTX 5000' in torch.cuda.get_device_name(0))" | grep -q "True"; then
  echo "RTX 5000 GPU detected - installing pre-built flash-attention wheel..."

  # Try multiple approaches to install flash-attn
  echo "Attempt 1: Using pre-built wheel with prefer-binary..."
  pip install "flash-attn<2.3.5" --prefer-binary --no-build-isolation || \

  echo "Attempt 2: Using specific version 2.3.3..."
  pip install flash-attn==2.3.3 --prefer-binary --no-build-isolation || \

  echo "Attempt 3: Trying explicit CUDA installation..."
  pip install "flash-attn<2.3.5" --prefer-binary --extra-index-url https://download.pytorch.org/whl/cu121 || \

  echo "Skipping flash-attention installation - this is fine, other optimizations will still work"
else
  echo "RTX 4000 or other GPU detected - flash-attention not recommended for this GPU"
  echo "Skipping flash-attention installation"
fi

# Install alternative optimizations
echo "Installing alternative memory optimizations..."
pip install -U xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121

echo "===================================================================="
echo "Flash-attention fix complete! If installation failed, this is okay,"
echo "as the system will fall back to using xformers optimizations instead."
echo "===================================================================="
EOF

# Make the scripts executable
chmod +x fix_dependencies.sh
chmod +x fix_flash_attn.sh

# Run the comprehensive fix
./fix_dependencies.sh
```

## Google Drive Integration

### Mount Google Drive

```python
# Mount Google Drive in Paperspace
from google.colab import drive
drive.mount('/content/drive')

# Verify that Google Drive is mounted
!ls -la /content/drive/MyDrive
```

### Create Directory Structure in Google Drive

```python
# Create directories in Google Drive for persistent storage
!mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant
!mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant/models
!mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant/datasets
!mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant/checkpoints
```

### Verify Google Drive Integration

```python
# Check if Google Drive is properly mounted and directories exist
!ls -la /content/drive/MyDrive/Jarvis_AI_Assistant

# Verify the symlink to Google Drive storage is working
!ls -la /notebooks/google_drive_storage
```

### Sync Between Paperspace Storage and Google Drive

```python
# Sync from Paperspace storage to Google Drive
!python sync_to_drive.py --direction to_drive

# Sync from Google Drive to Paperspace storage
!python sync_to_drive.py --direction from_drive
```

## Model Training

### RTX4000 GPU (8GB VRAM) Training Commands

```python
# Train with a smaller model on RTX4000
!python src/generative_ai_module/jarvis_unified.py \
    --mode train \
    --model deepseek-ai/deepseek-coder-1.3b-base \
    --datasets pile \
    --max-samples 200 \
    --epochs 1 \
    --batch-size 2 \
    --gradient-accumulation-steps 8 \
    --sequence-length 512 \
    --load-in-4bit \
    --use-unsloth \
    --memory-file /content/drive/MyDrive/Jarvis_AI_Assistant/memory.json
```

### RTX5000 GPU (16GB VRAM) Training Commands

```python
# Train with a larger model on RTX5000
!python src/generative_ai_module/jarvis_unified.py \
    --mode train \
    --model deepseek-ai/deepseek-coder-6.7b-base \
    --datasets pile \
    --max-samples 300 \
    --epochs 1 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --sequence-length 1024 \
    --load-in-4bit \
    --use-unsloth \
    --memory-file /content/drive/MyDrive/Jarvis_AI_Assistant/memory.json
```

## Interactive Mode

### Interactive Mode with Pre-trained Model

```python
# Run interactive chat session with DeepSeek model
!python src/generative_ai_module/jarvis_unified.py \
    --mode interactive \
    --model deepseek-ai/deepseek-coder-1.3b-instruct \
    --load-in-4bit \
    --use-unsloth \
    --memory-file /content/drive/MyDrive/Jarvis_AI_Assistant/chat_history.json
```

### Interactive Mode with Fine-tuned Model

```python
# Run interactive chat with your fine-tuned model from Google Drive
!python src/generative_ai_module/jarvis_unified.py \
    --mode interactive \
    --model-path /content/drive/MyDrive/Jarvis_AI_Assistant/models/pile_best \
    --load-in-4bit \
    --use-unsloth \
    --memory-file /content/drive/MyDrive/Jarvis_AI_Assistant/chat_history.json
```

## Complete Pipeline

### End-to-End Pipeline Example

```python
# Step 1: Initial setup
!git clone https://github.com/your-username/Jarvis-AI-Assistant.git
%cd Jarvis-AI-Assistant
!bash setup_paperspace.sh

# Step 2: Fix any dependency issues
!bash fix_dependencies.sh

# Step 3: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant/models

# Step 4: Verify environment
import torch
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')
print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
print(f'BF16 support: {torch.cuda.is_bf16_supported()}')

# Step 5: Run memory analyzer to determine suitable model size
!python analyze_gpu_memory.py

# Step 6: Train the model
# For RTX4000 (smaller model)
!python src/generative_ai_module/jarvis_unified.py \
    --mode train \
    --model deepseek-ai/deepseek-coder-1.3b-base \
    --datasets pile \
    --max-samples 200 \
    --epochs 1 \
    --batch-size 2 \
    --gradient-accumulation-steps 8 \
    --sequence-length 512 \
    --load-in-4bit \
    --use-unsloth \
    --memory-file /content/drive/MyDrive/Jarvis_AI_Assistant/memory.json

# Step 7: Sync trained model to Google Drive
!python sync_to_drive.py --direction to_drive

# Step 8: Run interactive session
!python src/generative_ai_module/jarvis_unified.py \
    --mode interactive \
    --model deepseek-ai/deepseek-coder-1.3b-instruct \
    --load-in-4bit \
    --use-unsloth \
    --memory-file /content/drive/MyDrive/Jarvis_AI_Assistant/chat_history.json
```

## Troubleshooting

### Fix NumPy Version Conflicts (Critical Issue)

NumPy version conflicts are one of the most common issues on Paperspace. If you see errors about NumPy 2.x versus NumPy 1.x compatibility, run:

```bash
# The emergency NumPy fix script
./fix_numpy_errors.sh

# If that doesn't work, run the comprehensive dependency fix
./fix_numpy.sh
```

If you see an error like `Cannot uninstall numpy 2.0.0: no RECORD file was found for numpy`, this indicates a corrupted NumPy installation. Use the more aggressive fix:

```bash
# Forcefully remove corrupted NumPy
sudo rm -rf /usr/local/lib/python3.11/dist-packages/numpy*
pip install numpy==1.26.4 --no-deps --force-reinstall
```

### Fix CUDA and Library Issues

```python
# Run the CUDA compatibility fixer
!python fix_bitsandbytes.py

# If you encounter undefined CUDA symbols, try:
import os
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:/usr/lib64-nvidia'
```

### Fix PyTorch Compatibility Issues

```python
# Check current installations
!pip list | grep -E "torch|bitsandbytes|unsloth|peft|accelerate|xformers"

# Fix PyTorch and associated libraries
!pip uninstall -y torch torchvision torchaudio
!pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu121
```

### Memory Management

```python
# Monitor GPU memory usage
!nvidia-smi

# Check current memory usage with PyTorch
import torch
print(f'GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB')
print(f'GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB')

# Clear CUDA cache if you're running out of memory
torch.cuda.empty_cache()
```

### Common Issues and Solutions

#### bitsandbytes Installation Failure

If the GitHub wheel link is failing:

```bash
# Alternative installation for bitsandbytes
pip install bitsandbytes==0.41.0
```

#### tokenizers Version Conflict

```bash
# Install compatible tokenizers version
pip uninstall -y tokenizers
pip install tokenizers==0.14.1
```

#### flash-attn Installation Failure

For RTX4000, you can skip flash-attention entirely (it's not needed). For RTX5000, use our dedicated fix script:

```bash
# Run the flash-attention specific fix script
./fix_flash_attn.sh

# Or manually try:
pip uninstall -y flash-attn
pip install "flash-attn<2.3.5" --prefer-binary --no-build-isolation

# If that fails, try a specific version:
pip install flash-attn==2.3.3 --prefer-binary --no-build-isolation

# If compilation from source still fails, that's okay!
# The system will use xformers optimizations instead:
pip install -U xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121
```

#### Torch Version Conflicts

```bash
# Fix the torch version, then reinstall unsloth without dependencies
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu121
pip install unsloth --no-deps
```

### Verify Fixed Installation

```python
# Verify that everything is working correctly
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')

import bitsandbytes as bnb
print(f'bitsandbytes version: {bnb.__version__}')

try:
    from unsloth import FastLanguageModel
    print("Unsloth imported successfully")
except ImportError as e:
    print(f"Unsloth import error: {e}")
```

---

## Storage Structure Reference

- **Google Drive Path**: `/content/drive/MyDrive/Jarvis_AI_Assistant`
  - Models: `/content/drive/MyDrive/Jarvis_AI_Assistant/models`
  - Datasets: `/content/drive/MyDrive/Jarvis_AI_Assistant/datasets`
  - Checkpoints: `/content/drive/MyDrive/Jarvis_AI_Assistant/checkpoints`
- **Paperspace Path**: `/storage/Jarvis_AI_Assistant`

  - Models: `/storage/Jarvis_AI_Assistant/models`
  - Datasets: `/storage/Jarvis_AI_Assistant/datasets`
  - Checkpoints: `/storage/Jarvis_AI_Assistant/checkpoints`

- **Symlinks**:
  - Google Drive: `/notebooks/google_drive_storage` → `/content/drive/MyDrive/Jarvis_AI_Assistant`
  - Paperspace Storage: `/notebooks/Jarvis_AI_Assistant_storage` → `/storage/Jarvis_AI_Assistant`
