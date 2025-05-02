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

If you encounter the dependency issues mentioned, run this fixed installation script:

```bash
# Create a fixed dependency installation script
cat > fix_dependencies.sh << 'EOF'
#!/bin/bash

echo "Fixing dependency issues for Jarvis AI Assistant on Paperspace..."

# Clean environment first
pip uninstall -y bitsandbytes unsloth peft accelerate flash-attn xformers

# Fix torch version to ensure compatibility
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu121

# Install compatible bitsandbytes
pip install bitsandbytes==0.41.0

# Install other dependencies with pinned versions
pip install accelerate==0.27.0
pip install peft==0.6.0
pip install tokenizers==0.14.1
pip install transformers==4.36.2
pip install einops==0.7.0

# Install xformers compatible with PyTorch 2.1.2
pip install -U xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121

# Skip flash-attention for RTX4000, install for RTX5000
if python -c "import torch; print('RTX 5000' in torch.cuda.get_device_name(0))" | grep -q "True"; then
  pip install -U "flash-attn<2.3.5" --no-build-isolation
else
  echo "Skipping flash-attention for RTX4000 GPU"
fi

# Install unsloth with modified dependencies
pip install "unsloth>=2025.3.0" --no-deps
pip install trl==0.7.10 datasets==2.19.0

# Fix CUDA library paths if needed
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/lib64-nvidia' >> ~/.bashrc
source ~/.bashrc

echo "Dependency fixes complete"
EOF

# Make it executable and run it
chmod +x fix_dependencies.sh
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

For RTX4000, you can skip flash-attention:

```bash
# Skip flash-attention for RTX4000
# For RTX5000, use an older version:
pip install "flash-attn<2.3.5" --no-build-isolation
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
