# Jarvis AI Assistant - GPU Commands

This document provides ready-to-use commands for running Jarvis AI Assistant on Google Colab with A100 GPUs or Paperspace with RTX4000/5000 GPUs.

## Table of Contents

- [Initial Setup](#initial-setup)
- [Model Training](#model-training)
- [Interactive Chat](#interactive-chat)
- [Text Generation](#text-generation)
- [Complete Pipeline](#complete-pipeline)
- [Paperspace Specific Commands](#paperspace-specific-commands)
- [Troubleshooting](#troubleshooting)

## Initial Setup

### Clone Repository and Setup Environment

```python
# Clone the repository
!git clone https://github.com/your-username/Jarvis-AI-Assistant.git
%cd Jarvis-AI-Assistant

# Setup the environment with GPU optimizations
!bash colab_setup.sh

# Mount Google Drive for persistent storage (Colab only)
from google.colab import drive
drive.mount('/content/drive')

# Create directories in Google Drive for persistent storage (Colab only)
!mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant/models
!mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant/datasets
!mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant/checkpoints

# Verify GPU availability
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
    print(f'BF16 support: {torch.cuda.is_bf16_supported()}')
```

### Verify Dependencies

```python
# Check key dependencies
!pip list | grep -E "torch|transformers|peft|accelerate|unsloth|bitsandbytes|flash-attn"

# Fix bitsandbytes if needed
!python fix_bitsandbytes.py
```

## Model Training

### Fine-tune with HuggingFace Pre-trained Models (Colab A100)

```python
# Train with DeepSeek model using Unsloth and 4-bit quantization on A100
!python src/generative_ai_module/jarvis_unified.py \
    --mode train \
    --model deepseek-ai/deepseek-coder-6.7b-base \
    --datasets pile openassistant \
    --max-samples 1000 \
    --epochs 1 \
    --batch-size 8 \
    --load-in-4bit \
    --use-unsloth \
    --memory-file /content/drive/MyDrive/Jarvis_AI_Assistant/memory.json
```

### Memory-Constrained Training

```python
# Train with smaller batch size and fewer samples for memory constraints
!python src/generative_ai_module/jarvis_unified.py \
    --mode train \
    --model deepseek-ai/deepseek-coder-6.7b-base \
    --datasets pile \
    --max-samples 500 \
    --epochs 1 \
    --batch-size 2 \
    --gradient-accumulation-steps 8 \
    --load-in-4bit \
    --use-unsloth \
    --memory-file /content/drive/MyDrive/Jarvis_AI_Assistant/memory.json
```

## Interactive Chat

### Interactive Mode with Pre-trained Model

```python
# Run interactive chat session with DeepSeek model
!python src/generative_ai_module/jarvis_unified.py \
    --mode interactive \
    --model deepseek-ai/deepseek-coder-6.7b-instruct \
    --load-in-4bit \
    --use-unsloth \
    --memory-file /content/drive/MyDrive/Jarvis_AI_Assistant/chat_history.json
```

### Interactive Mode with Fine-tuned Model

```python
# Run interactive chat with your fine-tuned model
!python src/generative_ai_module/jarvis_unified.py \
    --mode interactive \
    --model-path /content/drive/MyDrive/Jarvis_AI_Assistant/models/pile_best \
    --load-in-4bit \
    --use-unsloth \
    --memory-file /content/drive/MyDrive/Jarvis_AI_Assistant/chat_history.json
```

### Continue Previous Conversation

```python
# Continue a previous conversation
!python src/generative_ai_module/jarvis_unified.py \
    --mode interactive \
    --model deepseek-ai/deepseek-coder-6.7b-instruct \
    --history /content/drive/MyDrive/Jarvis_AI_Assistant/chat_history.json \
    --output /content/drive/MyDrive/Jarvis_AI_Assistant/chat_history.json
```

## Text Generation

### Generate Text from Prompt

```python
# Generate text from a single prompt
!python src/generative_ai_module/jarvis_unified.py \
    --mode generate \
    --model deepseek-ai/deepseek-coder-6.7b-instruct \
    --prompt "Write a Python function to calculate Fibonacci numbers" \
    --max-length 500 \
    --temperature 0.7
```

### Generate from Fine-tuned Model

```python
# Generate text using your fine-tuned model
!python src/generative_ai_module/jarvis_unified.py \
    --mode generate \
    --model-path /content/drive/MyDrive/Jarvis_AI_Assistant/models/pile_best \
    --prompt "Create a class for implementing a binary search tree in Python" \
    --max-length 800 \
    --temperature 0.8
```

## Complete Pipeline

### End-to-End Pipeline Example (Colab)

```python
# Step 1: Initial setup
!git clone https://github.com/your-username/Jarvis-AI-Assistant.git
%cd Jarvis-AI-Assistant
!bash colab_setup.sh

# Step 2: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant/models

# Step 3: Verify GPU
import torch
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')

# Step 4: Train the model (optional)
!python src/generative_ai_module/jarvis_unified.py \
    --mode train \
    --model deepseek-ai/deepseek-coder-6.7b-base \
    --datasets pile \
    --max-samples 500 \
    --epochs 1 \
    --batch-size 4 \
    --load-in-4bit \
    --use-unsloth

# Step 5: Run interactive session
!python src/generative_ai_module/jarvis_unified.py \
    --mode interactive \
    --model deepseek-ai/deepseek-coder-6.7b-instruct \
    --memory-file /content/drive/MyDrive/Jarvis_AI_Assistant/chat_history.json
```

## Paperspace Specific Commands

### Setup on Paperspace

```python
# Clone repository and run setup script
!git clone https://github.com/your-username/Jarvis-AI-Assistant.git
%cd Jarvis-AI-Assistant
!bash colab_setup.sh  # Will automatically detect Paperspace environment

# Verify GPU
import torch
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')
print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
```

### RTX4000 GPU (8GB VRAM) Commands

```python
# Fine-tune smaller model on RTX4000 (memory-optimized settings)
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
    --memory-file /storage/Jarvis_AI_Assistant/memory.json

# Interactive mode with smaller model on RTX4000
!python src/generative_ai_module/jarvis_unified.py \
    --mode interactive \
    --model deepseek-ai/deepseek-coder-1.3b-instruct \
    --load-in-4bit \
    --use-unsloth \
    --memory-file /storage/Jarvis_AI_Assistant/chat_history.json
```

### RTX5000 GPU (16GB VRAM) Commands

```python
# Fine-tune on RTX5000 (optimized settings)
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
    --memory-file /storage/Jarvis_AI_Assistant/memory.json

# Interactive mode on RTX5000
!python src/generative_ai_module/jarvis_unified.py \
    --mode interactive \
    --model deepseek-ai/deepseek-coder-6.7b-instruct \
    --load-in-4bit \
    --use-unsloth \
    --memory-file /storage/Jarvis_AI_Assistant/chat_history.json
```

### Paperspace Persistent Storage

The setup script automatically creates directories in Paperspace's persistent storage:

```python
# Verify persistent storage directories
!ls -la /storage/Jarvis_AI_Assistant

# Copy models to persistent storage (if needed)
!cp -r models/ /storage/Jarvis_AI_Assistant/models/

# Working with symlinked storage
!ls -la /notebooks/Jarvis_AI_Assistant_storage  # This is a symlink to /storage/Jarvis_AI_Assistant
```

## Troubleshooting

### Fix CUDA and bitsandbytes Issues

```python
# Run the fixer script
!python fix_bitsandbytes.py

# If that doesn't work, try reinstalling bitsandbytes
!pip install --force-reinstall --no-deps https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-any.whl
```

### Check Memory Usage

```python
# Monitor GPU memory usage
!nvidia-smi

# For detailed memory analysis
import torch
print(f'GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB')
print(f'GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB')
print(f'Max memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB')
```

### Clear CUDA Cache

```python
# Clear CUDA cache if you're running out of memory
import torch
torch.cuda.empty_cache()
!nvidia-smi
```

### Common Command Line Arguments

- `--mode`: Choose between 'train', 'interactive', or 'generate'
- `--model`: The pre-trained model to use
- `--model-path`: Path to a fine-tuned model
- `--load-in-4bit`: Use 4-bit quantization (recommended for all NVIDIA GPUs)
- `--use-unsloth`: Enable Unsloth optimizations
- `--datasets`: Datasets to train on (e.g., pile, openassistant, gpteacher)
- `--batch-size`: Training batch size
- `--gradient-accumulation-steps`: Number of steps to accumulate gradients (higher for RTX GPUs)
- `--sequence-length`: Maximum sequence length (shorter for RTX GPUs)
- `--max-samples`: Limit number of training samples
- `--max-length`: Maximum length of generated text
- `--memory-file`: Path to save/load conversation memory
- `--history`: Path to previous chat history
- `--output`: Path to save chat history
