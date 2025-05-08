# Jarvis AI Assistant Training Guide

This guide explains how to train the Jarvis AI Assistant models, focusing on DeepSeek Coder training.

## Repository Organization

The repository is organized as follows:

- **Main Scripts**:
  - `train_jarvis.sh`: Main training script that supports different model types
  - `fix_deepseek_training.py`: Comprehensive fix for DeepSeek training issues
  - `fix_transformer_issues.py`: Fix for transformer model issues
  - `fix_attention_mask.py`: Fix for attention mask issues
  - `gpu_utils.py`: GPU monitoring and management utilities

- **Directories**:
  - `src/`: Source code for the Jarvis AI Assistant
  - `setup/`: Setup scripts for different environments
  - `tests/`: Test scripts
  - `Jarvis_AI_Assistant/`: Output directory for models and checkpoints

## Training Models

### DeepSeek Coder Training

To train a DeepSeek Coder model:

```bash
./train_jarvis.sh --model-type code-unified
```

This will:
1. Create a `unified_deepseek_training.py` script with all necessary fixes
2. Run the training with optimized parameters for your GPU

### Text Model Training

To train a text generation model:

```bash
./train_jarvis.sh --model-type text
```

### CNN-Enhanced Text Model Training

To train a CNN-enhanced text model:

```bash
./train_jarvis.sh --model-type cnn-text
```

## Fixing Issues

If you encounter issues during training, you can run the comprehensive fix script:

```bash
./fix_deepseek_training.py
```

This script fixes:
1. Unsloth parameter issues:
   - max_seq_length parameter issue
   - device_map parameter issue
   - use_gradient_checkpointing parameter issue
   - random_state parameter issue
2. Dataset processing issues:
   - Tokenization issues
   - Data collation issues
   - Tensor creation issues
3. Attention mask issues

## GPU Optimization

The training script automatically optimizes parameters based on your GPU:

- **A6000 (48+ GiB VRAM)**:
  - Batch size: 8
  - Sequence length: 2048
  - 8-bit quantization

- **A6000 (24-48 GiB VRAM)**:
  - Batch size: 4
  - Sequence length: 1024
  - 8-bit quantization

- **A6000 (<24 GiB VRAM), A4000, RTX5000**:
  - Batch size: 1-2
  - Sequence length: 512
  - 4-bit quantization

You can override these settings with command-line arguments:

```bash
./train_jarvis.sh --model-type code-unified --gpu-type A6000 --vram 50
```

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM) Errors**:
   - Reduce batch size: `--batch-size 1`
   - Use 4-bit quantization: `--load-in-4bit`
   - Reduce sequence length: `--max-length 512`
   - Increase gradient accumulation steps: `--gradient-accumulation-steps 32`

2. **Attention Mask Errors**:
   - Run `fix_attention_mask.py` to patch the LlamaModel.forward method

3. **Dataset Processing Errors**:
   - Run `fix_deepseek_training.py` to fix dataset processing issues

4. **Unsloth Compatibility Issues**:
   - Run `fix_deepseek_training.py` to fix Unsloth parameter issues

### GPU Monitoring

You can monitor GPU usage during training:

```bash
python gpu_utils.py monitor --interval 5 --log-file gpu_memory_log.txt
```

To clear GPU memory:

```bash
python gpu_utils.py clear
```

## Setup Scripts

The `setup/` directory contains scripts for setting up different environments:

- `setup.sh`: Main setup script
- `setup_paperspace.sh`: Setup for Paperspace environment
- `fix_rtx5000.sh`: Optimizations for RTX5000 GPU
- `fix_dependencies.sh`: Fix dependency issues

## Additional Resources

- `RTX5000_Jarvis_Guide.md`: Guide for training on RTX5000 GPU
- `RTX5000_TRAINING.md`: Training parameters for RTX5000 GPU
- `REFACTORING.md`: Information about code refactoring
