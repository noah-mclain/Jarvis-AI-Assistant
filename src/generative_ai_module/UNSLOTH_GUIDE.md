# Unsloth for DeepSeek Optimization

This guide explains how Unsloth is used to optimize DeepSeek-Coder fine-tuning in the Jarvis AI Assistant.

## What is Unsloth?

[Unsloth](https://github.com/unslothai/unsloth) is an optimization library that makes fine-tuning large language models:

- 2x faster training
- Up to 80% less memory usage
- Support for much longer context lengths
- Compatible with different hardware (CUDA, MPS, CPU)

## How to Use

### Command Line Usage

When using the fine-tuning script from the command line, Unsloth optimization is automatically applied:

```bash
python src/generative_ai_module/run_finetune.py
```

Important flags to customize your fine-tuning:

```bash
# For NVIDIA GPUs, 4-bit quantization is used by default
# For Apple Silicon (M1/M2/M3), 8-bit quantization is used automatically

# Specify custom sequence length (possible with Unsloth memory savings)
python src/generative_ai_module/run_finetune.py --sequence-length 4096

# Customize batch size
python src/generative_ai_module/run_finetune.py --batch-size 16

# Use a specific language subset
python src/generative_ai_module/run_finetune.py --subset java --all-subsets False
```

### Interactive Mode

When running without arguments, the script will use Unsloth optimization with hardware-appropriate defaults:

```bash
python src/generative_ai_module/run_finetune.py
```

You'll be prompted about whether to train both DeepSeek and text models:

```
Do you want to train both DeepSeek code model and text models? (y/n):
```

## Benefits

### 1. Memory Efficiency

Unsloth uses optimized kernels and quantization techniques to reduce memory usage:

- **NVIDIA GPUs**: Can fit models up to 70% larger than standard methods
- **Apple Silicon**: Can train larger models on M1/M2/M3 Macs
- **CPU**: Even enables training on CPU when GPU is unavailable

### 2. Faster Training

Unsloth provides significant speedups:

- Up to 2x faster training on most hardware
- More efficient gradient computation
- Optimized LoRA implementation

### 3. Longer Context

Unsloth enables working with significantly longer sequences:

- Standard fine-tuning might be limited to 512-1024 tokens
- With Unsloth, you can potentially use 2048, 4096, or even longer sequences

## Architecture Support

The current implementation supports:

- DeepSeek-Coder models (6.7B, etc.)
- QLoRA fine-tuning with 4-bit and 8-bit quantization
- Hardware detection for NVIDIA GPUs, Apple Silicon, and CPU
- Automatic handling of test, validation and training splits

## Implementation Details

Unsloth is now the default optimization method used through these components:

1. `unsloth_deepseek.py` - Core implementation with improved error handling and dataset preparation
2. `run_finetune.py` - Entry point with hardware-specific optimizations
3. `code_preprocessing.py` - Enhanced dataset preparation with proper train/validation/test splits

## Dataset Handling Improvements

The implementation includes robust dataset handling:

1. **Proper Splitting**: Data is split into train, validation, and test sets consistently
2. **Data Cleaning**: Empty or malformed examples are automatically removed
3. **Error Recovery**: Graceful handling of errors during training
4. **Format Conversion**: Automatically converts tokenized data for Unsloth compatibility

## Troubleshooting

If you encounter issues:

1. **Out of Memory**: Try reducing batch size (`--batch-size 4`) or sequence length (`--sequence-length 1024`)
2. **Slow Training**: Ensure your datasets aren't too large (`--max-samples 1000`)
3. **Apple Silicon Issues**: The code automatically uses 8-bit mode for Apple Silicon

## Advanced Configuration

For advanced users, you can directly use the Unsloth APIs:

```python
from src.generative_ai_module.unsloth_deepseek import get_unsloth_model, finetune_with_unsloth

# Load optimized model
model, tokenizer = get_unsloth_model(
    model_name="deepseek-ai/deepseek-coder-6.7b-base",
    max_seq_length=2048,
    load_in_4bit=True
)

# Custom fine-tuning
finetune_with_unsloth(
    train_dataset=your_dataset,
    eval_dataset=your_eval_dataset,
    max_steps=100,
    per_device_train_batch_size=4
)
```

## Hardware-Specific Optimizations

The system automatically applies hardware-specific optimizations:

1. **NVIDIA GPUs**: Uses 4-bit quantization, larger batch sizes, longer sequences
2. **Apple Silicon**: Uses 8-bit quantization, smaller batch sizes, reduced samples
3. **CPU**: Falls back to 8-bit with minimal batch sizes for compatibility

## References

- [Unsloth GitHub Repository](https://github.com/unslothai/unsloth)
- [Unsloth Documentation](https://unsloth.ai/)
