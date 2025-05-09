# Jarvis AI Assistant Setup Scripts

This directory contains the consolidated setup scripts for Jarvis AI Assistant. The scripts have been organized and consolidated to reduce duplication and make the setup process more streamlined.

## Main Scripts

### 1. `consolidated_unified_setup.sh`

This is the main setup script that handles the entire setup process. It:

- Detects the environment (Colab, Paperspace, or standard)
- Sets up the Python environment
- Cleans up any existing installations
- Installs all dependencies
- Configures GPU optimizations
- Sets up minimal Unsloth implementation
- Sets up spaCy with minimal tokenizer
- Applies attention mask fixes for DeepSeek models
- Creates import fixes

Usage:

```bash
./setup/consolidated_unified_setup.sh
```

### 2. `consolidated_install_dependencies.sh`

This script handles the installation of all dependencies with specific compatible versions. It:

- Installs NumPy 1.26.4 (foundation package)
- Installs PyTorch 2.1.2 with CUDA 12.1
- Installs core scientific packages
- Installs Hugging Face ecosystem
- Installs optimization libraries
- Installs xFormers with enhanced attention support
- Installs additional dependencies for enhanced attention mechanisms
- Installs Unsloth 2024.8
- Installs Flash Attention 2.5.5

Usage:

```bash
./setup/consolidated_install_dependencies.sh
```

### 3. `consolidated_fix_unsloth.sh`

This script sets up a minimal Unsloth implementation that works without dependency conflicts. It:

- Creates a custom minimal Unsloth implementation
- Applies the fixed Unsloth to Python files
- Adds the custom Unsloth to PYTHONPATH
- Verifies the minimal Unsloth functionality

Usage:

```bash
./setup/consolidated_fix_unsloth.sh
```

### 4. `consolidated_fix_spacy.sh`

This script sets up spaCy with a minimal tokenizer that works in Paperspace environments. It:

- Detects if running in Paperspace
- Creates a minimal spaCy tokenizer
- Installs spaCy with specific compatible versions
- Tests the installation and minimal tokenizer

Usage:

```bash
./setup/consolidated_fix_spacy.sh
```

### 5. `train_jarvis.sh`

This is the main training script for Jarvis AI Assistant. It:

- Activates the Python environment and Unsloth
- Parses command line arguments
- Sets up the environment and creates directories
- Checks for required Python packages
- Clears CUDA cache
- Sets environment variables for optimal memory usage
- Applies attention mask fix for DeepSeek models
- Runs the training process

Usage:

```bash
./setup/train_jarvis.sh --gpu-type A6000 --vram 50 --model-type code
```

## Fix Scripts

### Attention Mask Fix Scripts

The following scripts are used to fix attention mask issues in DeepSeek models:

- `fix_transformers_attention_mask.py`: General attention mask fixes
- `fix_attention_mask_params.py`: Parameter-specific attention mask fixes
- `fix_tensor_size_mismatch.py`: Tensor size mismatch fixes
- `fix_attention_dimension_mismatch.py`: Attention dimension mismatch fixes
- `fix_tuple_unpacking_error.py`: Fixes for "too many values to unpack" error
- `comprehensive_attention_mask_fix.py`: Comprehensive attention mask fix
- `fix_all_attention_issues.py`: All-in-one fix script
- `ultimate_attention_fix.py`: Ultimate fix for all attention-related issues

### Other Fix Scripts

- `fix_transformers_utils.py`: Fixes missing transformers.utils module
- `fix_deepseek_model.py`: Fixes issues with DeepSeek model in transformers
- `fix_bitsandbytes_version.py`: Fixes bitsandbytes version issues for 4-bit quantization

#### bitsandbytes Version Fix

The `fix_bitsandbytes_version.py` script addresses an issue with 4-bit quantization in older versions of bitsandbytes.

When using 4-bit quantization with older versions of bitsandbytes (< 0.42.0), you may encounter the following error:

```bash
Calling `to()` is not supported for `4-bit` quantized models with the installed version of bitsandbytes. The current device is `cuda:0`. If you intended to move the model, please install bitsandbytes >= 0.43.2.
```

The script:

1. Checks the current bitsandbytes version
2. Upgrades to a compatible version (0.42.0 or newer) if needed
3. Adds a `__version__` attribute if it's missing

You can run the script directly:

```bash
python setup/fix_bitsandbytes_version.py
```

Or it will be automatically run by the training script if needed.

These scripts are called by the main setup script and don't need to be run individually in most cases.

## Setup Process

1. Run the main setup script:

   ```bash
   ./setup/consolidated_unified_setup.sh
   ```

2. Train a model:
   ```bash
   ./setup/train_jarvis.sh --model-type code
   ```

## Troubleshooting

If you encounter issues:

1. Check the logs for error messages
2. Try running the individual fix scripts:
   ```bash
   ./setup/consolidated_install_dependencies.sh
   ./setup/consolidated_fix_unsloth.sh
   ./setup/consolidated_fix_spacy.sh
   ```
3. If you're having issues with spaCy, use the minimal tokenizer:
   ```python
   from src.generative_ai_module.minimal_spacy_tokenizer import tokenize
   tokens = tokenize("Your text here")
   ```
4. If you're having issues with Unsloth, make sure the custom Unsloth is in your PYTHONPATH:
   ```bash
   source /notebooks/custom_unsloth/activate_minimal_unsloth.sh
   ```
