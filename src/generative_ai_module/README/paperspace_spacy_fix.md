# Paperspace-specific spaCy Fix for Jarvis AI Assistant

This document explains how to fix spaCy issues in Paperspace environments, particularly the `ParametricAttention_v2` error that can cause segmentation faults.

## Quick Solution

If you're experiencing spaCy errors in Paperspace, use one of these solutions:

### Solution 1: Run the isolated installation script (recommended)

```bash
bash setup/install_spacy_isolated.sh
```

### Solution 2: Run the Python fix script

```bash
python src/generative_ai_module/paperspace_spacy_fix.py
```

### Solution 3: Use the check_spacy.py with Paperspace flag

```bash
python src/generative_ai_module/check_spacy.py --fix --paperspace
```

## Understanding the Problem

In Paperspace environments, spaCy can encounter a critical error:

```
ImportError: cannot import name 'ParametricAttention_v2' from 'thinc.api'
```

This happens because:

1. Paperspace has specific CUDA/GPU configurations that may conflict with some components
2. There's a version mismatch between spaCy and its thinc dependency
3. Certain advanced neural network components in spaCy trigger segmentation faults

## How Our Fix Works

Our solution uses multiple approaches to resolve the issue:

### 1. Isolated Installation

The `install_spacy_isolated.sh` script:

- Downloads specific wheels for spaCy and all dependencies
- Installs each package with `--no-deps` to avoid dependency resolution conflicts
- Uses compatible versions (spaCy 3.7.4 + thinc 8.1.10)
- Cleans up afterward

### 2. Import Patching

The `paperspace_spacy_fix.py` script:

- Directly manipulates Python's import system
- Creates a dummy `thinc.api` module with a mock `ParametricAttention_v2` object
- Prevents segmentation faults by bypassing problematic imports

### 3. Tokenizer-Only Mode

The most reliable approach is to use only spaCy's tokenizer component:

- Avoids the neural network components that cause segmentation faults
- Still provides essential text tokenization functionality
- Works reliably across different Paperspace environments

## How to Use spaCy in Paperspace

After applying the fix, use spaCy with this pattern to avoid segmentation faults:

```python
import spacy

# Load the model
nlp = spacy.load("en_core_web_sm")

# Use ONLY the tokenizer component
text = "Your text here"
tokens = [t.text for t in nlp.tokenizer(text)]

# Work with tokens safely
print(tokens)
```

**IMPORTANT:** Avoid using these components in Paperspace as they may cause segmentation faults:

- `nlp.pipe()`
- Entity recognition (`doc.ents`)
- Dependency parsing
- POS tagging
- Neural network components

## Testing Your Fix

We've included several test scripts:

1. `test_spacy_tokenize_only.py` - Tests only the tokenizer (safest)
2. `test_spacy_minimal.py` - Basic test with minimal functionality
3. `test_spacy_paperspace.py` - Includes the import fix hack

## Automatic Fallback in Jarvis AI

The Jarvis AI codebase has been updated to detect Paperspace environments and automatically:

1. Apply the import fix
2. Switch to tokenizer-only mode
3. Use proper fallback mechanisms if spaCy is unavailable

This is handled in `prompt_enhancer.py` which includes Paperspace-specific detection and fallbacks.

## Troubleshooting

If you still experience issues:

1. Run `python src/generative_ai_module/check_spacy.py --paperspace --fix` for diagnostics
2. Try completely uninstalling spaCy: `pip uninstall -y spacy thinc`
3. Install with the isolated script: `bash setup/install_spacy_isolated.sh`
4. If all else fails, the codebase will fall back to basic tokenization without spaCy

## For Developers

When modifying code that uses spaCy:

1. Always check if the tokenizer is available before using it
2. Provide fallback mechanisms for when spaCy is unavailable
3. In Paperspace, only use the tokenizer component
4. Test your changes in both local and Paperspace environments
