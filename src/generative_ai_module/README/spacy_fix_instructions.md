# Fixing spaCy Installation Issues

This guide explains how to fix spaCy installation issues in different environments, particularly Paperspace which may experience segmentation faults.

## Quick Solutions

### For Paperspace Environments

If you experience a segmentation fault or `ParametricAttention_v2` error:

```bash
# Option 1: Run our Python fix script (recommended)
cd /notebooks  # or your project root
python src/generative_ai_module/fix_spacy_paperspace.py

# Option 2: Run the bash script
bash setup/fix_spacy_paperspace.sh
```

### For Regular Environments

For non-Paperspace environments:

```bash
# Option 1: Run the setup script
python setup/setup_spacy.py

# Option 2: Install directly
pip install spacy==3.7.4
python -m spacy download en_core_web_sm
```

## Testing Your spaCy Installation

After installation, verify everything works:

```bash
# Run the minimal test (safest option for Paperspace)
python test_spacy_minimal.py

# Or use the standard test
python test_spacy.py
```

## Common Issues and Solutions

### Missing `ParametricAttention_v2` Error

This error occurs due to a mismatch between spaCy and thinc versions:

```
cannot import name 'ParametricAttention_v2' from 'thinc.api'
```

**Solution**: The fix scripts will install spaCy 3.7.4 with thinc 8.1.10, which work together correctly.

### Segmentation Fault

If you get a segmentation fault (core dumped):

1. It's likely happening in complex spaCy components
2. The `fix_spacy_paperspace.py` script avoids this by using separate processes for testing
3. Use the minimal test which only uses the tokenizer component

### Model Not Found

If you get an error about the model not found:

```
[E050] Can't find model 'en_core_web_sm'
```

**Solution**: Install the model with:

```bash
python -m spacy download en_core_web_sm
```

Or directly from the URL:

```bash
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0.tar.gz
```

## Understanding the Fix

The fix works by:

1. Uninstalling all conflicting packages
2. Installing dependencies in the correct order
3. Using compatible versions (spaCy 3.7.4 + thinc 8.1.10)
4. Testing installation with basic functionality first

If you continue to experience issues, try running your code with minimal spaCy functionality or use the fallback mechanisms already built into the Jarvis AI codebase.
