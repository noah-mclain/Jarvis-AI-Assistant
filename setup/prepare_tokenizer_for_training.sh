#!/bin/bash

echo "======================================================================"
echo "🔧 Preparing Minimal Tokenizer for Training"
echo "======================================================================"

# Function to check if a command succeeded
check_success() {
  if [ $? -ne 0 ]; then
    echo "❌ Error: $1 failed"
    exit 1
  else
    echo "✅ $1 successful"
  fi
}

# Create a temporary test script
echo "Creating tokenizer test script..."
cat > /notebooks/test_minimal_tokenizer.py << 'EOF'
#!/usr/bin/env python3
"""
Simple test for the minimal spaCy tokenizer in Paperspace
"""

import sys
import os
import importlib.util

print("=================================================================")
print("MINIMAL SPACY TOKENIZER TEST FOR PAPERSPACE")
print("=================================================================")

# Add the notebooks directory to sys.path
if '/notebooks' not in sys.path:
    sys.path.insert(0, '/notebooks')

try:
    # Test direct import
    from src.generative_ai_module.minimal_spacy_tokenizer import tokenize, tokenizer
    
    print("\nTokenizer availability:", "✅ AVAILABLE" if tokenizer.is_available else "❌ UNAVAILABLE")
    
    # Test tokenization
    text = "This tokenizer will be used for Jarvis AI training in Paperspace!"
    tokens = tokenize(text)
    
    print(f"\nSample text: {text}")
    print(f"Tokenized result: {tokens}")
    print(f"Token count: {len(tokens)}")
    
    # Check for paperspace environment
    from src.generative_ai_module.utils import is_paperspace_environment
    print(f"\nIs Paperspace environment: {is_paperspace_environment()}")
    
    print("\n✅ The tokenizer is working correctly!")
    sys.exit(0)
except Exception as e:
    print(f"\n❌ Error: {e}")
    sys.exit(1)
EOF

chmod +x /notebooks/test_minimal_tokenizer.py
check_success "Creating test script"

# Run the tokenizer test
echo "Testing minimal tokenizer..."
cd /notebooks
python test_minimal_tokenizer.py
check_success "Tokenizer test"

# Ensure spaCy is properly installed
echo "Checking spaCy installation..."
python -c "
import sys
try:
    import spacy
    print(f'spaCy version: {spacy.__version__}')
    print('✅ spaCy is installed')
except ImportError:
    print('❌ spaCy is not installed')
    sys.exit(1)
except Exception as e:
    print(f'❌ Error importing spaCy: {e}')
    sys.exit(1)
"
check_success "spaCy check"

# Create a final test script that uses minimal_spacy_tokenizer in training context
echo "Testing tokenizer in training context..."
python -c "
import sys
sys.path.insert(0, '/notebooks')

try:
    # Import minimal tokenizer
    from src.generative_ai_module.minimal_spacy_tokenizer import tokenize
    print('✅ Imported minimal tokenizer')
    
    # Import basic training components
    from src.generative_ai_module.utils import is_paperspace_environment
    print(f'✅ Paperspace environment: {is_paperspace_environment()}')
    
    # Try a simple tokenization
    text = 'Testing the tokenizer for Jarvis AI training'
    tokens = tokenize(text)
    print(f'✅ Tokenized text: {tokens}')
    
    print('\\n✅ Tokenizer is ready for training!')
    sys.exit(0)
except Exception as e:
    print(f'❌ Error: {e}')
    sys.exit(1)
"
check_success "Training context test"

# Set up environment variables for training
echo "Setting up environment for training..."
export CUDA_VISIBLE_DEVICES=0
export CODE_SUBSET="jarvis_code_instructions"
echo "export CUDA_VISIBLE_DEVICES=0" >> ~/.bashrc
echo "export CODE_SUBSET=\"jarvis_code_instructions\"" >> ~/.bashrc
check_success "Environment setup"

echo ""
echo "======================================================================"
echo "✅ Minimal tokenizer is ready for training!"
echo ""
echo "Run your training with:"
echo ""
echo "cd /notebooks"
echo "python src/generative_ai_module/train_models.py \\"
echo "    --model-type code \\"
echo "    --use-deepseek \\"
echo "    --code-subset \$CODE_SUBSET \\"
echo "    --batch-size 4 \\"
echo "    --epochs 3 \\"
echo "    --learning-rate 2e-5 \\"
echo "    --warmup-steps 100 \\"
echo "    --load-in-4bit"
echo ""
echo "======================================================================" 