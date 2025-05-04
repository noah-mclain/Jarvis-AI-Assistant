#!/bin/bash

echo "======================================================================"
echo "🔧 Complete Paperspace Setup for Jarvis AI Training"
echo "======================================================================"

# Function to check if a command succeeded
check_success() {
  if [ $? -ne 0 ]; then
    echo "❌ Error: $1 failed"
    echo "Continuing with setup process..."
  else
    echo "✅ $1 successful"
  fi
}

# Change to the notebooks directory
cd /notebooks

# Step 1: Fix spaCy installation
echo ""
echo "Step 1: Setting up spaCy with minimal tokenizer..."
echo ""

# Run the spaCy fix script if it exists
if [ -f "/notebooks/setup/fix_spacy_for_paperspace.sh" ]; then
    bash /notebooks/setup/fix_spacy_for_paperspace.sh
    check_success "spaCy installation"
else
    # Create a minimal installation
    echo "Setup script not found, running minimal spaCy setup..."
    
    # Uninstall potentially conflicting packages
    pip uninstall -y spacy thinc spacy-legacy spacy-loggers
    
    # Install minimal dependencies
    pip install spacy==3.7.4 --no-deps
    pip install pydantic==1.10.13 --no-deps
    pip install thinc==8.1.10 --no-deps
    
    # Install the model
    pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0.tar.gz --no-deps
    
    check_success "Manual spaCy installation"
fi

# Step 2: Fix train_models.py
echo ""
echo "Step 2: Fixing train_models.py..."
echo ""

# Create a backup of the original file
cp -f /notebooks/src/generative_ai_module/train_models.py /notebooks/src/generative_ai_module/train_models.py.bak
check_success "Backup creation"

# Fix the import issue by adding is_paperspace_environment - works with both new and existing import
if grep -q "from .utils import get_storage_path" /notebooks/src/generative_ai_module/train_models.py; then
    # Fix by adding to the existing import line
    sed -i 's/from .utils import \(.*\)/from .utils import \1, is_paperspace_environment/' /notebooks/src/generative_ai_module/train_models.py
    check_success "Import fix (existing import)"
else
    # Add a new import if it doesn't exist yet
    sed -i '1i from .utils import is_paperspace_environment' /notebooks/src/generative_ai_module/train_models.py
    check_success "Import fix (new import)"
fi

# Step 3: Create a minimal test to verify everything is working
echo ""
echo "Step 3: Testing setup..."
echo ""

# Create a test script
cat > /notebooks/test_setup.py << 'EOF'
#!/usr/bin/env python3
"""
Test Jarvis AI training setup in Paperspace
"""

import sys
import os

# Add notebooks to path
sys.path.insert(0, '/notebooks')

def test_imports():
    """Test all required imports"""
    
    try:
        # Test minimal tokenizer
        from src.generative_ai_module.minimal_spacy_tokenizer import tokenize
        print("✅ Minimal tokenizer import successful")
        
        # Test utils
        from src.generative_ai_module.utils import is_paperspace_environment
        print(f"✅ Utils import successful (Paperspace: {is_paperspace_environment()})")
        
        # Test train_models
        from src.generative_ai_module.train_models import main
        print("✅ train_models import successful")
        
        return True
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_tokenization():
    """Test tokenization functionality"""
    
    try:
        from src.generative_ai_module.minimal_spacy_tokenizer import tokenize
        
        # Test with sample text
        sample = "Testing the Jarvis AI Assistant tokenizer in Paperspace environment!"
        tokens = tokenize(sample)
        
        print(f"✅ Tokenization successful: {tokens}")
        return True
    except Exception as e:
        print(f"❌ Tokenization test failed: {e}")
        return False

def main():
    """Run all tests"""
    
    print("=" * 70)
    print("JARVIS AI TRAINING SETUP TEST")
    print("=" * 70)
    
    imports_ok = test_imports()
    tokenization_ok = test_tokenization()
    
    print("\nSummary:")
    print(f"Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"Tokenization: {'✅ PASS' if tokenization_ok else '❌ FAIL'}")
    
    if imports_ok and tokenization_ok:
        print("\n✅ Setup is complete! You can now run training.")
        return 0
    else:
        print("\n❌ Setup has issues that need to be fixed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

python /notebooks/test_setup.py
check_success "Setup test"

# Step 4: Set up environment for training
echo ""
echo "Step 4: Setting up training environment..."
echo ""

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export CODE_SUBSET="jarvis_code_instructions"

# Add to bashrc for persistence across sessions
echo "export CUDA_VISIBLE_DEVICES=0" >> ~/.bashrc
echo "export CODE_SUBSET=\"jarvis_code_instructions\"" >> ~/.bashrc

check_success "Environment setup"

# Final instructions
echo ""
echo "======================================================================"
echo "✅ Jarvis AI Assistant is ready for training!"
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
echo "If you encounter any issues, try running:"
echo "python /notebooks/test_setup.py"
echo ""
echo "======================================================================" 