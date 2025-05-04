#!/bin/bash

echo "======================================================================"
echo "🔧 Complete SpaCy Fix for Jarvis AI Assistant"
echo "======================================================================"

# Function to check if a command succeeded
check_success() {
  if [ $? -ne 0 ]; then
    echo "❌ Error: $1 failed"
    echo "Continuing with installation..."
  else
    echo "✅ $1 successful"
  fi
}

# Create a clean environment for spaCy
echo "Removing existing spaCy-related packages..."
pip uninstall -y spacy thinc catalogue wasabi srsly weasel cloudpathlib murmurhash cymem preshed blis langcodes pydantic pydantic-core typer
check_success "Uninstallation"

# Install all dependencies in a specific order
echo "Installing spaCy ecosystem dependencies..."

# Core dependencies first in correct order
echo "1. Installing foundation dependencies..."
pip install -U pip setuptools wheel
check_success "Foundation packages"

# Install core numerical/scientific packages
echo "2. Installing numerical libraries..."
pip install numpy==1.26.4 --no-deps
pip install numpy==1.26.4
check_success "NumPy"

# Install thinc and its dependencies
echo "3. Installing thinc and its dependencies..."
pip install cymem==2.0.11 --no-deps 
pip install preshed==3.0.9 --no-deps
pip install murmurhash==1.0.12 --no-deps
pip install blis==0.7.11 --no-deps
pip install thinc==8.1.10 --no-deps
pip install thinc==8.1.10
check_success "Thinc core"

# Install pydantic and dependencies
echo "4. Installing pydantic (old version, for compatibility)..."
pip install typing-extensions==4.10.0 --no-deps
pip install pydantic==1.10.13 --no-deps
check_success "Pydantic"

# Install spaCy support libraries
echo "5. Installing spaCy supporting libraries..."
pip install catalogue==2.0.10 --no-deps
pip install wasabi==1.1.3 --no-deps
pip install srsly==2.4.8 --no-deps
pip install langcodes==3.3.0 --no-deps
pip install typer==0.9.0 --no-deps
pip install cloudpathlib==0.16.0 --no-deps
pip install weasel==0.3.4 --no-deps
check_success "SpaCy supporting libraries"

# Now install spaCy itself
echo "6. Installing spaCy core..."
pip install spacy==3.7.4 --no-deps
pip install spacy==3.7.4
check_success "SpaCy core"

# Install spaCy English model
echo "7. Installing English language model..."
python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0-py3-none-any.whl --no-deps
check_success "English model"

# Verify installation
echo "======================================================================"
echo "Verifying installation..."
echo "======================================================================"

python -c "
import sys
try:
    import spacy
    print(f'SpaCy version: {spacy.__version__}')
    try:
        # Try loading the model
        nlp = spacy.load('en_core_web_sm')
        print('English model loaded successfully')
        
        # Try basic processing
        doc = nlp('Jarvis AI Assistant is processing this text with spaCy.')
        print('Basic text processing completed')
        
        # Try POS tagging and entity recognition
        print('Sample POS tagging:')
        for token in list(doc)[:5]:
            print(f'  - {token.text}: {token.pos_}')
        
        print('\nSpaCy is working correctly with all dependencies!')
    except Exception as e:
        print(f'Error with model: {str(e)}')
except Exception as e:
    print(f'Error importing spaCy: {str(e)}')
    import traceback
    traceback.print_exc()
"

echo "======================================================================"
echo "SpaCy installation with all dependencies complete!"
echo "You can now run the Jarvis AI training pipeline with spaCy support."
echo "======================================================================" 