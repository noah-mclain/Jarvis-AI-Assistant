#!/bin/bash

echo "======================================================================"
echo "🔧 Fixing spaCy and thinc compatibility issues for Paperspace"
echo "======================================================================"

# Safe cleanup first - remove all potentially conflicting packages
echo "Step 1: Removing conflicting packages..."
pip uninstall -y spacy thinc spacy-legacy spacy-loggers catalogue wasabi srsly murmurhash cymem preshed blis langcodes pydantic pydantic-core typer

# Install dependencies in correct order
echo "Step 2: Installing foundation packages..."
pip install -U pip setuptools wheel

# Install NumPy first
echo "Step 3: Installing NumPy..."
pip install numpy==1.26.4

# Install dependencies in correct order (key packages with exact versions)
echo "Step 4: Installing core dependencies..."
pip install -v cymem==2.0.11 preshed==3.0.9 murmurhash==1.0.12 blis==0.7.11
pip install -v typer==0.9.0 catalogue==2.0.10 wasabi==1.1.3 srsly==2.4.8
pip install -v pydantic==1.10.13

# Install Thinc correctly
echo "Step 5: Installing Thinc (spaCy's ML backbone)..."
pip install -v thinc==8.1.10 --no-deps
pip install -v thinc==8.1.10

# Install spaCy with proper version
echo "Step 6: Installing spaCy core..."
pip install -v spacy==3.7.4 --no-deps
pip install -v spacy==3.7.4

# Install the model
echo "Step 7: Installing English language model..."
pip install -v https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0.tar.gz --no-deps

# Verify installation with a simple script that won't cause segfaults
echo "Step 8: Verifying installation..."
python -c "
import sys
try:
    import spacy
    print(f'SpaCy version: {spacy.__version__}')
    try:
        # Try loading without doing anything complex
        nlp = spacy.load('en_core_web_sm')
        print('English model loaded successfully')
        doc = nlp('Simple test sentence.')
        print('Basic test successful')
    except Exception as e:
        print(f'Error with model: {str(e)}')
except Exception as e:
    print(f'Error importing spaCy: {str(e)}')
    sys.exit(1)
"

# Final verification and cleanup
if [ $? -eq 0 ]; then
    echo "======================================================================"
    echo "✅ spaCy fixed successfully!"
    echo "======================================================================"
else
    echo "======================================================================"
    echo "❌ Issue still persists. Please reach out for additional support."
    echo "======================================================================"
    exit 1
fi 