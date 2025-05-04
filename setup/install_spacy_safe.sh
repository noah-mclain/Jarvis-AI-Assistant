#!/bin/bash

echo "===================================================================="
echo "Installing spaCy without dependency conflicts"
echo "===================================================================="

# Install spaCy with no dependencies first
pip install spacy==3.7.4 --no-deps

# Install minimal dependencies that spaCy needs but won't conflict
pip install wasabi==1.1.3 --no-deps
pip install srsly==2.4.8 --no-deps
pip install catalogue==2.0.10 --no-deps 
pip install typer==0.9.0 --no-deps
pip install weasel==0.3.4 --no-deps
pip install setuptools==65.5.0 --no-deps

# Install en_core_web_sm language model - special handling
# This avoids installing a different spaCy version
python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0.tar.gz --no-deps

# Verify installation
python -c "
import spacy
print(f'✅ spaCy version {spacy.__version__} installed')
try:
    nlp = spacy.load('en_core_web_sm')
    print('✅ English model loaded successfully')
    doc = nlp('This is a test sentence.')
    print('✅ Basic processing works')
except Exception as e:
    print(f'❌ Error loading model: {e}')
"

echo "===================================================================="
echo "spaCy installation complete!"
echo "===================================================================="
