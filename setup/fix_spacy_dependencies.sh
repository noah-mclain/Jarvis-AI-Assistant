#!/bin/bash

echo "======================================================================"
echo "🔧 Fixing spaCy dependencies compatibility issues"
echo "======================================================================"

# Uninstall current versions
echo "Uninstalling current incompatible versions..."
pip uninstall -y spacy thinc catalogue wasabi srsly

# Install correct versions in correct order
echo "Installing compatible thinc version first..."
pip install thinc==8.1.10

echo "Installing spaCy with compatible dependencies..."
pip install spacy==3.7.4

# Install the small English model for basic functionality
echo "Installing en_core_web_sm model..."
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0-py3-none-any.whl

echo "======================================================================"
echo "✅ SpaCy dependencies fixed!"
echo "======================================================================" 