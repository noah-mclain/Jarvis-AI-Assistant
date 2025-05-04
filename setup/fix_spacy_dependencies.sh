#!/bin/bash

echo "======================================================================"
echo "🔧 Fixing spaCy dependencies compatibility issues"
echo "======================================================================"

# Uninstall current versions
echo "Uninstalling current incompatible versions..."
pip uninstall -y spacy thinc catalogue wasabi srsly weasel

# Install correct versions in correct order
echo "Installing compatible thinc version first..."
pip install thinc==8.1.10 --no-deps

# Install cloudpathlib dependency for weasel
echo "Installing cloudpathlib dependency..."
pip install cloudpathlib==0.16.0 --no-deps

echo "Installing spaCy with compatible dependencies..."
pip install spacy==3.7.4 --no-deps

# Install supporting libraries
echo "Installing additional dependencies..."
pip install wasabi==1.1.3 --no-deps
pip install srsly==2.4.8 --no-deps
pip install catalogue==2.0.10 --no-deps
pip install typer==0.9.0 --no-deps
pip install weasel==0.3.4 --no-deps

# Install the small English model for basic functionality
echo "Installing en_core_web_sm model..."
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0-py3-none-any.whl --no-deps

echo "======================================================================"
echo "✅ SpaCy dependencies fixed!"
echo "======================================================================" 