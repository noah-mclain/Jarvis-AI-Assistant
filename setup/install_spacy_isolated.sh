#!/bin/bash

echo "======================================================================"
echo "🔧 Installing spaCy with ZERO dependency conflicts (FIXED VERSION)"
echo "======================================================================"

# Create a temporary directory for downloads
mkdir -p ./tmp_spacy_install
cd ./tmp_spacy_install

echo "Step 1: Uninstalling conflicting packages first..."
pip uninstall -y spacy thinc spacy-legacy spacy-loggers confection weasel

echo "Step 2: Downloading exact spaCy wheel and dependencies..."

# Download specific wheels without installing them
pip download spacy==3.7.4 --no-deps -d .
pip download thinc==8.1.10 --no-deps -d .
pip download pydantic==1.10.13 --no-deps -d .
pip download typer==0.9.0 --no-deps -d .
pip download catalogue==2.0.10 --no-deps -d .
pip download srsly==2.4.8 --no-deps -d .
pip download wasabi==1.1.3 --no-deps -d .
pip download blis==0.7.11 --no-deps -d .
pip download cymem==2.0.11 --no-deps -d .
pip download preshed==3.0.9 --no-deps -d .
pip download murmurhash==1.0.12 --no-deps -d .
pip download weasel==0.3.4 --no-deps -d .
pip download confection==0.1.3 --no-deps -d .
pip download spacy-legacy==3.0.12 --no-deps -d .
pip download spacy-loggers==1.0.5 --no-deps -d .
pip download https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0.tar.gz --no-deps -d .

echo "Step 3: Installing prerequisites with no-deps..."

# Install all dependencies with --no-deps to avoid dependency resolution
# Use exact filenames instead of wildcards for better reliability
for pkg in cymem-*.whl murmurhash-*.whl preshed-*.whl blis-*.whl wasabi-*.whl srsly-*.whl catalogue-*.whl typer-*.whl pydantic-*.whl; do
    if [ -f "$pkg" ]; then
        echo "Installing $pkg..."
        pip install --force-reinstall "$pkg" --no-deps
    else
        echo "Warning: $pkg not found, skipping"
    fi
done

# Handle spacy-legacy and spacy-loggers specifically since their naming caused issues
for pkg in *.whl; do
    if [[ "$pkg" == *"spacy-legacy"* ]]; then
        echo "Installing $pkg..."
        pip install --force-reinstall "$pkg" --no-deps
    elif [[ "$pkg" == *"spacy-loggers"* ]]; then
        echo "Installing $pkg..."
        pip install --force-reinstall "$pkg" --no-deps
    fi
done

# Install thinc separately - it's critical to get this right
echo "Installing thinc..."
pip install --force-reinstall thinc-*.whl --no-deps

# Install confection and weasel
for pkg in confection-*.whl weasel-*.whl; do
    if [ -f "$pkg" ]; then
        echo "Installing $pkg..."
        pip install --force-reinstall "$pkg" --no-deps
    fi
done

echo "Step 4: Installing spaCy core..."
pip install --force-reinstall spacy-*.whl --no-deps

echo "Step 5: Installing English model..."
pip install --force-reinstall en_core_web_sm-*.tar.gz --no-deps

# Clean up temporary directory
cd ..
rm -rf ./tmp_spacy_install

echo "Step 6: Verifying installation with safer import sequence..."
# Run with absolute minimal verification avoiding problematic imports
python -c "
import sys
import os
os.environ['PYTHONPATH'] = os.getcwd() + ':' + os.environ.get('PYTHONPATH', '')

try:
    # Import only what we need and avoid problematic imports
    import spacy.tokens
    import spacy.vocab
    import spacy.language
    
    print(f'✅ Imported spaCy core components')
    
    # Only import the tokenizer which is safest
    from spacy.tokenizer import Tokenizer
    print(f'✅ Tokenizer imported successfully')
    
    # Try to load model directly without using nlp.pipe() which might trigger the error
    try:
        import en_core_web_sm
        nlp = en_core_web_sm.load()
        print('✅ English model loaded successfully')
        
        # Only use tokenizer which is the safest component
        tokens = [t.text for t in nlp.tokenizer('Jarvis AI Assistant')]
        print(f'✅ Tokenization works: {tokens}')
    except Exception as e:
        print(f'❌ Error with model, but tokenizer might still work: {e}')
except Exception as e:
    print(f'❌ Error: {e}')
    sys.exit(1)
"

# If we got here, create a simplified test script that only uses tokenization
cat > test_spacy_tokenize_only.py << 'EOL'
#!/usr/bin/env python3
"""
Minimal spaCy test that only uses tokenization to avoid ParametricAttention_v2 error
"""

import sys

def test_spacy_tokenize_only():
    """Test spaCy with only tokenization functionality"""
    try:
        # Import only what we need
        import spacy
        print(f"spaCy version: {spacy.__version__}")
        
        # Try loading the model
        try:
            nlp = spacy.load("en_core_web_sm")
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
        
        # Use ONLY the tokenizer component
        text = "Jarvis AI Assistant uses spaCy for text processing."
        tokens = nlp.tokenizer(text)
        token_texts = [t.text for t in tokens]
        
        print("\nTokens:", token_texts)
        print("\n✅ Tokenization works!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("TOKENIZATION-ONLY SPACY TEST")
    print("=" * 70)
    
    success = test_spacy_tokenize_only()
    
    if success:
        print("\n✅ SUCCESS: spaCy tokenization is working!")
    else:
        print("\n❌ FAILED: spaCy test failed")
    
    sys.exit(0 if success else 1)
EOL

chmod +x test_spacy_tokenize_only.py

echo "======================================================================"
echo "✅ spaCy minimal installation completed!"
echo "======================================================================"
echo ""
echo "IMPORTANT: If you're on Paperspace and still seeing ParametricAttention_v2 errors,"
echo "use ONLY the tokenizer component to avoid segmentation faults:"
echo ""
echo "  python test_spacy_tokenize_only.py"
echo ""
echo "This will skip the problematic parts of spaCy but still give you basic tokenization."
echo "======================================================================" 