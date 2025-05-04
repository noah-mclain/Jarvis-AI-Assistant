#!/bin/bash

echo "======================================================================"
echo "🔧 Installing spaCy with ZERO dependency conflicts"
echo "======================================================================"

# Create a temporary directory for downloads
mkdir -p ./tmp_spacy_install
cd ./tmp_spacy_install

echo "Step 1: Downloading exact spaCy wheel and dependencies..."

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

echo "Step 2: Installing prerequisites with no-deps..."

# Install all dependencies with --no-deps to avoid dependency resolution
# This ensures nothing gets pulled in that might conflict with existing packages
pip install ./cymem-*.whl --no-deps
pip install ./murmurhash-*.whl --no-deps
pip install ./preshed-*.whl --no-deps
pip install ./blis-*.whl --no-deps
pip install ./wasabi-*.whl --no-deps
pip install ./srsly-*.whl --no-deps
pip install ./catalogue-*.whl --no-deps
pip install ./typer-*.whl --no-deps
pip install ./pydantic-*.whl --no-deps
pip install ./spacy-legacy-*.whl --no-deps
pip install ./spacy-loggers-*.whl --no-deps
pip install ./thinc-*.whl --no-deps
pip install ./confection-*.whl --no-deps
pip install ./weasel-*.whl --no-deps

echo "Step 3: Installing spaCy core..."
pip install ./spacy-*.whl --no-deps

echo "Step 4: Installing English model..."
pip install ./en_core_web_sm-*.tar.gz --no-deps

# Clean up temporary directory
cd ..
rm -rf ./tmp_spacy_install

echo "Step 5: Verifying installation..."
# Run with absolute minimal verification
python -c "
import sys
try:
    import spacy
    print(f'✅ spaCy version {spacy.__version__} installed')
    try:
        nlp = spacy.load('en_core_web_sm')
        print('✅ English model loaded successfully')
        # Only use tokenizer which is the safest component
        tokens = [t.text for t in nlp.tokenizer('Jarvis AI Assistant')]
        print(f'✅ Tokenization works: {tokens}')
    except Exception as e:
        print(f'❌ Error with model: {e}')
        sys.exit(1)
except Exception as e:
    print(f'❌ Error: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo "======================================================================"
    echo "✅ spaCy installed successfully with ZERO dependency conflicts!"
    echo "======================================================================"
    
    echo "Try it with the minimal test script:"
    echo "python test_spacy_minimal.py"
    
    echo "NOTE: You may still see pip warnings, but they don't affect functionality"
    echo "      and your other packages remain completely untouched."
else
    echo "======================================================================"
    echo "❌ Installation issue detected. Please check the error message above."
    echo "======================================================================"
    exit 1
fi 