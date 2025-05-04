#!/bin/bash

echo "===================================================================="
echo "Installing missing spaCy dependencies without conflicts"
echo "===================================================================="

# Install core thinc package (required by spaCy)
pip install thinc==8.2.1 --no-deps

# Install minimal dependencies for thinc
pip install blis==0.7.11 --no-deps
pip install murmurhash==1.0.10 --no-deps
pip install cymem==2.0.8 --no-deps
pip install preshed==3.0.9 --no-deps

# Install other missing spaCy dependencies
pip install langcodes==3.3.0 --no-deps
pip install pydantic==1.10.13 --no-deps  # Important: use 1.10.x not 2.x
pip install smart-open==6.4.0 --no-deps

# Install compatibility shim for different pydantic versions
pip install pydantic-core==2.14.5 --no-deps

# Verify installation
python -c "
try:
    import spacy
    print(f'✅ spaCy version {spacy.__version__} installed')
    try:
        nlp = spacy.load('en_core_web_sm')
        print('✅ English model loaded successfully')
        doc = nlp('This is a test sentence.')
        print('✅ Basic processing works')
        print('✅ Parts of speech:', [(token.text, token.pos_) for token in doc])
    except Exception as e:
        print(f'❌ Error loading model: {e}')
except Exception as e:
    print(f'❌ Error importing spaCy: {e}')
    import traceback
    traceback.print_exc()
"

echo "===================================================================="
echo "spaCy dependency installation complete!"
echo "===================================================================="
