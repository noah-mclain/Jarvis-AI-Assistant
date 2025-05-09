#!/bin/bash

echo "======================================================================"
echo "🔧 Jarvis AI Assistant - Running All spaCy Fixes"
echo "======================================================================"

# Run both fix scripts
echo "Running fix_spacy_for_paperspace.sh..."
chmod +x setup/fix_spacy_for_paperspace.sh
./setup/fix_spacy_for_paperspace.sh

echo "Running consolidated_fix_spacy.sh..."
chmod +x setup/consolidated_fix_spacy.sh
./setup/consolidated_fix_spacy.sh

# Create a Python script to test if the fixes worked
cat > setup/verify_spacy_fix.py << 'EOF'
#!/usr/bin/env python3
"""
Verify that the spaCy fixes worked
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("verify_spacy_fix")

def verify_fix():
    """Verify that the spaCy fixes worked"""
    try:
        # Add the project root to sys.path if needed
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Try to import the minimal_spacy module
        try:
            from src.generative_ai_module.minimal_spacy import tokenize
            logger.info("✅ Successfully imported minimal_spacy module")
            
            # Test tokenization
            test_text = "Jarvis AI Assistant is verifying the spaCy fix!"
            tokens = tokenize(test_text)
            
            logger.info(f"Tokenized result: {tokens}")
            logger.info(f"Token count: {len(tokens)}")
            
            # Try to import spaCy directly
            try:
                import spacy
                logger.info(f"✅ Successfully imported spaCy {spacy.__version__}")
                
                # Try to create a blank model
                try:
                    nlp = spacy.blank("en")
                    logger.info("✅ Successfully created blank model")
                    
                    # Test tokenization with spaCy
                    doc = nlp(test_text)
                    spacy_tokens = [token.text for token in doc]
                    
                    logger.info(f"spaCy tokenization: {spacy_tokens}")
                    logger.info(f"spaCy token count: {len(spacy_tokens)}")
                    
                    return True
                except Exception as e:
                    logger.error(f"❌ Error creating blank model: {e}")
                    return False
            except Exception as e:
                logger.error(f"❌ Error importing spaCy: {e}")
                return False
        except Exception as e:
            logger.error(f"❌ Error importing minimal_spacy: {e}")
            return False
    except Exception as e:
        logger.error(f"❌ Error verifying fix: {e}")
        return False

if __name__ == "__main__":
    success = verify_fix()
    sys.exit(0 if success else 1)
EOF

# Make the verification script executable
chmod +x setup/verify_spacy_fix.py

# Run the verification script
echo "Verifying that the spaCy fixes worked..."
python setup/verify_spacy_fix.py

echo "======================================================================"
echo "✅ All spaCy fixes complete!"
echo ""
echo "To use the minimal spaCy tokenizer in your code:"
echo "from src.generative_ai_module.minimal_spacy import tokenize"
echo ""
echo "Example:"
echo "tokens = tokenize('Your text here')"
echo "======================================================================"
