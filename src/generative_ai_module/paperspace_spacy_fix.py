#!/usr/bin/env python3
"""
Paperspace-specific spaCy Fix - Bypasses ParametricAttention_v2 Error

This script:
1. Creates a custom import path for spaCy that avoids problematic modules
2. Directly manipulates Python's sys.modules to prevent problematic imports
3. Enables tokenization functionality without triggering segmentation faults

Usage:
    python src/generative_ai_module/paperspace_spacy_fix.py
"""

import os
import sys
import subprocess
import importlib
import logging
import time
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("paperspace_spacy_fix.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("paperspace_spacy_fix")

def run_command(cmd, check=False):
    """
    Run a shell command safely, logging output and errors.
    Returns success status and output.
    """
    logger.info(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        if result.stdout:
            logger.info(f"Output: {result.stdout[:100]}...")
        if result.stderr:
            logger.warning(f"Stderr: {result.stderr[:100]}...")
        return result.returncode == 0, result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        if hasattr(e, 'stderr'):
            logger.error(e.stderr)
        return False, str(e)
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return False, str(e)

def fix_spacy_imports():
    """Fix spaCy imports by manipulating sys.modules to avoid ParametricAttention_v2"""
    logger.info("Creating safe spaCy import environment...")
    
    try:
        # Create dummy modules to prevent problematic imports
        class DummyModule:
            """Dummy module to replace problematic imports"""
            def __init__(self, name):
                self.__name__ = name
            
            def __getattr__(self, attr):
                # Return self for nested attributes
                return self
                
        # We'll manipulate sys.modules to prevent problematic imports
        # This is a hack but it's effective for avoiding segfaults
        if 'thinc.api' in sys.modules:
            logger.info("Removing existing thinc.api module...")
            del sys.modules['thinc.api']
        
        # Insert a dummy module for thinc.api
        dummy_thinc_api = DummyModule('thinc.api')
        sys.modules['thinc.api'] = dummy_thinc_api
        
        # Add ParametricAttention_v2 to the dummy module
        dummy_thinc_api.ParametricAttention_v2 = object()
        
        logger.info("Created dummy ParametricAttention_v2 in thinc.api")
        return True
    except Exception as e:
        logger.error(f"Error fixing imports: {e}")
        return False

def test_tokenization():
    """Test if basic tokenization works"""
    logger.info("Testing tokenization functionality...")
    
    try:
        # Import spacy (should work with our fixes)
        import spacy
        logger.info(f"Imported spaCy version: {spacy.__version__}")
        
        # Try to load the model
        try:
            nlp = spacy.load("en_core_web_sm")
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
        
        # Try tokenization only
        text = "This is a test of the Jarvis AI Assistant tokenization."
        tokens = [t.text for t in nlp.tokenizer(text)]
        logger.info(f"Tokenization successful: {tokens}")
        
        return True
    except Exception as e:
        logger.error(f"Error in tokenization test: {e}")
        return False

def create_test_script():
    """Create a test script for tokenization"""
    script_content = """#!/usr/bin/env python3
\"\"\"
Paperspace-safe spaCy tokenization test
\"\"\"

import sys
import os

def test_spacy_tokenize():
    \"\"\"Test spaCy tokenization only\"\"\"
    try:
        # Add the hack to avoid ParametricAttention_v2 error
        import sys
        class DummyModule:
            def __getattr__(self, attr):
                return self
        if 'thinc.api' in sys.modules:
            del sys.modules['thinc.api']
        sys.modules['thinc.api'] = DummyModule()
        sys.modules['thinc.api'].ParametricAttention_v2 = object()
        
        # Now import spacy
        import spacy
        print(f"spaCy version: {spacy.__version__}")
        
        # Load model
        nlp = spacy.load("en_core_web_sm")
        print("Model loaded successfully")
        
        # Tokenize only
        text = "Testing Jarvis AI Assistant with spaCy tokenization."
        tokens = [t.text for t in nlp.tokenizer(text)]
        print(f"\\nTokens: {tokens}")
        
        print("\\nTokenization test passed!")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("PAPERSPACE-SAFE SPACY TOKENIZATION TEST")
    print("=" * 70)
    
    success = test_spacy_tokenize()
    
    if success:
        print("\\n✅ SUCCESS: spaCy tokenization works!")
    else:
        print("\\n❌ FAILED: spaCy tokenization test failed")
    
    sys.exit(0 if success else 1)
"""
    
    test_path = "test_spacy_paperspace.py"
    with open(test_path, "w") as f:
        f.write(script_content)
    
    os.chmod(test_path, 0o755)  # Make executable
    logger.info(f"Created Paperspace-safe test script: {test_path}")
    return test_path

def main():
    """Main function"""
    print("=" * 70)
    print("🔧 PAPERSPACE-SPECIFIC SPACY FIX")
    print("=" * 70)
    
    # Fix imports
    if fix_spacy_imports():
        logger.info("Successfully fixed problematic imports")
    else:
        logger.error("Failed to fix imports")
    
    # Test tokenization
    if test_tokenization():
        logger.info("Tokenization works correctly")
        
        # Create test script
        test_path = create_test_script()
        
        print("\n" + "=" * 70)
        print("✅ spaCy tokenization is now working in Paperspace!")
        print("=" * 70)
        print(f"\nTo use tokenization, run: python {test_path}")
        print("\nIMPORTANT: You can safely use tokenization, but avoid using")
        print("          other spaCy components that might cause segfaults.")
        print("=" * 70)
        
        return 0
    else:
        print("\n" + "=" * 70)
        print("❌ spaCy fix was not completely successful")
        print("=" * 70)
        print("\nTry running the isolation script again:")
        print("  bash setup/install_spacy_isolated.sh")
        print("\nOr use only the basic tokenizer in your code.")
        print("=" * 70)
        
        return 1

if __name__ == "__main__":
    sys.exit(main()) 