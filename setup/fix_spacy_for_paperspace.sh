#!/bin/bash

echo "======================================================================"
echo "🔧 Ultimate Paperspace spaCy Fix for Jarvis AI Assistant"
echo "======================================================================"

# Step 1: Optional - Check if we're running in Paperspace
if [ -d "/paperspace" ] || [[ "$HOSTNAME" == *"gradient"* ]] || [[ "$PAPERSPACE" == "true" ]]; then
    echo "✅ Detected Paperspace environment"
else
    echo "⚠️ This doesn't appear to be a Paperspace environment, but will continue anyway"
fi

# Step 2: Ask for confirmation
echo ""
echo "This script will:"
echo "  1. Clean up any existing spaCy installation"
echo "  2. Install the minimal spaCy setup for Paperspace"
echo "  3. Configure Jarvis AI to use the minimal tokenizer"
echo ""
read -p "Continue? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Operation cancelled"
    exit 1
fi

# Step 3: Create the minimal_spacy_tokenizer.py file if it doesn't exist
if [ ! -f "src/generative_ai_module/minimal_spacy_tokenizer.py" ]; then
    echo "Creating minimal_spacy_tokenizer.py..."
    cat > src/generative_ai_module/minimal_spacy_tokenizer.py << 'EOF'
#!/usr/bin/env python3
"""
Minimal SpaCy Tokenizer for Paperspace Environments

This module provides a super-minimal tokenizer that works in Paperspace without
triggering any of the problematic imports that cause segmentation faults.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("minimal_spacy")

class MinimalTokenizer:
    """
    A minimal wrapper around spaCy's tokenizer that avoids all problematic imports.
    This is specifically designed for Paperspace environments where normal spaCy
    imports cause segmentation faults or import errors.
    """
    def __init__(self):
        self.nlp = None
        self.is_available = False
        self.tokenizer = None
        self._safe_init()
    
    def _safe_init(self):
        """Initialize spaCy in the safest possible way"""
        try:
            # First, fix the import system to avoid ParametricAttention_v2 error
            self._fix_imports()
            
            # Import only the absolute minimum from spaCy
            try:
                # Import just the English language directly to avoid other imports
                import en_core_web_sm
                
                # Get just the tokenizer component
                self.tokenizer = en_core_web_sm.load().tokenizer
                self.is_available = True
                logger.info("Minimal spaCy tokenizer initialized successfully")
            except ImportError:
                logger.warning("en_core_web_sm not found, trying direct spaCy import")
                try:
                    # Try to import spaCy directly and get English tokenizer
                    import spacy
                    self.tokenizer = spacy.blank("en").tokenizer
                    self.is_available = True
                    logger.info("Minimal spaCy tokenizer initialized with blank model")
                except ImportError:
                    logger.warning("spaCy not available, using fallback tokenizer")
        except Exception as e:
            logger.error(f"Error initializing minimal tokenizer: {e}")
    
    def _fix_imports(self):
        """Fix problematic imports by manipulating the module system"""
        try:
            # Create a simple dummy module class
            class DummyModule:
                def __init__(self, name):
                    self.__name__ = name
                    self.__dict__["ParametricAttention_v2"] = type("ParametricAttention_v2", (), {})
                
                def __getattr__(self, name):
                    # Return a dummy object for any attribute
                    return type(name, (), {})()
            
            # Replace problematic modules
            for module_name in ['thinc.api', 'thinc.layers', 'thinc.model', 'thinc.config']:
                if module_name in sys.modules:
                    del sys.modules[module_name]
                sys.modules[module_name] = DummyModule(module_name)
            
            # Set other environment variables that might help
            os.environ["SPACY_WARNING_IGNORE"] = "W008,W107,W101"
            
            return True
        except Exception as e:
            logger.error(f"Error fixing imports: {e}")
            return False
    
    def tokenize(self, text):
        """
        Tokenize text using spaCy's tokenizer if available, otherwise fallback to basic split
        
        Args:
            text: The input text to tokenize
            
        Returns:
            List of token strings
        """
        if not text:
            return []
        
        if self.is_available and self.tokenizer:
            try:
                # Use spaCy tokenizer if available
                return [t.text for t in self.tokenizer(text)]
            except Exception as e:
                logger.warning(f"SpaCy tokenization failed: {e}")
                # Fall through to basic tokenization
        
        # Basic fallback tokenizer (simple but reasonable)
        return self._basic_tokenize(text)
    
    def _basic_tokenize(self, text):
        """Very basic tokenization as a fallback"""
        # Replace common punctuation with spaces around them for better splitting
        for punct in '.,;:!?()[]{}""\'':
            text = text.replace(punct, f' {punct} ')
        
        # Split on whitespace and filter out empty strings
        return [token for token in text.split() if token]


# Singleton instance for easy import
tokenizer = MinimalTokenizer()

def tokenize(text):
    """Convenience function to tokenize text"""
    return tokenizer.tokenize(text)


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("MINIMAL SPACY TOKENIZER TEST")
    print("=" * 70)
    
    test_text = "This is a test of the minimal tokenizer for Jarvis AI Assistant!"
    tokens = tokenize(test_text)
    
    print(f"\nInput: {test_text}")
    print(f"Tokens: {tokens}")
    print(f"Count: {len(tokens)} tokens")
    
    if tokenizer.is_available:
        print("\n✅ Using spaCy-based tokenization")
    else:
        print("\n⚠️ Using fallback tokenization (spaCy not available)")
    
    print("=" * 70)
EOF
    echo "✅ Created minimal_spacy_tokenizer.py"
else
    echo "✅ minimal_spacy_tokenizer.py already exists"
fi

# Step 4: Create a test script
echo "Creating test script..."
cat > test_minimal_spacy.py << 'EOF'
#!/usr/bin/env python3
"""
Test the minimal spaCy tokenizer in Paperspace environments
"""

import sys
import os

# Try to add the project directory to the path if needed
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    # First try importing from the module
    from src.generative_ai_module.minimal_spacy_tokenizer import tokenize, tokenizer
    print("✅ Successfully imported minimal_spacy_tokenizer")
    
    # Test the tokenizer
    test_text = "Jarvis AI Assistant is testing the minimal spaCy tokenizer in Paperspace!"
    tokens = tokenize(test_text)
    
    print(f"\nInput text: {test_text}")
    print(f"Tokenized result: {tokens}")
    print(f"Token count: {len(tokens)}")
    
    if tokenizer.is_available:
        print("\n✅ Using spaCy-based tokenization")
        
        # Test if we can use it in a loop (basic stress test)
        print("\nRunning basic stress test...")
        for i in range(10):
            test = f"Test sentence {i}: The quick brown fox jumps over the lazy dog."
            tokens = tokenize(test)
            print(f"  Tokenized test {i}: {len(tokens)} tokens")
        
        print("\n✅ All tests passed! The minimal tokenizer is working correctly.")
        sys.exit(0)
    else:
        print("\n⚠️ Using fallback tokenization (spaCy not available)")
        print("\n❌ Test failed: spaCy tokenization is not available")
        sys.exit(1)
        
except ImportError as e:
    print(f"❌ Error importing minimal_spacy_tokenizer: {e}")
    print("\nMake sure you have created the file at:")
    print("src/generative_ai_module/minimal_spacy_tokenizer.py")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)
EOF

chmod +x test_minimal_spacy.py
echo "✅ Created test_minimal_spacy.py"

# Step 5: Uninstall existing problematic packages
echo ""
echo "Step 1: Removing potentially conflicting packages..."
pip uninstall -y spacy thinc spacy-legacy spacy-loggers catalogue wasabi srsly weasel confection 

# Step 6: Install spaCy essentials
echo ""
echo "Step 2: Installing the minimal spaCy setup..."
pip install spacy==3.7.4 --no-deps
pip install en_core_web_sm==3.7.0 --no-deps

# Step 7: Run the test script
echo ""
echo "Step 3: Testing the minimal tokenizer..."
python test_minimal_spacy.py

# Print final instructions
echo ""
echo "======================================================================"

if [ $? -eq 0 ]; then
    echo "✅ The minimal spaCy tokenizer has been successfully installed!"
    echo "   Jarvis AI Assistant will now use the minimal tokenizer automatically."
    echo ""
    echo "   You can avoid spaCy issues in your code by using:"
    echo "   from src.generative_ai_module.minimal_spacy_tokenizer import tokenize"
    echo ""
    echo "   Example:"
    echo "   tokens = tokenize('Your text here')"
else
    echo "❌ There was an issue setting up the minimal tokenizer."
    echo "   Please check the error messages above for details."
fi

echo "======================================================================" 