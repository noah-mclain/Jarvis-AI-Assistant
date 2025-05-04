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
            
            # Try to initialize without any module patching first
            try:
                self._init_with_direct_import()
                return  # If this works, we're done
            except Exception as e:
                logger.warning(f"Direct import failed: {e}, trying fallback methods")
            
            # If direct import failed, try with minimal patching
            try:
                self._init_with_minimal_patching()
            except Exception as e:
                logger.warning(f"Minimal patching failed: {e}")
                # Continue to fallback tokenizer
        except Exception as e:
            logger.error(f"Error initializing minimal tokenizer: {e}")
    
    def _init_with_direct_import(self):
        """Try to initialize spaCy directly without any module patching"""
        import spacy
        self.nlp = spacy.blank("en")
        self.tokenizer = self.nlp.tokenizer
        self.is_available = True
        logger.info("Minimal spaCy tokenizer initialized with direct import")
    
    def _init_with_minimal_patching(self):
        """Initialize with minimal module patching"""
        # Create dummy modules only as needed
        class DummyModule:
            def __init__(self, name):
                self.__name__ = name
                # Add ParametricAttention_v2 attribute to avoid the common error
                if name == "thinc.api":
                    self.ParametricAttention_v2 = type("ParametricAttention_v2", (), {})
            
            def __getattr__(self, attr_name):
                # For 'registry()' error, return a callable that accepts no args
                if attr_name == "registry":
                    return lambda: None
                # Return dummy objects for other attributes
                return type(attr_name, (), {})
        
        # Patch just the minimum modules needed
        for module_name in ['thinc.api']:
            if module_name in sys.modules:
                del sys.modules[module_name]
            sys.modules[module_name] = DummyModule(module_name)
        
        # Try importing again
        import spacy
        
        # Use only the most minimal functionality
        self.nlp = spacy.blank("en")
        self.tokenizer = self.nlp.tokenizer
        self.is_available = True
        logger.info("Minimal spaCy tokenizer initialized with minimal patching")
    
    def _fix_imports(self):
        """Fix problematic imports by manipulating the module system"""
        try:
            # Set environment variables that might help
            os.environ["SPACY_WARNING_IGNORE"] = "W008,W107,W101"
            
            return True
        except Exception as e:
            logger.error(f"Error setting environment variables: {e}")
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