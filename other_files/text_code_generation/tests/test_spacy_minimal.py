#!/usr/bin/env python3
"""
Minimal spaCy Test Script for Paperspace

This script tests if spaCy is correctly installed while avoiding features
that might cause segmentation faults in Paperspace environments.
"""

import sys
import os

def test_spacy_minimal():
    """Test spaCy with minimal functionality"""
    try:
        # Try importing spaCy
        import spacy
        print(f"spaCy version: {spacy.__version__}")
        
        # Try loading the model without advanced features
        try:
            nlp = spacy.load("en_core_web_sm")
            print("✅ Model loaded successfully")
        except OSError:
            print("❌ Model not found. Try installing it with:")
            print("python -m spacy download en_core_web_sm")
            return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
        
        # Basic tokenization only - avoiding complex pipeline components
        text = "Testing spaCy without features that might cause segmentation faults."
        print("\nProcessing text:", text)
        
        # Process with safe options
        tokens = []
        for token in nlp.tokenizer(text):
            tokens.append(token.text)
        
        print("\nTokens:")
        for token in tokens[:10]:  # Show first 10 tokens
            print(f"  {token}")
        
        print("\n✅ spaCy test completed successfully!")
        return True
        
    except ImportError:
        print("❌ spaCy is not installed")
        print("To install: pip install spacy==3.7.4")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main entry point"""
    print("=" * 70)
    print("MINIMAL SPACY TEST (PAPERSPACE-SAFE)")
    print("=" * 70)
    
    success = test_spacy_minimal()
    
    print("\n" + "=" * 70)
    if success:
        print("✅ SUCCESS: spaCy is working correctly with minimal functionality")
    else:
        print("❌ FAILED: spaCy test failed")
        print("Run src/generative_ai_module/fix_spacy_paperspace.py to fix the installation")
    print("=" * 70)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 