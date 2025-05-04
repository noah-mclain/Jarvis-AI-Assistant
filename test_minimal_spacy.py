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
        for i in range(5):
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
