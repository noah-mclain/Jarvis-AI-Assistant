#!/usr/bin/env python3
"""
Simple test for the minimal spaCy tokenizer

This script tests the minimal tokenizer in isolation from other Jarvis components
to demonstrate it works correctly without causing import issues.
"""

import sys
import os
import importlib.util

# Directly import the minimal_spacy_tokenizer module to avoid loading the entire package
module_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "src/generative_ai_module/minimal_spacy_tokenizer.py"
)
spec = importlib.util.spec_from_file_location("minimal_spacy_tokenizer", module_path)
minimal_spacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(minimal_spacy)

# Get the tokenize function and tokenizer instance
tokenize = minimal_spacy.tokenize
tokenizer = minimal_spacy.tokenizer

print("=================================================================")
print("MINIMAL SPACY TOKENIZER TEST (PAPERSPACE-SAFE)")
print("=================================================================")

# Test with some sample texts
samples = [
    "This is a simple test sentence.",
    "The Jarvis AI Assistant can process natural language effectively!",
    "In Paperspace environments, spaCy sometimes has import issues and segfaults.",
    "This tokenizer works without those problems, providing basic NLP functionality."
]

print("\nTokenizer availability:", "✅ AVAILABLE" if tokenizer.is_available else "❌ UNAVAILABLE")
print("Using fallback:", "No (using spaCy)" if tokenizer.is_available else "Yes (basic splitting)")

# Process each sample
for i, sample in enumerate(samples):
    print(f"\n[Sample {i+1}]")
    print(f"Input: {sample}")
    tokens = tokenize(sample)
    print(f"Tokens ({len(tokens)}): {tokens}")

print("\n=================================================================")
print("TEST COMPLETE")
print("=================================================================") 