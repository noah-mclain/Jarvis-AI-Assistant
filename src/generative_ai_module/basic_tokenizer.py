"""
Basic character-level tokenizer for use with the preprocessed datasets
"""



import os
import sys
import torch
import json
import argparse

# Common characters mapped to token IDs based on inspection of datasets
CHAR_TO_TOKEN = {
    ' ': 5,   # Space - most common token
    'e': 70,  # Common letter
    't': 74,  # Common letter
    'a': 84,  # Common letter
    'o': 89,  # Common letter
    'i': 78,  # Common letter
    'n': 46,  # Common letter
    's': 87,  # Common letter
    'r': 83,  # Common letter
    'h': 19,  # Common letter
    'l': 88,  # Common letter
    'd': 81,  # Common letter
    '\n': 33, # Newline
    ',': 42,  # Comma
    '.': 35,  # Period
    'c': 82,  # More letters
    'm': 73,  # More letters
    'u': 72,  # More letters
    'f': 77,  # More letters
    'p': 76,  # More letters
    'g': 71,  # More letters
    'w': 75,  # More letters
    'y': 79,  # More letters
    'b': 90,  # More letters
    'v': 86,  # More letters
    'k': 85,  # More letters
    '?': 40,  # Question mark
    '!': 39,  # Exclamation
    "'": 36,  # Apostrophe
    '"': 37,  # Quote
    '-': 38,  # Dash
    ':': 41,  # Colon
    ';': 44,  # Semicolon
    '(': 45,  # Open paren
    ')': 47,  # Close paren
    '<': 53,  # Less than (for tags like <PERSONA>)
    '>': 54,  # Greater than
    '/': 48,  # Slash
    '\\': 49, # Backslash
    '+': 43,  # Plus
    '=': 50,  # Equals
    '_': 51,  # Underscore
    '*': 52,  # Asterisk
    # Special token sequences
    'P': 55,  # For tags
    'E': 56,  # For tags
    'R': 57,  # For tags
    'S': 58,  # For tags 
    'O': 59,  # For tags
    'N': 60,  # For tags
    'A': 61,  # For tags
    'D': 62,  # For tags
    'I': 63,  # For tags
    'L': 64,  # For tags
    'G': 65,  # For tags
    'U': 66,  # For tags
    'T': 67,  # For tags
    # Add more as needed
}

TOKEN_TO_CHAR = {v: k for k, v in CHAR_TO_TOKEN.items()} | {
    0: '<PAD>',
    1: '<START>',
    2: '<END>',
    3: '<UNK>',
    4: '<SEP>',
}

class BasicTokenizer:
    """Simple character-level tokenizer for the preprocessed datasets"""
    
    def __init__(self, vocab_size=104):
        self.vocab_size = vocab_size
        self.char_to_token = CHAR_TO_TOKEN
        self.token_to_char = TOKEN_TO_CHAR
        
        # Default token for unknown characters
        self.unk_token = 3
    
    def encode(self, text):
        """Convert text to token IDs"""
        if text is None:
            return []
        
        tokens = []
        for char in text:
            if char in self.char_to_token:
                tokens.append(self.char_to_token[char])
            else:
                # Add the unknown token
                tokens.append(self.char_to_token.get('<UNK>', 3))
                print(f"Warning: Unknown character '{char}' (ord={ord(char)}) encountered during encoding")
        
        return tokens
    
    def decode(self, tokens):
        """Convert token IDs back to text"""
        if not tokens:
            return ""
        
        text = []
        for token in tokens:
            # Get the character for this token
            char = TOKEN_TO_CHAR.get(token)
            if char is not None:
                text.append(char)
            else:
                # Treat any unknown token as unknown character
                text.append(f"<{token}>")
        
        return "".join(text)
    
    def save(self, filepath):
        """Save tokenizer mappings to a JSON file"""
        data = {
            'vocab_size': self.vocab_size,
            'char_to_token': self.char_to_token,
            'unk_token': self.unk_token
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Tokenizer saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load tokenizer from a JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        tokenizer = cls(vocab_size=data['vocab_size'])
        tokenizer.char_to_token = data['char_to_token']
        tokenizer.unk_token = data['unk_token']

        tokenizer.token_to_char = {
            v: k for k, v in tokenizer.char_to_token.items()
        } | {
            0: '<PAD>',
            1: '<START>',
            2: '<END>',
            3: '<UNK>',
            4: '<SEP>',
        }
        return tokenizer

def test_tokenizer():
    """Test the tokenizer with some examples"""
    tokenizer = BasicTokenizer()
    
    test_texts = [
        "Hello, world!",
        "This is a test.",
        "<PERSONA>\n- I am a software engineer.\n<DIALOGUE>\nUSER: What do you do for a living?\nASSISTANT: I work as a software engineer.",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
    ]
    
    for text in test_texts:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        
        print(f"\nOriginal: {text}")
        print(f"Tokens: {tokens[:20]}... (length: {len(tokens)})")
        print(f"Decoded: {decoded}")
        print(f"Match: {text == decoded}")
    
    # Save and load test
    tokenizer.save("test_tokenizer.json")
    loaded_tokenizer = BasicTokenizer.load("test_tokenizer.json")
    
    # Test loaded tokenizer
    test_text = "Test loaded tokenizer"
    tokens = loaded_tokenizer.encode(test_text)
    decoded = loaded_tokenizer.decode(tokens)
    
    print(f"\nLoaded tokenizer test:")
    print(f"Original: {test_text}")
    print(f"Decoded: {decoded}")
    print(f"Match: {test_text == decoded}")
    
    # Clean up
    os.remove("test_tokenizer.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the basic tokenizer")
    parser.add_argument("--test", action="store_true", help="Run tokenizer tests")
    parser.add_argument("--text", help="Text to encode and decode")
    args = parser.parse_args()
    
    if args.test:
        test_tokenizer()
    elif args.text:
        tokenizer = BasicTokenizer()
        tokens = tokenizer.encode(args.text)
        decoded = tokenizer.decode(tokens)
        
        print(f"Original: {args.text}")
        print(f"Tokens: {tokens}")
        print(f"Decoded: {decoded}")
        print(f"Match: {args.text == decoded}")
    else:
        print("Use --test to run tokenizer tests or --text to encode/decode a specific text") 