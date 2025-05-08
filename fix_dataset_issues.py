#!/usr/bin/env python3
"""
Direct fix for dataset processing and tokenization issues in unified_deepseek_training.py.
This script directly modifies the existing file to fix the 'Unable to create tensor' error.
"""

import os
import re
import shutil
from pathlib import Path

def fix_unified_deepseek():
    """Fix dataset processing and tokenization in unified_deepseek_training.py"""
    
    # Path to the file
    file_path = "/notebooks/unified_deepseek_training.py"
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"❌ {file_path} not found")
        return False
    
    # Create a backup
    backup_path = f"{file_path}.dataset.bak"
    shutil.copy2(file_path, backup_path)
    print(f"Created backup at {backup_path}")
    
    # Read the file
    with open(file_path, "r") as f:
        content = f.read()
    
    # Fix 1: Modify the tokenize_function to handle potential issues
    pattern = r'(\s+)def tokenize_function\(examples\):\s*return tokenizer\(\s*examples\["text"\],\s*truncation=True,\s*padding="max_length",\s*max_length=args\.max_length,\s*return_tensors="pt"\s*\)'
    
    replacement = r'\1def tokenize_function(examples):\n\1    """Tokenize examples with proper handling of potential issues"""\n\1    # Ensure all texts are strings\n\1    texts = [str(text) if not isinstance(text, str) else text for text in examples["text"]]\n\1    \n\1    # Tokenize without return_tensors to avoid the "too many dimensions" error\n\1    return tokenizer(\n\1        texts,\n\1        truncation=True,\n\1        padding="max_length",\n\1        max_length=args.max_length,\n\1        return_tensors=None\n\1    )'
    
    # Check if the pattern is found
    if not re.search(pattern, content):
        print(f"Warning: tokenize_function pattern not found in {file_path}")
        
        # Try a more general pattern
        pattern = r'(\s+)def tokenize_function\(examples\):\s*return tokenizer\([^)]+\)'
        if not re.search(pattern, content):
            print(f"Error: alternative tokenize_function pattern not found in {file_path}")
            return False
    
    # Apply the fix
    content = re.sub(pattern, replacement, content)
    
    # Fix 2: Update the data collator to handle potential issues
    pattern = r'(\s+)# Create data collator\s*data_collator = DataCollatorForLanguageModeling\(\s*tokenizer=tokenizer,\s*mlm=False\s*\)'
    
    replacement = r'\1# Create a custom data collator that handles potential issues\n\1class SafeDataCollator(DataCollatorForLanguageModeling):\n\1    def __call__(self, features):\n\1        try:\n\1            # Try the standard collation\n\1            return super().__call__(features)\n\1        except ValueError as e:\n\1            # If there\'s an error, log it and try a more robust approach\n\1            logger.warning(f"Data collation error: {e}")\n\1            \n\1            # Convert all features to the same format\n\1            batch = {}\n\1            for key in features[0].keys():\n\1                if key in ["input_ids", "attention_mask", "labels"]:\n\1                    batch[key] = []\n\1                    for feature in features:\n\1                        # Ensure the feature is a list\n\1                        if isinstance(feature[key], list):\n\1                            batch[key].append(feature[key])\n\1                        else:\n\1                            batch[key].append([feature[key]])\n\1            \n\1            # Pad the sequences\n\1            return self.tokenizer.pad(batch, return_tensors="pt")\n\1\n\1# Create data collator with custom handling\n\1data_collator = SafeDataCollator(\n\1    tokenizer=tokenizer,\n\1    mlm=False\n\1)'
    
    # Check if the pattern is found
    if not re.search(pattern, content):
        print(f"Warning: data_collator pattern not found in {file_path}")
        
        # Try a more general pattern
        pattern = r'(\s+)data_collator = DataCollatorForLanguageModeling\([^)]+\)'
        if not re.search(pattern, content):
            print(f"Error: alternative data_collator pattern not found in {file_path}")
            return False
    
    # Apply the fix
    content = re.sub(pattern, replacement, content)
    
    # Write the fixed content back to the file
    with open(file_path, "w") as f:
        f.write(content)
    
    print(f"✅ Successfully fixed dataset processing in {file_path}")
    return True

def main():
    """Main function"""
    print("=" * 50)
    print("FIXING DATASET PROCESSING ISSUES")
    print("=" * 50)
    
    success = fix_unified_deepseek()
    
    if success:
        print("\nDataset processing fix applied successfully!")
        print("You can now run the training script again.")
    else:
        print("\nFailed to apply dataset processing fix.")
        print("Please check the error messages above.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
