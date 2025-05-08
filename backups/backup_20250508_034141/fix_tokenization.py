#!/usr/bin/env python3
"""
Fix for tokenization issues in unified_deepseek_training.py.

This script fixes the 'Unable to create tensor' error by ensuring proper padding and truncation.
"""

import os
import re
import shutil
from pathlib import Path

def fix_tokenization_function():
    """Fix the tokenization function in unified_deepseek_training.py"""
    
    # Path to the file
    file_path = "/notebooks/unified_deepseek_training.py"
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"❌ {file_path} not found")
        return False
    
    # Create a backup
    backup_path = f"{file_path}.tokenization.bak"
    shutil.copy2(file_path, backup_path)
    print(f"Created backup at {backup_path}")
    
    # Read the file
    with open(file_path, "r") as f:
        content = f.read()
    
    # Fix the tokenize_function
    pattern = r'(\s+)def tokenize_function\(examples\):\s*return tokenizer\(\s*examples\["text"\],\s*truncation=True,\s*padding="max_length",\s*max_length=args\.max_length,\s*return_tensors="pt"\s*\)'
    
    replacement = r'\1def tokenize_function(examples):\n\1    """Tokenize examples with proper padding and truncation"""\n\1    # Process one example at a time to avoid the "too many dimensions \'str\'" error\n\1    result = {"input_ids": [], "attention_mask": []}\n\1    \n\1    for text in examples["text"]:\n\1        # Ensure text is a string\n\1        if not isinstance(text, str):\n\1            text = str(text)\n\1        \n\1        # Tokenize with truncation but without padding or return_tensors\n\1        encoded = tokenizer(\n\1            text,\n\1            truncation=True,\n\1            max_length=args.max_length,\n\1            padding=False,\n\1            return_tensors=None\n\1        )\n\1        \n\1        result["input_ids"].append(encoded["input_ids"])\n\1        result["attention_mask"].append(encoded["attention_mask"])\n\1    \n\1    return result'
    
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
    
    # Also update the map function to not use batched=True
    pattern = r'(\s+)train_dataset = train_dataset\.map\(tokenize_function, batched=True\)'
    replacement = r'\1# Process examples individually to avoid dimension errors\n\1train_dataset = train_dataset.map(tokenize_function, batched=True, batch_size=1)'
    
    content = re.sub(pattern, replacement, content)
    
    pattern = r'(\s+)eval_dataset = eval_dataset\.map\(tokenize_function, batched=True\)'
    replacement = r'\1# Process examples individually to avoid dimension errors\n\1eval_dataset = eval_dataset.map(tokenize_function, batched=True, batch_size=1)'
    
    content = re.sub(pattern, replacement, content)
    
    # Update the data collator to handle the new format
    pattern = r'(\s+)# Create data collator\s*data_collator = DataCollatorForLanguageModeling\(\s*tokenizer=tokenizer,\s*mlm=False\s*\)'
    
    replacement = r'\1# Create data collator with proper padding\n\1data_collator = DataCollatorForLanguageModeling(\n\1    tokenizer=tokenizer,\n\1    mlm=False,\n\1    padding="max_length",\n\1    max_length=args.max_length\n\1)'
    
    content = re.sub(pattern, replacement, content)
    
    # Write the fixed content back to the file
    with open(file_path, "w") as f:
        f.write(content)
    
    print(f"✅ Successfully fixed tokenization in {file_path}")
    return True

def main():
    """Main function"""
    print("=" * 50)
    print("FIXING TOKENIZATION ISSUES")
    print("=" * 50)
    
    success = fix_tokenization_function()
    
    if success:
        print("\nTokenization fix applied successfully!")
        print("You can now run the training script again.")
    else:
        print("\nFailed to apply tokenization fix.")
        print("Please check the error messages above.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
