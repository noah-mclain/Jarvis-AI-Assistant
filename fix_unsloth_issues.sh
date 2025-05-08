#!/bin/bash
# Fix Unsloth issues script
# This script checks and fixes issues with Unsloth implementation

echo "===== Fixing Unsloth Implementation Issues ====="

# First, check the Unsloth implementation
echo "Checking Unsloth implementation..."
python check_unsloth_implementation.py

# Then, fix the custom_unsloth implementation
echo "Fixing custom_unsloth implementation..."
python fix_custom_unsloth.py

# Fix the unified_deepseek_training.py file
echo "Fixing unified_deepseek_training.py..."
cat > fix_unified_deepseek.py << 'EOL'
#!/usr/bin/env python3
"""
Fix for unified_deepseek_training.py to resolve parameter issues.
"""
import os
import re
from pathlib import Path

def fix_unified_deepseek():
    """Fix the unified_deepseek_training.py file"""
    
    # Check if the file exists
    unified_path = Path("unified_deepseek_training.py")
    if not unified_path.exists():
        print(f"Error: {unified_path} not found")
        return False
    
    # Read the file
    with open(unified_path, "r") as f:
        content = f.read()
    
    # Fix the FastLanguageModel.from_pretrained call
    pattern = r'(\s+)model, tokenizer = FastLanguageModel\.from_pretrained\(\s*model_name=args\.model_name,\s*max_seq_length=args\.max_length,\s*load_in_4bit=args\.load_in_4bit,\s*load_in_8bit=args\.load_in_8bit(?:,\s*device_map="auto")?\s*\)'
    
    replacement = r'\1# Note: FastLanguageModel.from_pretrained already handles device_map internally\n\1model, tokenizer = FastLanguageModel.from_pretrained(\n\1    model_name=args.model_name,\n\1    load_in_4bit=args.load_in_4bit,\n\1    load_in_8bit=args.load_in_8bit\n\1)\n\n\1# Set max sequence length after model is loaded\n\1model.config.max_position_embeddings = args.max_length\n\1tokenizer.model_max_length = args.max_length'
    
    # Apply the fix
    fixed_content = re.sub(pattern, replacement, content)
    
    # Write the fixed content back to the file
    with open(unified_path, "w") as f:
        f.write(fixed_content)
    
    print(f"✅ Successfully fixed {unified_path}")
    return True

if __name__ == "__main__":
    fix_unified_deepseek()
EOL

chmod +x fix_unified_deepseek.py
python fix_unified_deepseek.py

echo "===== Fix Complete ====="
echo "You can now run the training script again."
