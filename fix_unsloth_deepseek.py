#!/usr/bin/env python3
"""
Fix for Unsloth DeepSeek model loading issue.

This script patches the unified_deepseek_training.py file to fix the
'LlamaForCausalLM.__init__() got an unexpected keyword argument 'max_seq_length'' error.

The issue is that the max_seq_length parameter is being passed directly to the model initialization,
but it should be handled differently for DeepSeek models.
"""

import os
import sys
import re
from pathlib import Path

def fix_unified_deepseek_training():
    """Fix the unified_deepseek_training.py file to properly handle max_seq_length"""
    
    # Check if the file exists
    if not os.path.exists("unified_deepseek_training.py"):
        print("Error: unified_deepseek_training.py not found")
        return False
    
    # Read the file
    with open("unified_deepseek_training.py", "r") as f:
        content = f.read()
    
    # Fix the FastLanguageModel.from_pretrained call
    # Original:
    # model, tokenizer = FastLanguageModel.from_pretrained(
    #     model_name=args.model_name,
    #     max_seq_length=args.max_length,
    #     load_in_4bit=args.load_in_4bit,
    #     load_in_8bit=args.load_in_8bit
    # )
    
    # New version:
    fixed_content = re.sub(
        r'model, tokenizer = FastLanguageModel\.from_pretrained\(\s*'
        r'model_name=args\.model_name,\s*'
        r'max_seq_length=args\.max_length,\s*'
        r'load_in_4bit=args\.load_in_4bit,\s*'
        r'load_in_8bit=args\.load_in_8bit\s*'
        r'\)',
        'model, tokenizer = FastLanguageModel.from_pretrained(\n'
        '        model_name=args.model_name,\n'
        '        # max_seq_length is handled internally by FastLanguageModel\n'
        '        load_in_4bit=args.load_in_4bit,\n'
        '        load_in_8bit=args.load_in_8bit,\n'
        '        # Set device_map to auto to let Unsloth handle device placement\n'
        '        device_map="auto"\n'
        '    )\n'
        '    \n'
        '    # Set max sequence length after model is loaded\n'
        '    model.config.max_position_embeddings = args.max_length\n'
        '    tokenizer.model_max_length = args.max_length',
        content
    )
    
    # Write the fixed content back to the file
    with open("unified_deepseek_training.py", "w") as f:
        f.write(fixed_content)
    
    print("✅ Successfully fixed unified_deepseek_training.py")
    return True

def main():
    """Main function"""
    print("=" * 50)
    print("FIXING UNSLOTH DEEPSEEK MODEL LOADING ISSUE")
    print("=" * 50)
    
    success = fix_unified_deepseek_training()
    
    if success:
        print("\nFix applied successfully!")
        print("You can now run the training script again.")
    else:
        print("\nFailed to apply the fix.")
        print("Please check the error message above.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
