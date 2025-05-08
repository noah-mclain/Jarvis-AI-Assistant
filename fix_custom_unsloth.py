#!/usr/bin/env python3
"""
Fix for custom_unsloth implementation to resolve the device_map parameter issue.

This script patches the custom_unsloth implementation to fix the
'got multiple values for keyword argument 'device_map'' error.
"""

import os
import sys
import re
from pathlib import Path

def fix_custom_unsloth():
    """Fix the custom_unsloth implementation to handle device_map parameter correctly"""
    
    # Check if the custom_unsloth directory exists
    custom_unsloth_path = Path("/notebooks/custom_unsloth")
    if not custom_unsloth_path.exists():
        print(f"Error: {custom_unsloth_path} not found")
        return False
    
    # Find the models/__init__.py file
    models_init_path = custom_unsloth_path / "unsloth" / "models" / "__init__.py"
    if not models_init_path.exists():
        print(f"Error: {models_init_path} not found")
        return False
    
    # Read the file
    with open(models_init_path, "r") as f:
        content = f.read()
    
    # Check if the file contains the get_model_and_tokenizer function
    if "def get_model_and_tokenizer" not in content:
        print(f"Error: get_model_and_tokenizer function not found in {models_init_path}")
        return False
    
    # Fix the device_map parameter issue
    # Look for the AutoModelForCausalLM.from_pretrained call
    pattern = r'(\s+)model = AutoModelForCausalLM\.from_pretrained\(\s*([^,]+),\s*([^)]+)\)'
    
    # Check if the pattern is found
    if not re.search(pattern, content):
        print(f"Error: AutoModelForCausalLM.from_pretrained call not found in {models_init_path}")
        return False
    
    # Function to process the match and remove device_map if it's in kwargs
    def process_match(match):
        indent = match.group(1)
        model_name = match.group(2)
        kwargs = match.group(3)
        
        # Remove device_map from kwargs if it's there
        kwargs_lines = kwargs.split('\n')
        filtered_kwargs = []
        for line in kwargs_lines:
            if 'device_map=' not in line:
                filtered_kwargs.append(line)
        
        # Join the filtered kwargs
        filtered_kwargs_str = '\n'.join(filtered_kwargs)
        
        # Add a comment explaining the change
        comment = f"{indent}# device_map is handled by FastLanguageModel, so we don't pass it here\n"
        
        # Return the fixed code
        return f"{comment}{indent}model = AutoModelForCausalLM.from_pretrained(\n{indent}    {model_name},\n{filtered_kwargs_str})"
    
    # Apply the fix
    fixed_content = re.sub(pattern, process_match, content, flags=re.DOTALL)
    
    # Write the fixed content back to the file
    with open(models_init_path, "w") as f:
        f.write(fixed_content)
    
    print(f"✅ Successfully fixed {models_init_path}")
    
    # Now fix the unified_deepseek_training.py file
    unified_path = Path("/notebooks/unified_deepseek_training.py")
    if not unified_path.exists():
        print(f"Error: {unified_path} not found")
        return False
    
    # Read the file
    with open(unified_path, "r") as f:
        content = f.read()
    
    # Fix the FastLanguageModel.from_pretrained call
    pattern = r'(\s+)model, tokenizer = FastLanguageModel\.from_pretrained\(\s*([^)]+)\)'
    
    # Check if the pattern is found
    if not re.search(pattern, content):
        print(f"Error: FastLanguageModel.from_pretrained call not found in {unified_path}")
        return False
    
    # Function to process the match and remove device_map and max_seq_length
    def process_match(match):
        indent = match.group(1)
        kwargs = match.group(2)
        
        # Remove device_map and max_seq_length from kwargs
        kwargs_lines = kwargs.split('\n')
        filtered_kwargs = []
        for line in kwargs_lines:
            if 'device_map=' not in line and 'max_seq_length=' not in line:
                filtered_kwargs.append(line)
        
        # Join the filtered kwargs
        filtered_kwargs_str = '\n'.join(filtered_kwargs)
        
        # Add a comment explaining the change
        comment = f"{indent}# device_map and max_seq_length are handled internally by FastLanguageModel\n"
        
        # Return the fixed code
        return f"{comment}{indent}model, tokenizer = FastLanguageModel.from_pretrained(\n{filtered_kwargs_str})"
    
    # Apply the fix
    fixed_content = re.sub(pattern, process_match, content, flags=re.DOTALL)
    
    # Add code to set max_seq_length after model is loaded
    if "model.config.max_position_embeddings = args.max_length" not in fixed_content:
        pattern = r'(\s+)model, tokenizer = FastLanguageModel\.from_pretrained\([^)]+\)'
        replacement = r'\1model, tokenizer = FastLanguageModel.from_pretrained(\2)\n\n\1# Set max sequence length after model is loaded\n\1model.config.max_position_embeddings = args.max_length\n\1tokenizer.model_max_length = args.max_length'
        fixed_content = re.sub(pattern, replacement, fixed_content, flags=re.DOTALL)
    
    # Write the fixed content back to the file
    with open(unified_path, "w") as f:
        f.write(fixed_content)
    
    print(f"✅ Successfully fixed {unified_path}")
    return True

def main():
    """Main function"""
    print("=" * 50)
    print("FIXING CUSTOM UNSLOTH IMPLEMENTATION")
    print("=" * 50)
    
    success = fix_custom_unsloth()
    
    if success:
        print("\nFix applied successfully!")
        print("You can now run the training script again.")
    else:
        print("\nFailed to apply the fix.")
        print("Please check the error message above.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
