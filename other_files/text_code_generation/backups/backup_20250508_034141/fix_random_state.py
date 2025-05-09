#!/usr/bin/env python3
"""
Fix for random_state parameter issue in unified_deepseek_training.py.

This script fixes the 'LoraConfig.__init__() got an unexpected keyword argument 'random_state'' error.
"""

import os
import re
from pathlib import Path

def fix_unified_deepseek():
    """Fix the unified_deepseek_training.py file to remove random_state from LoraConfig"""
    
    # Check if the file exists
    unified_path = Path("unified_deepseek_training.py")
    if not unified_path.exists():
        print(f"Error: {unified_path} not found")
        return False
    
    # Read the file
    with open(unified_path, "r") as f:
        content = f.read()
    
    # Fix the FastLanguageModel.get_peft_model call
    pattern = r'(\s+)model = FastLanguageModel\.get_peft_model\(\s*model,\s*r=16,\s*lora_alpha=32,\s*lora_dropout=0\.05,\s*target_modules=\[\s*"q_proj", "k_proj", "v_proj", "o_proj",\s*"gate_proj", "up_proj", "down_proj"\s*\],\s*(?:use_gradient_checkpointing=True,\s*)?random_state=42,?\s*\)'
    
    replacement = r'\1# Note: random_state is not a valid parameter for LoraConfig\n\1model = FastLanguageModel.get_peft_model(\n\1    model,\n\1    r=16,  # LoRA rank\n\1    lora_alpha=32,\n\1    lora_dropout=0.05,\n\1    target_modules=[\n\1        "q_proj", "k_proj", "v_proj", "o_proj",\n\1        "gate_proj", "up_proj", "down_proj"\n\1    ]\n\1)'
    
    # Apply the fix
    fixed_content = re.sub(pattern, replacement, content)
    
    # Write the fixed content back to the file
    with open(unified_path, "w") as f:
        f.write(fixed_content)
    
    print(f"✅ Successfully fixed {unified_path}")
    return True

def fix_custom_unsloth():
    """Fix the custom_unsloth implementation to remove random_state from get_peft_model"""
    
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
    
    # Check if the file contains the get_peft_model function
    if "def get_peft_model" not in content:
        print(f"Error: get_peft_model function not found in {models_init_path}")
        return False
    
    # Fix the get_peft_model function to handle random_state
    pattern = r'def get_peft_model\(model, ([^)]+)random_state=None,([^)]+)\):'
    
    # Check if the pattern is found
    if not re.search(pattern, content):
        print(f"Warning: random_state parameter not found in get_peft_model function in {models_init_path}")
        
        # Try an alternative pattern
        pattern = r'def get_peft_model\(model, ([^)]+)random_state=None\):'
        if not re.search(pattern, content):
            print(f"Warning: alternative random_state pattern not found in get_peft_model function in {models_init_path}")
            
            # Try to find the function signature
            pattern = r'def get_peft_model\([^)]+\):'
            match = re.search(pattern, content)
            if match:
                print(f"Found function signature: {match.group(0)}")
            
            return True
    
    # Replace the pattern
    replacement = r'def get_peft_model(model, \1\2):\n    # random_state is not a valid parameter for LoraConfig'
    
    # Apply the fix
    fixed_content = re.sub(pattern, replacement, content)
    
    # Also fix the actual LoraConfig call
    pattern = r'(\s+)peft_config = LoraConfig\(\s*([^)]+)random_state=random_state,\s*([^)]+)\)'
    
    # Check if the pattern is found
    if not re.search(pattern, content):
        print(f"Warning: random_state not found in LoraConfig call in {models_init_path}")
        
        # Try an alternative pattern
        pattern = r'(\s+)peft_config = LoraConfig\(\s*([^)]+)random_state=random_state\s*\)'
        if not re.search(pattern, content):
            print(f"Warning: alternative random_state pattern not found in LoraConfig call in {models_init_path}")
            return True
    
    # Replace the pattern
    replacement = r'\1# random_state is not a valid parameter for LoraConfig\n\1peft_config = LoraConfig(\n\1    \2\3)'
    
    # Apply the fix
    fixed_content = re.sub(pattern, replacement, fixed_content)
    
    # Write the fixed content back to the file
    with open(models_init_path, "w") as f:
        f.write(fixed_content)
    
    print(f"✅ Successfully fixed {models_init_path}")
    return True

def main():
    """Main function"""
    print("=" * 50)
    print("FIXING RANDOM_STATE PARAMETER ISSUE")
    print("=" * 50)
    
    # Fix the unified_deepseek_training.py file
    unified_success = fix_unified_deepseek()
    
    # Fix the custom_unsloth implementation
    custom_success = fix_custom_unsloth()
    
    if unified_success and custom_success:
        print("\nFix applied successfully!")
        print("You can now run the training script again.")
    else:
        print("\nFailed to apply some fixes.")
        print("Please check the error messages above.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
