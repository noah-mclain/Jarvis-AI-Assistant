#!/usr/bin/env python3
"""
Fix for LoraConfig parameter issue in unified_deepseek_training.py.

This script fixes the 'LoraConfig.__init__() got an unexpected keyword argument 'use_gradient_checkpointing'' error.
"""

import os
import re
from pathlib import Path

def fix_unified_deepseek():
    """Fix the unified_deepseek_training.py file to remove use_gradient_checkpointing from LoraConfig"""
    
    # Check if the file exists
    unified_path = Path("unified_deepseek_training.py")
    if not unified_path.exists():
        print(f"Error: {unified_path} not found")
        return False
    
    # Read the file
    with open(unified_path, "r") as f:
        content = f.read()
    
    # Fix the FastLanguageModel.get_peft_model call
    pattern = r'(\s+)model = FastLanguageModel\.get_peft_model\(\s*model,\s*r=16,\s*lora_alpha=32,\s*lora_dropout=0\.05,\s*target_modules=\[\s*"q_proj", "k_proj", "v_proj", "o_proj",\s*"gate_proj", "up_proj", "down_proj"\s*\],\s*use_gradient_checkpointing=True,\s*random_state=42,\s*\)'
    
    replacement = r'\1# Note: use_gradient_checkpointing should be set in TrainingArguments, not LoraConfig\n\1model = FastLanguageModel.get_peft_model(\n\1    model,\n\1    r=16,  # LoRA rank\n\1    lora_alpha=32,\n\1    lora_dropout=0.05,\n\1    target_modules=[\n\1        "q_proj", "k_proj", "v_proj", "o_proj",\n\1        "gate_proj", "up_proj", "down_proj"\n\1    ],\n\1    random_state=42\n\1)'
    
    # Apply the fix
    fixed_content = re.sub(pattern, replacement, content)
    
    # Also update the TrainingArguments to enable gradient checkpointing
    pattern = r'(\s+)training_args = TrainingArguments\(\s*output_dir=args\.output_dir,\s*num_train_epochs=args\.epochs,\s*per_device_train_batch_size=args\.batch_size,\s*per_device_eval_batch_size=args\.batch_size,\s*gradient_accumulation_steps=args\.gradient_accumulation_steps,\s*learning_rate=args\.learning_rate,\s*weight_decay=0\.01,\s*warmup_steps=args\.warmup_steps,\s*logging_steps=10,\s*save_steps=100,\s*evaluation_strategy="steps",\s*eval_steps=100,\s*save_total_limit=3,\s*bf16=args\.bf16,\s*fp16=not args\.bf16 and torch\.cuda\.is_available\(\),\s*remove_unused_columns=False,\s*dataloader_num_workers=args\.num_workers,\s*dataloader_pin_memory=True,\s*group_by_length=True,\s*\)'
    
    replacement = r'\1training_args = TrainingArguments(\n\1    output_dir=args.output_dir,\n\1    num_train_epochs=args.epochs,\n\1    per_device_train_batch_size=args.batch_size,\n\1    per_device_eval_batch_size=args.batch_size,\n\1    gradient_accumulation_steps=args.gradient_accumulation_steps,\n\1    learning_rate=args.learning_rate,\n\1    weight_decay=0.01,\n\1    warmup_steps=args.warmup_steps,\n\1    logging_steps=10,\n\1    save_steps=100,\n\1    evaluation_strategy="steps",\n\1    eval_steps=100,\n\1    save_total_limit=3,\n\1    bf16=args.bf16,\n\1    fp16=not args.bf16 and torch.cuda.is_available(),\n\1    remove_unused_columns=False,\n\1    dataloader_num_workers=args.num_workers,\n\1    dataloader_pin_memory=True,\n\1    group_by_length=True,\n\1    # Enable gradient checkpointing for memory efficiency\n\1    gradient_checkpointing=True\n\1)'
    
    # Apply the fix
    fixed_content = re.sub(pattern, replacement, fixed_content)
    
    # Write the fixed content back to the file
    with open(unified_path, "w") as f:
        f.write(fixed_content)
    
    print(f"✅ Successfully fixed {unified_path}")
    return True

def fix_custom_unsloth():
    """Fix the custom_unsloth implementation to remove use_gradient_checkpointing from get_peft_model"""
    
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
    
    # Fix the get_peft_model function to handle use_gradient_checkpointing
    pattern = r'def get_peft_model\(model, ([^)]+)use_gradient_checkpointing=False,([^)]+)\):'
    
    # Check if the pattern is found
    if not re.search(pattern, content):
        print(f"Warning: use_gradient_checkpointing parameter not found in get_peft_model function in {models_init_path}")
        return True
    
    # Replace the pattern
    replacement = r'def get_peft_model(model, \1\2):\n    # use_gradient_checkpointing is handled by TrainingArguments'
    
    # Apply the fix
    fixed_content = re.sub(pattern, replacement, content)
    
    # Also fix the actual LoraConfig call
    pattern = r'(\s+)peft_config = LoraConfig\(\s*([^)]+)use_gradient_checkpointing=use_gradient_checkpointing,\s*([^)]+)\)'
    
    # Check if the pattern is found
    if not re.search(pattern, content):
        print(f"Warning: use_gradient_checkpointing not found in LoraConfig call in {models_init_path}")
        return True
    
    # Replace the pattern
    replacement = r'\1# use_gradient_checkpointing is handled by TrainingArguments\n\1peft_config = LoraConfig(\n\1    \2\3)'
    
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
    print("FIXING LORA CONFIG PARAMETER ISSUE")
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
