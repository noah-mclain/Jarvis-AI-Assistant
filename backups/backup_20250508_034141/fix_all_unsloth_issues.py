#!/usr/bin/env python3
"""
Comprehensive fix for all Unsloth issues in DeepSeek training.

This script fixes:
1. The max_seq_length parameter issue
2. The device_map parameter issue
3. The use_gradient_checkpointing parameter issue
4. The random_state parameter issue
"""

import os
import sys
import re
from pathlib import Path

def fix_unified_deepseek():
    """Fix all issues in the unified_deepseek_training.py file"""
    
    # Check if the file exists
    unified_path = Path("/notebooks/unified_deepseek_training.py")
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
    
    # Fix the FastLanguageModel.get_peft_model call
    pattern = r'(\s+)model = FastLanguageModel\.get_peft_model\(\s*model,\s*r=16,\s*lora_alpha=32,\s*lora_dropout=0\.05,\s*target_modules=\[\s*"q_proj", "k_proj", "v_proj", "o_proj",\s*"gate_proj", "up_proj", "down_proj"\s*\],\s*(?:use_gradient_checkpointing=True,\s*)?(?:random_state=42,?\s*)?\)'
    
    replacement = r'\1# Note: use_gradient_checkpointing and random_state are not valid parameters for LoraConfig\n\1model = FastLanguageModel.get_peft_model(\n\1    model,\n\1    r=16,  # LoRA rank\n\1    lora_alpha=32,\n\1    lora_dropout=0.05,\n\1    target_modules=[\n\1        "q_proj", "k_proj", "v_proj", "o_proj",\n\1        "gate_proj", "up_proj", "down_proj"\n\1    ]\n\1)'
    
    # Apply the fix
    fixed_content = re.sub(pattern, replacement, fixed_content)
    
    # Update the TrainingArguments to enable gradient checkpointing
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
    """Fix all issues in the custom_unsloth implementation"""
    
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
    
    # Fix the get_peft_model function to handle use_gradient_checkpointing and random_state
    # First, check the function signature
    pattern = r'def get_peft_model\(model, ([^)]+)\):'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        params = match.group(1)
        # Remove use_gradient_checkpointing and random_state from params
        params_lines = params.split('\n')
        filtered_params = []
        for line in params_lines:
            if 'use_gradient_checkpointing=' not in line and 'random_state=' not in line:
                filtered_params.append(line)
        
        # Join the filtered params
        filtered_params_str = '\n'.join(filtered_params)
        
        # Replace the function signature
        fixed_content = re.sub(pattern, f"def get_peft_model(model, {filtered_params_str}):", fixed_content, flags=re.DOTALL)
    
    # Fix the LoraConfig call
    pattern = r'(\s+)peft_config = LoraConfig\(\s*([^)]+)\)'
    match = re.search(pattern, fixed_content, re.DOTALL)
    if match:
        indent = match.group(1)
        params = match.group(2)
        
        # Remove use_gradient_checkpointing and random_state from params
        params_lines = params.split('\n')
        filtered_params = []
        for line in params_lines:
            if 'use_gradient_checkpointing=' not in line and 'random_state=' not in line:
                filtered_params.append(line)
        
        # Join the filtered params
        filtered_params_str = '\n'.join(filtered_params)
        
        # Replace the LoraConfig call
        comment = f"{indent}# use_gradient_checkpointing and random_state are not valid parameters for LoraConfig\n"
        fixed_content = re.sub(pattern, f"{comment}{indent}peft_config = LoraConfig(\n{indent}    {filtered_params_str})", fixed_content, flags=re.DOTALL)
    
    # Write the fixed content back to the file
    with open(models_init_path, "w") as f:
        f.write(fixed_content)
    
    print(f"✅ Successfully fixed {models_init_path}")
    return True

def main():
    """Main function"""
    print("=" * 50)
    print("FIXING ALL UNSLOTH ISSUES")
    print("=" * 50)
    
    # Fix the unified_deepseek_training.py file
    unified_success = fix_unified_deepseek()
    
    # Fix the custom_unsloth implementation
    custom_success = fix_custom_unsloth()
    
    if unified_success and custom_success:
        print("\nAll fixes applied successfully!")
        print("You can now run the training script again.")
    else:
        print("\nFailed to apply some fixes.")
        print("Please check the error messages above.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
