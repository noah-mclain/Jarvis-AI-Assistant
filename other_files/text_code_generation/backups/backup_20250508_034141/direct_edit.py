#!/usr/bin/env python3
"""
Direct edit of existing files to fix Unsloth issues.
This script directly modifies the unified_deepseek_training.py file and custom_unsloth implementation.
"""

import os
import re
import shutil
from pathlib import Path

def fix_unified_deepseek():
    """Fix the unified_deepseek_training.py file"""
    
    # Path to the file
    file_path = "/notebooks/unified_deepseek_training.py"
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"❌ {file_path} not found")
        return False
    
    # Create a backup
    backup_path = f"{file_path}.bak"
    shutil.copy2(file_path, backup_path)
    print(f"Created backup at {backup_path}")
    
    # Read the file
    with open(file_path, "r") as f:
        content = f.read()
    
    # Fix 1: FastLanguageModel.from_pretrained call
    pattern1 = r'(\s+)model, tokenizer = FastLanguageModel\.from_pretrained\(\s*model_name=args\.model_name,\s*max_seq_length=args\.max_length,\s*load_in_4bit=args\.load_in_4bit,\s*load_in_8bit=args\.load_in_8bit(?:,\s*device_map="auto")?\s*\)'
    
    replacement1 = r'\1# Note: FastLanguageModel.from_pretrained already handles device_map internally\n\1model, tokenizer = FastLanguageModel.from_pretrained(\n\1    model_name=args.model_name,\n\1    load_in_4bit=args.load_in_4bit,\n\1    load_in_8bit=args.load_in_8bit\n\1)\n\n\1# Set max sequence length after model is loaded\n\1model.config.max_position_embeddings = args.max_length\n\1tokenizer.model_max_length = args.max_length'
    
    content = re.sub(pattern1, replacement1, content)
    
    # Fix 2: FastLanguageModel.get_peft_model call
    pattern2 = r'(\s+)model = FastLanguageModel\.get_peft_model\(\s*model,\s*r=16,\s*lora_alpha=32,\s*lora_dropout=0\.05,\s*target_modules=\[\s*"q_proj", "k_proj", "v_proj", "o_proj",\s*"gate_proj", "up_proj", "down_proj"\s*\],\s*(?:use_gradient_checkpointing=True,\s*)?(?:random_state=42,?\s*)?\)'
    
    replacement2 = r'\1# Note: use_gradient_checkpointing and random_state are not valid parameters for LoraConfig\n\1model = FastLanguageModel.get_peft_model(\n\1    model,\n\1    r=16,  # LoRA rank\n\1    lora_alpha=32,\n\1    lora_dropout=0.05,\n\1    target_modules=[\n\1        "q_proj", "k_proj", "v_proj", "o_proj",\n\1        "gate_proj", "up_proj", "down_proj"\n\1    ]\n\1)'
    
    content = re.sub(pattern2, replacement2, content)
    
    # Fix 3: TrainingArguments to enable gradient checkpointing
    pattern3 = r'(\s+)training_args = TrainingArguments\(\s*output_dir=args\.output_dir,\s*num_train_epochs=args\.epochs,\s*per_device_train_batch_size=args\.batch_size,\s*per_device_eval_batch_size=args\.batch_size,\s*gradient_accumulation_steps=args\.gradient_accumulation_steps,\s*learning_rate=args\.learning_rate,\s*weight_decay=0\.01,\s*warmup_steps=args\.warmup_steps,\s*logging_steps=10,\s*save_steps=100,\s*evaluation_strategy="steps",\s*eval_steps=100,\s*save_total_limit=3,\s*bf16=args\.bf16,\s*fp16=not args\.bf16 and torch\.cuda\.is_available\(\),\s*remove_unused_columns=False,\s*dataloader_num_workers=args\.num_workers,\s*dataloader_pin_memory=True,\s*group_by_length=True,?\s*\)'
    
    replacement3 = r'\1training_args = TrainingArguments(\n\1    output_dir=args.output_dir,\n\1    num_train_epochs=args.epochs,\n\1    per_device_train_batch_size=args.batch_size,\n\1    per_device_eval_batch_size=args.batch_size,\n\1    gradient_accumulation_steps=args.gradient_accumulation_steps,\n\1    learning_rate=args.learning_rate,\n\1    weight_decay=0.01,\n\1    warmup_steps=args.warmup_steps,\n\1    logging_steps=10,\n\1    save_steps=100,\n\1    evaluation_strategy="steps",\n\1    eval_steps=100,\n\1    save_total_limit=3,\n\1    bf16=args.bf16,\n\1    fp16=not args.bf16 and torch.cuda.is_available(),\n\1    remove_unused_columns=False,\n\1    dataloader_num_workers=args.num_workers,\n\1    dataloader_pin_memory=True,\n\1    group_by_length=True,\n\1    # Enable gradient checkpointing for memory efficiency\n\1    gradient_checkpointing=True\n\1)'
    
    content = re.sub(pattern3, replacement3, content)
    
    # Write the fixed content back to the file
    with open(file_path, "w") as f:
        f.write(content)
    
    print(f"✅ Successfully fixed {file_path}")
    return True

def fix_custom_unsloth():
    """Fix the custom_unsloth implementation"""
    
    # Path to the file
    models_init_path = "/notebooks/custom_unsloth/unsloth/models/__init__.py"
    
    # Check if the file exists
    if not os.path.exists(models_init_path):
        print(f"❌ {models_init_path} not found")
        return False
    
    # Create a backup
    backup_path = f"{models_init_path}.bak"
    shutil.copy2(models_init_path, backup_path)
    print(f"Created backup at {backup_path}")
    
    # Read the file
    with open(models_init_path, "r") as f:
        content = f.read()
    
    # Fix 1: Remove device_map from AutoModelForCausalLM.from_pretrained
    pattern1 = r'(\s+)model = AutoModelForCausalLM\.from_pretrained\(\s*([^,]+),\s*([^)]+)\)'
    
    def process_match1(match):
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
    
    content = re.sub(pattern1, process_match1, content, flags=re.DOTALL)
    
    # Fix 2: Remove use_gradient_checkpointing and random_state from get_peft_model
    pattern2 = r'def get_peft_model\(model, ([^)]+)\):'
    match2 = re.search(pattern2, content, re.DOTALL)
    if match2:
        params = match2.group(1)
        # Remove use_gradient_checkpointing and random_state from params
        params_lines = params.split('\n')
        filtered_params = []
        for line in params_lines:
            if 'use_gradient_checkpointing=' not in line and 'random_state=' not in line:
                filtered_params.append(line)
        
        # Join the filtered params
        filtered_params_str = '\n'.join(filtered_params)
        
        # Replace the function signature
        content = re.sub(pattern2, f"def get_peft_model(model, {filtered_params_str}):\n    # use_gradient_checkpointing and random_state are handled by TrainingArguments", content, flags=re.DOTALL)
    
    # Fix 3: Remove use_gradient_checkpointing and random_state from LoraConfig
    pattern3 = r'(\s+)peft_config = LoraConfig\(\s*([^)]+)\)'
    match3 = re.search(pattern3, content, re.DOTALL)
    if match3:
        indent = match3.group(1)
        params = match3.group(2)
        
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
        content = re.sub(pattern3, f"{comment}{indent}peft_config = LoraConfig(\n{indent}    {filtered_params_str})", content, flags=re.DOTALL)
    
    # Write the fixed content back to the file
    with open(models_init_path, "w") as f:
        f.write(content)
    
    print(f"✅ Successfully fixed {models_init_path}")
    return True

def main():
    """Main function"""
    print("=" * 50)
    print("DIRECTLY EDITING FILES TO FIX UNSLOTH ISSUES")
    print("=" * 50)
    
    # Fix the unified_deepseek_training.py file
    unified_success = fix_unified_deepseek()
    
    # Fix the custom_unsloth implementation
    custom_success = fix_custom_unsloth()
    
    if unified_success or custom_success:
        print("\nEdits applied successfully!")
        print("You can now run the training script again.")
    else:
        print("\nNo files were edited.")
        print("Please check the error messages above.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
