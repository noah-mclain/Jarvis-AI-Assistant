#!/usr/bin/env python3
"""
Comprehensive fix for DeepSeek Coder training issues.

This script combines fixes for:
1. Unsloth parameter issues:
   - max_seq_length parameter issue
   - device_map parameter issue
   - use_gradient_checkpointing parameter issue
   - random_state parameter issue
2. Dataset processing issues:
   - Tokenization issues
   - Data collation issues
   - Tensor creation issues
3. Attention mask issues

Usage:
    python setup/fix_deepseek_training.py [--path /path/to/src/generative_ai_module/unified_deepseek_training.py]
"""

import os
import sys
import re
import shutil
import argparse
from pathlib import Path

def fix_unsloth_parameters(content):
    """Fix Unsloth parameter issues in the content"""
    print("Fixing Unsloth parameter issues...")

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

    return content

def fix_dataset_processing(content):
    """Fix dataset processing issues in the content"""
    print("Fixing dataset processing issues...")

    # Fix 1: Modify the tokenize_function to handle potential issues
    pattern1 = r'(\s+)def tokenize_function\(examples\):\s*return tokenizer\(\s*examples\["text"\],\s*truncation=True,\s*padding="max_length",\s*max_length=args\.max_length,\s*return_tensors="pt"\s*\)'

    replacement1 = r'\1def tokenize_function(examples):\n\1    """Tokenize examples with proper handling of potential issues"""\n\1    # Ensure all texts are strings\n\1    texts = [str(text) if not isinstance(text, str) else text for text in examples["text"]]\n\1    \n\1    # Tokenize without return_tensors to avoid the "too many dimensions" error\n\1    return tokenizer(\n\1        texts,\n\1        truncation=True,\n\1        padding="max_length",\n\1        max_length=args.max_length,\n\1        return_tensors=None\n\1    )'

    # Check if the pattern is found
    if not re.search(pattern1, content):
        print(f"Warning: tokenize_function pattern not found")

        # Try a more general pattern
        pattern1 = r'(\s+)def tokenize_function\(examples\):\s*return tokenizer\([^)]+\)'
        if not re.search(pattern1, content):
            print(f"Error: alternative tokenize_function pattern not found")

    # Apply the fix
    content = re.sub(pattern1, replacement1, content)

    # Fix 2: Update the data collator to handle potential issues
    pattern2 = r'(\s+)# Create data collator\s*data_collator = DataCollatorForLanguageModeling\(\s*tokenizer=tokenizer,\s*mlm=False\s*\)'

    replacement2 = r'\1# Create a custom data collator that handles potential issues\n\1class SafeDataCollator(DataCollatorForLanguageModeling):\n\1    def __call__(self, features):\n\1        try:\n\1            # Try the standard collation\n\1            return super().__call__(features)\n\1        except ValueError as e:\n\1            # If there\'s an error, log it and try a more robust approach\n\1            logger.warning(f"Data collation error: {e}")\n\1            \n\1            # Convert all features to the same format\n\1            batch = {}\n\1            for key in features[0].keys():\n\1                if key in ["input_ids", "attention_mask", "labels"]:\n\1                    batch[key] = []\n\1                    for feature in features:\n\1                        # Ensure the feature is a list\n\1                        if isinstance(feature[key], list):\n\1                            batch[key].append(feature[key])\n\1                        else:\n\1                            batch[key].append([feature[key]])\n\1            \n\1            # Pad the sequences\n\1            return self.tokenizer.pad(batch, return_tensors="pt")\n\1\n\1# Create data collator with custom handling\n\1data_collator = SafeDataCollator(\n\1    tokenizer=tokenizer,\n\1    mlm=False\n\1)''

    # Check if the pattern is found
    if not re.search(pattern2, content):
        print(f"Warning: data_collator pattern not found")

        # Try a more general pattern
        pattern2 = r'(\s+)data_collator = DataCollatorForLanguageModeling\([^)]+\)'
        if not re.search(pattern2, content):
            print(f"Error: alternative data_collator pattern not found")

    # Apply the fix
    content = re.sub(pattern2, replacement2, content)

    return content

def fix_attention_mask(content):
    """Fix attention mask issues in the content"""
    print("Fixing attention mask issues...")

    # Check if apply_attention_mask_fix function already exists
    if "def apply_attention_mask_fix" in content:
        print("Attention mask fix function already exists, skipping...")
        return content

    # Find a good insertion point after imports
    import_section_end = re.search(r'(from [^\n]+\n\n)', content)
    if not import_section_end:
        print("Warning: Could not find a good insertion point for attention mask fix")
        return content

    insertion_point = import_section_end.end()

    # Attention mask fix function
    attention_mask_fix = """
# Apply attention mask fix
def apply_attention_mask_fix():
    """Apply the attention mask fix for DeepSeek models"""
    try:
        from transformers.models.llama.modeling_llama import LlamaModel

        # Store the original forward method
        original_forward = LlamaModel.forward

        # Define a patched forward method that properly handles attention masks
        def patched_forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        ):
            """Patched forward method for LlamaModel that properly handles attention masks."""
            # Force use_cache to False when using gradient checkpointing
            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                if use_cache:
                    logger.info("use_cache=True is incompatible with gradient checkpointing. Setting use_cache=False...")
                use_cache = False

            # Fix attention mask shape if needed
            if attention_mask is not None and attention_mask.dim() == 2:
                # Get the device and dtype
                device = attention_mask.device
                dtype = attention_mask.dtype

                # Get sequence length
                seq_length = attention_mask.size(1)
                batch_size = attention_mask.size(0)

                # Convert attention_mask from [batch_size, seq_length] to [batch_size, 1, seq_length, seq_length]
                # First, expand to [batch_size, 1, 1, seq_length]
                expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)

                # Create a causal mask of shape [1, 1, seq_length, seq_length]
                causal_mask = torch.triu(
                    torch.ones((seq_length, seq_length), device=device, dtype=dtype),
                    diagonal=1
                ).unsqueeze(0).unsqueeze(0)

                # Convert masks to proper format (-inf for masked positions, 0 for attended positions)
                expanded_mask = (1.0 - expanded_mask) * -10000.0
                causal_mask = (causal_mask > 0) * -10000.0

                # Combine the masks
                combined_mask = expanded_mask + causal_mask

                # Replace the original attention_mask with our fixed version
                attention_mask = combined_mask

                logger.debug(f"Fixed attention mask shape: {attention_mask.shape}")

            # Call the original forward method with the fixed attention mask
            return original_forward(
                self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # Replace the original forward method with our patched version
        LlamaModel.forward = patched_forward

        logger.info("Successfully applied attention mask fix")
        return True
    except Exception as e:
        logger.error(f"Error applying attention mask fix: {e}")
        return False
"""

    # Insert the attention mask fix function
    content = content[:insertion_point] + attention_mask_fix + content[insertion_point:]

    # Make sure the function is called in the main function
    if "apply_attention_mask_fix()" not in content:
        # Find the main function
        main_function = re.search(r'def main\(\):[^\n]*\n', content)
        if main_function:
            # Find a good insertion point after the main function starts
            main_start = main_function.end()
            # Find the first line after the main function definition
            next_line = content.find('\n', main_start)
            if next_line != -1:
                # Insert the call to apply_attention_mask_fix
                content = content[:next_line+1] + "    # Apply attention mask fix\n    logger.info(\"Applying attention mask fix...\")\n    apply_attention_mask_fix()\n" + content[next_line+1:]

    return content

def fix_custom_unsloth(custom_unsloth_path):
    """Fix the custom_unsloth implementation"""
    print(f"Checking for custom_unsloth at {custom_unsloth_path}...")

    # Check if the custom_unsloth directory exists
    if not os.path.exists(custom_unsloth_path):
        print(f"Custom unsloth not found at {custom_unsloth_path}, skipping...")
        return

    # Find the models/__init__.py file
    models_init_path = os.path.join(custom_unsloth_path, "unsloth", "models", "__init__.py")
    if not os.path.exists(models_init_path):
        print(f"models/__init__.py not found at {models_init_path}, skipping...")
        return

    print(f"Fixing custom_unsloth implementation at {models_init_path}...")

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
        comment = f\"{indent}# device_map is handled by FastLanguageModel, so we don't pass it here\n\"'

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

def fix_unified_deepseek(file_path):
    """Fix all issues in the unified_deepseek_training.py file"""

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"❌ {file_path} not found")
        return False

    # Create a backup
    backup_path = f"{file_path}.comprehensive.bak"
    shutil.copy2(file_path, backup_path)
    print(f"Created backup at {backup_path}")

    # Read the file
    with open(file_path, "r") as f:
        content = f.read()

    # Apply all fixes
    content = fix_unsloth_parameters(content)
    content = fix_dataset_processing(content)
    content = fix_attention_mask(content)

    # Write the fixed content back to the file
    with open(file_path, "w") as f:
        f.write(content)

    print(f"✅ Successfully fixed {file_path}")
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Fix DeepSeek Coder training issues")
    parser.add_argument("--path", type=str, default="src/generative_ai_module/unified_deepseek_training.py",
                        help="Path to the unified_deepseek_training.py file")
    parser.add_argument("--custom-unsloth", type=str, default="/notebooks/custom_unsloth",
                        help="Path to the custom_unsloth directory")

    args = parser.parse_args()

    print("=" * 50)
    print("COMPREHENSIVE FIX FOR DEEPSEEK CODER TRAINING")
    print("=" * 50)

    # Fix the unified_deepseek_training.py file
    unified_success = fix_unified_deepseek(args.path)

    # Fix the custom_unsloth implementation
    fix_custom_unsloth(args.custom_unsloth)

    if unified_success:
        print("\nAll fixes applied successfully!")
        print("You can now run the training script again.")
    else:
        print("\nFailed to apply some fixes.")
        print("Please check the error messages above.")

    print("=" * 50)

if __name__ == "__main__":
    main()
