#!/usr/bin/env python3
"""
Fix for dataset processing issues in unified_deepseek_training.py.

This script fixes the 'Unable to create tensor' error by modifying how datasets are processed.
"""

import os
import re
import shutil
from pathlib import Path

def fix_dataset_processing():
    """Fix the dataset processing in unified_deepseek_training.py"""
    
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
    
    # Add a function to clean and prepare the dataset
    # Find the train_with_unsloth function
    pattern = r'def train_with_unsloth\(args\):'
    match = re.search(pattern, content)
    
    if not match:
        print(f"Error: train_with_unsloth function not found in {file_path}")
        return False
    
    # Add the clean_dataset function before train_with_unsloth
    insertion_point = match.start()
    
    clean_dataset_function = """
def clean_dataset(dataset, text_field="text"):
    """Clean and prepare dataset for training"""
    def clean_example(example):
        # Ensure text is a string
        if text_field in example:
            if not isinstance(example[text_field], str):
                example[text_field] = str(example[text_field])
            
            # Remove any problematic characters or patterns
            # This is a basic cleaning - you may need to customize based on your data
            example[text_field] = example[text_field].replace("\\r", "\\n")
            
            # Ensure the text is not empty
            if not example[text_field].strip():
                example[text_field] = "Empty text"
        
        return example
    
    # Apply cleaning to each example
    return dataset.map(clean_example)

"""
    
    # Insert the clean_dataset function
    content = content[:insertion_point] + clean_dataset_function + content[insertion_point:]
    
    # Modify the dataset loading code to clean the datasets
    pattern = r'(\s+)(logger\.info\(f"Train dataset size: \{len\(train_dataset\)\}"\)\s*logger\.info\(f"Eval dataset size: \{len\(eval_dataset\)\}"\))'
    
    replacement = r'\1# Clean and prepare datasets\n\1train_dataset = clean_dataset(train_dataset)\n\1eval_dataset = clean_dataset(eval_dataset)\n\n\1\2'
    
    content = re.sub(pattern, replacement, content)
    
    # Modify the tokenize_function to handle potential issues
    pattern = r'(\s+)def tokenize_function\(examples\):\s*return tokenizer\(\s*examples\["text"\],\s*truncation=True,\s*padding="max_length",\s*max_length=args\.max_length,\s*return_tensors="pt"\s*\)'
    
    replacement = r'\1def tokenize_function(examples):\n\1    """Tokenize examples with proper handling of potential issues"""\n\1    # Ensure all texts are strings\n\1    texts = [str(text) if not isinstance(text, str) else text for text in examples["text"]]\n\1    \n\1    # Tokenize without return_tensors to avoid the "too many dimensions" error\n\1    return tokenizer(\n\1        texts,\n\1        truncation=True,\n\1        padding="max_length",\n\1        max_length=args.max_length,\n\1        return_tensors=None\n\1    )'
    
    content = re.sub(pattern, replacement, content)
    
    # Update the data collator
    pattern = r'(\s+)# Create data collator\s*data_collator = DataCollatorForLanguageModeling\(\s*tokenizer=tokenizer,\s*mlm=False\s*\)'
    
    replacement = r'\1# Create a custom data collator that handles potential issues\n\1class SafeDataCollator(DataCollatorForLanguageModeling):\n\1    def __call__(self, features):\n\1        try:\n\1            # Try the standard collation\n\1            return super().__call__(features)\n\1        except ValueError as e:\n\1            # If there's an error, log it and try a more robust approach\n\1            logger.warning(f"Data collation error: {e}")\n\1            \n\1            # Convert all features to the same format\n\1            batch = {}\n\1            for key in features[0].keys():\n\1                if key in ["input_ids", "attention_mask", "labels"]:\n\1                    batch[key] = []\n\1                    for feature in features:\n\1                        # Ensure the feature is a list\n\1                        if isinstance(feature[key], list):\n\1                            batch[key].append(feature[key])\n\1                        else:\n\1                            batch[key].append([feature[key]])\n\1            \n\1            # Pad the sequences\n\1            return self.tokenizer.pad(batch, return_tensors="pt")\n\1\n\1# Create data collator with custom handling\n\1data_collator = SafeDataCollator(\n\1    tokenizer=tokenizer,\n\1    mlm=False\n\1)'
    
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
    
    success = fix_dataset_processing()
    
    if success:
        print("\nDataset processing fix applied successfully!")
        print("You can now run the training script again.")
    else:
        print("\nFailed to apply dataset processing fix.")
        print("Please check the error messages above.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
