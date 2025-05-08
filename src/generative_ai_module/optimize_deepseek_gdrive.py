#!/usr/bin/env python3
"""
DeepSeek-Coder Fine-tuning with Google Drive Integration for Paperspace Gradient

This script optimizes DeepSeek-Coder fine-tuning to work within Paperspace Gradient's
15GB storage limits by offloading models and checkpoints to Google Drive.

Usage:
    python optimize_deepseek_gdrive.py \
        --gdrive-folder-id YOUR_FOLDER_ID \
        --output-dir models/deepseek_optimized \
        --quantize 4 \
        --max-steps 500 \
        --batch-size 4
"""

import os
import sys
import argparse
import json
import time
import tempfile
import torch
from pathlib import Path

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, parent_dir)

# Import Google Drive integration
from src.generative_ai_module.google_drive_storage import (
    GoogleDriveStorage,
    CheckpointStrategy,
    get_gdrive_checkpoint_callback
)

# Check for unsloth availability
try:
    import unsloth
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="DeepSeek-Coder Fine-tuning with Google Drive Integration")

    # Google Drive configuration
    parser.add_argument("--gdrive-folder-id", type=str, required=True,
                      help="Google Drive folder ID for storing model files")
    parser.add_argument("--service-account-file", type=str, default=None,
                      help="Path to Google service account credentials file (for headless auth)")

    # Model configuration
    parser.add_argument("--model-name", type=str, default="deepseek-ai/deepseek-coder-6.7b-base",
                      help="Base model name to fine-tune")
    parser.add_argument("--output-dir", type=str, default="models/deepseek_optimized",
                      help="Local directory for temporary model output")
    parser.add_argument("--quantize", type=int, choices=[0, 4, 8], default=4,
                      help="Quantization level: 0=None, 4=4-bit, 8=8-bit")

    # Training parameters
    parser.add_argument("--batch-size", type=int, default=4,
                      help="Training batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                      help="Gradient accumulation steps")
    parser.add_argument("--max-steps", type=int, default=1000,
                      help="Maximum training steps")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                      help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=2048,
                      help="Maximum sequence length")

    # Dataset options
    parser.add_argument("--max-samples", type=int, default=5000,
                      help="Maximum number of samples to use for training")
    parser.add_argument("--use-mini-dataset", action="store_true",
                      help="Use mini dataset for quick testing")

    # Checkpoint configuration
    parser.add_argument("--checkpoint-strategy", type=str,
                      choices=["improvement", "regular", "hybrid", "all"],
                      default="improvement",
                      help="Strategy for saving checkpoints")
    parser.add_argument("--max-checkpoints", type=int, default=3,
                      help="Maximum number of checkpoints to keep")

    # Advanced options
    parser.add_argument("--eval-steps", type=int, default=100,
                      help="Steps between evaluations")
    parser.add_argument("--save-steps", type=int, default=100,
                      help="Steps between saving checkpoints")

    return parser.parse_args()

def setup_google_drive(args):
    """Set up Google Drive storage"""
    print("\n=== Setting up Google Drive integration ===")

    try:
        gdrive = GoogleDriveStorage(
            folder_id=args.gdrive_folder_id,
            service_account_file=args.service_account_file,
            create_subfolders=True
        )
        print("Google Drive integration initialized successfully")
        return gdrive
    except Exception as e:
        print(f"Error setting up Google Drive integration: {e}")

        # Ask user if they want to continue without Google Drive
        if input("Continue without Google Drive integration? (y/n): ").lower() != "y":
            sys.exit(1)
        return None

def prepare_dataset(args):
    """Prepare dataset for fine-tuning"""
    print("\n=== Preparing dataset ===")

    if args.use_mini_dataset:
        from src.generative_ai_module.finetune_deepseek import create_mini_dataset
        train_dataset, eval_dataset = create_mini_dataset(args.max_seq_length)

        # Convert tokenized dataset to text format for Unsloth if needed
        if UNSLOTH_AVAILABLE:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

            # Create text dataset from tokenized dataset
            def convert_to_text(dataset):
                texts = []
                for item in dataset:
                    input_ids = item["input_ids"]
                    # Convert tensor to list if needed
                    if hasattr(input_ids, "cpu"):
                        input_ids = input_ids.cpu().tolist()
                    text = tokenizer.decode(input_ids)
                    texts.append({"text": text})
                return texts

            train_texts = convert_to_text(train_dataset)
            eval_texts = convert_to_text(eval_dataset)

            # Import Dataset class
            from datasets import Dataset
            train_dataset = Dataset.from_list(train_texts)
            eval_dataset = Dataset.from_list(eval_texts)
    else:
        from src.generative_ai_module.code_preprocessing import load_and_preprocess_dataset

        train_dataset, eval_dataset = load_and_preprocess_dataset(
            max_samples=args.max_samples,
            sequence_length=args.max_seq_length,
            subset="python",  # Using Python subset for reliability
            all_subsets=False,  # Avoid using all subsets to conserve memory
            return_raw=True  # Get raw text instead of tokenized for Unsloth
        )

    print(f"Dataset prepared: {len(train_dataset)} training samples, {len(eval_dataset)} evaluation samples")
    return train_dataset, eval_dataset

def fine_tune_with_unsloth(args, train_dataset, eval_dataset, gdrive=None):
    """Fine-tune using Unsloth with Google Drive integration"""
    if not UNSLOTH_AVAILABLE:
        print("Error: Unsloth is required for optimized fine-tuning")
        print("Install with: pip install unsloth")
        return None

    print("\n=== Starting fine-tuning with Unsloth ===")

    # Configure quantization
    load_in_4bit = args.quantize == 4
    load_in_8bit = args.quantize == 8

    if load_in_4bit:
        print("Using 4-bit quantization (87% reduced model size)")
    elif load_in_8bit:
        print("Using 8-bit quantization (75% reduced model size)")
    else:
        print("Using full precision model (no quantization)")

    # Load model with Unsloth
    print(f"Loading model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit
        # Don't set trust_remote_code here as it's already set by FastLanguageModel internally
    )

    # Set up LoRA
    print("Configuring LoRA adapters")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        use_gradient_checkpointing=True,
        random_state=42,
    )

    # Format dataset for Unsloth
    def format_instruction(sample):
        """Format data as instruction tuning pairs"""
        # We need to combine all the text in the dataset into instruction/response pairs
        # The deepseek-coder model uses this format for instruction tuning

        # If the sample is already in the right format, return it as is
        if isinstance(sample, dict) and "text" in sample:
            text = sample["text"]

            # Check if it already has instruction/response format
            if "### Instruction:" in text and "### Response:" in text:
                return {"text": text}

            # Otherwise, create a synthetic instruction/response pair
            # Extract the first 1/3 as "instruction" and the rest as "response"
            split_point = len(text) // 3
            instruction = text[:split_point]
            response = text[split_point:]

            # Format with deepseek-coder instruction format
            return {
                "text": f"### Instruction: {instruction}\n\n### Response: {response}"
            }

        # If the sample has instruction and response fields, format accordingly
        if isinstance(sample, dict) and "instruction" in sample and "response" in sample:
            return {
                "text": f"### Instruction: {sample['instruction']}\n\n### Response: {sample['response']}"
            }

        # Default case - just return the sample as is
        return sample

    print("Formatting datasets for instruction tuning")
    train_dataset = train_dataset.map(format_instruction)
    eval_dataset = eval_dataset.map(format_instruction)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up Google Drive checkpoint callback if available
    callbacks = []
    if gdrive:
        print("Setting up Google Drive checkpoint integration")
        checkpoint_strategy = CheckpointStrategy(args.checkpoint_strategy)
        gdrive_callback = get_gdrive_checkpoint_callback(
            gdrive,
            strategy=checkpoint_strategy,
            max_checkpoints=args.max_checkpoints
        )
        callbacks.append(gdrive_callback)

    # Start training
    print(f"Starting training with batch size {args.batch_size}, "
          f"gradient accumulation {args.gradient_accumulation_steps}")

    start_time = time.time()
    trainer = FastLanguageModel.get_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_task="CAUSAL_LM",
        args=dict(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=min(100, int(args.max_steps * 0.1)),  # 10% of total steps
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            fp16=not load_in_4bit and not load_in_8bit,  # Only use fp16 for full precision
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=1,  # Only keep one local checkpoint
            load_best_model_at_end=True,
            report_to="none",  # Disable wandb/tensorboard
            ddp_find_unused_parameters=False,
            group_by_length=True,
        ),
        callbacks=callbacks,
        force_download_tokenizer=True,
    )

    # Train the model
    trainer.train()

    # Calculate training time
    training_time = time.time() - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")

    # Save final model and tokenizer
    print("\nSaving model locally...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Upload the final model to Google Drive if available
    if gdrive:
        print("\nUploading final model to Google Drive...")
        file_id = gdrive.upload_model(args.output_dir, f"deepseek_ft_{time.strftime('%Y%m%d_%H%M%S')}")
        if file_id:
            print(f"Model uploaded successfully to Google Drive")

    return trainer, model, tokenizer

def main():
    """Main function to run the script"""
    args = parse_args()

    print("\n" + "="*80)
    print("DeepSeek-Coder Fine-tuning with Google Drive Integration")
    print("Optimized for Paperspace Gradient")
    print("="*80 + "\n")

    # Set up Google Drive integration
    gdrive = setup_google_drive(args)

    # Prepare dataset
    train_dataset, eval_dataset = prepare_dataset(args)

    # Fine-tune with Unsloth
    trainer, model, tokenizer = fine_tune_with_unsloth(args, train_dataset, eval_dataset, gdrive)

    # Test the model with a simple prompt
    print("\n=== Testing fine-tuned model ===")
    test_prompt = "### Instruction: Write a function to calculate factorial of a number.\n\n### Response:"

    # Generate with fine-tuned model
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\nSample generated code:")
    print("-" * 40)
    print(generated_text)
    print("-" * 40)

    print("\nFine-tuning complete! Your model is saved both locally and on Google Drive.")

    # Display Google Drive info
    if gdrive and gdrive.folder_id:
        print(f"\nYour model is stored in Google Drive. Access it at:")
        print(f"https://drive.google.com/drive/folders/{gdrive.folder_id}")

    # Final advice for Paperspace Gradient users
    if os.path.exists("/storage"):
        print("\n=== Important for Paperspace Gradient ===")
        print("To avoid storage issues on your next run:")
        print("1. Use the same Google Drive folder ID")
        print("2. Clean up local files in /storage/models with 'rm -rf /storage/models/*'")
        print("3. When loading your model later, use the Google Drive integration to download it")

if __name__ == "__main__":
    main()