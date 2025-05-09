#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Jarvis AI Assistant - Model Fine-Tuning
This script handles the fine-tuning of large language models.
Optimized for Google Colab with A100 GPU using both Unsloth and bitsandbytes.
"""

import os
import sys
import torch
import argparse
import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import time
import json
from datetime import datetime
from .utils import get_storage_path, ensure_directory_exists, sync_to_gdrive, sync_from_gdrive, is_paperspace_environment, setup_logging, sync_logs

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("finetune_output.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("jarvis-finetune")

# Check for CUDA
if torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA Version: {torch.version.cuda}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    logger.info(f"BF16 support: {torch.cuda.is_bf16_supported()}")
else:
    device = torch.device("cpu")
    logger.warning("CUDA not available, using CPU. Fine-tuning will be extremely slow.")

# Try to import the necessary modules with helpful error messages
try:
    from datasets import load_dataset, Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import bitsandbytes as bnb
    from unsloth import FastLanguageModel
    logger.info(f"Unsloth version: {FastLanguageModel.__version__ if hasattr(FastLanguageModel, '__version__') else 'Unknown'}")
except ImportError as e:
    logger.error(f"Error importing required modules: {e}")
    logger.error("Please run the setup script first: !bash colab_setup.sh")
    sys.exit(1)

def format_training_examples(examples: dict) -> List[str]:
    """Format training examples for instruction fine-tuning"""
    formatted_examples = []
    
    for i in range(len(examples["input"])):
        instruction = examples["instruction"][i]
        input_text = examples["input"][i]
        output = examples["output"][i]
        
        # Format based on whether input is provided or not
        if input_text.strip():
            formatted_text = f"USER: {instruction}\n{input_text}\nASSISTANT: {output}"
        else:
            formatted_text = f"USER: {instruction}\nASSISTANT: {output}"
            
        formatted_examples.append(formatted_text)
    
    return formatted_examples

def create_dataset_from_json(json_file: str) -> Dataset:
    """Create a dataset from a JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Convert to the expected format if necessary
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        if "instruction" in data[0] and "output" in data[0]:
            # It's already in the expected format
            return Dataset.from_list(data)
        else:
            # Need to transform the data
            transformed_data = []
            for item in data:
                transformed_item = {
                    "instruction": item.get("prompt", ""),
                    "input": item.get("context", ""),
                    "output": item.get("response", "")
                }
                transformed_data.append(transformed_item)
            return Dataset.from_list(transformed_data)
    else:
        raise ValueError("JSON file format not recognized")

def load_training_data(
    dataset_name: Optional[str] = None,
    dataset_config: Optional[str] = None,
    json_file: Optional[str] = None,
    split: str = "train",
    max_samples: Optional[int] = None
) -> Dataset:
    """Load the training dataset from HuggingFace or local JSON file"""
    if json_file:
        logger.info(f"Loading dataset from JSON file: {json_file}")
        dataset = create_dataset_from_json(json_file)
    elif dataset_name:
        logger.info(f"Loading dataset {dataset_name} from HuggingFace")
        if dataset_config:
            dataset = load_dataset(dataset_name, dataset_config, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
    else:
        raise ValueError("Either dataset_name or json_file must be provided")
    
    # Limit samples if specified
    if max_samples and len(dataset) > max_samples:
        logger.info(f"Limiting dataset to {max_samples} samples (from {len(dataset)})")
        dataset = dataset.select(range(max_samples))
    
    return dataset

def preprocess_and_tokenize(
    dataset: Dataset,
    tokenizer,
    max_length: int = 2048,
    text_column: str = "text"
) -> Dataset:
    """Preprocess and tokenize the dataset"""
    
    # Check if the dataset needs formatting
    if all(col in dataset.column_names for col in ["instruction", "input", "output"]):
        logger.info("Formatting dataset for instruction tuning")
        formatted_dataset = dataset.map(
            lambda examples: {"text": format_training_examples(examples)},
            batched=True,
            remove_columns=dataset.column_names
        )
    elif text_column in dataset.column_names:
        logger.info(f"Using existing '{text_column}' column")
        formatted_dataset = dataset
    else:
        raise ValueError(f"Dataset must contain either 'instruction'/'input'/'output' columns or a '{text_column}' column")
    
    # Tokenize the dataset
    logger.info("Tokenizing dataset")
    
    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
    
    tokenized_dataset = formatted_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[text_column]
    )
    
    return tokenized_dataset

def create_mini_dataset(sequence_length=2048, num_samples=10):
    """Create a mini dataset for testing"""
    logger.info(f"Creating mini dataset with {num_samples} samples")
    
    # Example data for code generation
    instructions = [
        "Write a Python function to calculate Fibonacci numbers recursively",
        "Create a function to sort a list using quicksort algorithm",
        "Implement a binary search tree in Python",
        "Write a function to check if a string is a palindrome",
        "Create a Python class for a simple banking system",
        "Write a function to find the longest common substring of two strings",
        "Implement a priority queue in Python",
        "Create a decorator that measures the execution time of a function",
        "Write a Python function to flatten a nested list",
        "Implement a simple HTTP server in Python"
    ]
    
    # Create minimal sample dataset
    data = []
    for i in range(min(num_samples, len(instructions))):
        data.append({
            "instruction": instructions[i],
            "input": "",
            "output": f"# Example output for: {instructions[i]}\ndef example_function():\n    pass"
        })
    
    # Add more samples if needed by repeating instructions
    if num_samples > len(instructions):
        for i in range(len(instructions), num_samples):
            idx = i % len(instructions)
            data.append({
                "instruction": f"Variation of: {instructions[idx]}",
                "input": "",
                "output": f"# Example output for variation\ndef example_function_{i}():\n    pass"
            })
    
    return Dataset.from_list(data)

def main():
    """
    Main function to run the finetuning process.
    """
    parser = argparse.ArgumentParser(description="Run finetuning for LLMs with customizable parameters")
    
    # Model and dataset parameters
    parser.add_argument("--model_name", type=str, default="deepseek-ai/deepseek-coder-6.7b-base", 
                        help="Name of the base model to finetune")
    parser.add_argument("--dataset_path", type=str, default=None, 
                        help="Path to the dataset to use for finetuning")
    parser.add_argument("--dataset_name", type=str, default=None, 
                        help="Name of the dataset on HuggingFace Hub to use for finetuning")
    
    # Output directory
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="Directory to save the finetuned model")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=3, 
                        help="Number of epochs to train for")
    
    # Model arguments
    parser.add_argument("--load-in-4bit", action="store_true", default=True,
                        help="Load model in 4-bit precision (recommended for A100 GPUs)")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Load model in 8-bit precision (alternative to 4-bit)")
    
    # Dataset arguments
    parser.add_argument("--max-seq-length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to use (for memory constraints)")
    parser.add_argument("--use-mini-dataset", action="store_true",
                        help="Use a small synthetic dataset for testing")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size per device")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--use-unsloth", action="store_true", default=True,
                        help="Use Unsloth for faster training")
    parser.add_argument("--flash-attention", action="store_true", default=True,
                        help="Use Flash Attention for faster training on A100")
    
    # LoRA arguments
    parser.add_argument("--lora-r", type=int, default=64,
                        help="LoRA rank parameter")
    parser.add_argument("--lora-alpha", type=int, default=16,
                        help="LoRA alpha parameter")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="LoRA dropout probability")
    
    # A100 optimization arguments
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use bfloat16 precision (recommended for A100)")
    parser.add_argument("--optim", type=str, default="adamw_torch_fused",
                        help="Optimizer to use")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"finetune_{timestamp}.log"
    setup_logging(log_file)
    
    logger.info("Starting fine-tuning process")
    
    # First, sync from Google Drive to get latest datasets and checkpoints if we're in Paperspace
    if is_paperspace_environment():
        logger.info("Running in Paperspace environment, syncing from Google Drive...")
        # Ensure our directories exist by syncing from Google Drive first
        sync_from_gdrive("datasets")
        sync_from_gdrive("checkpoints")
        sync_from_gdrive("models")
        logger.info("Synced latest datasets, checkpoints, and models from Google Drive")
    
    # If output_dir is not specified, use the storage path utility
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_short = args.model_name.split('/')[-1]
        args.output_dir = get_storage_path("models", f"{model_name_short}_finetuned_{timestamp}")
    
    # Ensure output directory exists
    ensure_directory_exists("models", os.path.basename(args.output_dir))
    
    # Check for A100 GPU and set optimal parameters
    is_a100 = False
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0).lower()
        is_a100 = "a100" in gpu_name
        if is_a100:
            logger.info("A100 GPU detected - applying optimal settings")
            # Export environment variables for optimal A100 performance
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
            os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
            os.environ["TOKENIZERS_PARALLELISM"] = "true"
        else:
            logger.info(f"Using GPU: {gpu_name} (not an A100)")
    
    # Log settings
    logger.info(f"Starting fine-tuning with settings: {args}")
    
    # Load tokenizer and model
    logger.info(f"Loading tokenizer from {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Set quantization config
    if args.load_in_4bit:
        logger.info("Using 4-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    elif args.load_in_8bit:
        logger.info("Using 8-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    else:
        quantization_config = None
    
    # Load dataset
    if args.use_mini_dataset:
        logger.info("Using mini dataset for testing")
        dataset = create_mini_dataset(sequence_length=args.max_seq_length)
    else:
        dataset = load_training_data(
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_path,
            max_samples=args.max_samples
        )
    
    # Load model with Unsloth optimization (if requested)
    if args.use_unsloth:
        logger.info(f"Loading model from {args.model_name} with Unsloth optimizations")
        # Use Unsloth's FastLanguageModel for optimized training
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_name,
            max_seq_length=args.max_seq_length,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            use_flash_attention_2=args.flash_attention and is_a100
        )
        
        # Apply LoRA with Unsloth optimization
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = FastLanguageModel.get_peft_model(
            model,
            lora_config,
            use_gradient_checkpointing=True
        )
        logger.info("Applied LoRA with Unsloth optimizations")
    else:
        logger.info(f"Loading model from {args.model_name} with standard HF approach")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            trust_remote_code=True,
            use_flash_attention_2=args.flash_attention and is_a100
        )
        
        # Enable gradient checkpointing to save memory
        model.gradient_checkpointing_enable()
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # Apply LoRA
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
    
    # Preprocess and tokenize dataset
    tokenized_dataset = preprocess_and_tokenize(
        dataset,
        tokenizer,
        max_length=args.max_seq_length
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Calculate optimal steps and warmup
    steps_per_epoch = max(1, len(tokenized_dataset) // args.batch_size)
    max_steps = args.epochs * steps_per_epoch
    warmup_steps = min(100, int(max_steps * 0.1))  # 10% of total steps, capped at 100
    
    logger.info(f"Training for {args.epochs} epochs, {steps_per_epoch} steps per epoch, {max_steps} total steps")
    logger.info(f"Using warmup for {warmup_steps} steps")
    
    # Setup training arguments optimized for A100
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
        learning_rate=args.learning_rate,
        max_grad_norm=0.3,
        warmup_steps=warmup_steps,
        lr_scheduler_type="constant",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=args.bf16 and torch.cuda.is_bf16_supported(),
        fp16=not (args.bf16 and torch.cuda.is_bf16_supported()),
        logging_steps=10,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        push_to_hub=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    
    # Log memory usage before training
    if torch.cuda.is_available():
        logger.info(f"GPU memory allocated before training: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        logger.info(f"GPU memory reserved before training: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    # Start training
    logger.info("Starting training...")
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    
    # Log training completion
    training_duration = (end_time - start_time) / 60
    logger.info(f"Training completed in {training_duration:.2f} minutes")
    
    # Log memory usage after training
    if torch.cuda.is_available():
        logger.info(f"GPU memory allocated after training: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        logger.info(f"GPU memory reserved after training: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    # Save the model and tokenizer
    output_folder = os.path.join(
        args.output_dir,
        f"jarvis-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    os.makedirs(output_folder, exist_ok=True)
    
    logger.info(f"Saving model to {output_folder}")
    model.save_pretrained(output_folder)
    tokenizer.save_pretrained(output_folder)
    
    # Save training configuration
    config_file = os.path.join(output_folder, "training_config.json")
    with open(config_file, "w") as f:
        json.dump({
            "model_name": args.model_name,
            "load_in_4bit": args.load_in_4bit,
            "load_in_8bit": args.load_in_8bit,
            "max_seq_length": args.max_seq_length,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "use_unsloth": args.use_unsloth,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "training_duration_minutes": training_duration,
            "dataset_samples": len(dataset),
            "steps_per_epoch": steps_per_epoch,
            "total_steps": max_steps,
            "gpu_info": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    logger.info(f"Saved training configuration to {config_file}")
    
    # After training is complete, sync to Google Drive
    if is_paperspace_environment():
        logger.info("Training complete, syncing results to Google Drive...")
        sync_to_gdrive("models")
        sync_to_gdrive("metrics")
        sync_to_gdrive("checkpoints")
        sync_logs()  # Sync log files
        logger.info("Sync complete!")
    
    logger.info(f"Finetuning complete! Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()