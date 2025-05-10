#!/usr/bin/env python3
"""
Train Text Model for Jarvis AI Assistant

This script trains a DeepSeek text model for Jarvis AI Assistant.
"""

import os
import sys
import logging
import argparse
import torch
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add the parent directory to the path so we can import from the root
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def train_text_model(gpu_type="A6000", vram_size=50):
    """
    Train a DeepSeek text model for Jarvis AI Assistant
    
    Args:
        gpu_type (str): Type of GPU (A6000, A4000, RTX5000)
        vram_size (int): VRAM size in GiB
    """
    logger.info(f"Training text model with GPU type {gpu_type} and VRAM size {vram_size} GiB")
    
    # Import required modules
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
        from transformers import DataCollatorForLanguageModeling
        from datasets import load_dataset
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Please run setup/consolidated_unified_setup.sh to install all dependencies")
        sys.exit(1)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. Cannot train text model.")
        sys.exit(1)
    
    # Set model parameters based on GPU type and VRAM size
    if gpu_type == "A6000" and vram_size >= 48:
        # A6000 with 48+ GiB VRAM
        model_name = "deepseek-ai/deepseek-llm-7b-base"
        batch_size = 4
        gradient_accumulation_steps = 4
        learning_rate = 2e-5
        use_4bit = True
    elif gpu_type == "A4000" or (gpu_type == "A6000" and vram_size < 48):
        # A4000 or A6000 with less than 48 GiB VRAM
        model_name = "deepseek-ai/deepseek-llm-1.3b-base"
        batch_size = 2
        gradient_accumulation_steps = 8
        learning_rate = 5e-5
        use_4bit = True
    else:
        # Default to smaller model for other GPUs
        model_name = "deepseek-ai/deepseek-llm-1.3b-base"
        batch_size = 1
        gradient_accumulation_steps = 16
        learning_rate = 1e-5
        use_4bit = True
    
    logger.info(f"Using model: {model_name}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Using 4-bit quantization: {use_4bit}")
    
    # Create output directory
    output_dir = "models/text"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer
    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load model
    logger.info("Loading model")
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            
            # Configure 4-bit quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            # Load model with 4-bit quantization
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        except ImportError:
            logger.warning("BitsAndBytesConfig not available. Using 16-bit precision instead.")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
    else:
        # Load model with 16-bit precision
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    
    # Load dataset
    logger.info("Loading dataset")
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        logger.info(f"Dataset loaded with {len(dataset)} examples")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.error("Using a small synthetic dataset instead")
        
        # Create a small synthetic dataset
        dataset = {
            "train": [
                {"text": "The capital of France is Paris. It is known for the Eiffel Tower."},
                {"text": "Machine learning is a subset of artificial intelligence that involves training algorithms to make predictions or decisions based on data."},
                {"text": "The solar system consists of the Sun and the objects that orbit it, including eight planets, their moons, asteroids, comets, and other small bodies."}
            ]
        }
    
    # Tokenize dataset
    logger.info("Tokenizing dataset")
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=500,
        logging_dir="logs",
        logging_steps=10,
        save_steps=1000,
        save_total_limit=2,
        fp16=True,
        dataloader_num_workers=4,
        report_to="tensorboard"
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    
    # Train model
    logger.info("Training model")
    trainer.train()
    
    # Save model
    logger.info("Saving model")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Training complete")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train text model for Jarvis AI Assistant")
    parser.add_argument("gpu_type", type=str, help="Type of GPU (A6000, A4000, RTX5000)")
    parser.add_argument("vram_size", type=int, help="VRAM size in GiB")
    args = parser.parse_args()
    
    # Train text model
    train_text_model(args.gpu_type, args.vram_size)
