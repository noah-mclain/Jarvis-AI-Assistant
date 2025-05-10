#!/usr/bin/env python3
"""
Train CNN Text Model for Jarvis AI Assistant

This script trains a CNN-enhanced text model for Jarvis AI Assistant.
"""

import os
import sys
import logging
import argparse
import torch
import torch.nn as nn
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

# Define CNN-enhanced language model
class CNNLanguageModel(nn.Module):
    def __init__(self, base_model, vocab_size, hidden_size=768, num_cnn_layers=3):
        super(CNNLanguageModel, self).__init__()
        self.base_model = base_model
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_cnn_layers = num_cnn_layers
        
        # CNN layers
        self.cnn_layers = nn.ModuleList([
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
            for _ in range(num_cnn_layers)
        ])
        
        # Layer normalization and activation
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size)
            for _ in range(num_cnn_layers)
        ])
        self.activation = nn.GELU()
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids, attention_mask=None):
        # Get base model outputs
        outputs = self.base_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # Use the last hidden state
        
        # Apply CNN layers
        x = hidden_states.permute(0, 2, 1)  # [batch_size, hidden_size, seq_len]
        
        for i in range(self.num_cnn_layers):
            residual = x
            x = self.cnn_layers[i](x)
            x = x + residual  # Residual connection
            x = x.permute(0, 2, 1)  # [batch_size, seq_len, hidden_size]
            x = self.layer_norms[i](x)
            x = self.activation(x)
            x = x.permute(0, 2, 1)  # [batch_size, hidden_size, seq_len]
        
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, hidden_size]
        
        # Project to vocabulary
        logits = self.output_projection(x)
        
        return logits, hidden_states

def train_cnn_text_model(gpu_type="A6000", vram_size=50, use_improved_preprocessor=False):
    """
    Train a CNN-enhanced text model for Jarvis AI Assistant
    
    Args:
        gpu_type (str): Type of GPU (A6000, A4000, RTX5000)
        vram_size (int): VRAM size in GiB
        use_improved_preprocessor (bool): Whether to use the improved preprocessor
    """
    logger.info(f"Training CNN text model with GPU type {gpu_type} and VRAM size {vram_size} GiB")
    logger.info(f"Using improved preprocessor: {use_improved_preprocessor}")
    
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
        logger.error("CUDA is not available. Cannot train CNN text model.")
        sys.exit(1)
    
    # Set model parameters based on GPU type and VRAM size
    if gpu_type == "A6000" and vram_size >= 48:
        # A6000 with 48+ GiB VRAM
        model_name = "google/flan-ul2"
        batch_size = 2
        gradient_accumulation_steps = 8
        learning_rate = 1e-5
        use_4bit = True
        num_cnn_layers = 3
    elif gpu_type == "A4000" or (gpu_type == "A6000" and vram_size < 48):
        # A4000 or A6000 with less than 48 GiB VRAM
        model_name = "google/flan-t5-large"
        batch_size = 2
        gradient_accumulation_steps = 8
        learning_rate = 2e-5
        use_4bit = True
        num_cnn_layers = 2
    else:
        # Default to smaller model for other GPUs
        model_name = "google/flan-t5-base"
        batch_size = 1
        gradient_accumulation_steps = 16
        learning_rate = 5e-5
        use_4bit = True
        num_cnn_layers = 1
    
    logger.info(f"Using model: {model_name}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Using 4-bit quantization: {use_4bit}")
    logger.info(f"Number of CNN layers: {num_cnn_layers}")
    
    # Create output directory
    output_dir = "models/cnn-text"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer
    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load base model
    logger.info("Loading base model")
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
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
        except ImportError:
            logger.warning("BitsAndBytesConfig not available. Using 16-bit precision instead.")
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
    else:
        # Load model with 16-bit precision
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    # Create CNN-enhanced model
    logger.info("Creating CNN-enhanced model")
    model = CNNLanguageModel(
        base_model=base_model,
        vocab_size=tokenizer.vocab_size,
        hidden_size=base_model.config.hidden_size,
        num_cnn_layers=num_cnn_layers
    )
    
    # Load dataset
    logger.info("Loading dataset")
    try:
        # Load multiple datasets
        datasets = []
        
        # Load writing prompts dataset
        try:
            writing_prompts = load_dataset("roneneldan/writing_prompts", split="train[:1000]")
            datasets.append(writing_prompts)
            logger.info(f"Writing prompts dataset loaded with {len(writing_prompts)} examples")
        except Exception as e:
            logger.warning(f"Failed to load writing prompts dataset: {e}")
        
        # Load CNN Daily Mail dataset
        try:
            cnn_daily = load_dataset("cnn_dailymail", "3.0.0", split="train[:1000]")
            datasets.append(cnn_daily)
            logger.info(f"CNN Daily Mail dataset loaded with {len(cnn_daily)} examples")
        except Exception as e:
            logger.warning(f"Failed to load CNN Daily Mail dataset: {e}")
        
        # Load WikiText dataset
        try:
            wikitext = load_dataset("wikitext", "wikitext-103-v1", split="train[:1000]")
            datasets.append(wikitext)
            logger.info(f"WikiText dataset loaded with {len(wikitext)} examples")
        except Exception as e:
            logger.warning(f"Failed to load WikiText dataset: {e}")
        
        # Load BookCorpus dataset
        try:
            bookcorpus = load_dataset("bookcorpus", split="train[:1000]")
            datasets.append(bookcorpus)
            logger.info(f"BookCorpus dataset loaded with {len(bookcorpus)} examples")
        except Exception as e:
            logger.warning(f"Failed to load BookCorpus dataset: {e}")
        
        # Load OpenWebText dataset
        try:
            openwebtext = load_dataset("openwebtext", split="train[:1000]")
            datasets.append(openwebtext)
            logger.info(f"OpenWebText dataset loaded with {len(openwebtext)} examples")
        except Exception as e:
            logger.warning(f"Failed to load OpenWebText dataset: {e}")
        
        if not datasets:
            raise ValueError("No datasets were loaded successfully")
        
        # Combine datasets
        from datasets import concatenate_datasets
        dataset = concatenate_datasets(datasets)
        logger.info(f"Combined dataset with {len(dataset)} examples")
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
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
        # Handle different dataset formats
        if "text" in examples:
            text_field = "text"
        elif "article" in examples:
            text_field = "article"
        elif "wp_text" in examples:
            text_field = "wp_text"
        elif "content" in examples:
            text_field = "content"
        else:
            # Use the first field that contains string data
            for field in examples:
                if isinstance(examples[field][0], str):
                    text_field = field
                    break
            else:
                raise ValueError("Could not find a text field in the dataset")
        
        return tokenizer(examples[text_field], truncation=True, max_length=512)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    
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
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    tokenizer.save_pretrained(output_dir)
    
    # Save model configuration
    with open(os.path.join(output_dir, "config.txt"), "w") as f:
        f.write(f"Base model: {model_name}\n")
        f.write(f"Number of CNN layers: {num_cnn_layers}\n")
        f.write(f"Hidden size: {base_model.config.hidden_size}\n")
        f.write(f"Vocabulary size: {tokenizer.vocab_size}\n")
    
    logger.info("Training complete")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train CNN text model for Jarvis AI Assistant")
    parser.add_argument("gpu_type", type=str, help="Type of GPU (A6000, A4000, RTX5000)")
    parser.add_argument("vram_size", type=int, help="VRAM size in GiB")
    parser.add_argument("use_improved_preprocessor", type=int, nargs="?", default=0, help="Whether to use the improved preprocessor (0 or 1)")
    args = parser.parse_args()
    
    # Train CNN text model
    train_cnn_text_model(args.gpu_type, args.vram_size, bool(args.use_improved_preprocessor))
