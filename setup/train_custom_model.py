#!/usr/bin/env python3
"""
Train Custom Encoder-Decoder Model for Jarvis AI Assistant

This script trains a custom encoder-decoder model that uses the CNN model as a feature extractor.
It has been enhanced with memory optimizations and better performance.
"""

import os
import sys
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
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

# Check if we're running in a Paperspace environment
def is_paperspace_environment():
    """Check if running in Paperspace Gradient environment"""
    return os.path.exists("/notebooks") or os.path.exists("/storage")

# Define custom encoder-decoder model with improved memory efficiency
class CustomEncoderDecoder(nn.Module):
    def __init__(self, cnn_model, hidden_size=768, num_encoder_layers=3, num_decoder_layers=3, dropout=0.1):
        super(CustomEncoderDecoder, self).__init__()
        self.cnn_model = cnn_model
        self.hidden_size = hidden_size
        
        # Freeze CNN model
        for param in self.cnn_model.parameters():
            param.requires_grad = False
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        
        # Flag for gradient checkpointing
        self.use_gradient_checkpointing = False
    
    def forward(self, input_ids, decoder_input_ids, attention_mask=None, decoder_attention_mask=None):
        # Get CNN model features - optimized to handle memory better
        with torch.no_grad():
            # Clear cache if needed
            if hasattr(torch.cuda, 'empty_cache') and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Process with CNN model
            _, encoder_hidden_states = self.cnn_model(input_ids, attention_mask=attention_mask)
        
        # Apply encoder with gradient checkpointing if enabled
        if self.use_gradient_checkpointing and self.training:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
                
            encoder_output = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.encoder),
                encoder_hidden_states
            )
        else:
            encoder_output = self.encoder(encoder_hidden_states)
        
        # Get decoder input embeddings
        decoder_hidden_states = self.cnn_model.base_model.get_input_embeddings()(decoder_input_ids)
        
        # Create masks for decoder
        tgt_mask = self._generate_square_subsequent_mask(decoder_hidden_states.size(1)).to(decoder_hidden_states.device)
        tgt_key_padding_mask = None if decoder_attention_mask is None else ~decoder_attention_mask.bool()
        memory_key_padding_mask = None if attention_mask is None else ~attention_mask.bool()
        
        # Apply decoder with gradient checkpointing if enabled
        if self.use_gradient_checkpointing and self.training:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
                
            decoder_output = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.decoder),
                decoder_hidden_states,
                encoder_output,
                tgt_mask,
                None,  # memory_mask
                tgt_key_padding_mask,
                memory_key_padding_mask
            )
        else:
            decoder_output = self.decoder(
                decoder_hidden_states,
                encoder_output,
                tgt_mask=tgt_mask,
                memory_mask=None,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
        
        # Project to output
        output = self.output_projection(decoder_output)
        
        return output
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save memory during training"""
        self.use_gradient_checkpointing = True
        logger.info("Gradient checkpointing enabled for encoder-decoder model")
        
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing"""
        self.use_gradient_checkpointing = False
        logger.info("Gradient checkpointing disabled for encoder-decoder model")

def train_custom_model(args):
    """
    Train a custom encoder-decoder model for Jarvis AI Assistant
    
    Args:
        args: Command line arguments
    """
    logger.info(f"Training custom encoder-decoder model")
    logger.info(f"CNN model path: {args.cnn_model_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Max samples: {args.max_samples}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Weight decay: {args.weight_decay}")
    logger.info(f"Hidden size: {args.hidden_size}")
    logger.info(f"Number of encoder layers: {args.num_encoder_layers}")
    logger.info(f"Number of decoder layers: {args.num_decoder_layers}")
    logger.info(f"Dropout: {args.dropout}")
    logger.info(f"Log every: {args.log_every}")
    logger.info(f"Using improved preprocessor: {args.use_improved_preprocessor}")
    logger.info(f"Gradient checkpointing: {args.gradient_checkpointing}")
    logger.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    
    # Import required modules
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from datasets import load_dataset
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Please run setup/consolidated_unified_setup.sh to install all dependencies")
        sys.exit(1)
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("CUDA is not available. Training will be slow on CPU.")
        if args.force_gpu:
            logger.error("Force GPU was specified but CUDA is not available. Exiting.")
            sys.exit(1)
    else:
        # Log GPU info
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        try:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU Memory: {vram_gb:.2f} GB")
            
            # Auto-adjust batch size based on VRAM if needed
            if args.auto_batch_size and vram_gb < 24 and args.batch_size > 8:
                old_batch_size = args.batch_size
                if vram_gb < 8:
                    args.batch_size = 2
                elif vram_gb < 16:
                    args.batch_size = 4
                else:
                    args.batch_size = 8
                    
                logger.info(f"Auto-adjusted batch size from {old_batch_size} to {args.batch_size} based on available VRAM")
                
                # Increase gradient accumulation to compensate
                old_accum = args.gradient_accumulation_steps
                args.gradient_accumulation_steps = max(old_accum, old_batch_size // args.batch_size)
                logger.info(f"Auto-adjusted gradient accumulation steps from {old_accum} to {args.gradient_accumulation_steps}")
        except Exception as e:
            logger.warning(f"Failed to get GPU memory: {e}")
    
    # Check if CNN model exists
    if not os.path.exists(args.cnn_model_path):
        logger.error(f"CNN model not found at {args.cnn_model_path}")
        logger.error("Please run train_jarvis.sh with --model-type cnn-text first")
        sys.exit(1)
    
    # Clean up before loading models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load CNN model
    logger.info("Loading CNN model")
    try:
        # Load CNN model configuration
        config_path = os.path.join(os.path.dirname(args.cnn_model_path), "config.txt")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_lines = f.readlines()
            
            # Parse configuration
            base_model_name = None
            for line in config_lines:
                if line.startswith("Base model:"):
                    base_model_name = line.split(":", 1)[1].strip()
                    break
            
            if base_model_name:
                logger.info(f"CNN model base model: {base_model_name}")
            else:
                logger.warning("Could not find base model name in config.txt")
                base_model_name = "google/flan-ul2"
        else:
            logger.warning(f"Config file not found at {config_path}")
            base_model_name = "google/flan-ul2"
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Load base model with memory optimizations
        logger.info(f"Loading base model {base_model_name} with optimizations")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        # Create CNN model
        from setup.train_cnn_text_model import CNNLanguageModel
        cnn_model = CNNLanguageModel(
            base_model=base_model,
            vocab_size=tokenizer.vocab_size,
            hidden_size=base_model.config.hidden_size,
            num_cnn_layers=3
        )
        
        # Load CNN model weights
        logger.info(f"Loading CNN model weights from {args.cnn_model_path}")
        cnn_state_dict = torch.load(args.cnn_model_path, map_location=device)
        cnn_model.load_state_dict(cnn_state_dict)
        cnn_model.eval()  # Set to evaluation mode
        
        logger.info("CNN model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load CNN model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create custom encoder-decoder model
    logger.info("Creating custom encoder-decoder model")
    model = CustomEncoderDecoder(
        cnn_model=cnn_model,
        hidden_size=args.hidden_size,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dropout=args.dropout
    )
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    
    # Move model to device
    model = model.to(device)
    
    # Define optimizer with weight decay
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    
    # Set up optimizer with weight decay
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Set up learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # Load dataset
    logger.info("Loading dataset")
    try:
        # Load multiple datasets
        datasets = []
        
        # Load writing prompts dataset
        try:
            writing_prompts = load_dataset("roneneldan/writing_prompts", split=f"train[:{args.max_samples}]")
            datasets.append(writing_prompts)
            logger.info(f"Writing prompts dataset loaded with {len(writing_prompts)} examples")
        except Exception as e:
            logger.warning(f"Failed to load writing prompts dataset: {e}")
        
        # Load CNN Daily Mail dataset
        try:
            cnn_daily = load_dataset("cnn_dailymail", "3.0.0", split=f"train[:{args.max_samples}]")
            datasets.append(cnn_daily)
            logger.info(f"CNN Daily Mail dataset loaded with {len(cnn_daily)} examples")
        except Exception as e:
            logger.warning(f"Failed to load CNN Daily Mail dataset: {e}")
        
        # Load WikiText dataset
        try:
            wikitext = load_dataset("wikitext", "wikitext-103-v1", split=f"train[:{args.max_samples}]")
            datasets.append(wikitext)
            logger.info(f"WikiText dataset loaded with {len(wikitext)} examples")
        except Exception as e:
            logger.warning(f"Failed to load WikiText dataset: {e}")
        
        # Load BookCorpus dataset
        try:
            bookcorpus = load_dataset("bookcorpus", split=f"train[:{args.max_samples}]")
            datasets.append(bookcorpus)
            logger.info(f"BookCorpus dataset loaded with {len(bookcorpus)} examples")
        except Exception as e:
            logger.warning(f"Failed to load BookCorpus dataset: {e}")
        
        # Load OpenWebText dataset
        try:
            openwebtext = load_dataset("openwebtext", split=f"train[:{args.max_samples}]")
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
        
        # Tokenize input and output
        inputs = tokenizer(examples[text_field], truncation=True, max_length=512, padding="max_length")
        
        # Create decoder input by shifting the input
        decoder_inputs = {
            "input_ids": [ids[1:] + [tokenizer.pad_token_id] for ids in inputs["input_ids"]],
            "attention_mask": inputs["attention_mask"]
        }
        
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "decoder_input_ids": decoder_inputs["input_ids"],
            "decoder_attention_mask": decoder_inputs["attention_mask"]
        }
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    
    # Create data loader
    from torch.utils.data import DataLoader, TensorDataset
    
    # Convert dataset to tensors
    input_ids = torch.tensor(tokenized_dataset["input_ids"])
    attention_mask = torch.tensor(tokenized_dataset["attention_mask"])
    decoder_input_ids = torch.tensor(tokenized_dataset["decoder_input_ids"])
    decoder_attention_mask = torch.tensor(tokenized_dataset["decoder_attention_mask"])
    
    # Create tensor dataset
    tensor_dataset = TensorDataset(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask)
    
    # Create data loader
    data_loader = DataLoader(tensor_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Train model
    logger.info("Training model")
    model.train()
    
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        
        total_loss = 0.0
        
        for i, (input_ids_batch, attention_mask_batch, decoder_input_ids_batch, decoder_attention_mask_batch) in enumerate(data_loader):
            # Move tensors to GPU
            input_ids_batch = input_ids_batch.to(torch.device("cuda"))
            attention_mask_batch = attention_mask_batch.to(torch.device("cuda"))
            decoder_input_ids_batch = decoder_input_ids_batch.to(torch.device("cuda"))
            decoder_attention_mask_batch = decoder_attention_mask_batch.to(torch.device("cuda"))
            
            # Forward pass
            outputs = model(
                input_ids=input_ids_batch,
                attention_mask=attention_mask_batch,
                decoder_input_ids=decoder_input_ids_batch,
                decoder_attention_mask=decoder_attention_mask_batch
            )
            
            # Calculate loss
            loss = criterion(outputs.view(-1, args.hidden_size), decoder_input_ids_batch.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update total loss
            total_loss += loss.item()
            
            # Log progress
            if (i + 1) % args.log_every == 0:
                logger.info(f"Epoch {epoch + 1}/{args.epochs}, Batch {i + 1}/{len(data_loader)}, Loss: {loss.item():.4f}")
        
        # Log epoch results
        avg_loss = total_loss / len(data_loader)
        logger.info(f"Epoch {epoch + 1}/{args.epochs}, Average Loss: {avg_loss:.4f}")
    
    # Save model
    logger.info("Saving model")
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
    
    # Save model configuration
    with open(os.path.join(args.output_dir, "config.txt"), "w") as f:
        f.write(f"CNN model path: {args.cnn_model_path}\n")
        f.write(f"Hidden size: {args.hidden_size}\n")
        f.write(f"Number of encoder layers: {args.num_encoder_layers}\n")
        f.write(f"Number of decoder layers: {args.num_decoder_layers}\n")
        f.write(f"Dropout: {args.dropout}\n")
    
    logger.info("Training complete")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train custom encoder-decoder model for Jarvis AI Assistant")
    parser.add_argument("--cnn-model-path", type=str, required=True, help="Path to the CNN model")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--max-samples", type=int, default=5000, help="Maximum number of samples per dataset")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--hidden-size", type=int, default=768, help="Hidden size")
    parser.add_argument("--num-encoder-layers", type=int, default=3, help="Number of encoder layers")
    parser.add_argument("--num-decoder-layers", type=int, default=3, help="Number of decoder layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--log-every", type=int, default=10, help="Log every N batches")
    parser.add_argument("--use-improved-preprocessor", action="store_true", help="Use improved preprocessor")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--auto-batch-size", action="store_true", help="Auto-adjust batch size based on available VRAM")
    parser.add_argument("--force-gpu", action="store_true", help="Force GPU even if CUDA is not available")
    args = parser.parse_args()
    
    # Train custom model
    train_custom_model(args)
