"""
Consolidated Generation Pipeline

This module provides a unified interface for text and code generation, combining
the functionality of text_generator.py, code_generator.py, and unified_generation_pipeline.py.

Features:
1. Text generation with character-level or tokenized models
2. Code generation with DeepSeek models
3. Training pipeline for both text and code models
4. Unified interface for all generation tasks
5. Memory-efficient processing
6. GPU/CPU/MPS support
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import json
import datetime
import time
import string
import logging
import gc
import functools
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Union, Tuple

# Try to import optional dependencies with graceful fallbacks
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from transformers import BitsAndBytesConfig
    from peft import get_peft_model, LoraConfig, TaskType
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers library not available. Some features will be limited.")

# Local imports
from .utils import (
    setup_gpu_for_training,
    force_cuda_device,
    is_paperspace_environment,
    is_zipfile,
    process_zip
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ConsolidatedGenerationPipeline")

# Define infinity for use in the code
infinity = float('inf')

# Execution time measurement utility function
def print_execution_time(func):
    """Decorator to print the execution time of a function"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.2f} seconds to execute")
        return result
    return wrapper

class CombinedModel(nn.Module):
    """
    Combined model for text generation with LSTM and embedding layers.

    This model can handle both one-hot encoded inputs and token indices.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(CombinedModel, self).__init__()

        # Add embedding layer for tokenized inputs
        self.embedding = nn.Embedding(input_size, hidden_size)

        # Simple LSTM model
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, hidden=None):
        # Add safety checks for tensor types and shapes
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(x)}")

        # Handle different input shapes:
        # x shape can be either:
        # [batch_size, seq_len, input_size] (3D tensor) - one-hot encoded
        # [batch_size, seq_len] (2D tensor) - token indices
        # [batch_size, input_size] (2D tensor) - single time step one-hot
        # [seq_len] (1D tensor) - single sample token indices

        # Handle 1D tensor (single sequence of tokens)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension [1, seq_len]

        # Handle unexpected higher dimensions (> 3D)
        if x.dim() > 3:
            x = x.view(x.size(0), x.size(1), -1)  # Flatten extra dimensions to 3D

        # Process input based on dimensionality and dtype
        if x.dim() == 2:
            if x.dtype in [torch.long, torch.int64]:
                # Input is token indices [batch_size, seq_len]
                # Ensure values are within valid embedding range
                vocab_size = self.embedding.num_embeddings
                if x.max() >= vocab_size:
                    # Clip indices to prevent out-of-bounds errors
                    x = torch.clamp(x, 0, vocab_size - 1)
                # Pass through embedding layer
                x = self.embedding(x)
            else:
                # If input is [batch_size, features] one-hot, add sequence dimension
                x = x.unsqueeze(1)
                # And convert to float in case it's not
                x = x.float()
        elif x.dim() == 3:
            # 3D tensor inputs are assumed to be one-hot encodings
            # No embedding needed, just make sure it's float
            x = x.float()

        # Execute forward pass with error handling
        try:
            # Pass through LSTM
            lstm_out, hidden = self.lstm(x, hidden)

            # Get output from last time step only
            last_output = lstm_out[:, -1, :] if lstm_out.dim() == 3 else lstm_out
            # Apply dropout and linear layer
            last_output = self.dropout(last_output)
            output = self.fc(last_output)

            return output, hidden
        except RuntimeError as e:
            # Handle specific runtime errors with more informative messages
            if "device-side assert triggered" in str(e):
                error_msg = (f"CUDA error: Device-side assert triggered. Input shape: {x.shape}, "
                           f"dtype: {x.dtype}. This might be caused by invalid indices or values.")
                raise RuntimeError(error_msg) from e
            elif "expected hidden[0] size" in str(e):
                error_msg = (f"LSTM hidden state size mismatch. Input shape: {x.shape}, "
                           f"hidden state shapes: {hidden[0].shape}, {hidden[1].shape} if hidden else 'None'")
                raise RuntimeError(error_msg) from e
            elif "CUDA out of memory" in str(e):
                # Try to free memory and provide a helpful message
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                error_msg = "CUDA out of memory error. Try reducing batch size or sequence length."
                raise RuntimeError(error_msg) from e
            else:
                # Re-raise the original error
                raise

class ConsolidatedGenerationPipeline:
    """
    Unified pipeline for text and code generation.

    This class combines the functionality of TextGenerator, CodeGenerator,
    and UnifiedGenerationPipeline to provide a single interface for all
    generation tasks.
    """

    # Default configuration values
    DEFAULT_CONFIG = {
        "hidden_size": 256,
        "num_layers": 2,
        "learning_rate": 0.001,
        "batch_size": 64,
        "sequence_length": 100,
        "epochs": 20,
        "temperature": 0.8,
        "top_k": 5,
        "top_p": 0.9,
        "max_length": 1000,
        "device": "auto"  # 'auto', 'cuda', 'cpu', or 'mps'
    }

    # Supported model types
    MODEL_TYPES = {
        "text": "Character-level text generation model",
        "code": "Code generation model using DeepSeek",
        "custom": "Custom model with user-defined architecture"
    }

    def __init__(self, model_type="text", model_path=None, config=None):
        """
        Initialize the generation pipeline.

        Args:
            model_type: Type of model to use ('text', 'code', or 'custom')
            model_path: Path to a saved model (optional)
            config: Configuration dictionary (optional)
        """
        # Set up basic attributes
        self.model_type = model_type
        self.model_path = model_path
        self.logger = logging.getLogger("ConsolidatedGenerationPipeline")

        # Set up configuration
        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)

        # Set up device
        self._setup_device()

        # Initialize model and tokenizer attributes
        self.model = None
        self.tokenizer = None

        # Character mappings for text generation
        self.chars = sorted(list(string.printable))
        self.char_to_index = {ch: i for i, ch in enumerate(self.chars)}
        self.index_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.unknown_token = '<UNK>'

        # Add unknown token if not present
        if self.unknown_token not in self.char_to_index:
            self.char_to_index[self.unknown_token] = len(self.chars)
            self.index_to_char[len(self.chars)] = self.unknown_token
            self.vocab_size += 1

        # Load model if path is provided
        if model_path:
            self.load_model(model_path)

        self.logger.info(f"Initialized {model_type} generation pipeline")

    def _setup_device(self):
        """Set up the device for model training and inference."""
        device = self.config["device"]

        if device == "auto":
            # Automatically select the best available device
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.logger.info(f"Using device: {self.device}")

        # Set up GPU for training if using CUDA
        if self.device == "cuda":
            setup_gpu_for_training()

    def create_model(self):
        """Create a new model based on the model type."""
        if self.model_type == "text":
            # Create a character-level text generation model
            hidden_size = self.config["hidden_size"]
            num_layers = self.config["num_layers"]

            self.model = CombinedModel(
                input_size=self.vocab_size,
                hidden_size=hidden_size,
                output_size=self.vocab_size,
                num_layers=num_layers
            )

        elif self.model_type == "code" and TRANSFORMERS_AVAILABLE:
            # Create a code generation model using DeepSeek
            self._create_code_model()

        elif self.model_type == "custom":
            # Custom model should be set by the user
            self.logger.warning("Custom model type selected. Use set_custom_model() to set the model.")

        else:
            if not TRANSFORMERS_AVAILABLE and self.model_type == "code":
                self.logger.error("Transformers library not available. Cannot create code model.")
            else:
                self.logger.error(f"Unknown model type: {self.model_type}")

        # Move model to the selected device
        if self.model is not None:
            self.model.to(self.device)

    def _create_code_model(self):
        """Create a code generation model using DeepSeek."""
        if not TRANSFORMERS_AVAILABLE:
            self.logger.error("Transformers library not available. Cannot create code model.")
            return

        try:
            # Configure quantization for efficient loading
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

            # Load the model with quantization
            model_id = "deepseek-ai/deepseek-coder-6.7b-instruct"
            self.logger.info(f"Loading DeepSeek model: {model_id}")

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )

            # Load the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True
            )

            self.logger.info("DeepSeek model and tokenizer loaded successfully")

        except Exception as e:
            self.logger.error(f"Error creating code model: {str(e)}")
            # Fall back to a simpler model if DeepSeek fails
            self.logger.info("Falling back to a simpler model")
            self.model_type = "text"
            self.create_model()

    def set_custom_model(self, model, tokenizer=None):
        """
        Set a custom model for generation.

        Args:
            model: Custom model to use
            tokenizer: Optional tokenizer for the model
        """
        self.model = model
        if tokenizer:
            self.tokenizer = tokenizer

        # Move model to the selected device
        if self.model is not None:
            self.model.to(self.device)

        self.logger.info("Custom model set successfully")

    def load_model(self, model_path):
        """
        Load a saved model from disk.

        Args:
            model_path: Path to the saved model
        """
        try:
            if not os.path.exists(model_path):
                self.logger.error(f"Model path does not exist: {model_path}")
                return False

            # Load the model based on type
            if self.model_type == "text":
                # Load character-level model
                self._load_text_model(model_path)
            elif self.model_type == "code" and TRANSFORMERS_AVAILABLE:
                # Load code model
                self._load_code_model(model_path)
            else:
                self.logger.error(f"Cannot load model of type {self.model_type}")
                return False

            self.logger.info(f"Model loaded successfully from {model_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False

    def _load_text_model(self, model_path):
        """
        Load a text generation model.

        Args:
            model_path: Path to the model file or directory

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Loading text model from {model_path}")
            
            # Check if model_path is a directory
            if os.path.isdir(model_path):
                # Check if it's a Hugging Face model directory
                if os.path.exists(os.path.join(model_path, "config.json")):
                    logger.info("Detected Hugging Face model directory")
                    
                    if TRANSFORMERS_AVAILABLE:
                        # Try to load as a FLAN-UL2 or other HF model
                        try:
                            # Configure quantization for memory efficiency
                            quantization_config = None
                            if torch.cuda.is_available() and getattr(self, "_use_4bit", False):
                                logger.info("Loading model with 4-bit quantization")
                                quantization_config = BitsAndBytesConfig(
                                    load_in_4bit=True,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.float16,
                                    bnb_4bit_use_double_quant=True
                                )
                            elif torch.cuda.is_available() and getattr(self, "_use_8bit", False):
                                logger.info("Loading model with 8-bit quantization")
                                quantization_config = BitsAndBytesConfig(
                                    load_in_8bit=True
                                )
                                
                            # Load tokenizer and model
                            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                            self.model = AutoModelForCausalLM.from_pretrained(
                                model_path,
                                device_map="auto",
                                quantization_config=quantization_config,
                                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                                low_cpu_mem_usage=True
                            )
                            
                            # Load vocabulary
                            self.vocab = list(self.tokenizer.get_vocab().keys())
                            self.vocab_size = len(self.vocab)
                            
                            logger.info(f"Successfully loaded Hugging Face model with vocab size {self.vocab_size}")
                            self.model_loaded = True
                            self.is_hf_model = True
                            
                            # Clear CUDA cache to free memory
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                gc.collect()
                                
                            return True
                        except Exception as e:
                            logger.error(f"Error loading Hugging Face model: {e}")
                            # Fall back to character-level model
                
                # Try to load a character-level model saved with torch.save
                model_file = os.path.join(model_path, "model.pt")
                if os.path.exists(model_file):
                    return self._load_char_level_model(model_file)
                    
                # Try to find any .pt files in the directory
                pt_files = [f for f in os.listdir(model_path) if f.endswith(".pt")]
                if pt_files:
                    model_file = os.path.join(model_path, pt_files[0])
                    return self._load_char_level_model(model_file)
                    
                logger.error(f"No valid model found in directory {model_path}")
                return False
            
            # If model_path is a file, load it directly
            elif os.path.isfile(model_path):
                return self._load_char_level_model(model_path)
                
            else:
                logger.error(f"Model path {model_path} does not exist")
                return False
                
        except Exception as e:
            logger.error(f"Error loading text model: {e}")
            return False
            
    def _load_char_level_model(self, model_file):
        """Load a character-level model saved with torch.save"""
        try:
            # Load model state
            state_dict = torch.load(model_file, map_location=self.device)
            
            # Check if it's a complete model or just the state dict
            if isinstance(state_dict, dict) and "model" in state_dict:
                # Extract model state dict and metadata
                model_state = state_dict["model"]
                self.vocab = state_dict.get("vocab", list(string.printable))
                self.vocab_size = len(self.vocab)
                self.char_to_idx = {char: i for i, char in enumerate(self.vocab)}
                self.idx_to_char = {i: char for i, char in enumerate(self.vocab)}
                
                # Create new model with the right dimensions
                input_size = output_size = self.vocab_size
                hidden_size = state_dict.get("hidden_size", self.config["hidden_size"])
                num_layers = state_dict.get("num_layers", self.config["num_layers"])
                
                self.model = CombinedModel(input_size, hidden_size, output_size, num_layers)
                self.model.load_state_dict(model_state)
                self.model.to(self.device)
                self.model.eval()
                
                logger.info(f"Successfully loaded character-level model with vocab size {self.vocab_size}")
                self.model_loaded = True
                self.is_hf_model = False
                return True
                
            else:
                # Assume it's just the model state dict
                # We need to guess the model dimensions
                self.vocab = list(string.printable)
                self.vocab_size = len(self.vocab)
                self.char_to_idx = {char: i for i, char in enumerate(self.vocab)}
                self.idx_to_char = {i: char for i, char in enumerate(self.vocab)}
                
                # Try to infer model dimensions from state dict
                fc_weight_shape = None
                for key, value in state_dict.items():
                    if "fc.weight" in key:
                        fc_weight_shape = value.shape
                        break
                        
                if fc_weight_shape is not None:
                    output_size, hidden_size = fc_weight_shape
                    input_size = output_size  # Assume input size = output size for char-level models
                    num_layers = self.config["num_layers"]  # Just use default
                    
                    self.model = CombinedModel(input_size, hidden_size, output_size, num_layers)
                    self.model.load_state_dict(state_dict)
                    self.model.to(self.device)
                    self.model.eval()
                    
                    logger.info(f"Successfully loaded character-level model with inferred dimensions")
                    self.model_loaded = True
                    self.is_hf_model = False
                    return True
                    
                else:
                    logger.error("Could not infer model dimensions from state dict")
                    return False
                    
        except Exception as e:
            logger.error(f"Error loading character-level model: {e}")
            return False

    def _load_code_model(self, model_path):
        """Load a code generation model."""
        if not TRANSFORMERS_AVAILABLE:
            self.logger.error("Transformers library not available. Cannot load code model.")
            return

        try:
            # Load the model and tokenizer
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                trust_remote_code=True
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )

        except Exception as e:
            self.logger.error(f"Error loading code model: {str(e)}")

    def save_model(self, model_path):
        """
        Save the model to disk.

        Args:
            model_path: Path to save the model
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            # Save based on model type
            if self.model_type == "text":
                self._save_text_model(model_path)
            elif self.model_type == "code" and TRANSFORMERS_AVAILABLE:
                self._save_code_model(model_path)
            else:
                self.logger.error(f"Cannot save model of type {self.model_type}")
                return False

            self.logger.info(f"Model saved successfully to {model_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            return False

    def _save_text_model(self, model_path):
        """Save a text generation model."""
        # Get model parameters
        if isinstance(self.model, CombinedModel):
            hidden_size = self.model.lstm.hidden_size
            num_layers = self.model.lstm.num_layers
        else:
            hidden_size = self.config['hidden_size']
            num_layers = self.config['num_layers']

        # Create checkpoint with all necessary data
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'char_to_index': self.char_to_index,
            'index_to_char': self.index_to_char,
            'vocab_size': self.vocab_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'timestamp': datetime.datetime.now().isoformat()
        }

        # Save the checkpoint
        torch.save(checkpoint, model_path)

    def _save_code_model(self, model_path):
        """Save a code generation model."""
        if not TRANSFORMERS_AVAILABLE:
            self.logger.error("Transformers library not available. Cannot save code model.")
            return

        try:
            # Save the model and tokenizer
            self.model.save_pretrained(model_path)
            if self.tokenizer:
                self.tokenizer.save_pretrained(model_path)

        except Exception as e:
            self.logger.error(f"Error saving code model: {str(e)}")

    @print_execution_time
    def train(self, dataset, epochs=None, learning_rate=None, batch_size=None):
        """
        Train the model on a dataset.

        Args:
            dataset: Dataset to train on (list of text sequences)
            epochs: Number of training epochs (default: config value)
            learning_rate: Learning rate (default: config value)
            batch_size: Batch size (default: config value)

        Returns:
            Dict with training metrics
        """
        # Set parameters or use defaults
        epochs = epochs or self.config["epochs"]
        learning_rate = learning_rate or self.config["learning_rate"]
        batch_size = batch_size or self.config["batch_size"]

        # If VRAM is limited, auto-adjust batch size
        if torch.cuda.is_available():
            try:
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"Detected {vram_gb:.2f} GB VRAM")
                
                # Auto-adjust batch size based on VRAM
                if vram_gb < 16 and batch_size > 16:
                    old_batch_size = batch_size
                    batch_size = min(16, batch_size)
                    logger.info(f"Auto-adjusted batch size from {old_batch_size} to {batch_size} based on available VRAM")
                    
            except Exception as e:
                logger.warning(f"Failed to auto-adjust batch size: {e}")

        if self.model_type == "code" and TRANSFORMERS_AVAILABLE:
            return self._train_code_model(dataset, epochs, learning_rate, batch_size)
        elif hasattr(self, "is_hf_model") and self.is_hf_model and TRANSFORMERS_AVAILABLE:
            return self._train_hf_model(dataset, epochs, learning_rate, batch_size)
        else:
            return self._train_char_model(dataset, epochs, learning_rate, batch_size)

    def _train_hf_model(self, dataset, epochs, learning_rate, batch_size):
        """
        Train a Hugging Face model (like FLAN-UL2) with memory optimizations.
        
        Args:
            dataset: Dataset to train on
            epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            
        Returns:
            Dict with training metrics
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers library not available. Cannot train Hugging Face model.")
            return {"error": "Transformers library not available"}
            
        logger.info(f"Training Hugging Face model for {epochs} epochs with batch size {batch_size}")
        
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from transformers import Trainer, TrainingArguments
            
            # Apply LoRA for memory-efficient fine-tuning
            if not hasattr(self.model, "is_peft_model") or not self.model.is_peft_model:
                logger.info("Applying LoRA adapters for memory-efficient training")
                
                # Prepare model for k-bit training if using quantization
                if getattr(self, "_use_4bit", False) or getattr(self, "_use_8bit", False):
                    self.model = prepare_model_for_kbit_training(self.model)
                
                # Define LoRA configuration
                lora_config = LoraConfig(
                    r=16,  # Rank
                    lora_alpha=32,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM"
                )
                
                # Apply LoRA adapters
                self.model = get_peft_model(self.model, lora_config)
                
            # Prepare the dataset
            from datasets import Dataset as HFDataset
            
            # Process the dataset into the right format
            if isinstance(dataset, list):
                # If it's a list of strings, convert to a proper dataset
                processed_data = []
                for text in dataset:
                    try:
                        inputs = self.tokenizer(text, truncation=True, max_length=self.config["sequence_length"])
                        inputs["labels"] = inputs["input_ids"].copy()
                        processed_data.append(inputs)
                    except Exception as e:
                        logger.warning(f"Error tokenizing text: {e}")
                        continue
                
                # Convert to HF Dataset
                train_dataset = HFDataset.from_list(processed_data)
                
            elif hasattr(dataset, "__getitem__") and hasattr(dataset, "__len__"):
                # If it's already a dataset-like object, convert to HF Dataset if needed
                if not isinstance(dataset, HFDataset):
                    train_dataset = HFDataset.from_dict({
                        "input_ids": [item["input_ids"] for item in dataset],
                        "attention_mask": [item["attention_mask"] for item in dataset],
                        "labels": [item["labels"] for item in dataset],
                    })
                else:
                    train_dataset = dataset
            else:
                raise ValueError("Dataset must be a list of strings or a dataset-like object")
            
            # Set up training arguments
            training_args = TrainingArguments(
                output_dir="./results",
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=4,  # Accumulate gradients for effective larger batch size
                learning_rate=learning_rate,
                weight_decay=0.01,
                warmup_ratio=0.1,
                logging_steps=10,
                save_steps=100,
                save_total_limit=2,
                fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
                remove_unused_columns=False
            )
            
            # Create the trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                tokenizer=self.tokenizer
            )
            
            # Train the model
            train_result = trainer.train()
            
            # Get metrics
            metrics = train_result.metrics
            
            # Set the model to eval mode
            self.model.eval()
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error training Hugging Face model: {e}")
            # Attempt to clear CUDA cache in case of OOM error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            return {"error": str(e)}

    def generate_text(self, seed_text, max_length=None, temperature=None, top_k=None, top_p=None):
        """
        Generate text based on a seed text.

        Args:
            seed_text: Text to start generation from
            max_length: Maximum length of generated text (default: from config)
            temperature: Temperature for sampling (default: from config)
            top_k: Top-k sampling parameter (default: from config)
            top_p: Top-p sampling parameter (default: from config)

        Returns:
            Generated text
        """
        # Use provided parameters or defaults from config
        max_length = max_length or self.config['max_length']
        temperature = temperature or self.config['temperature']
        top_k = top_k or self.config['top_k']
        top_p = top_p or self.config['top_p']

        # Create model if it doesn't exist
        if self.model is None:
            self.create_model()

        # Set model to evaluation mode
        self.model.eval()

        # Generate based on model type
        if self.model_type == "text":
            return self._generate_text_with_char_model(
                seed_text, max_length, temperature, top_k, top_p
            )
        elif self.model_type == "code" and TRANSFORMERS_AVAILABLE:
            return self._generate_text_with_code_model(
                seed_text, max_length, temperature, top_k, top_p
            )
        else:
            self.logger.error(f"Cannot generate text with model type {self.model_type}")
            return seed_text

    def _generate_text_with_char_model(self, seed_text, max_length, temperature, top_k, top_p):
        """Generate text with a character-level model."""
        # Ensure we have a valid seed text
        if not seed_text:
            seed_text = random.choice(list(self.char_to_index.keys()))

        # Convert seed text to indices
        input_indices = [self.char_to_index.get(ch, self.char_to_index.get(self.unknown_token, 0))
                        for ch in seed_text]

        # Initialize generation
        generated_text = seed_text

        # Generate text
        with torch.no_grad():
            # Initialize hidden state
            hidden = None

            # Generate one character at a time
            for _ in range(max_length):
                # Prepare input tensor
                x = torch.tensor([input_indices[-self.config['sequence_length']:]], dtype=torch.long).to(self.device)

                # Forward pass
                output, hidden = self.model(x, hidden)

                # Apply temperature
                output = output / temperature

                # Apply top-k sampling
                if top_k > 0:
                    indices_to_remove = output < torch.topk(output, top_k)[0][..., -1, None]
                    output[indices_to_remove] = -float('Inf')

                # Apply top-p sampling
                if 0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(output, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    output[0, indices_to_remove] = -float('Inf')

                # Sample from the distribution
                probabilities = F.softmax(output, dim=-1)
                next_char_index = torch.multinomial(probabilities, 1).item()

                # Convert to character and append to generated text
                next_char = self.index_to_char.get(next_char_index, self.unknown_token)
                generated_text += next_char

                # Update input indices for next iteration
                input_indices.append(next_char_index)

        return generated_text

    def _generate_text_with_code_model(self, seed_text, max_length, temperature, top_k, top_p):
        """Generate text with a code model."""
        if not TRANSFORMERS_AVAILABLE:
            self.logger.error("Transformers library not available. Cannot generate with code model.")
            return seed_text

        try:
            # Prepare inputs
            inputs = self.tokenizer(seed_text, return_tensors="pt").to(self.device)

            # Generate
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

            # Decode and return
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text

        except Exception as e:
            self.logger.error(f"Error generating with code model: {str(e)}")
            return seed_text

    def generate_code(self, prompt, max_length=None, temperature=None, top_k=None, top_p=None):
        """
        Generate code based on a prompt.

        Args:
            prompt: Prompt to generate code from
            max_length: Maximum length of generated code (default: from config)
            temperature: Temperature for sampling (default: from config)
            top_k: Top-k sampling parameter (default: from config)
            top_p: Top-p sampling parameter (default: from config)

        Returns:
            Generated code
        """
        # For code generation, we use the same method as text generation
        # but with a specialized prompt format

        # Format prompt for code generation
        if self.model_type == "code" and TRANSFORMERS_AVAILABLE:
            # Format prompt for DeepSeek model
            formatted_prompt = f"""
You are an expert programmer. Write code to solve the following problem:

{prompt}

```
"""
            return self.generate_text(
                formatted_prompt, max_length, temperature, top_k, top_p
            )
        else:
            # Fall back to regular text generation
            return self.generate_text(
                f"// {prompt}\n", max_length, temperature, top_k, top_p
            )

    def evaluate(self, test_dataset):
        """
        Evaluate the model on a test dataset.

        Args:
            test_dataset: Dataset to evaluate on (list of batches)

        Returns:
            Dictionary with evaluation metrics
        """
        # Create model if it doesn't exist
        if self.model is None:
            self.create_model()

        # Set model to evaluation mode
        self.model.eval()

        # Evaluation metrics
        metrics = {
            'loss': 0,
            'accuracy': 0,
            'perplexity': 0,
            'num_batches': 0,
            'num_samples': 0
        }

        # Evaluate
        with torch.no_grad():
            for input_batch, target_batch in tqdm(test_dataset, desc="Evaluating"):
                # Move data to device
                input_batch = input_batch.to(self.device)
                target_batch = target_batch.to(self.device)

                # Forward pass
                output, _ = self.model(input_batch)

                # Calculate loss
                loss = F.cross_entropy(output, target_batch)

                # Calculate accuracy
                _, predicted = torch.max(output, 1)
                correct = (predicted == target_batch).sum().item()
                accuracy = correct / target_batch.size(0)

                # Update metrics
                metrics['loss'] += loss.item()
                metrics['accuracy'] += accuracy
                metrics['num_batches'] += 1
                metrics['num_samples'] += target_batch.size(0)

                # Free memory
                del input_batch, target_batch, output
                if self.device == "cuda":
                    torch.cuda.empty_cache()

        # Calculate average metrics
        if metrics['num_batches'] > 0:
            metrics['loss'] /= metrics['num_batches']
            metrics['accuracy'] /= metrics['num_batches']
            metrics['perplexity'] = np.exp(metrics['loss'])

        return metrics

    @print_execution_time
    def train_with_gradient_accumulation(self, dataset, epochs=None, learning_rate=None,
                                        batch_size=None, gradient_accumulation_steps=4):
        """
        Train the model with gradient accumulation for larger effective batch sizes.

        Args:
            dataset: Dataset to train on (list of batches)
            epochs: Number of epochs to train for (default: from config)
            learning_rate: Learning rate (default: from config)
            batch_size: Batch size (default: from config)
            gradient_accumulation_steps: Number of steps to accumulate gradients

        Returns:
            Dictionary with training metrics
        """
        # Use provided parameters or defaults from config
        epochs = epochs or self.config['epochs']
        learning_rate = learning_rate or self.config['learning_rate']
        batch_size = batch_size or self.config['batch_size']

        # Create model if it doesn't exist
        if self.model is None:
            self.create_model()

        # Set model to training mode
        self.model.train()

        # Create optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training metrics
        metrics = {
            'epochs': epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'effective_batch_size': batch_size * gradient_accumulation_steps,
            'losses': [],
            'start_time': datetime.datetime.now().isoformat()
        }

        # Training loop
        try:
            for epoch in range(epochs):
                epoch_loss = 0
                batch_count = 0
                accumulated_steps = 0

                # Process each batch
                for input_batch, target_batch in tqdm(dataset, desc=f"Epoch {epoch+1}/{epochs}"):
                    # Move data to device
                    input_batch = input_batch.to(self.device)
                    target_batch = target_batch.to(self.device)

                    # Forward pass
                    output, _ = self.model(input_batch)

                    # Calculate loss
                    loss = F.cross_entropy(output, target_batch)

                    # Scale the loss by gradient accumulation steps
                    scaled_loss = loss / gradient_accumulation_steps

                    # Backward pass
                    scaled_loss.backward()

                    # Update metrics
                    epoch_loss += loss.item()
                    batch_count += 1
                    accumulated_steps += 1

                    # Optimize only after accumulating enough gradients
                    if accumulated_steps % gradient_accumulation_steps == 0:
                        # Clip gradients to prevent exploding gradients
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                        # Optimizer step
                        optimizer.step()
                        optimizer.zero_grad()

                        # Log progress
                        if batch_count % 10 == 0:
                            self.logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_count}/{len(dataset)}, "
                                           f"Loss: {loss.item():.4f}")

                    # Free memory
                    del input_batch, target_batch, output, loss, scaled_loss
                    if self.device == "cuda":
                        torch.cuda.empty_cache()

                # Make sure to optimize at the end of epoch if there are remaining steps
                if accumulated_steps % gradient_accumulation_steps != 0:
                    optimizer.step()
                    optimizer.zero_grad()

                # Calculate average loss for the epoch
                avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
                metrics['losses'].append(avg_loss)

                # Log progress
                self.logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

            # Set model back to evaluation mode
            self.model.eval()

            # Update metrics
            metrics['end_time'] = datetime.datetime.now().isoformat()

            return metrics

        except Exception as e:
            self.logger.error(f"Error during training with gradient accumulation: {str(e)}")
            # Set model back to evaluation mode
            self.model.eval()
            return metrics

    def train_batch(self, batch, optimizer=None, criterion=None):
        """
        Train on a single batch.

        Args:
            batch: Tuple of (input_batch, target_batch)
            optimizer: Optimizer to use (default: create new Adam optimizer)
            criterion: Loss function (default: CrossEntropyLoss)

        Returns:
            Loss value
        """
        # Create model if it doesn't exist
        if self.model is None:
            self.create_model()

        # Set model to training mode
        self.model.train()

        # Create optimizer if not provided
        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])

        # Create criterion if not provided
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        # Unpack batch
        input_batch, target_batch = batch

        # Move data to device
        input_batch = input_batch.to(self.device)
        target_batch = target_batch.to(self.device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output, _ = self.model(input_batch)

        # Calculate loss
        loss = criterion(output, target_batch)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Get loss value
        loss_value = loss.item()

        # Free memory
        del input_batch, target_batch, output, loss
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return loss_value
