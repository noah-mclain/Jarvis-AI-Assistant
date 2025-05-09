"""
Consolidated DeepSeek Training Module

This module provides a unified interface for training and fine-tuning DeepSeek models.
It consolidates functionality from:
- deepseek_handler.py
- unsloth_deepseek.py
- finetune_deepseek.py
- finetune_deepseek_examples.py
- finetune_on_mini.py
- fixed_run_finetune.py
- run_finetune.py
- unified_deepseek_training.py

Features:
- Unsloth optimization for faster training
- Memory-efficient training with quantization
- Google Drive integration for model storage
- Robust error handling and recovery
- Support for various hardware configurations
"""

import os
import sys
import time
import torch
import logging
import gc
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from transformers import BitsAndBytesConfig
    from peft import get_peft_model, LoraConfig, TaskType
    from datasets import Dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers library not available. DeepSeek training will be limited.")

# Try to import Unsloth
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    logger.warning("Unsloth not available. Training will use standard methods.")

# Import local modules
try:
    from .storage_manager import StorageManager, sync_to_gdrive, sync_from_gdrive
except ImportError:
    logger.warning("Storage manager not available. Storage optimizations will be limited.")
    sync_to_gdrive = lambda *args, **kwargs: None
    sync_from_gdrive = lambda *args, **kwargs: None

# Utility functions
def is_paperspace_environment():
    """Check if running in Paperspace Gradient environment"""
    return os.path.exists("/notebooks") or os.path.exists("/storage")

def setup_gpu_for_training():
    """Set up GPU for training"""
    if torch.cuda.is_available():
        device = "cuda"
        # Clear GPU cache
        torch.cuda.empty_cache()
        gc.collect()
        # Log GPU info
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = "cpu"
        logger.info("Using CPU (no GPU available)")

    return device

class DeepSeekTrainer:
    """
    Unified trainer for DeepSeek models.

    This class provides methods for:
    - Loading DeepSeek models with various optimizations
    - Fine-tuning with Unsloth or standard methods
    - Saving and loading models with storage optimization
    - Generating text with fine-tuned models
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/deepseek-coder-6.7b-base",
        use_unsloth: bool = True,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        force_gpu: bool = False,
        output_dir: str = "models/deepseek_finetuned"
    ):
        """
        Initialize the DeepSeek trainer.

        Args:
            model_name: Name or path of the DeepSeek model
            use_unsloth: Whether to use Unsloth optimization
            load_in_4bit: Whether to load in 4-bit quantization
            load_in_8bit: Whether to load in 8-bit quantization
            force_gpu: Whether to force GPU usage
            output_dir: Directory to save fine-tuned models
        """
        self.model_name = model_name
        self.use_unsloth = use_unsloth and UNSLOTH_AVAILABLE
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.force_gpu = force_gpu
        self.output_dir = output_dir

        # Set up device
        self.device = setup_gpu_for_training() if not force_gpu else "cuda"

        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None

        logger.info(f"Initialized DeepSeek trainer with model: {model_name}")
        logger.info(f"Using Unsloth: {self.use_unsloth}")
        logger.info(f"Device: {self.device}")

    def load_model(self, model_to_load=None):
        """
        Load the DeepSeek model.

        Args:
            model_to_load: Model name or path to load (defaults to self.model_name)

        Returns:
            Tuple of (model, tokenizer)
        """
        model_to_load = model_to_load or self.model_name

        try:
            # Load with Unsloth if available and requested
            if self.use_unsloth:
                logger.info("Loading model with Unsloth optimization")
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_to_load,
                    max_seq_length=2048,
                    dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
                    load_in_4bit=self.load_in_4bit,
                    load_in_8bit=self.load_in_8bit
                )
            else:
                # Standard loading
                model_kwargs = {
                    "trust_remote_code": True
                }

                # Add quantization parameters if needed
                if self.load_in_4bit and self.device == "cuda":
                    model_kwargs["load_in_4bit"] = True
                elif self.load_in_8bit and self.device == "cuda":
                    model_kwargs["load_in_8bit"] = True

                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    model_to_load,
                    trust_remote_code=True
                )

                # Load model
                model = AutoModelForCausalLM.from_pretrained(
                    model_to_load,
                    **model_kwargs
                )

            self.model = model
            self.tokenizer = tokenizer

            logger.info(f"Successfully loaded model: {model_to_load}")
            return model, tokenizer

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def fine_tune(
        self,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        output_dir: Optional[str] = None,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        warmup_steps: int = 100,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        max_steps: int = -1,
        save_steps: int = 500,
        eval_steps: int = 100,
        logging_steps: int = 10,
        gradient_accumulation_steps: int = 1,
        sequence_length: int = 2048,
        save_total_limit: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fine-tune the DeepSeek model.

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            output_dir: Directory to save the fine-tuned model
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            max_steps: Maximum number of training steps (-1 for no limit)
            save_steps: Save checkpoint every X steps
            eval_steps: Evaluate every X steps
            logging_steps: Log every X steps
            gradient_accumulation_steps: Number of gradient accumulation steps
            sequence_length: Maximum sequence length
            save_total_limit: Maximum number of checkpoints to keep
            **kwargs: Additional arguments

        Returns:
            Dictionary with training metrics
        """
        # Set output directory
        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Load model if not already loaded
        if self.model is None or self.tokenizer is None:
            self.load_model()

        # Choose the appropriate training method
        if self.use_unsloth:
            return self._fine_tune_with_unsloth(
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                output_dir=output_dir,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                warmup_steps=warmup_steps,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                max_steps=max_steps,
                save_steps=save_steps,
                eval_steps=eval_steps,
                logging_steps=logging_steps,
                gradient_accumulation_steps=gradient_accumulation_steps,
                sequence_length=sequence_length,
                save_total_limit=save_total_limit,
                **kwargs
            )
        else:
            return self._fine_tune_standard(
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                output_dir=output_dir,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                warmup_steps=warmup_steps,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                max_steps=max_steps,
                save_steps=save_steps,
                eval_steps=eval_steps,
                logging_steps=logging_steps,
                gradient_accumulation_steps=gradient_accumulation_steps,
                sequence_length=sequence_length,
                save_total_limit=save_total_limit,
                **kwargs
            )

    def _fine_tune_with_unsloth(self,
                          train_dataset,
                          eval_dataset=None,
                          output_dir=None,
                          epochs=3,
                          batch_size=4,
                          learning_rate=2e-5,
                          warmup_steps=100,
                          lora_r=16,
                          lora_alpha=32,
                          lora_dropout=0.05,
                          max_steps=-1,
                          save_steps=500,
                          eval_steps=100,
                          logging_steps=10,
                          gradient_accumulation_steps=1,
                          sequence_length=2048,
                          save_total_limit=3,
                          **kwargs):
        """
        Fine-tune with Unsloth optimization.

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            output_dir: Directory to save the fine-tuned model
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            max_steps: Maximum number of training steps (-1 for no limit)
            save_steps: Save checkpoint every X steps
            eval_steps: Evaluate every X steps
            logging_steps: Log every X steps
            gradient_accumulation_steps: Number of gradient accumulation steps
            sequence_length: Maximum sequence length
            save_total_limit: Maximum number of checkpoints to keep
            **kwargs: Additional arguments

        Returns:
            Dictionary with training metrics
        """
        if not UNSLOTH_AVAILABLE:
            logger.warning("Unsloth not available. Falling back to standard fine-tuning.")
            return self._fine_tune_standard(
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                output_dir=output_dir,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                warmup_steps=warmup_steps,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                max_steps=max_steps,
                save_steps=save_steps,
                eval_steps=eval_steps,
                logging_steps=logging_steps,
                gradient_accumulation_steps=gradient_accumulation_steps,
                sequence_length=sequence_length,
                save_total_limit=save_total_limit,
                **kwargs
            )

        logger.info("Fine-tuning with Unsloth optimization")

        # Make sure model and tokenizer are loaded
        if self.model is None or self.tokenizer is None:
            self.load_model()

        # Apply LoRA
        try:
            # Get target modules for LoRA
            target_modules = kwargs.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

            # Add LoRA adapters
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=lora_r,
                target_modules=target_modules,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                use_gradient_checkpointing=kwargs.get("gradient_checkpointing", True),
                random_state=kwargs.get("seed", 42),
                use_rslora=kwargs.get("use_rslora", False),
                loftq_config=None
            )

            logger.info(f"Added LoRA adapters with rank {lora_r}")

            # Set up training arguments
            training_args = {
                "output_dir": output_dir,
                "num_train_epochs": epochs,
                "per_device_train_batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "learning_rate": learning_rate,
                "max_steps": max_steps,
                "logging_steps": logging_steps,
                "save_steps": save_steps,
                "save_total_limit": save_total_limit,
                "group_by_length": True,
                "warmup_steps": warmup_steps,
                "lr_scheduler_type": kwargs.get("lr_scheduler_type", "cosine"),
                "weight_decay": kwargs.get("weight_decay", 0.01),
                "optim": kwargs.get("optim", "adamw_torch"),
                "fp16": kwargs.get("fp16", False),
                "bf16": kwargs.get("bf16", True),
                "max_grad_norm": kwargs.get("max_grad_norm", 0.3),
                "remove_unused_columns": False,
                "report_to": kwargs.get("report_to", "none"),
            }

            # Add evaluation arguments if eval_dataset is provided
            if eval_dataset is not None:
                training_args.update({
                    "evaluation_strategy": "steps",
                    "eval_steps": eval_steps,
                    "per_device_eval_batch_size": batch_size,
                })

            # Create Trainer
            from transformers import Trainer, TrainingArguments

            trainer = Trainer(
                model=self.model,
                args=TrainingArguments(**training_args),
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=kwargs.get("data_collator", None),
            )

            # Train the model
            logger.info("Starting training...")
            train_result = trainer.train()

            # Save the model
            logger.info(f"Saving model to {output_dir}")
            trainer.save_model(output_dir)

            # Save training metrics
            metrics = train_result.metrics
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

            # Sync to Google Drive if in Paperspace environment
            if is_paperspace_environment():
                try:
                    from .storage_manager import sync_to_gdrive
                    logger.info("Syncing model to Google Drive...")
                    sync_to_gdrive("models")
                except ImportError:
                    logger.warning("Storage manager not available. Model not synced to Google Drive.")

            return {
                "status": "success",
                "method": "unsloth",
                "metrics": metrics,
                "output_dir": output_dir
            }

        except Exception as e:
            logger.error(f"Error during Unsloth fine-tuning: {str(e)}")
            logger.info("Falling back to standard fine-tuning")
            return self._fine_tune_standard(
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                output_dir=output_dir,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                warmup_steps=warmup_steps,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                max_steps=max_steps,
                save_steps=save_steps,
                eval_steps=eval_steps,
                logging_steps=logging_steps,
                gradient_accumulation_steps=gradient_accumulation_steps,
                sequence_length=sequence_length,
                save_total_limit=save_total_limit,
                **kwargs
            )

    def _fine_tune_standard(self,
                           train_dataset,
                           eval_dataset=None,
                           output_dir=None,
                           epochs=3,
                           batch_size=4,
                           learning_rate=2e-5,
                           warmup_steps=100,
                           lora_r=16,
                           lora_alpha=32,
                           lora_dropout=0.05,
                           max_steps=-1,
                           save_steps=500,
                           eval_steps=100,
                           logging_steps=10,
                           gradient_accumulation_steps=1,
                           sequence_length=2048,
                           save_total_limit=3,
                           **kwargs):
        """
        Fine-tune with standard methods.

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            output_dir: Directory to save the fine-tuned model
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            max_steps: Maximum number of training steps (-1 for no limit)
            save_steps: Save checkpoint every X steps
            eval_steps: Evaluate every X steps
            logging_steps: Log every X steps
            gradient_accumulation_steps: Number of gradient accumulation steps
            sequence_length: Maximum sequence length
            save_total_limit: Maximum number of checkpoints to keep
            **kwargs: Additional arguments

        Returns:
            Dictionary with training metrics
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers library not available. Cannot fine-tune model.")
            return {"status": "error", "error": "Transformers library not available"}

        logger.info("Fine-tuning with standard methods")

        # Make sure model and tokenizer are loaded
        if self.model is None or self.tokenizer is None:
            self.load_model()

        try:
            # Apply LoRA
            from peft import LoraConfig, get_peft_model

            # Get target modules for LoRA
            target_modules = kwargs.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

            # Create LoRA configuration
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )

            # Apply LoRA to the model
            self.model = get_peft_model(self.model, lora_config)
            logger.info(f"Added LoRA adapters with rank {lora_r}")

            # Set up training arguments
            training_args = {
                "output_dir": output_dir,
                "num_train_epochs": epochs,
                "per_device_train_batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "learning_rate": learning_rate,
                "max_steps": max_steps,
                "logging_steps": logging_steps,
                "save_steps": save_steps,
                "save_total_limit": save_total_limit,
                "group_by_length": True,
                "warmup_steps": warmup_steps,
                "lr_scheduler_type": kwargs.get("lr_scheduler_type", "cosine"),
                "weight_decay": kwargs.get("weight_decay", 0.01),
                "optim": kwargs.get("optim", "adamw_torch"),
                "fp16": kwargs.get("fp16", False),
                "bf16": kwargs.get("bf16", True),
                "max_grad_norm": kwargs.get("max_grad_norm", 0.3),
                "remove_unused_columns": False,
                "report_to": kwargs.get("report_to", "none"),
            }

            # Add evaluation arguments if eval_dataset is provided
            if eval_dataset is not None:
                training_args.update({
                    "evaluation_strategy": "steps",
                    "eval_steps": eval_steps,
                    "per_device_eval_batch_size": batch_size,
                })

            # Create Trainer
            from transformers import Trainer, TrainingArguments

            # Apply attention mask fix if needed
            if kwargs.get("fix_attention_mask", True):
                logger.info("Applying attention mask fix")

                # Patch the attention mask handling in the model
                def patched_unmask_unattended(self, attention_mask, unmasked_value=0.0):
                    """Patch for the attention mask handling in DeepSeek models"""
                    # Check if attention_mask is None
                    if attention_mask is None:
                        return None

                    # Get the device and dtype of the attention mask
                    device = attention_mask.device
                    dtype = attention_mask.dtype

                    # Create a causal mask
                    batch_size, seq_length = attention_mask.shape
                    causal_mask = torch.triu(
                        torch.ones((seq_length, seq_length), dtype=dtype, device=device) * unmasked_value,
                        diagonal=1,
                    )

                    # Expand the causal mask to match the batch size
                    causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)

                    # Create the full attention mask
                    if len(attention_mask.shape) == 2:
                        # Expand attention_mask to 3D
                        expanded_mask = attention_mask.unsqueeze(1).expand(-1, seq_length, -1)
                        # Combine with causal mask
                        full_attention_mask = expanded_mask + causal_mask
                    else:
                        # If attention_mask is already 3D or 4D, just add the causal mask
                        full_attention_mask = attention_mask + causal_mask

                    # Return the full attention mask
                    return full_attention_mask

                # Apply the patch if the model has the method
                if hasattr(self.model, "unmask_unattended"):
                    self.model.unmask_unattended = patched_unmask_unattended.__get__(self.model)
                    logger.info("Successfully applied attention mask fix")
                else:
                    logger.warning("Model does not have unmask_unattended method, skipping attention mask fix")

            # Create Trainer
            trainer = Trainer(
                model=self.model,
                args=TrainingArguments(**training_args),
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=kwargs.get("data_collator", None),
            )

            # Train the model
            logger.info("Starting training...")
            train_result = trainer.train()

            # Save the model
            logger.info(f"Saving model to {output_dir}")
            trainer.save_model(output_dir)

            # Save training metrics
            metrics = train_result.metrics
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

            # Sync to Google Drive if in Paperspace environment
            if is_paperspace_environment():
                try:
                    from .storage_manager import sync_to_gdrive
                    logger.info("Syncing model to Google Drive...")
                    sync_to_gdrive("models")
                except ImportError:
                    logger.warning("Storage manager not available. Model not synced to Google Drive.")

            return {
                "status": "success",
                "method": "standard",
                "metrics": metrics,
                "output_dir": output_dir
            }

        except Exception as e:
            logger.error(f"Error during standard fine-tuning: {str(e)}")
            return {"status": "error", "error": str(e)}

    def generate(self, prompt, max_length=1000, temperature=0.7, top_p=0.9, top_k=50):
        """
        Generate text with the fine-tuned model.

        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter

        Returns:
            Generated text
        """
        # Load model if not already loaded
        if self.model is None or self.tokenizer is None:
            self.load_model()

        # Set model to evaluation mode
        self.model.eval()

        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode and return
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text

        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            return prompt

# Create a singleton instance for easy import
deepseek_trainer = DeepSeekTrainer()

# Expose key functions for backward compatibility
def fine_tune_deepseek(*args, **kwargs):
    """Fine-tune a DeepSeek model"""
    return deepseek_trainer.fine_tune(*args, **kwargs)

def load_deepseek_model(*args, **kwargs):
    """Load a DeepSeek model"""
    return deepseek_trainer.load_model(*args, **kwargs)

def generate_with_deepseek(*args, **kwargs):
    """Generate text with a DeepSeek model"""
    return deepseek_trainer.generate(*args, **kwargs)
