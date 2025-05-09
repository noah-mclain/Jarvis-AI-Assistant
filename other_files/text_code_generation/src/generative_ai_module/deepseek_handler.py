"""
DeepSeek Handler Module

This module provides a unified interface for working with DeepSeek models in the Jarvis AI Assistant.
It includes functionality for fine-tuning, optimization, and storage management of DeepSeek models.

Features:
- Fine-tuning DeepSeek models with Unsloth optimization
- Storage optimization for different environments
- Google Drive integration for model persistence
- Helper functions for working with DeepSeek models in Paperspace

Requirements:
- transformers
- unsloth (optional but recommended)
- torch
- peft
"""

import os
import sys
import logging
import json
import time
import shutil
import tempfile
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime

# Add the parent directory to the path to make the module importable
if __name__ == "__main__":
    # When run as a script
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import utility functions
try:
    from src.generative_ai_module.utils import (
        setup_logging, ensure_directory_exists,
        sync_to_gdrive, sync_from_gdrive,
        is_paperspace_environment
    )
except ImportError:
    # Fallback definitions if utils is not available
    def setup_logging(log_file=None):
        """Set up logging configuration"""
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(file_handler)
        return logger

    def ensure_directory_exists(directory):
        """Ensure a directory exists, creating it if necessary"""
        os.makedirs(directory, exist_ok=True)
        return directory

    def is_paperspace_environment():
        """Check if running in Paperspace Gradient"""
        return os.environ.get('PAPERSPACE') == 'true' or os.environ.get('PAPERSPACE_ENVIRONMENT') == 'true'

    def sync_to_gdrive(local_path, remote_path=None):
        """Stub for syncing to Google Drive"""
        logger.warning("sync_to_gdrive not available - utils module not imported")
        return False

    def sync_from_gdrive(remote_path, local_path=None):
        """Stub for syncing from Google Drive"""
        logger.warning("sync_from_gdrive not available - utils module not imported")
        return False

# Try to import optional dependencies
try:
    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        TrainingArguments, Trainer,
        DataCollatorForLanguageModeling
    )
    from peft import LoraConfig, get_peft_model

    # Try importing unsloth
    try:
        from unsloth import FastLanguageModel
        UNSLOTH_AVAILABLE = True
    except ImportError:
        logger.warning("Unsloth not available. Fine-tuning will use standard methods.")
        UNSLOTH_AVAILABLE = False

except ImportError as e:
    logger.error(f"Required dependencies not available: {e}")
    logger.error("Please install required packages with: pip install transformers torch peft")
    UNSLOTH_AVAILABLE = False

class DeepSeekHandler:
    """
    Class for handling DeepSeek models, including fine-tuning, optimization, and storage.
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
        Initialize the DeepSeek handler

        Args:
            model_name: Name of the DeepSeek model to use
            use_unsloth: Whether to use Unsloth for optimization (if available)
            load_in_4bit: Whether to use 4-bit quantization
            load_in_8bit: Whether to use 8-bit quantization
            force_gpu: Whether to force GPU usage
            output_dir: Directory to save fine-tuned models
        """
        self.model_name = model_name
        self.use_unsloth = use_unsloth and UNSLOTH_AVAILABLE
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.force_gpu = force_gpu
        self.output_dir = output_dir

        # Set up the device
        self.device = self._get_device()

        # Log configuration
        logger.info(f"Initialized DeepSeekHandler with model: {model_name}")
        logger.info(f"Using Unsloth: {self.use_unsloth}")
        logger.info(f"Quantization: 4-bit={load_in_4bit}, 8-bit={load_in_8bit}")
        logger.info(f"Device: {self.device}")

        # Initialize model and tokenizer to None (load when needed)
        self.model = None
        self.tokenizer = None

    def _get_device(self) -> str:
        """Get the appropriate device for training/inference"""
        if torch.cuda.is_available() and (self.force_gpu or not is_apple_silicon()):
            return "cuda"
        elif is_apple_silicon() and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def load_model(self, model_path: Optional[str] = None) -> Tuple[Any, Any]:
        """
        Load the model and tokenizer

        Args:
            model_path: Path to a fine-tuned model, or None to load the base model

        Returns:
            Tuple of (model, tokenizer)
        """
        # Use the provided path or the default model name
        model_to_load = model_path if model_path else self.model_name
        logger.info(f"Loading model from {model_to_load}")

        try:
            # Load with Unsloth if available and requested
            if self.use_unsloth and UNSLOTH_AVAILABLE:
                logger.info("Loading model with Unsloth optimization")
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_to_load,
                    max_seq_length=2048,
                    dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
                    load_in_4bit=self.load_in_4bit,
                    load_in_8bit=self.load_in_8bit
                    # Don't set device_map here as it's already set by FastLanguageModel internally
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

                # Add device mapping if on CUDA
                if self.device == "cuda":
                    model_kwargs["device_map"] = "auto"

                # Load the tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_to_load, trust_remote_code=True)
                tokenizer.pad_token = tokenizer.eos_token

                # Load the model
                model = AutoModelForCausalLM.from_pretrained(
                    model_to_load,
                    **model_kwargs
                )

                # Move model to device if not using device_map="auto"
                if self.device != "cuda" or (not self.load_in_4bit and not self.load_in_8bit):
                    model = model.to(self.device)

            # Store the model and tokenizer
            self.model = model
            self.tokenizer = tokenizer

            return model, tokenizer

        except Exception as e:
            logger.error(f"Error loading model: {e}")
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
        Fine-tune the DeepSeek model

        Args:
            train_dataset: Dataset for training
            eval_dataset: Dataset for evaluation (optional)
            output_dir: Directory to save the fine-tuned model
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            lora_r: LoRA attention dimension
            lora_alpha: LoRA alpha parameter
            lora_dropout: Dropout probability for LoRA
            max_steps: Maximum number of training steps (-1 to use epochs)
            save_steps: Save checkpoint every X steps
            eval_steps: Evaluate every X steps
            logging_steps: Log every X steps
            gradient_accumulation_steps: Number of steps to accumulate gradients
            sequence_length: Maximum sequence length
            save_total_limit: Maximum number of checkpoints to keep

        Returns:
            Dictionary with training metrics
        """
        # Use the provided output directory or the default
        output_dir = output_dir or self.output_dir
        ensure_directory_exists(output_dir)

        # Load model if not already loaded
        if self.model is None or self.tokenizer is None:
            self.load_model()

        # Prepare for LoRA fine-tuning
        start_time = time.time()

        try:
            if self.use_unsloth and UNSLOTH_AVAILABLE:
                # Unsloth-optimized LoRA
                logger.info("Setting up Unsloth LoRA fine-tuning")
                model, tokenizer = self.model, self.tokenizer

                # Apply LoRA using Unsloth
                model = FastLanguageModel.get_peft_model(
                    model,
                    r=lora_r,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias="none",
                )
            else:
                # Standard LoRA setup
                logger.info("Setting up standard LoRA fine-tuning")

                # Create LoRA configuration
                peft_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                )

                # Apply LoRA to the model
                model = get_peft_model(self.model, peft_config)

            # Print trainable parameters info
            model.print_trainable_parameters()

            # Create training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
                logging_steps=logging_steps,
                save_steps=save_steps,
                eval_steps=eval_steps if eval_dataset else None,
                evaluation_strategy="steps" if eval_dataset else "no",
                save_strategy="steps",
                save_total_limit=save_total_limit,
                max_steps=max_steps if max_steps > 0 else None,
                warmup_steps=warmup_steps,
                weight_decay=0.01,
                report_to="tensorboard",
                fp16=self.device == "cuda",
                load_best_model_at_end=True if eval_dataset else False,
                remove_unused_columns=False,
                group_by_length=True,
                use_mps_device=self.device == "mps",
                auto_find_batch_size=True if self.device == "cuda" else False,
            )

            # Create the trainer with data collator
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

            trainer = Trainer(
                model=model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                args=training_args,
                data_collator=data_collator,
            )

            # Start training
            logger.info("Starting fine-tuning...")
            trainer.train()

            # Save the model
            logger.info(f"Saving fine-tuned model to {output_dir}")
            trainer.save_model(output_dir)

            # Save tokenizer
            tokenizer.save_pretrained(output_dir)

            # Calculate training time
            training_time = time.time() - start_time
            logger.info(f"Fine-tuning completed in {training_time:.2f} seconds")

            # Save training config
            train_config = {
                "model_name": self.model_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "sequence_length": sequence_length,
                "training_time": training_time,
                "finished_at": datetime.now().isoformat()
            }

            with open(os.path.join(output_dir, "training_config.json"), "w") as f:
                json.dump(train_config, f, indent=2)

            # Sync to Google Drive if in Paperspace
            if is_paperspace_environment():
                logger.info("Syncing fine-tuned model to Google Drive")
                sync_to_gdrive(output_dir)

            # Return metrics
            train_metrics = {
                "training_time": training_time,
                "epochs": epochs,
                "final_loss": trainer.state.log_history[-1].get("loss", None) if trainer.state.log_history else None,
            }

            if eval_dataset:
                eval_metrics = trainer.evaluate()
                train_metrics.update(eval_metrics)

            return train_metrics

        except Exception as e:
            logger.error(f"Error during fine-tuning: {e}")
            raise

    def optimize_storage(
        self,
        output_dir: str,
        quantize_bits: int = 4,
        use_external_storage: bool = True,
        storage_type: str = "gdrive",
        remote_path: str = "DeepSeek_Models"
    ) -> Dict[str, Any]:
        """
        Optimize storage for DeepSeek models

        Args:
            output_dir: Directory to save the optimized model
            quantize_bits: Number of bits for quantization (4 or 8)
            use_external_storage: Whether to use external storage
            storage_type: Type of external storage ('gdrive')
            remote_path: Path in external storage

        Returns:
            Dictionary with optimization results
        """
        ensure_directory_exists(output_dir)

        start_time = time.time()
        model_size_before = 0
        model_size_after = 0

        try:
            # Load the model if not already loaded
            if self.model is None or self.tokenizer is None:
                self.load_model()

            # Calculate original model size
            with tempfile.TemporaryDirectory() as temp_dir:
                self.model.save_pretrained(temp_dir)
                model_size_before = get_directory_size(temp_dir)

            # Perform quantization
            if quantize_bits in (4, 8):
                logger.info(f"Quantizing model to {quantize_bits}-bit precision")

                if self.use_unsloth and UNSLOTH_AVAILABLE:
                    # Unsloth quantization
                    if quantize_bits == 4:
                        model, tokenizer = FastLanguageModel.from_pretrained(
                            self.model_name,
                            max_seq_length=2048,
                            dtype=torch.bfloat16,
                            load_in_4bit=True
                        )
                    else:  # 8-bit
                        model, tokenizer = FastLanguageModel.from_pretrained(
                            self.model_name,
                            max_seq_length=2048,
                            dtype=torch.bfloat16,
                            load_in_8bit=True
                        )
                else:
                    # Standard quantization
                    from transformers import BitsAndBytesConfig

                    tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=quantize_bits == 4,
                        load_in_8bit=quantize_bits == 8,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True
                    )

                    model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        quantization_config=quantization_config,
                        device_map="auto",
                        trust_remote_code=True
                    )

                # Save the quantized model
                logger.info(f"Saving quantized model to {output_dir}")
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)

                # Calculate new model size
                model_size_after = get_directory_size(output_dir)
            else:
                # No quantization, just save the model
                logger.info(f"Saving model without quantization to {output_dir}")
                self.model.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)

                # Calculate new model size
                model_size_after = get_directory_size(output_dir)

            # Sync to external storage if requested
            if use_external_storage:
                if storage_type == "gdrive":
                    logger.info(f"Syncing model to Google Drive: {remote_path}")
                    sync_to_gdrive(output_dir, remote_path)

            # Calculate optimization time
            optimization_time = time.time() - start_time

            # Create optimization summary
            optimization_results = {
                "model_name": self.model_name,
                "quantization_bits": quantize_bits,
                "original_size_mb": model_size_before / (1024 * 1024),
                "optimized_size_mb": model_size_after / (1024 * 1024),
                "size_reduction_percent": 100 * (1 - model_size_after / max(1, model_size_before)),
                "optimization_time_seconds": optimization_time,
                "use_external_storage": use_external_storage,
                "storage_type": storage_type if use_external_storage else None,
                "remote_path": remote_path if use_external_storage else None
            }

            # Save optimization summary
            with open(os.path.join(output_dir, "optimization_summary.json"), "w") as f:
                json.dump(optimization_results, f, indent=2)

            return optimization_results

        except Exception as e:
            logger.error(f"Error optimizing storage: {e}")
            raise

    def generate_code(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        num_return_sequences: int = 1
    ) -> str:
        """
        Generate code using the DeepSeek model

        Args:
            prompt: Text prompt for code generation
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Top-p sampling parameter
            num_return_sequences: Number of sequences to return

        Returns:
            Generated code
        """
        # Load model if not already loaded
        if self.model is None or self.tokenizer is None:
            self.load_model()

        # Format the prompt for DeepSeek model
        formatted_prompt = f"### Instruction: Write code for this task:\n{prompt}\n\n### Response:"

        # Tokenize the prompt
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")

        # Move inputs to the correct device
        if hasattr(self.model, "device"):
            device = self.model.device
        else:
            device = next(self.model.parameters()).device

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate code
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode and return generated code (excluding the prompt)
        generated_texts = []
        for output in outputs:
            generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
            # Extract just the generated response part
            response_marker = "### Response:"
            if response_marker in generated_text:
                generated_text = generated_text.split(response_marker, 1)[1].strip()
            generated_texts.append(generated_text)

        # Return a single string or list based on num_return_sequences
        if num_return_sequences == 1:
            return generated_texts[0]
        else:
            return generated_texts


# Helper functions

def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon (M1/M2/M3)"""
    import platform
    return platform.system() == "Darwin" and platform.machine() == "arm64"

def get_directory_size(directory: str) -> int:
    """Get the total size of a directory in bytes"""
    total_size = 0
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            # Skip if it's a symbolic link
            if not os.path.islink(filepath):
                total_size += os.path.getsize(filepath)
    return total_size

def create_mini_dataset(input_file: str, output_file: str, n_samples: int = 100) -> bool:
    """
    Create a mini dataset from a larger dataset file

    Args:
        input_file: Path to the input dataset file
        output_file: Path to save the mini dataset
        n_samples: Number of samples to include

    Returns:
        bool: Whether the creation was successful
    """
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)

        # Handle different dataset formats
        if isinstance(data, list):
            mini_data = data[:n_samples]
        elif isinstance(data, dict) and "data" in data:
            mini_data = {
                "data": data["data"][:n_samples],
                "metadata": data.get("metadata", {})
            }
        else:
            logger.error(f"Unsupported dataset format in {input_file}")
            return False

        # Save the mini dataset
        with open(output_file, 'w') as f:
            json.dump(mini_data, f, indent=2)

        logger.info(f"Created mini dataset with {len(mini_data) if isinstance(mini_data, list) else len(mini_data['data'])} samples")
        return True

    except Exception as e:
        logger.error(f"Error creating mini dataset: {e}")
        return False

# Command-line interface for the module
def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="DeepSeek model handling utilities")

    # Sub-commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Fine-tuning command
    finetune_parser = subparsers.add_parser("finetune", help="Fine-tune a DeepSeek model")
    finetune_parser.add_argument("--model", type=str, default="deepseek-ai/deepseek-coder-6.7b-base",
                              help="Name or path of the base model")
    finetune_parser.add_argument("--dataset", type=str, required=True,
                              help="Path to the dataset file")
    finetune_parser.add_argument("--output-dir", type=str, default="models/deepseek_finetuned",
                              help="Directory to save the fine-tuned model")
    finetune_parser.add_argument("--epochs", type=int, default=3,
                              help="Number of training epochs")
    finetune_parser.add_argument("--batch-size", type=int, default=4,
                              help="Batch size for training")
    finetune_parser.add_argument("--learning-rate", type=float, default=2e-5,
                              help="Learning rate for training")
    finetune_parser.add_argument("--sequence-length", type=int, default=2048,
                              help="Maximum sequence length")
    finetune_parser.add_argument("--load-in-4bit", action="store_true",
                              help="Use 4-bit quantization during training")
    finetune_parser.add_argument("--use-unsloth", action="store_true",
                              help="Use Unsloth for optimization")
    finetune_parser.add_argument("--force-gpu", action="store_true",
                              help="Force GPU usage")
    finetune_parser.add_argument("--eval-split", type=float, default=0.1,
                              help="Fraction of data to use for evaluation")
    finetune_parser.add_argument("--max-samples", type=int, default=None,
                              help="Maximum number of samples to use from the dataset")
    finetune_parser.add_argument("--all-subsets", action="store_true",
                              help="Use all subsets in the code dataset")
    finetune_parser.add_argument("--subset", type=str, default=None,
                              help="Specific subset to use from the code dataset")

    # Optimization command
    optimize_parser = subparsers.add_parser("optimize", help="Optimize a DeepSeek model for storage")
    optimize_parser.add_argument("--model", type=str, default="deepseek-ai/deepseek-coder-6.7b-base",
                             help="Name or path of the model to optimize")
    optimize_parser.add_argument("--output-dir", type=str, default="models/deepseek_optimized",
                             help="Directory to save the optimized model")
    optimize_parser.add_argument("--quantize-bits", type=int, choices=[4, 8], default=4,
                             help="Number of bits for quantization")
    optimize_parser.add_argument("--use-external-storage", action="store_true",
                             help="Use external storage for the model")
    optimize_parser.add_argument("--storage-type", type=str, choices=["gdrive"], default="gdrive",
                             help="Type of external storage")
    optimize_parser.add_argument("--remote-path", type=str, default="DeepSeek_Models",
                             help="Path in external storage")

    # Generation command
    generate_parser = subparsers.add_parser("generate", help="Generate code using a DeepSeek model")
    generate_parser.add_argument("--model-path", type=str, default=None,
                             help="Path to a fine-tuned model (or None to use the base model)")
    generate_parser.add_argument("--prompt", type=str, required=True,
                             help="Prompt for code generation")
    generate_parser.add_argument("--max-tokens", type=int, default=512,
                             help="Maximum number of tokens to generate")
    generate_parser.add_argument("--temperature", type=float, default=0.7,
                             help="Sampling temperature")
    generate_parser.add_argument("--output-file", type=str, default=None,
                             help="File to save the generated code (or None to print to stdout)")

    # Mini dataset command
    mini_parser = subparsers.add_parser("create-mini", help="Create a mini dataset for testing")
    mini_parser.add_argument("--input-file", type=str, required=True,
                          help="Path to the input dataset file")
    mini_parser.add_argument("--output-file", type=str, required=True,
                          help="Path to save the mini dataset")
    mini_parser.add_argument("--n-samples", type=int, default=100,
                          help="Number of samples to include")

    return parser.parse_args()

def main():
    """Main entry point for the script"""
    args = parse_args()

    if args.command == "finetune":
        # Fine-tune a DeepSeek model

        # Load the dataset
        try:
            from src.generative_ai_module.code_preprocessing import load_and_preprocess_dataset

            logger.info(f"Loading dataset from {args.dataset}")

            if os.path.exists(args.dataset):
                # Load from a file
                with open(args.dataset, 'r') as f:
                    data = json.load(f)

                # TODO: Process the dataset into train and eval sets
                # For now, just print a message
                logger.info("Loading dataset from file not implemented yet")

            else:
                # Use the built-in dataset loader
                train_dataset, eval_dataset = load_and_preprocess_dataset(
                    max_samples=args.max_samples,
                    sequence_length=args.sequence_length,
                    subset=args.subset,
                    all_subsets=args.all_subsets
                )

                # Create the DeepSeek handler
                handler = DeepSeekHandler(
                    model_name=args.model,
                    use_unsloth=args.use_unsloth,
                    load_in_4bit=args.load_in_4bit,
                    force_gpu=args.force_gpu,
                    output_dir=args.output_dir
                )

                # Fine-tune the model
                metrics = handler.fine_tune(
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    output_dir=args.output_dir,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    sequence_length=args.sequence_length
                )

                # Print metrics
                logger.info("Fine-tuning completed with metrics:")
                for key, value in metrics.items():
                    logger.info(f"  {key}: {value}")

        except Exception as e:
            logger.error(f"Error during fine-tuning: {e}")
            return 1

    elif args.command == "optimize":
        # Optimize a DeepSeek model for storage
        handler = DeepSeekHandler(
            model_name=args.model,
            use_unsloth=True
        )

        # Optimize the model
        try:
            results = handler.optimize_storage(
                output_dir=args.output_dir,
                quantize_bits=args.quantize_bits,
                use_external_storage=args.use_external_storage,
                storage_type=args.storage_type,
                remote_path=args.remote_path
            )

            # Print results
            logger.info("Storage optimization completed with results:")
            for key, value in results.items():
                logger.info(f"  {key}: {value}")

        except Exception as e:
            logger.error(f"Error during storage optimization: {e}")
            return 1

    elif args.command == "generate":
        # Generate code using a DeepSeek model
        handler = DeepSeekHandler(
            use_unsloth=True,
            force_gpu=True
        )

        # Load the model
        try:
            handler.load_model(args.model_path)

            # Generate code
            generated_code = handler.generate_code(
                prompt=args.prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature
            )

            # Save or print the generated code
            if args.output_file:
                with open(args.output_file, 'w') as f:
                    f.write(generated_code)
                logger.info(f"Generated code saved to {args.output_file}")
            else:
                print("\nGenerated Code:")
                print("=" * 40)
                print(generated_code)
                print("=" * 40)

        except Exception as e:
            logger.error(f"Error during code generation: {e}")
            return 1

    elif args.command == "create-mini":
        # Create a mini dataset for testing
        try:
            success = create_mini_dataset(
                input_file=args.input_file,
                output_file=args.output_file,
                n_samples=args.n_samples
            )

            if success:
                logger.info(f"Mini dataset created successfully: {args.output_file}")
            else:
                logger.error("Failed to create mini dataset")
                return 1

        except Exception as e:
            logger.error(f"Error creating mini dataset: {e}")
            return 1

    else:
        logger.error("Please specify a command: finetune, optimize, generate, or create-mini")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())