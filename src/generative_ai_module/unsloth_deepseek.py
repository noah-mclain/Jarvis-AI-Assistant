"""
DeepSeek-Coder Fine-tuning with Unsloth Optimization

This module provides functions to fine-tune DeepSeek-Coder models using Unsloth optimization.
Unsloth speeds up LLM training and inference while reducing memory usage.

Functions:
    - get_unsloth_model: Load a DeepSeek model optimized with Unsloth
    - finetune_with_unsloth: Fine-tune a DeepSeek model using Unsloth
    - evaluate_model: Evaluate a fine-tuned model on test data
    - create_text_dataset_from_tokenized: Convert tokenized dataset to text format
    - preprocess_for_unsloth: Preprocess code dataset for Unsloth compatibility
"""

# Import unsloth first, before transformers and other libraries
# This ensures all optimizations are properly applied

from typing import Optional, Dict, List, Any, Union
import sys
import logging
import os
import time
import datetime
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define minimal dependencies that will always be available
try:
    import torch
    import numpy as np
except ImportError as e:
    logger.error(f"Critical dependency missing: {e}")
    # Create stub versions for torch and numpy to prevent immediate errors
    class StubModule:
        def __getattr__(self, _):
            return lambda *args, **kwargs: None

    torch = StubModule()
    torch.cuda = StubModule()
    torch.cuda.is_available = lambda: False
    torch.Tensor = list
    torch.device = lambda x: x
    torch.nn = StubModule()
    torch.no_grad = lambda: StubModule()
    np = StubModule()

# Default flags in case imports fail
UNSLOTH_AVAILABLE = False
TRL_HAS_SFT_CONFIG = False
TRANSFORMERS_AVAILABLE = False

# Import unsloth with fallback mechanisms
try:
    import unsloth
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
    # Try to import DeepSeek support - this might fail in minimal installations
    try:
        from unsloth.models import FastDeepseekV2ForCausalLM
        DEEPSEEK_NATIVE_SUPPORT = True
        logger.info("Loaded Unsloth with native DeepSeek support")
    except ImportError:
        DEEPSEEK_NATIVE_SUPPORT = False
        logger.warning("Using minimal Unsloth without specialized DeepSeek support")
except ImportError as e:
    logger.warning(f"Unsloth not available: {e}")
    UNSLOTH_AVAILABLE = False
    DEEPSEEK_NATIVE_SUPPORT = False
    # Create stub for FastLanguageModel
    class FastLanguageModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise RuntimeError("Unsloth not available")

        @staticmethod
        def get_peft_model(*args, **kwargs):
            raise RuntimeError("Unsloth not available")

# Try to import Dataset class or create stub
try:
    from datasets import Dataset
except ImportError:
    logger.warning("datasets library not available")
    # Create stub Dataset class
    class Dataset:
        def __init__(self, data=None):
            self.data = data or {}

        def __getitem__(self, idx):
            return self.data.get(idx, {})

        def __len__(self):
            return len(self.data)

# Try to import tqdm
try:
    from tqdm import tqdm
except ImportError:
    # Simple tqdm replacement
    def tqdm(iterable, **kwargs):
        desc = kwargs.get('desc', '')
        logger.info(f"Starting: {desc}")
        for i, item in enumerate(iterable):
            if i % 10 == 0:
                logger.info(f"{desc}: {i}/{len(iterable) if hasattr(iterable, '__len__') else '?'}")
            yield item

# Conditional imports for transformers components
try:
    from transformers import AutoTokenizer, TrainingArguments
    TRANSFORMERS_AVAILABLE = True
    # Try to import from TRL with version compatibility handling
    try:
        # First, try to fix the top_k_top_p_filtering import issue
        try:
            # Check if we need to add the function
            try:
                from transformers import top_k_top_p_filtering
                logger.info("top_k_top_p_filtering is already available in transformers")
            except ImportError:
                # Add the function to transformers
                logger.info("Adding top_k_top_p_filtering to transformers")

                # Find transformers path
                import transformers
                import inspect
                import os

                transformers_path = os.path.dirname(inspect.getfile(transformers))

                # Check if we have the generation utils module
                try:
                    from transformers.generation.logits_process import TopKLogitsWarper, TopPLogitsWarper

                    # Define the missing function
                    def top_k_top_p_filtering(
                        logits,
                        top_k=0,
                        top_p=1.0,
                        filter_value=-float("Inf"),
                        min_tokens_to_keep=1,
                    ):
                        """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering."""
                        if top_k > 0:
                            logits = TopKLogitsWarper(top_k=top_k, filter_value=filter_value,
                                                     min_tokens_to_keep=min_tokens_to_keep)(None, logits)

                        if 0 <= top_p < 1.0:
                            logits = TopPLogitsWarper(top_p=top_p, filter_value=filter_value,
                                                     min_tokens_to_keep=min_tokens_to_keep)(None, logits)

                        return logits

                    # Add the function to transformers namespace
                    transformers.top_k_top_p_filtering = top_k_top_p_filtering

                    # Also add to generation utils if it exists
                    if hasattr(transformers, 'generation'):
                        if hasattr(transformers.generation, 'utils'):
                            transformers.generation.utils.top_k_top_p_filtering = top_k_top_p_filtering

                    logger.info("Successfully added top_k_top_p_filtering to transformers")
                except Exception as patch_error:
                    logger.warning(f"Failed to add top_k_top_p_filtering: {patch_error}")
        except Exception as fix_error:
            logger.warning(f"Error fixing transformers imports: {fix_error}")

        # Now try to import TRL
        import trl
        from trl import SFTTrainer

        # Check if SFTConfig exists in this version of TRL
        try:
            from trl import SFTConfig
            TRL_HAS_SFT_CONFIG = True
            logger.info(f"Using TRL version {getattr(trl, '__version__', 'unknown')} with SFTConfig")
        except ImportError:
            TRL_HAS_SFT_CONFIG = False
            logger.info(f"Using TRL version {getattr(trl, '__version__', 'unknown')} without SFTConfig")

        from peft import LoraConfig
    except ImportError as e:
        logger.warning(f"TRL/PEFT not available: {e}")

        # Create stub LoraConfig for imports
        class LoraConfig:
            def __init__(self, *args, **kwargs):
                pass
except ImportError as e:
    logger.warning(f"Transformers not available: {e}")
    TRANSFORMERS_AVAILABLE = False

    # Create stub AutoTokenizer for imports
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return None

# Import storage optimization utilities with fallback stubs
try:
    from .storage_optimization import (
        create_checkpoint_strategy, manage_checkpoints,
        setup_streaming_dataset, compress_dataset, optimize_storage_for_model,
        setup_google_drive, setup_s3_storage, upload_to_gdrive, upload_to_s3
    )
except ImportError as e:
    logger.warning(f"Storage optimization utilities not available: {e}")
    # Define placeholder functions for storage optimization
    def create_checkpoint_strategy(*args, **kwargs): return {"mode": "none"}
    def manage_checkpoints(*args, **kwargs): pass
    def setup_streaming_dataset(*args, **kwargs): return None
    def compress_dataset(*args, **kwargs): return None
    def optimize_storage_for_model(*args, **kwargs): return None
    def setup_google_drive(*args, **kwargs): return None
    def setup_s3_storage(*args, **kwargs): return None
    def upload_to_gdrive(*args, **kwargs): return None
    def upload_to_s3(*args, **kwargs): return None

def get_unsloth_model(
    model_name: str = "deepseek-ai/deepseek-coder-6.7b-base",
    model_dir: Optional[str] = None,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    use_peft: bool = True,
    r: int = 16,
    target_modules: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    alpha: int = 16,
    dropout: float = 0.1
) -> tuple:
    """
    Load a DeepSeek-Coder model with Unsloth optimization.

    Args:
        model_name: The model name or path to load
        model_dir: Directory containing fine-tuned weights (None for base model)
        max_seq_length: Maximum sequence length for the model
        load_in_4bit: Whether to load the model in 4-bit quantization
        load_in_8bit: Whether to load the model in 8-bit quantization
        use_peft: Whether to use PEFT/LoRA for fine-tuning
        r: Rank for LoRA fine-tuning
        target_modules: Which modules to apply LoRA to
        alpha: LoRA alpha parameter
        dropout: Dropout rate for LoRA

    Returns:
        Tuple of (model, tokenizer)
    """
    # Check if we have GPU support
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. This will be extremely slow.")
    else:
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

    try:
        # Load model and tokenizer with Unsloth optimization
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit
            # Don't set device_map here as it's already set by FastLanguageModel internally
        )

        # Apply LoRA if using PEFT
        if use_peft:
            lora_config = LoraConfig(
                r=r,
                target_modules=target_modules,
                lora_alpha=alpha,
                lora_dropout=dropout,
                task_type="CAUSAL_LM",
            )

            # Apply LoRA config
            model = FastLanguageModel.get_peft_model(
                model,
                lora_config,
                # For inference, we can disable gradient checkpointing to save memory
                inference_mode=(model_dir is not None)
            )

        # Load fine-tuned weights if provided
        if model_dir and os.path.exists(model_dir):
            model.load_adapter(model_dir, adapter_name="default")
            logger.info(f"Loaded fine-tuned weights from {model_dir}")

        return model, tokenizer

    except Exception as e:
        logger.error(f"Error loading model with Unsloth: {e}")
        logger.warning("Falling back to standard transformers loading")

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            # Load with standard transformers
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit
            )

            # Load LoRA weights if provided
            if model_dir and os.path.exists(model_dir) and use_peft:
                try:
                    from peft import PeftModel
                    model = PeftModel.from_pretrained(model, model_dir)
                    logger.info(f"Loaded fine-tuned weights from {model_dir}")
                except Exception as peft_error:
                    logger.error(f"Error loading LoRA weights: {peft_error}")

            return model, tokenizer

        except Exception as fallback_error:
            logger.error(f"Fallback loading also failed: {fallback_error}")
            raise RuntimeError("Failed to load model in any way. Please check your installation.")

def finetune_with_unsloth(
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    model_name: str = "deepseek-ai/deepseek-coder-6.7b-base",
    output_dir: str = "models/deepseek_unsloth",
    max_seq_length: int = 2048,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-5,
    max_steps: int = 500,
    logging_steps: int = 10,
    save_steps: int = 100,
    warmup_steps: int = 50,
    weight_decay: float = 0.01,
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    r: int = 16,
    target_modules: list[str] = None,
    save_total_limit: int = 3,
    # Storage optimization parameters
    use_storage_optimization: bool = False,
    storage_config: dict = None
):
    """
    Fine-tune a DeepSeek-Coder model with Unsloth optimization.

    Args:
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        model_name: The model name or path to fine-tune
        output_dir: Directory to save fine-tuned model
        max_seq_length: Maximum sequence length for the model
        per_device_train_batch_size: Batch size per device during training
        gradient_accumulation_steps: Number of updates steps to accumulate before backward pass
        learning_rate: Learning rate for training
        max_steps: Maximum number of training steps
        logging_steps: Log metrics every X steps
        save_steps: Save model checkpoint every X steps
        warmup_steps: Number of steps for learning rate warm-up
        weight_decay: Weight decay for regularization
        load_in_4bit: Whether to load the model in 4-bit quantization
        load_in_8bit: Whether to load the model in 8-bit quantization
        r: Rank for LoRA fine-tuning
        target_modules: Which modules to apply LoRA to
        save_total_limit: Maximum number of checkpoints to save
        use_storage_optimization: Whether to use storage optimization strategies
        storage_config: Configuration for storage optimization

    Returns:
        Dictionary with training metrics
    """
    start_time = time.time()

    # Set default target modules for DeepSeek-Coder if not specified
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # Set up storage optimization if requested
    if use_storage_optimization:
        if storage_config is None:
            storage_config = {
                "checkpoint_strategy": "improvement",
                "max_checkpoints": 2,
                "use_external_storage": False,
                "storage_type": "gdrive",
                "remote_path": "DeepSeek_Models"
            }

        # Create checkpoint strategy
        checkpoint_strategy = create_checkpoint_strategy(
            total_steps=max_steps,
            save_mode=storage_config.get("checkpoint_strategy", "improvement"),
            save_interval=save_steps,
            max_checkpoints=storage_config.get("max_checkpoints", 2)
        )

        # Setup external storage if enabled
        remote_storage_func = None
        if storage_config.get("use_external_storage", False):
            storage_type = storage_config.get("storage_type", "gdrive")
            if storage_type == "gdrive":
                drive_client = setup_google_drive()
                if drive_client:
                    remote_path = storage_config.get("remote_path", "DeepSeek_Models")
                    remote_storage_func = lambda path: upload_to_gdrive(path, drive_folder=remote_path)
            elif storage_type == "s3":
                s3_client = setup_s3_storage(
                    storage_config.get("aws_access_key_id"),
                    storage_config.get("aws_secret_access_key")
                )
                if s3_client and storage_config.get("s3_bucket"):
                    remote_path = storage_config.get("remote_path", "deepseek_models")
                    s3_bucket = storage_config.get("s3_bucket")
                    remote_storage_func = lambda path: upload_to_s3(
                        s3_client, path, s3_bucket, s3_prefix=f"{remote_path}/"
                    )
    else:
        checkpoint_strategy = None
        remote_storage_func = None

    # Load model and tokenizer with Unsloth optimization
    model, tokenizer = get_unsloth_model(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        use_peft=True,
        r=r,
        target_modules=target_modules
    )

    # Create training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_steps=max_steps,
        logging_steps=logging_steps,
        # Adjust save behavior based on storage optimization
        save_steps=save_steps if not use_storage_optimization else max_steps + 1,  # Disable default saving if using custom strategy
        save_total_limit=save_total_limit if not use_storage_optimization else None,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        lr_scheduler_type="cosine",
        fp16=not load_in_4bit,  # Use fp16 if not using 4-bit quantization
        bf16=False,
        optim="adamw_torch",
        report_to="none",  # Disable reporting to wandb or other services by default
        group_by_length=True,  # More efficient batching by sequence length
        save_strategy="steps" if not use_storage_optimization else "no",  # Disable default saving if using custom strategy
        remove_unused_columns=True,
        run_name="deepseek_unsloth"
    )

    # Custom evaluation callback for storage-optimized checkpointing
    if use_storage_optimization:
        from transformers.trainer_callback import TrainerCallback

        class StorageOptimizedCallback(TrainerCallback):
            def __init__(self, checkpoint_strategy, output_dir, remote_storage_func):
                self.checkpoint_strategy = checkpoint_strategy
                self.output_dir = output_dir
                self.remote_storage_func = remote_storage_func
                self.best_metric = float('inf')

            def on_step_end(self, args, state, control, **kwargs):
                # Check if we should save based on strategy
                if self.checkpoint_strategy:
                    should_save, checkpoint_path = manage_checkpoints(
                        self.checkpoint_strategy,
                        state.global_step,
                        self.output_dir,
                        remote_storage_func=self.remote_storage_func
                    )
                    if should_save:
                        print(f"Saving checkpoint at step {state.global_step} to {checkpoint_path}")
                        # Trigger trainer save
                        control.should_save = True
                        # If we have a custom path, we need to handle that in on_save
                        if checkpoint_path and checkpoint_path != self.output_dir:
                            state.best_model_checkpoint = checkpoint_path

            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                # Save if improvement detected and using "improvement" strategy
                if metrics and self.checkpoint_strategy and 'eval_loss' in metrics:
                    metric_value = metrics['eval_loss']
                    should_save, checkpoint_path = manage_checkpoints(
                        self.checkpoint_strategy,
                        state.global_step,
                        self.output_dir,
                        metric_value=metric_value,
                        remote_storage_func=self.remote_storage_func
                    )
                    if should_save:
                        print(f"Saving improved model at step {state.global_step} with eval_loss {metric_value:.4f}")
                        control.should_save = True
                        if checkpoint_path and checkpoint_path != self.output_dir:
                            state.best_model_checkpoint = checkpoint_path

    # Create SFT trainer
    # We need to set tokenizer_name explicitly here for DeepSeek
    if TRL_HAS_SFT_CONFIG:
        # Use the newer TRL API with SFTConfig
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",  # Use 'text' field for training
            max_seq_length=max_seq_length,
            args=training_args,
            packing=True,  # Enable packing for more efficient training
            tokenizer_name=model_name  # Set tokenizer name explicitly
        )
    else:
        # Use the older TRL API without SFTConfig for TRL 0.7.x
        logger.info("Using TRL 0.7.x compatibility mode for SFTTrainer")
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            packing=True,  # Enable packing for more efficient training
            dataset_text_field="text",  # Use 'text' field for training
            max_seq_length=max_seq_length,
        )

    # Add storage optimization callback if needed
    if use_storage_optimization:
        trainer.add_callback(StorageOptimizedCallback(
            checkpoint_strategy=checkpoint_strategy,
            output_dir=output_dir,
            remote_storage_func=remote_storage_func
        ))

    # Train the model
    print("Starting training...")
    trainer.train()

    # Save the fine-tuned model (only LoRA adapters, much smaller than full model)
    model.save_pretrained(output_dir)
    print(f"Model adapters saved to {output_dir}")

    # Store the base model name for future loading
    config_path = os.path.join(output_dir, "base_model_info.json")
    with open(config_path, 'w') as f:
        json.dump({
            "base_model": model_name,
            "date_trained": time.strftime("%Y-%m-%d %H:%M:%S"),
            "quantization": "4bit" if load_in_4bit else "8bit" if load_in_8bit else "none",
            "lora_rank": r,
        }, f, indent=2)

    # Upload final model to external storage if configured
    if use_storage_optimization and remote_storage_func:
        try:
            remote_path = remote_storage_func(output_dir)
            print(f"Uploaded final model to remote storage: {remote_path}")
        except Exception as e:
            print(f"Failed to upload final model to remote storage: {e}")

    # Create metrics dictionary
    metrics = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        "training_time_minutes": round((time.time() - start_time) / 60, 2),
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "lora_rank": r,
    }

    # Save metrics
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics

def evaluate_model(
    model_dir: str = None,
    test_dataset: Dataset = None,
    max_seq_length: int = 2048,
    batch_size: int = 4,
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    model=None,
    tokenizer=None,
) -> dict[str, float]:
    """
    Evaluate a fine-tuned model on test data.

    Args:
        model_dir: Directory containing fine-tuned model (if model not provided)
        test_dataset: Test dataset (can be dict or Dataset object)
        max_seq_length: Maximum sequence length for the model
        batch_size: Batch size for evaluation
        load_in_4bit: Whether to load the model in 4-bit quantization
        load_in_8bit: Whether to load the model in 8-bit quantization
        model: Pre-loaded model (optional)
        tokenizer: Pre-loaded tokenizer (optional)

    Returns:
        Dictionary with evaluation metrics
    """
    # Check if required dependencies are available
    if not TRANSFORMERS_AVAILABLE or not torch.cuda.is_available():
        error_msg = "Transformers or PyTorch CUDA not available for evaluation"
        logger.error(error_msg)
        return {
            "error": error_msg,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    start_time = time.time()

    # If no test dataset was provided, return empty metrics
    if test_dataset is None:
        logger.warning("No test dataset provided for evaluation")
        return {
            "error": "No test dataset provided",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    # Load the model and tokenizer if not provided
    if model is None or tokenizer is None:
        if model_dir is None:
            logger.error("Either model or model_dir must be provided")
            return {"error": "Either model or model_dir must be provided"}

        try:
            # Try to get base model name from config
            base_model_name = "deepseek-ai/deepseek-coder-6.7b-base"  # Default
            config_file = None

            # Look for adapter config file
            for filename in ["adapter_config.json", "config.json", "base_model_info.json"]:
                path = os.path.join(model_dir, filename)
                if os.path.exists(path):
                    config_file = path
                    break

            if config_file:
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                        if "base_model_name_or_path" in config:
                            base_model_name = config["base_model_name_or_path"]
                        elif "base_model" in config:
                            base_model_name = config["base_model"]
                except Exception as e:
                    logger.warning(f"Error loading config file {config_file}: {e}")

            logger.info(f"Using base model: {base_model_name}")

            # Load the fine-tuned model
            if UNSLOTH_AVAILABLE:
                try:
                    model, tokenizer = get_unsloth_model(
                        model_name=base_model_name,
                        model_dir=model_dir,
                        max_seq_length=max_seq_length,
                        load_in_4bit=load_in_4bit,
                        load_in_8bit=load_in_8bit
                    )
                except Exception as e:
                    logger.warning(f"Failed to load with Unsloth: {e}, falling back to transformers")
                    from transformers import AutoModelForCausalLM, AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(model_dir)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_dir,
                        device_map="auto",
                        torch_dtype=torch.float16,
                    )
            else:
                # Standard transformers loading
                from transformers import AutoModelForCausalLM, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(model_dir)
                model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
        except Exception as e:
            logger.error(f"Failed to load model for evaluation: {e}")
            return {
                "error": f"Failed to load model: {str(e)}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

    # Ensure model is in evaluation mode
    model.eval()

    # Get device
    device = model.device if hasattr(model, 'device') else next(model.parameters()).device
    logger.info(f"Evaluating on device: {device}")

    # Initialize metrics
    total_loss = 0.0
    num_samples = 0

    # Convert test_dataset to Dataset if it's a dictionary
    if isinstance(test_dataset, dict):
        if "input_ids" in test_dataset:
            # Handle tokenized dataset
            if tokenizer:
                test_dataset = create_text_dataset_from_tokenized(test_dataset, tokenizer)
            else:
                logger.error("Tokenizer required to convert tokenized dataset to text")
                return {"error": "Tokenizer required for tokenized dataset"}
        elif "text" in test_dataset:
            # Convert to Huggingface Dataset
            test_dataset = Dataset.from_dict(test_dataset)
        else:
            logger.error("Test dataset must contain 'input_ids' or 'text'")
            return {"error": "Invalid test dataset format"}

    try:
        # Process dataset in batches
        progress_bar = tqdm(range(0, len(test_dataset), batch_size), desc="Evaluating")
        for i in progress_bar:
            batch = test_dataset[i:min(i+batch_size, len(test_dataset))]

            # Get text data from batch
            if "text" in batch:
                batch_texts = batch["text"]
            elif "input_ids" in batch and tokenizer:
                batch_texts = [tokenizer.decode(ids) for ids in batch["input_ids"]]
            else:
                logger.error("Batch must contain 'text' or 'input_ids'")
                continue

            # Skip empty batches
            if not batch_texts or len(batch_texts) == 0:
                continue

            # Handle single string case
            if isinstance(batch_texts, str):
                batch_texts = [batch_texts]

            try:
                # Tokenize inputs
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_seq_length
                ).to(device)

                # Calculate loss
                with torch.no_grad():
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss

                # Update metrics
                batch_loss = loss.item() * len(batch_texts)
                total_loss += batch_loss
                num_samples += len(batch_texts)

                # Update progress bar
                progress_bar.set_postfix({"avg_loss": total_loss / max(1, num_samples)})

            except Exception as batch_error:
                logger.warning(f"Error processing batch: {batch_error}")
                continue

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return {
            "error": f"Evaluation failed: {str(e)}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    # Calculate final metrics
    if num_samples == 0:
        logger.warning("No samples were successfully evaluated")
        return {
            "error": "No samples successfully evaluated",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    avg_loss = total_loss / num_samples

    try:
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
    except OverflowError:
        # Handle overflow for very large loss values
        perplexity = float('inf')

    evaluation_time = time.time() - start_time

    metrics = {
        "loss": avg_loss,
        "perplexity": perplexity,
        "num_samples": num_samples,
        "evaluation_time_seconds": round(evaluation_time, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # Save metrics if model_dir is provided
    if model_dir:
        try:
            metrics_path = os.path.join(model_dir, "evaluation_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Saved evaluation metrics to {metrics_path}")
        except Exception as e:
            logger.warning(f"Failed to save metrics to file: {e}")

    return metrics

def create_text_dataset_from_tokenized(
    dataset: dict,
    tokenizer: any
) -> Dataset:
    """
    Convert a tokenized dataset to text format for Unsloth.

    Args:
        dataset: Tokenized dataset
        tokenizer: The tokenizer for the model

    Returns:
        Text-based dataset
    """
    # Decode tokens to texts
    texts = [tokenizer.decode(ids) for ids in dataset["input_ids"]]

    # Create a new dataset with texts
    return Dataset.from_dict({"text": texts})

def preprocess_for_unsloth(
    examples: dict,
    tokenizer: any,
    max_seq_length: int = 2048
) -> dict:
    """
    Preprocess dataset examples for Unsloth training.

    Args:
        examples: Dataset examples
        tokenizer: The tokenizer for the model
        max_seq_length: Maximum sequence length for the model

    Returns:
        Processed examples
    """
    # If we already have text data, just return it
    if "text" in examples:
        return examples

    # Convert input_ids back to text if needed
    if "input_ids" in examples:
        if not isinstance(examples["input_ids"], list):
            # Handle single example
            return {"text": tokenizer.decode(examples["input_ids"])}

        # Handle batch of examples
        texts = [tokenizer.decode(ids) for ids in examples["input_ids"]]
        return {"text": texts}
    # If we have prompt and completion fields (instruction format)
    if all(key in examples for key in ["instruction", "response"]):
        texts = []
        for i in range(len(examples["instruction"])):
            # Format as instruction-response pair
            prompt = examples["instruction"][i]
            response = examples["response"][i]

            # Create the formatted text
            if "### Instruction:" not in prompt:
                text = f"### Instruction: {prompt}\n\n### Response: {response}"
            else:
                # Already formatted, just combine
                text = f"{prompt}\n\n### Response: {response}"

            texts.append(text)

        return {"text": texts}

    # If we have raw code samples
    return {"text": examples["code"]} if "code" in examples else examples

def finetune_with_optimal_storage(
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    model_name: str = "deepseek-ai/deepseek-coder-6.7b-base",
    output_dir: str = "models/deepseek_unsloth",
    max_seq_length: int = 2048,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-5,
    max_steps: int = 500,
    warmup_steps: int = 50,
    quantize_bits: int = 4,  # Use 4-bit quantization by default for maximum storage efficiency
    r: int = 16,  # LoRA rank
    # Storage optimization parameters
    storage_type: str = "gdrive",  # "gdrive", "s3", or "local"
    remote_path: str = "DeepSeek_Models",
    checkpoint_strategy: str = "improvement",
    max_checkpoints: int = 2,
    # AWS parameters (if using S3)
    aws_access_key_id: str = None,
    aws_secret_access_key: str = None,
    s3_bucket: str = None,
):
    """
    Fine-tune a DeepSeek model with optimal storage usage, ideal for environments with limited storage.

    This function implements:
    1. Model quantization (4-bit or 8-bit)
    2. External cloud storage integration (Google Drive or S3)
    3. Efficient checkpointing strategies
    4. LoRA for parameter-efficient fine-tuning

    Args:
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        model_name: The model name or path to fine-tune
        output_dir: Directory to save fine-tuned model
        max_seq_length: Maximum sequence length for the model
        per_device_train_batch_size: Batch size per device during training
        gradient_accumulation_steps: Number of updates steps to accumulate before backward pass
        learning_rate: Learning rate for training
        max_steps: Maximum number of training steps
        warmup_steps: Number of steps for learning rate warm-up
        quantize_bits: Quantization precision (4 or 8)
        r: Rank for LoRA fine-tuning
        storage_type: Type of external storage ("gdrive", "s3", or "local")
        remote_path: Path in external storage
        checkpoint_strategy: Strategy for saving checkpoints ("improvement", "regular", or "hybrid")
        max_checkpoints: Maximum number of checkpoints to keep
        aws_access_key_id: AWS access key ID (for S3 storage)
        aws_secret_access_key: AWS secret access key (for S3 storage)
        s3_bucket: S3 bucket name (for S3 storage)

    Returns:
        Dictionary with training metrics
    """
    # Configure storage optimization
    use_external_storage = storage_type in ["gdrive", "s3"]

    # Setup storage configuration
    storage_config = {
        "checkpoint_strategy": checkpoint_strategy,
        "max_checkpoints": max_checkpoints,
        "use_external_storage": use_external_storage,
        "storage_type": storage_type,
        "remote_path": remote_path
    }

    # Add AWS credentials if using S3
    if storage_type == "s3":
        storage_config.update({
            "aws_access_key_id": aws_access_key_id,
            "aws_secret_access_key": aws_secret_access_key,
            "s3_bucket": s3_bucket
        })

    # Determine quantization settings
    load_in_4bit = quantize_bits == 4
    load_in_8bit = quantize_bits == 8

    print(f"Fine-tuning with optimal storage settings:")
    print(f"- Using {quantize_bits}-bit quantization")
    print(f"- LoRA rank: {r}")
    print(f"- External storage: {storage_type if use_external_storage else 'disabled'}")
    print(f"- Checkpoint strategy: {checkpoint_strategy} (max {max_checkpoints})")

    # Run the fine-tuning with storage optimization
    metrics = finetune_with_unsloth(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model_name=model_name,
        output_dir=output_dir,
        max_seq_length=max_seq_length,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        # Storage-optimized settings
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        r=r,
        # Storage optimization
        use_storage_optimization=True,
        storage_config=storage_config
    )

    # Ensure the metrics include storage optimization info
    metrics.update({
        "storage_optimization": {
            "quantize_bits": quantize_bits,
            "storage_type": storage_type,
            "checkpoint_strategy": checkpoint_strategy,
            "max_checkpoints": max_checkpoints,
            "external_storage_used": use_external_storage
        }
    })

    # Save updated metrics
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics

if __name__ == "__main__":
    # Simple test with mini dataset
    from finetune_deepseek import create_mini_dataset

    # Create minimal test dataset
    train_dataset, eval_dataset = create_mini_dataset(sequence_length=512)

    # We need to convert tokenized dataset to text format for Unsloth
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True)
    train_text_dataset = create_text_dataset_from_tokenized(train_dataset, tokenizer)
    eval_text_dataset = create_text_dataset_from_tokenized(eval_dataset, tokenizer)

    # Fine-tune with Unsloth
    finetune_with_unsloth(
        train_text_dataset,
        eval_text_dataset,
        output_dir="models/deepseek_unsloth_test",
        max_steps=10,  # Short test run
    )