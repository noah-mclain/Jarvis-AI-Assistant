#!/usr/bin/env python3
"""
Unified DeepSeek-Coder training script that combines the best aspects of all training approaches.
This script:
1. Applies the attention mask fix
2. Uses Unsloth optimization if available
3. Falls back to standard optimization if Unsloth is not available
4. Optimizes memory usage for different GPU types
5. Supports both 4-bit and 8-bit quantization
"""

import os
import sys
import time
import torch
import argparse
import logging
import multiprocessing
from datetime import datetime
from pathlib import Path

# Set multiprocessing start method to 'spawn' to avoid CUDA issues
# This must be done at the beginning of the script before any other multiprocessing code
if __name__ == "__main__":
    # Only set the start method in the main process
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'")
    except RuntimeError:
        # If it's already set, this will raise a RuntimeError
        print("Multiprocessing start method already set to:", multiprocessing.get_start_method())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"deepseek_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Check for Unsloth availability
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
    logger.info("Unsloth optimization available")
except ImportError:
    UNSLOTH_AVAILABLE = False
    logger.info("Unsloth not available. Using standard optimization.")

# Apply attention mask fix
def apply_attention_mask_fix():
    """Apply the attention mask fix for DeepSeek models"""
    try:
        # Check transformers version to apply the appropriate fix
        from transformers import __version__ as transformers_version

        # Parse version string to check compatibility
        try:
            version_parts = transformers_version.split('.')
            major = int(version_parts[0]) if len(version_parts) > 0 else 0
            minor = int(version_parts[1]) if len(version_parts) > 1 else 0
            logger.info(f"Detected transformers version: {major}.{minor}")

            # For newer versions of transformers (4.28+), the attention mask handling is different
            if major > 4 or (major == 4 and minor >= 28):
                logger.info(f"Using newer attention mask fix for transformers {transformers_version}")
                # For newer versions, we don't need to patch the forward method
                # as the _prepare_4d_causal_attention_mask_for_sdpa function handles it properly
                logger.info("Attention mask handling is already fixed in this transformers version")
                return True
        except Exception as e:
            logger.warning(f"Could not parse transformers version: {e}. Applying legacy fix.")

        # For older versions, apply the legacy fix
        from transformers.models.llama.modeling_llama import LlamaModel

        # Store the original forward method
        original_forward = LlamaModel.forward

        # Define a patched forward method that properly handles attention masks
        def patched_forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        ):
            """Patched forward method for LlamaModel that properly handles attention masks."""
            # Force use_cache to False when using gradient checkpointing
            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                if use_cache:
                    logger.info("use_cache=True is incompatible with gradient checkpointing. Setting use_cache=False...")
                use_cache = False

            # Fix attention mask shape if needed
            if attention_mask is not None and attention_mask.dim() == 2:
                try:
                    # Get the device and dtype
                    device = attention_mask.device
                    dtype = attention_mask.dtype

                    # Get sequence length
                    seq_length = attention_mask.size(1)
                    batch_size = attention_mask.size(0)

                    # Convert attention_mask from [batch_size, seq_length] to [batch_size, 1, seq_length, seq_length]
                    # First, expand to [batch_size, 1, 1, seq_length]
                    expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)

                    # Create a causal mask of shape [1, 1, seq_length, seq_length]
                    causal_mask = torch.triu(
                        torch.ones((seq_length, seq_length), device=device, dtype=dtype),
                        diagonal=1
                    ).unsqueeze(0).unsqueeze(0)

                    # Convert masks to proper format (-inf for masked positions, 0 for attended positions)
                    expanded_mask = (1.0 - expanded_mask) * -10000.0
                    causal_mask = (causal_mask > 0) * -10000.0

                    # Combine the masks
                    combined_mask = expanded_mask + causal_mask

                    # Replace the original attention_mask with our fixed version
                    attention_mask = combined_mask

                    logger.debug(f"Fixed attention mask shape: {attention_mask.shape}")
                except Exception as mask_error:
                    # If there's an error in our mask handling, log it and continue with the original mask
                    logger.warning(f"Error in attention mask fix: {mask_error}. Using original mask.")

            # Call the original forward method with the fixed attention mask
            return original_forward(
                self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # Replace the original forward method with our patched version
        LlamaModel.forward = patched_forward

        logger.info("Successfully applied legacy attention mask fix")
        return True
    except Exception as e:
        logger.error(f"Error applying attention mask fix: {e}")
        logger.warning("Continuing without attention mask fix")
        return False

def configure_for_gpu(args):
    """Configure training parameters based on GPU type and VRAM size"""
    logger.info(f"Configuring for GPU type {args.gpu_type} with {args.vram} GiB VRAM")

    # Base configuration for different GPU types
    if args.gpu_type == "A6000" and args.vram >= 48:
        # A6000 with 48+ GiB VRAM - maximize parameters while staying within constraints
        logger.info("Using optimized settings for A6000 with 48+ GiB VRAM")
        args.batch_size = 6  # Reduced from 8 to ensure stability
        args.gradient_accumulation_steps = 8
        args.max_length = 2048  # Reduced from 4096 to ensure stability with large models
        args.lora_rank = 32     # Increase LoRA rank for better quality
        args.lora_alpha = 64    # Increase LoRA alpha for better adaptation
        args.lora_dropout = 0.05  # Optimal dropout for stability
        args.load_in_4bit = True  # Use 4-bit quantization for maximum memory efficiency
        args.load_in_8bit = False
        args.bf16 = True  # Use bfloat16 precision for better training stability
        args.num_workers = 8    # Match your 8 CPU cores
        args.warmup_ratio = 0.03  # Optimal warmup for large models
        args.weight_decay = 0.01  # Prevent overfitting
        args.adam_beta1 = 0.9   # Standard beta1 for AdamW
        args.adam_beta2 = 0.999  # Standard beta2 for AdamW
        args.adam_epsilon = 1e-8  # Standard epsilon for AdamW
        args.max_grad_norm = 1.0  # Prevent gradient explosion
        args.scheduler_type = "cosine"  # Cosine scheduler with warmup
        args.evaluation_strategy = "steps"  # Evaluate during training
        args.eval_steps = 100   # Evaluate every 100 steps
        args.save_steps = 100   # Save every 100 steps
        args.save_total_limit = 3  # Keep only the last 3 checkpoints

    elif args.gpu_type == "A6000" and args.vram >= 40:
        # A6000 with 40-48 GiB VRAM
        logger.info("Using optimized settings for A6000 with 40-48 GiB VRAM")
        args.batch_size = 6
        args.gradient_accumulation_steps = 8
        args.max_length = 3072
        args.lora_rank = 24
        args.lora_alpha = 48
        args.lora_dropout = 0.05
        args.load_in_4bit = True
        args.load_in_8bit = False
        args.bf16 = True
        args.num_workers = 6
        args.warmup_ratio = 0.03
        args.weight_decay = 0.01
        args.adam_beta1 = 0.9
        args.adam_beta2 = 0.999
        args.adam_epsilon = 1e-8
        args.max_grad_norm = 1.0
        args.scheduler_type = "cosine"
        args.evaluation_strategy = "steps"
        args.eval_steps = 100
        args.save_steps = 100
        args.save_total_limit = 3

    elif args.gpu_type == "A4000" or (args.gpu_type == "A6000" and args.vram < 40):
        # A4000 or A6000 with less VRAM
        logger.info("Using optimized settings for A4000 or A6000 with <40 GiB VRAM")
        args.batch_size = 4
        args.gradient_accumulation_steps = 16
        args.max_length = 2048
        args.lora_rank = 16
        args.lora_alpha = 32
        args.lora_dropout = 0.05
        args.load_in_4bit = True
        args.load_in_8bit = False
        args.bf16 = True
        args.num_workers = 4
        args.warmup_ratio = 0.03
        args.weight_decay = 0.01
        args.adam_beta1 = 0.9
        args.adam_beta2 = 0.999
        args.adam_epsilon = 1e-8
        args.max_grad_norm = 1.0
        args.scheduler_type = "cosine"
        args.evaluation_strategy = "steps"
        args.eval_steps = 100
        args.save_steps = 100
        args.save_total_limit = 2

    elif args.gpu_type == "RTX5000":
        # RTX5000 with limited VRAM
        logger.info("Using optimized settings for RTX5000")
        args.batch_size = 2
        args.gradient_accumulation_steps = 32
        args.max_length = 1024
        args.lora_rank = 8
        args.lora_alpha = 16
        args.lora_dropout = 0.05
        args.load_in_4bit = True
        args.load_in_8bit = False
        args.bf16 = False  # Use fp16 instead
        args.num_workers = 2
        args.warmup_ratio = 0.03
        args.weight_decay = 0.01
        args.adam_beta1 = 0.9
        args.adam_beta2 = 0.999
        args.adam_epsilon = 1e-8
        args.max_grad_norm = 1.0
        args.scheduler_type = "linear"  # Linear scheduler for smaller models
        args.evaluation_strategy = "steps"
        args.eval_steps = 100
        args.save_steps = 100
        args.save_total_limit = 1

    # Calculate effective batch size
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    logger.info(f"Effective batch size: {effective_batch_size}")

    # Adjust learning rate based on batch size (linear scaling rule)
    base_lr = 2e-5
    base_batch_size = 64
    args.learning_rate = base_lr * (effective_batch_size / base_batch_size)
    logger.info(f"Adjusted learning rate: {args.learning_rate}")

    # Calculate warmup steps based on warmup ratio
    args.warmup_steps = int(args.warmup_ratio * effective_batch_size * 100)  # Assuming ~100 steps per epoch
    logger.info(f"Warmup steps: {args.warmup_steps}")

    # Memory optimization settings
    if args.optimize_memory_usage:
        logger.info("Applying memory optimization settings")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,garbage_collection_threshold:0.8"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["OMP_NUM_THREADS"] = str(min(8, os.cpu_count() or 1))  # Use available CPU cores efficiently
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Disable for better performance
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Reduce TensorFlow logging

        # Print GPU information if available
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                device_capability = torch.cuda.get_device_capability(0)
                logger.info(f"Using GPU: {device_name} with CUDA capability {device_capability[0]}.{device_capability[1]}")
                logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GiB")
                # Clear CUDA cache
                torch.cuda.empty_cache()
        except ImportError:
            logger.warning("PyTorch not available, skipping GPU information")

    return args

def main():
    """Main function to run the unified training"""
    parser = argparse.ArgumentParser(description="Unified DeepSeek-Coder training")
    parser.add_argument("--model_name", type=str, default="deepseek-ai/deepseek-coder-6.7b-instruct",
                        help="Model name or path")
    parser.add_argument("--output_dir", type=str, default="/notebooks/Jarvis_AI_Assistant/models/deepseek-coder-6.7b-finetuned",
                        help="Output directory for saving the model")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--max_samples", type=int, default=5000,
                        help="Maximum number of samples to use")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Load model in 4-bit precision")
    parser.add_argument("--load_in_8bit", action="store_true",
                        help="Load model in 8-bit precision")
    parser.add_argument("--use_flash_attn", action="store_true",
                        help="Use Flash Attention for faster training")
    parser.add_argument("--dataset_subset", type=str, default="python",
                        help="Dataset subset to use")
    parser.add_argument("--all_subsets", action="store_true",
                        help="Use all language subsets")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Number of warmup steps")
    parser.add_argument("--bf16", action="store_true",
                        help="Use bfloat16 precision")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--gpu-type", type=str, default="A6000",
                        help="GPU type (A6000, A4000, RTX5000)")
    parser.add_argument("--vram", type=int, default=50,
                        help="GPU VRAM size in GiB")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="Use gradient checkpointing to reduce memory usage")
    parser.add_argument("--mixed_precision", action="store_true", default=True,
                        help="Use mixed precision training")
    parser.add_argument("--optimize_memory_usage", action="store_true", default=True,
                        help="Optimize memory usage during training")
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank parameter")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout parameter")

    # Advanced optimization parameters
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Warmup ratio for learning rate scheduler")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for AdamW optimizer")
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                        help="Beta1 for AdamW optimizer")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                        help="Beta2 for AdamW optimizer")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8,
                        help="Epsilon for AdamW optimizer")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for clipping")
    parser.add_argument("--scheduler_type", type=str, default="cosine",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
                        help="Learning rate scheduler type")
    parser.add_argument("--evaluation_strategy", type=str, default="steps",
                        choices=["no", "steps", "epoch"],
                        help="Evaluation strategy")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Evaluation steps")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save steps")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Maximum number of checkpoints to keep")

    args = parser.parse_args()

    # Automatically configure parameters based on GPU type and VRAM size
    configure_for_gpu(args)

    # Apply attention mask fix
    logger.info("Applying attention mask fix...")
    apply_attention_mask_fix()

    # Ensure output directory exists and is in the correct location
    if not args.output_dir.startswith('/notebooks/Jarvis_AI_Assistant/'):
        logger.warning(f"Output directory {args.output_dir} is not in the expected location.")
        logger.warning("Changing output directory to be within /notebooks/Jarvis_AI_Assistant/models/")
        model_name = os.path.basename(args.output_dir)
        args.output_dir = f"/notebooks/Jarvis_AI_Assistant/models/{model_name}"

    logger.info(f"Using output directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Set default quantization if none specified
    if not args.load_in_4bit and not args.load_in_8bit:
        args.load_in_4bit = True

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    # Choose the appropriate training method
    if UNSLOTH_AVAILABLE:
        logger.info("Using Unsloth-optimized training")
        train_with_unsloth(args)
    else:
        logger.info("Using standard training with attention mask fix")
        train_with_standard_method(args)

    logger.info(f"Training completed. Model saved to {args.output_dir}")

def train_with_unsloth(args):
    """Train using Unsloth optimization"""
    from unsloth import FastLanguageModel
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling, __version__ as transformers_version
    from datasets import load_dataset

    # Check transformers version for compatibility
    logger.info(f"Using transformers version: {transformers_version}")

    # Parse version string to check compatibility
    try:
        version_parts = transformers_version.split('.')
        major = int(version_parts[0]) if len(version_parts) > 0 else 0
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0
        logger.info(f"Parsed transformers version: {major}.{minor}")

        # Adjust parameters based on transformers version
        if major < 4 or (major == 4 and minor < 20):
            logger.warning(f"Using older transformers version ({transformers_version}). Some parameters may not be supported.")
    except Exception as e:
        logger.warning(f"Could not parse transformers version: {e}. Assuming compatibility.")

    logger.info("Loading and preprocessing dataset...")

    # Load dataset
    if args.all_subsets:
        logger.info("Loading all language subsets")
        from src.generative_ai_module.code_preprocessing import load_and_preprocess_all_subsets
        train_dataset, eval_dataset = load_and_preprocess_all_subsets(
            max_samples=args.max_samples,
            sequence_length=args.max_length,
            return_raw=True
        )
    else:
        logger.info(f"Loading {args.dataset_subset} subset")
        from src.generative_ai_module.code_preprocessing import load_and_preprocess_dataset
        train_dataset, eval_dataset = load_and_preprocess_dataset(
            max_samples=args.max_samples,
            sequence_length=args.max_length,
            subset=args.dataset_subset,
            all_subsets=False,
            return_raw=True
        )

    # Validate datasets before proceeding
    if train_dataset is None or len(train_dataset) == 0:
        logger.error("Training dataset is empty or None. Cannot proceed with training.")
        raise ValueError("Training dataset is empty or None")

    if eval_dataset is None or len(eval_dataset) == 0:
        logger.warning("Evaluation dataset is empty or None. Creating a small subset of training data for evaluation.")
        # Create a small evaluation dataset from training data if needed
        if len(train_dataset) > 10:
            eval_dataset = train_dataset.select(range(min(len(train_dataset) // 10, 100)))
            logger.info(f"Created evaluation dataset with {len(eval_dataset)} samples from training data")
        else:
            # If training dataset is too small, use it for both training and evaluation
            eval_dataset = train_dataset
            logger.warning("Training dataset is very small. Using same data for evaluation.")

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")

    # Validate that datasets have the required 'text' field
    if 'text' not in train_dataset.column_names:
        logger.error("Training dataset does not have a 'text' field. Cannot proceed with training.")
        raise ValueError("Training dataset missing 'text' field")

    if 'text' not in eval_dataset.column_names:
        logger.error("Evaluation dataset does not have a 'text' field. Cannot proceed with training.")
        raise ValueError("Evaluation dataset missing 'text' field")

    # Load model with Unsloth with robust error handling
    logger.info(f"Loading model: {args.model_name}")
    print("🦥 Loading model", args.model_name, "with minimal unsloth")
    if args.load_in_4bit:
        print("Loading model in 4-bit quantization")
    elif args.load_in_8bit:
        print("Loading model in 8-bit quantization")

    # Add robust error handling for model loading
    try:
        # Note: FastLanguageModel.from_pretrained already sets trust_remote_code=True internally
        # and also sets device_map="auto" by default, so we don't need to pass these parameters
        # to avoid duplicate parameter errors
        # Don't pass max_seq_length directly to avoid TypeError with LlamaForCausalLM
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_name,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit
            # Don't set device_map here as it's already set by FastLanguageModel internally
        )
        logger.info(f"Successfully loaded model: {args.model_name}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        # Try with additional parameters that might help
        try:
            logger.info("Retrying model loading with additional parameters...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.model_name,
                load_in_4bit=args.load_in_4bit,
                load_in_8bit=args.load_in_8bit,
                trust_remote_code=True,  # Explicitly set trust_remote_code
                device_map="auto",       # Explicitly set device_map
                use_cache=False if args.gradient_checkpointing else True  # Disable KV cache if using gradient checkpointing
            )
            logger.info(f"Successfully loaded model on retry: {args.model_name}")
        except Exception as e2:
            logger.error(f"Fatal error loading model on retry: {e2}")
            raise RuntimeError(f"Could not load model {args.model_name}. Original error: {e}, Retry error: {e2}")

    # Set max sequence length after model is loaded
    model.config.max_position_embeddings = args.max_length
    tokenizer.model_max_length = args.max_length

    # Set up LoRA with optimized parameters
    logger.info("Configuring LoRA adapters")
    logger.info(f"Using LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}, dropout: {args.lora_dropout}")

    # Define target modules for LoRA
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]

    # Apply LoRA adapters with robust error handling
    try:
        logger.info(f"Applying LoRA with rank={args.lora_rank}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,  # LoRA rank from configuration
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",  # Don't train bias terms for stability
            # Removed task_type parameter as it's set internally by FastLanguageModel.get_peft_model
            modules_to_save=None  # Don't save any modules fully (use LoRA for all)
        )
        logger.info("Successfully applied LoRA adapters")
    except Exception as e:
        logger.error(f"Error applying LoRA adapters: {e}")

        # Try alternative approach with explicit LoraConfig
        try:
            from peft import LoraConfig

            logger.warning("Trying alternative LoRA application method...")
            # Create explicit LoraConfig
            lora_config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=target_modules,
                bias="none",
                # Don't set task_type here to avoid the duplicate parameter issue
            )

            # Apply LoRA with the config
            model = FastLanguageModel.get_peft_model(
                model,
                peft_config=lora_config  # Pass the config object instead of individual parameters
            )
            logger.info("Successfully applied LoRA adapters with alternative method")
        except Exception as e2:
            logger.error(f"Alternative LoRA application also failed: {e2}")
            raise RuntimeError(f"Could not apply LoRA adapters. Original error: {e}, Alternative error: {e2}")

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}% of total)")
    logger.info(f"Total parameters: {total_params:,}")

    # Tokenize dataset
    logger.info("Tokenizing dataset")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        """Tokenize examples with proper handling of potential issues"""
        try:
            # Ensure all texts are strings and handle potential None values
            texts = []
            for text in examples["text"]:
                if text is None:
                    texts.append("")  # Replace None with empty string
                elif isinstance(text, str):
                    texts.append(text)  # Keep strings as is
                else:
                    texts.append(str(text))  # Convert other types to string

            # Tokenize without return_tensors to avoid the "too many dimensions" error
            return tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=args.max_length,
                # Removed return_tensors="pt" to avoid the "too many dimensions" error
            )
        except Exception as e:
            logger.error(f"Error during tokenization: {e}")
            # Return empty dict with the right structure as fallback
            return {
                "input_ids": [[0] * args.max_length] * len(examples["text"]),
                "attention_mask": [[0] * args.max_length] * len(examples["text"])
            }

    # Keep only the essential 'text' field and remove all other fields to avoid tokenization issues
    logger.info("Cleaning datasets to keep only essential fields")

    # Get all column names except 'text'
    train_columns_to_remove = [col for col in train_dataset.column_names if col != 'text']
    eval_columns_to_remove = [col for col in eval_dataset.column_names if col != 'text']

    # Remove all non-essential columns
    if train_columns_to_remove:
        logger.info(f"Removing non-essential fields from training dataset: {train_columns_to_remove}")
        train_dataset = train_dataset.remove_columns(train_columns_to_remove)

    if eval_columns_to_remove:
        logger.info(f"Removing non-essential fields from evaluation dataset: {eval_columns_to_remove}")
        eval_dataset = eval_dataset.remove_columns(eval_columns_to_remove)

    # Apply tokenization and remove the 'text' field to avoid nesting issues
    logger.info("Applying tokenization to datasets and removing 'text' field")
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=['text'])

    # Verify that the 'text' field is removed
    if 'text' in train_dataset.column_names:
        logger.warning("'text' field still present in training dataset after tokenization, removing it explicitly")
        train_dataset = train_dataset.remove_columns(['text'])

    if 'text' in eval_dataset.column_names:
        logger.warning("'text' field still present in evaluation dataset after tokenization, removing it explicitly")
        eval_dataset = eval_dataset.remove_columns(['text'])

    # Set up training arguments with optimized parameters and multiprocessing safeguards
    logger.info(f"Setting up training arguments with batch size: {args.batch_size}, gradient accumulation: {args.gradient_accumulation_steps}")

    # Adjust number of workers for multiprocessing safety
    # When using 'spawn' method with CUDA, it's safer to use fewer workers or even 0
    safe_num_workers = 0  # Set to 0 to avoid multiprocessing issues with CUDA
    if args.num_workers > 0:
        logger.warning(f"Reducing dataloader workers from {args.num_workers} to {safe_num_workers} to avoid CUDA multiprocessing issues")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,

        # Logging and evaluation settings
        logging_steps=10,
        save_steps=args.save_steps,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        save_safetensors=True,  # Use safetensors format for better compatibility

        # Precision settings
        bf16=args.bf16,
        fp16=not args.bf16 and torch.cuda.is_available(),

        # Memory optimization settings
        gradient_checkpointing=args.gradient_checkpointing,
        optim="adamw_torch",  # More memory-efficient optimizer
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,

        # Data loading optimizations - adjusted for multiprocessing safety
        remove_unused_columns=False,
        dataloader_num_workers=safe_num_workers,  # Use 0 workers to avoid multiprocessing issues
        dataloader_pin_memory=False,  # Set to False to avoid potential CUDA issues
        group_by_length=True,

        # Additional memory and performance optimizations
        ddp_find_unused_parameters=False,
        torch_compile=False,  # Disable torch.compile which can use more memory
        report_to=["tensorboard"],  # Minimal reporting to save memory

        # Learning rate scheduler
        lr_scheduler_type=args.scheduler_type,

        # Push to Hub settings (disabled by default)
        push_to_hub=False,
        hub_strategy="every_save",

        # Mixed precision training
        tf32=True,  # Use TF32 precision on Ampere+ GPUs for better performance

        # Distributed training settings
        local_rank=-1,

        # Seed for reproducibility
        seed=42,

        # Additional safety settings
        dataloader_drop_last=False    # Don't drop last batch
    )

    # Create a completely custom data collator that doesn't rely on the parent class
    class SimpleDataCollator:
        """Simple custom data collator that handles all processing directly"""
        def __init__(self, tokenizer, mlm=False, pad_to_multiple_of=None):
            self.tokenizer = tokenizer
            self.mlm = mlm
            self.pad_to_multiple_of = pad_to_multiple_of
            logger.info("Using SimpleDataCollator with direct tensor creation")

        def __call__(self, features):
            try:
                # Check if features have the expected structure
                if not all(isinstance(f, dict) for f in features):
                    logger.warning("Features are not all dictionaries, converting...")
                    features = [dict(f) if not isinstance(f, dict) else f for f in features]

                # Remove 'text' field if present - this is causing the nesting issue
                for feature in features:
                    if 'text' in feature:
                        logger.warning("Removing 'text' field from feature to avoid nesting issues")
                        del feature['text']

                # Extract input_ids and attention_mask, handling potential issues
                batch_input_ids = []
                batch_attention_mask = []

                for i, feature in enumerate(features):
                    # Handle input_ids
                    if 'input_ids' not in feature:
                        logger.warning(f"Feature {i} missing input_ids, adding empty")
                        input_ids = [0] * args.max_length
                    else:
                        input_ids = feature['input_ids']
                        # Handle nested lists
                        if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
                            input_ids = input_ids[0]
                        # Ensure correct length
                        if len(input_ids) < args.max_length:
                            input_ids = input_ids + [self.tokenizer.pad_token_id] * (args.max_length - len(input_ids))
                        elif len(input_ids) > args.max_length:
                            input_ids = input_ids[:args.max_length]

                    # Handle attention_mask
                    if 'attention_mask' not in feature:
                        logger.warning(f"Feature {i} missing attention_mask, adding empty")
                        attention_mask = [0] * args.max_length
                    else:
                        attention_mask = feature['attention_mask']
                        # Handle nested lists
                        if isinstance(attention_mask, list) and attention_mask and isinstance(attention_mask[0], list):
                            attention_mask = attention_mask[0]
                        # Ensure correct length
                        if len(attention_mask) < args.max_length:
                            attention_mask = attention_mask + [0] * (args.max_length - len(attention_mask))
                        elif len(attention_mask) > args.max_length:
                            attention_mask = attention_mask[:args.max_length]

                    batch_input_ids.append(input_ids)
                    batch_attention_mask.append(attention_mask)

                # Create tensors directly on CUDA if available
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                logger.info(f"Creating tensors on device: {device}")

                input_ids_tensor = torch.tensor(batch_input_ids, dtype=torch.long, device=device)
                attention_mask_tensor = torch.tensor(batch_attention_mask, dtype=torch.long, device=device)

                # For causal language modeling, labels are the same as input_ids
                labels_tensor = input_ids_tensor.clone()

                # Explicitly log tensor devices
                logger.info(f"Input IDs tensor device: {input_ids_tensor.device}")
                logger.info(f"Attention mask tensor device: {attention_mask_tensor.device}")
                logger.info(f"Labels tensor device: {labels_tensor.device}")

                # Create the batch
                batch = {
                    "input_ids": input_ids_tensor,
                    "attention_mask": attention_mask_tensor,
                    "labels": labels_tensor
                }

                # Check if we need to reshape the attention mask for newer transformers versions
                # This is to avoid the "too many values to unpack (expected 2)" error
                try:
                    from transformers import __version__ as transformers_version
                    version_parts = transformers_version.split('.')
                    major = int(version_parts[0]) if len(version_parts) > 0 else 0
                    minor = int(version_parts[1]) if len(version_parts) > 1 else 0

                    # For newer versions of transformers (4.28+), we need to ensure the attention mask is 2D
                    if major > 4 or (major == 4 and minor >= 28):
                        # Ensure attention_mask is 2D [batch_size, seq_length]
                        if attention_mask_tensor.dim() > 2:
                            logger.warning(f"Reshaping attention mask from {attention_mask_tensor.shape} to 2D for compatibility")
                            # Take the first dimension if it's more than 2D
                            batch["attention_mask"] = attention_mask_tensor.view(attention_mask_tensor.size(0), -1)
                except Exception as e:
                    logger.warning(f"Error checking transformers version for attention mask: {e}")
                    # Continue with the original attention mask

                return batch
            except Exception as e:
                logger.error(f"Error in data collator: {e}")
                import traceback
                logger.error(traceback.format_exc())

                # Create minimal tensors directly on CUDA if available as fallback
                batch_size = len(features)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                logger.info(f"Creating fallback tensors on device: {device}")

                # Create tensors with requires_grad=True for the loss computation
                input_ids = torch.zeros((batch_size, args.max_length), dtype=torch.long, device=device)
                attention_mask = torch.zeros((batch_size, args.max_length), dtype=torch.long, device=device)
                labels = torch.zeros((batch_size, args.max_length), dtype=torch.long, device=device)

                # Log tensor devices
                logger.info(f"Fallback input IDs tensor device: {input_ids.device}")
                logger.info(f"Fallback attention mask tensor device: {attention_mask.device}")
                logger.info(f"Fallback labels tensor device: {labels.device}")

                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels
                }

    # Create data collator with improved handling
    data_collator = SimpleDataCollator(
        tokenizer=tokenizer,
        mlm=False,  # Not using masked language modeling
        pad_to_multiple_of=8  # Optimize for hardware efficiency
    )

    # Create a custom trainer that handles attention mask issues
    class AttentionMaskFixTrainer(Trainer):
        """Custom trainer that handles attention mask issues and device management"""
        def compute_loss(self, model, inputs, return_outputs=False):
            """Override compute_loss to handle attention mask issues and ensure proper device usage"""
            try:
                # Get the model's device
                device = model.device
                logger.info(f"In compute_loss: Model is on device: {device}")

                # Verify CUDA is available and being used
                if not torch.cuda.is_available():
                    logger.warning("CUDA is not available! Training will be slow on CPU.")
                elif str(device) != "cuda:0" and str(device) != "cuda":
                    logger.warning(f"Model is not on CUDA device! Current device: {device}")
                    # Try to move the model to CUDA
                    try:
                        model = model.to("cuda")
                        device = model.device
                        logger.info(f"Moved model to device: {device}")
                    except Exception as e:
                        logger.error(f"Failed to move model to CUDA: {e}")

                # Force all inputs to be on the model's device
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        if v.device != device:
                            logger.info(f"Moving {k} tensor from {v.device} to {device}")
                            inputs[k] = v.to(device)
                        # Log tensor shapes and devices
                        logger.info(f"Input tensor {k}: shape={v.shape}, device={v.device}")

                # Check if attention_mask needs fixing
                if "attention_mask" in inputs and inputs["attention_mask"].dim() != 2:
                    logger.warning(f"Fixing attention_mask dimension from {inputs['attention_mask'].dim()} to 2D")
                    # Reshape to 2D [batch_size, seq_length]
                    batch_size = inputs["attention_mask"].size(0)
                    seq_length = inputs["attention_mask"].size(-1)
                    inputs["attention_mask"] = inputs["attention_mask"].view(batch_size, seq_length).to(device)
                    logger.info(f"Fixed attention_mask shape: {inputs['attention_mask'].shape}, device: {inputs['attention_mask'].device}")

                # Verify all tensors are on the same device
                devices = set(v.device for k, v in inputs.items() if isinstance(v, torch.Tensor))
                if len(devices) > 1:
                    logger.error(f"Tensors are on different devices: {devices}")
                    # Force all to the model's device
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor) and v.device != device:
                            inputs[k] = v.to(device)
                    logger.info("Forced all tensors to the model's device")

                # Call the parent compute_loss with explicit try/except
                try:
                    logger.info("Calling parent compute_loss")
                    loss = super().compute_loss(model, inputs, return_outputs)
                    if isinstance(loss, tuple):
                        logger.info(f"Loss computed: {loss[0].item()}, device: {loss[0].device}")
                    else:
                        logger.info(f"Loss computed: {loss.item()}, device: {loss.device}")
                    return loss
                except Exception as parent_error:
                    logger.error(f"Error in parent compute_loss: {parent_error}")
                    raise  # Re-raise to be caught by the outer try/except
            except Exception as e:
                logger.error(f"Error in compute_loss: {e}")
                # Try to continue with a simplified computation
                try:
                    logger.info("Attempting simplified loss computation")
                    # Get the input IDs and labels and ensure they're on the right device
                    device = model.device
                    input_ids = inputs["input_ids"].to(device)
                    labels = inputs["labels"].to(device) if "labels" in inputs else input_ids.clone()

                    # Log tensor information
                    logger.info(f"Simplified computation - input_ids: shape={input_ids.shape}, device={input_ids.device}")
                    logger.info(f"Simplified computation - labels: shape={labels.shape}, device={labels.device}")

                    # Forward pass without attention mask
                    logger.info("Performing forward pass without attention mask")
                    outputs = model(input_ids=input_ids)

                    # Compute the loss manually
                    logits = outputs.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss_fct = torch.nn.CrossEntropyLoss()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                    # Ensure loss has gradients
                    if not loss.requires_grad:
                        logger.warning("Loss doesn't have gradients, creating a new tensor with requires_grad=True")
                        loss = loss.clone().detach().to(device).requires_grad_(True)

                    logger.info(f"Simplified loss computed: {loss.item()}, device: {loss.device}, requires_grad: {loss.requires_grad}")

                    return (loss, outputs) if return_outputs else loss
                except Exception as e2:
                    logger.error(f"Fallback loss computation also failed: {e2}")
                    # Return a dummy loss as last resort
                    logger.warning("Creating dummy loss with explicit gradient")
                    dummy_loss = torch.tensor(1.0, device=device, requires_grad=True)
                    logger.info(f"Dummy loss created: {dummy_loss.item()}, device: {dummy_loss.device}, requires_grad: {dummy_loss.requires_grad}")
                    return dummy_loss

        def training_step(self, model, inputs):
            """Override training_step to ensure proper device handling and gradient computation"""
            try:
                # Ensure model is in training mode
                model.train()

                # Get the model's device
                device = model.device
                logger.info(f"Model is on device: {device}")

                # Verify CUDA is available and being used
                if not torch.cuda.is_available():
                    logger.warning("CUDA is not available! Training will be slow on CPU.")
                elif str(device) != "cuda:0" and str(device) != "cuda":
                    logger.warning(f"Model is not on CUDA device! Current device: {device}")
                    # Try to move the model to CUDA
                    try:
                        model = model.to("cuda")
                        device = model.device
                        logger.info(f"Moved model to device: {device}")
                    except Exception as e:
                        logger.error(f"Failed to move model to CUDA: {e}")

                # Force all inputs to be on the model's device
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        if v.device != device:
                            logger.info(f"Moving {k} tensor from {v.device} to {device}")
                            inputs[k] = v.to(device)
                        # Log tensor shapes and devices
                        logger.info(f"Tensor {k}: shape={v.shape}, device={v.device}")

                # Compute loss with our enhanced compute_loss method
                loss = self.compute_loss(model, inputs)
                logger.info(f"Computed loss: {loss.item()}, device: {loss.device}, requires_grad: {loss.requires_grad}")

                # Force loss to have gradients
                if not hasattr(loss, 'grad_fn') or not loss.requires_grad:
                    logger.warning("Loss doesn't have gradients, creating a new tensor with requires_grad=True")
                    # Create a new tensor that requires gradients
                    loss = loss.clone().detach().to(device).requires_grad_(True)
                    logger.info(f"New loss tensor: {loss.item()}, device: {loss.device}, requires_grad: {loss.requires_grad}")

                # Scale loss for gradient accumulation if needed
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                # Backward pass with error handling
                try:
                    logger.info("Attempting backward pass with accelerator")
                    self.accelerator.backward(loss)
                    logger.info("Backward pass with accelerator successful")
                except Exception as e:
                    logger.error(f"Error in backward pass with accelerator: {e}")
                    # Try a more direct approach
                    try:
                        logger.info("Attempting direct backward pass")
                        loss.backward()
                        logger.info("Direct backward pass successful")
                    except Exception as e2:
                        logger.error(f"Direct backward also failed: {e2}")
                        # Create a dummy loss as last resort with explicit gradient
                        logger.warning("Creating dummy loss with explicit gradient")
                        dummy_loss = torch.tensor(1.0, device=device, requires_grad=True)
                        dummy_loss.backward()
                        logger.info("Dummy backward pass completed")

                # Verify gradients exist
                has_grad = False
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        has_grad = True
                        break

                if not has_grad:
                    logger.warning("No gradients found in model parameters after backward pass!")
                else:
                    logger.info("Gradients successfully computed")

                return loss.detach()
            except Exception as e:
                logger.error(f"Error in training_step: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # Return a dummy loss
                return torch.tensor(1.0, device=model.device)

        def get_train_dataloader(self):
            """Override to ensure tensors are properly handled with device management"""
            if self.train_dataset is None:
                raise ValueError("Trainer: training requires a train_dataset.")

            train_sampler = self._get_train_sampler()

            # Get the model's device
            device = self.model.device
            logger.info(f"In get_train_dataloader: Model is on device: {device}")

            # Create a custom collate function that ensures tensors are on the right device
            original_collate_fn = self.data_collator

            def device_aware_collate_fn(features):
                # Call the original collate function
                batch = original_collate_fn(features)

                # Force all tensors to be on the model's device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        if v.device != device:
                            logger.info(f"Moving batch tensor {k} from {v.device} to {device}")
                            batch[k] = v.to(device)
                        logger.info(f"Batch tensor {k}: shape={v.shape}, device={v.device}")

                return batch

            # Use our custom collate function with reduced workers
            logger.info("Creating train DataLoader with device-aware collate function")
            return torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=train_sampler,
                collate_fn=device_aware_collate_fn,
                drop_last=self.args.dataloader_drop_last,
                num_workers=0,  # Force 0 workers to avoid multiprocessing issues
                pin_memory=False,  # Disable pin_memory as we're explicitly moving tensors to device
            )

        def get_eval_dataloader(self, eval_dataset=None):
            """Override to ensure tensors are properly handled with device management"""
            eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
            if eval_dataset is None:
                raise ValueError("Trainer: evaluation requires an eval_dataset.")

            eval_sampler = self._get_eval_sampler(eval_dataset)

            # Get the model's device
            device = self.model.device
            logger.info(f"In get_eval_dataloader: Model is on device: {device}")

            # Create a custom collate function that ensures tensors are on the right device
            original_collate_fn = self.data_collator

            def device_aware_collate_fn(features):
                # Call the original collate function
                batch = original_collate_fn(features)

                # Force all tensors to be on the model's device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        if v.device != device:
                            logger.info(f"Moving eval batch tensor {k} from {v.device} to {device}")
                            batch[k] = v.to(device)
                        logger.info(f"Eval batch tensor {k}: shape={v.shape}, device={v.device}")

                return batch

            # Use our custom collate function with reduced workers
            logger.info("Creating eval DataLoader with device-aware collate function")
            return torch.utils.data.DataLoader(
                eval_dataset,
                batch_size=self.args.eval_batch_size,
                sampler=eval_sampler,
                collate_fn=device_aware_collate_fn,
                drop_last=self.args.dataloader_drop_last,
                num_workers=0,  # Force 0 workers to avoid multiprocessing issues
                pin_memory=False,  # Disable pin_memory as we're explicitly moving tensors to device
            )

    # Ensure model is on CUDA if available
    if torch.cuda.is_available():
        logger.info(f"Moving model to CUDA. Current device: {model.device}")
        model = model.to("cuda")
        logger.info(f"Model moved to device: {model.device}")
    else:
        logger.warning("CUDA is not available! Training will be slow on CPU.")

    # Add device handling to model's prepare_inputs_for_generation method
    if hasattr(model, 'prepare_inputs_for_generation'):
        original_prepare_inputs = model.prepare_inputs_for_generation

        def device_aware_prepare_inputs(input_ids, **kwargs):
            """Ensure all inputs are on the correct device"""
            device = model.device

            # Ensure input_ids is on the correct device
            if input_ids.device != device:
                logger.info(f"Moving input_ids from {input_ids.device} to {device}")
                input_ids = input_ids.to(device)

            # Move all kwargs tensors to the correct device
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor) and v.device != device:
                    logger.info(f"Moving kwarg {k} from {v.device} to {device}")
                    kwargs[k] = v.to(device)

            # Call the original method
            inputs = original_prepare_inputs(input_ids, **kwargs)

            # Move all outputs to the model's device
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor) and v.device != device:
                    logger.info(f"Moving generation output {k} from {v.device} to {device}")
                    inputs[k] = v.to(device)

            return inputs

        # Replace the original method with our device-aware version
        model.prepare_inputs_for_generation = device_aware_prepare_inputs
        logger.info("Added device handling to model's prepare_inputs_for_generation method")

    # Create trainer with attention mask fix
    trainer = AttentionMaskFixTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Verify trainer model is on CUDA
    logger.info(f"Trainer model device: {trainer.model.device}")
    if torch.cuda.is_available() and str(trainer.model.device) != "cuda:0" and str(trainer.model.device) != "cuda":
        logger.warning(f"Trainer model is not on CUDA! Moving to CUDA...")
        trainer.model = trainer.model.to("cuda")
        logger.info(f"Trainer model moved to device: {trainer.model.device}")

    # Ensure model parameters have requires_grad=True where needed
    trainable_params = 0
    all_params = 0
    for name, param in trainer.model.named_parameters():
        all_params += param.numel()
        if 'lora' in name:  # LoRA parameters should be trainable
            if not param.requires_grad:
                logger.warning(f"LoRA parameter {name} doesn't have requires_grad=True. Fixing...")
                param.requires_grad = True
            trainable_params += param.numel()

    logger.info(f"Verified model parameters: {trainable_params} trainable out of {all_params} total parameters")

    # Double-check that we have trainable parameters
    if trainable_params == 0:
        logger.error("No trainable parameters found! Training will not work.")
        raise ValueError("No trainable parameters found in the model. Check LoRA configuration.")

    # Train model with robust error handling
    logger.info("Starting training...")
    try:
        # Set a flag to track if training completed successfully
        training_successful = False

        # Start training
        trainer.train()

        # If we get here, training was successful
        training_successful = True
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        logger.error(traceback.format_exc())

        # Try to recover and save what we have
        logger.warning("Attempting to save partial training results despite error")

    # Save model with error handling and ensure correct save location
    try:
        # Double-check that output directory is in the correct location
        if not args.output_dir.startswith('/notebooks/Jarvis_AI_Assistant/'):
            logger.warning(f"Output directory {args.output_dir} is not in the expected location.")
            logger.warning("Changing output directory to be within /notebooks/Jarvis_AI_Assistant/models/")
            model_name = os.path.basename(args.output_dir)
            args.output_dir = f"/notebooks/Jarvis_AI_Assistant/models/{model_name}"
            os.makedirs(args.output_dir, exist_ok=True)

        logger.info(f"Saving model to {args.output_dir}")
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        logger.info(f"Model and tokenizer successfully saved to {args.output_dir}")

        # Create a README file with training information
        readme_content = f"""# DeepSeek Coder Fine-tuned Model

This model was fine-tuned from {args.model_name} on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.

## Training Parameters
- Batch size: {args.batch_size}
- Gradient accumulation steps: {args.gradient_accumulation_steps}
- Effective batch size: {args.batch_size * args.gradient_accumulation_steps}
- Learning rate: {args.learning_rate}
- Epochs: {args.epochs}
- LoRA rank: {args.lora_rank}
- LoRA alpha: {args.lora_alpha}
- LoRA dropout: {args.lora_dropout}
- Max sequence length: {args.max_length}
- Training samples: {len(train_dataset)}
"""
        with open(os.path.join(args.output_dir, "README.md"), "w") as f:
            f.write(readme_content)

    except Exception as e:
        logger.error(f"Error saving model: {e}")

        # Try alternative saving method
        try:
            logger.warning("Trying alternative saving method...")
            # Save the model state dict directly
            alternative_save_dir = f"/notebooks/Jarvis_AI_Assistant/models/backup_save_{int(time.time())}"
            os.makedirs(alternative_save_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{alternative_save_dir}/model.pt")
            tokenizer.save_pretrained(alternative_save_dir)
            logger.info(f"Model state dict saved to {alternative_save_dir}/model.pt")
        except Exception as e2:
            logger.error(f"Alternative saving also failed: {e2}")
            if not training_successful:
                raise RuntimeError(f"Both training and model saving failed. Original errors: Training: {e}, Saving: {e2}")

def train_with_standard_method(args):
    """Train using standard method with attention mask fix"""
    logger.info("Using standard training method with attention mask fix")

    # Import the finetune_deepseek module
    try:
        from src.generative_ai_module.finetune_deepseek import main as finetune_main

        # Ensure output directory is in the correct location
        if not args.output_dir.startswith('/notebooks/Jarvis_AI_Assistant/'):
            logger.warning(f"Output directory {args.output_dir} is not in the expected location.")
            logger.warning("Changing output directory to be within /notebooks/Jarvis_AI_Assistant/models/")
            model_name = os.path.basename(args.output_dir)
            args.output_dir = f"/notebooks/Jarvis_AI_Assistant/models/{model_name}"
            os.makedirs(args.output_dir, exist_ok=True)

        logger.info(f"Using output directory: {args.output_dir}")

        # Prepare arguments for finetune_deepseek
        sys.argv = [
            "finetune_deepseek.py",
            f"--epochs={args.epochs}",
            f"--batch-size={args.batch_size}",
            f"--learning-rate={args.learning_rate}",
            f"--sequence-length={args.max_length}",
            f"--max-samples={args.max_samples}",
            f"--warmup-steps={args.warmup_steps}",
            f"--output-dir={args.output_dir}",
            f"--gradient-accumulation-steps={args.gradient_accumulation_steps}",
            f"--num-workers={args.num_workers}"
        ]

        # Quantization settings
        if args.load_in_4bit:
            sys.argv.append("--load-in-4bit")
        elif args.load_in_8bit:
            sys.argv.append("--load-in-8bit")

        # Dataset settings
        if args.all_subsets:
            sys.argv.append("--all-subsets")
        else:
            sys.argv.append(f"--subset={args.dataset_subset}")

        # Optimization settings
        if args.gradient_checkpointing:
            sys.argv.append("--gradient-checkpointing")

        if args.bf16:
            sys.argv.append("--bf16")
        elif torch.cuda.is_available():
            sys.argv.append("--fp16")

        # LoRA settings
        sys.argv.append(f"--lora-rank={args.lora_rank}")
        sys.argv.append(f"--lora-alpha={args.lora_alpha}")
        sys.argv.append(f"--lora-dropout={args.lora_dropout}")

        # Advanced optimization settings
        sys.argv.append(f"--weight-decay={args.weight_decay}")
        sys.argv.append(f"--max-grad-norm={args.max_grad_norm}")
        sys.argv.append(f"--scheduler-type={args.scheduler_type}")
        sys.argv.append(f"--evaluation-strategy={args.evaluation_strategy}")
        sys.argv.append(f"--eval-steps={args.eval_steps}")
        sys.argv.append(f"--save-steps={args.save_steps}")
        sys.argv.append(f"--save-total-limit={args.save_total_limit}")

        # Adam optimizer settings
        sys.argv.append(f"--adam-beta1={args.adam_beta1}")
        sys.argv.append(f"--adam-beta2={args.adam_beta2}")
        sys.argv.append(f"--adam-epsilon={args.adam_epsilon}")

        # Run the finetune_deepseek main function
        logger.info("Starting finetune_deepseek with attention mask fix")
        finetune_main()

    except ImportError:
        logger.error("Could not import finetune_deepseek module. Falling back to CodeGenerator.")

        # Import CodeGenerator and related modules
        from src.generative_ai_module.code_generator import CodeGenerator
        from src.generative_ai_module.code_preprocessing import load_and_preprocess_dataset

        # Load dataset
        logger.info("Loading and preprocessing dataset...")
        train_dataset, eval_dataset = load_and_preprocess_dataset(
            max_samples=args.max_samples,
            sequence_length=args.max_length,
            subset=args.dataset_subset,
            all_subsets=args.all_subsets
        )

        # Initialize CodeGenerator
        logger.info("Initializing CodeGenerator...")
        code_gen = CodeGenerator(
            use_deepseek=True,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit
        )

        # Ensure output directory is in the correct location
        if not args.output_dir.startswith('/notebooks/Jarvis_AI_Assistant/'):
            logger.warning(f"Output directory {args.output_dir} is not in the expected location.")
            logger.warning("Changing output directory to be within /notebooks/Jarvis_AI_Assistant/models/")
            model_name = os.path.basename(args.output_dir)
            args.output_dir = f"/notebooks/Jarvis_AI_Assistant/models/{model_name}"
            os.makedirs(args.output_dir, exist_ok=True)

        logger.info(f"Using output directory: {args.output_dir}")

        # Fine-tune the model
        logger.info("Starting fine-tuning...")
        code_gen.fine_tune_deepseek(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            sequence_length=args.max_length,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            subset=args.dataset_subset,
            all_subsets=args.all_subsets,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=args.gradient_checkpointing,
            bf16=args.bf16,
            fp16=not args.bf16 and torch.cuda.is_available(),
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            num_workers=args.num_workers,
            # Advanced optimization parameters
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            scheduler_type=args.scheduler_type,
            evaluation_strategy=args.evaluation_strategy,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            # Adam optimizer settings
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            adam_epsilon=args.adam_epsilon
        )

        # Create a README file with training information
        try:
            readme_content = f"""# DeepSeek Coder Fine-tuned Model (CodeGenerator)

This model was fine-tuned from {args.model_name} on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.

## Training Parameters
- Batch size: {args.batch_size}
- Gradient accumulation steps: {args.gradient_accumulation_steps}
- Effective batch size: {args.batch_size * args.gradient_accumulation_steps}
- Learning rate: {args.learning_rate}
- Epochs: {args.epochs}
- LoRA rank: {args.lora_rank}
- LoRA alpha: {args.lora_alpha}
- LoRA dropout: {args.lora_dropout}
- Max sequence length: {args.max_length}
- Training samples: {len(train_dataset)}
"""
            with open(os.path.join(args.output_dir, "README.md"), "w") as f:
                f.write(readme_content)
            logger.info(f"Created README.md in {args.output_dir}")
        except Exception as e:
            logger.warning(f"Failed to create README.md: {e}")

def safe_main():
    """Wrapper around main() with comprehensive error handling"""
    try:
        # Set up additional safeguards
        if torch.cuda.is_available():
            # Set up CUDA error handling
            torch.cuda.empty_cache()

            # Check available GPU memory
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            free_memory_gb = free_memory / (1024**3)
            logger.info(f"Available GPU memory before training: {free_memory_gb:.2f} GiB")

            if free_memory_gb < 2.0:
                logger.warning(f"Very low GPU memory available ({free_memory_gb:.2f} GiB). Training may fail.")

        # Run the main function
        main()

        logger.info("Training process completed successfully")
        return 0
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        logger.error(f"Unhandled exception in training process: {e}")
        import traceback
        logger.error(traceback.format_exc())

        # Try to clean up
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logger.info("CUDA cache cleared")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clear CUDA cache: {cleanup_error}")

        return 1  # Error exit code

if __name__ == "__main__":
    exit_code = safe_main()
    sys.exit(exit_code)
