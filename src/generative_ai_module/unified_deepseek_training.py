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
from datetime import datetime
from pathlib import Path

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

        logger.info("Successfully applied attention mask fix")
        return True
    except Exception as e:
        logger.error(f"Error applying attention mask fix: {e}")
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
    parser.add_argument("--output_dir", type=str, default="models/deepseek-coder-6.7b-finetuned",
                        help="Output directory")
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

    # Ensure output directory exists
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
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    from datasets import load_dataset

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

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")

    # Load model with Unsloth
    logger.info(f"Loading model: {args.model_name}")
    print("🦥 Loading model", args.model_name, "with minimal unsloth")
    if args.load_in_4bit:
        print("Loading model in 4-bit quantization")
    elif args.load_in_8bit:
        print("Loading model in 8-bit quantization")

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

    # Apply LoRA adapters
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
        # Ensure all texts are strings
        texts = [str(text) if not isinstance(text, str) else text for text in examples["text"]]

        # Tokenize without return_tensors to avoid the "too many dimensions" error
        return tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
            # Removed return_tensors="pt" to avoid the "too many dimensions" error
        )

    # Remove any problematic fields that might cause issues during tokenization
    if 'repository_name' in train_dataset.features:
        logger.info("Removing repository_name field from datasets")
        train_dataset = train_dataset.remove_columns(['repository_name'])
    if 'repository_name' in eval_dataset.features:
        eval_dataset = eval_dataset.remove_columns(['repository_name'])

    # Apply tokenization
    logger.info("Applying tokenization to datasets")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    # Set up training arguments with optimized parameters
    logger.info(f"Setting up training arguments with batch size: {args.batch_size}, gradient accumulation: {args.gradient_accumulation_steps}")
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

        # Data loading optimizations
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
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
    )

    # Create data collator with improved handling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Not using masked language modeling
        pad_to_multiple_of=8  # Optimize for hardware efficiency
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train model
    logger.info("Starting training...")
    trainer.train()

    # Save model
    logger.info(f"Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

def train_with_standard_method(args):
    """Train using standard method with attention mask fix"""
    logger.info("Using standard training method with attention mask fix")

    # Import the finetune_deepseek module
    try:
        from src.generative_ai_module.finetune_deepseek import main as finetune_main

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

if __name__ == "__main__":
    main()
