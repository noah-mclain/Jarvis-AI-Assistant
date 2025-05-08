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

    args = parser.parse_args()

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
    # so we don't need to pass it explicitly to avoid the duplicate parameter error
    # Don't pass max_seq_length directly to avoid TypeError with LlamaForCausalLM
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        # Set device_map to auto to let Unsloth handle device placement
        device_map="auto"
    )

    # Set max sequence length after model is loaded
    model.config.max_position_embeddings = args.max_length
    tokenizer.model_max_length = args.max_length

    # Set up LoRA
    logger.info("Configuring LoRA adapters")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        use_gradient_checkpointing=True,
        random_state=42,
    )

    # Tokenize dataset
    logger.info("Tokenizing dataset")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
            return_tensors="pt"
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=args.warmup_steps,
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="steps",
        eval_steps=100,
        save_total_limit=3,
        bf16=args.bf16,
        fp16=not args.bf16 and torch.cuda.is_available(),
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        group_by_length=True,
    )

    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
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
            f"--output-dir={args.output_dir}"
        ]

        if args.load_in_4bit:
            sys.argv.append("--load-in-4bit")
        elif args.load_in_8bit:
            sys.argv.append("--load-in-8bit")

        if args.all_subsets:
            sys.argv.append("--all-subsets")
        else:
            sys.argv.append(f"--subset={args.dataset_subset}")

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
            all_subsets=args.all_subsets
        )

if __name__ == "__main__":
    main()
