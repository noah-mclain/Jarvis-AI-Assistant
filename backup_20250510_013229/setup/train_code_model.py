#!/usr/bin/env python3
"""
Train DeepSeek-Coder model for code generation.
"""

import os
import sys
import torch
import logging
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def train_code_model(gpu_type, vram_size):
    """
    Train DeepSeek-Coder model for code generation.

    Args:
        gpu_type (str): The type of GPU being used.
        vram_size (str): The amount of VRAM available.
    """
    logger.info(f"Starting code model training with GPU type: {gpu_type}, VRAM: {vram_size}")

    # Convert VRAM size to GB for calculations
    try:
        vram_gb = float(vram_size.replace("GiB", "").strip())
    except:
        vram_gb = 16  # Default if parsing fails
        logger.warning(f"Could not parse VRAM size '{vram_size}', using default: {vram_gb} GB")

    # Set batch size based on available VRAM
    if vram_gb >= 40:
        batch_size = 4
        use_4bit = True
    elif vram_gb >= 24:
        batch_size = 2
        use_4bit = True
    elif vram_gb >= 16:
        batch_size = 1
        use_4bit = True
    else:
        batch_size = 1
        use_4bit = True
        logger.warning(f"Low VRAM detected ({vram_gb} GB). Using minimal batch size and 4-bit quantization.")

    # Set up training parameters
    model_name = "deepseek-ai/deepseek-coder-6.7b-base"
    output_dir = "/notebooks/Jarvis_AI_Assistant/models/deepseek-coder-6.7b-finetuned"
    dataset_name = "code-search-net/code_search_net"
    all_subsets = True  # Use all language subsets
    subset = None  # Not used when all_subsets is True
    max_samples = 1000
    epochs = 3
    learning_rate = 3e-5
    max_length = 512
    gradient_accumulation_steps = max(16 // batch_size, 1)  # Adjust based on batch size

    logger.info(f"Training configuration:")
    logger.info(f"  - Model: {model_name}")
    logger.info(f"  - Dataset: {dataset_name} (using ALL language subsets)")
    logger.info(f"  - Batch size: {batch_size}")
    logger.info(f"  - Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"  - 4-bit quantization: {use_4bit}")
    logger.info(f"  - Max samples: {max_samples}")
    logger.info(f"  - Epochs: {epochs}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # Import code generator and call its methods directly
        sys.path.append("/notebooks")
        from src.generative_ai_module.code_generator import CodeGenerator

        # Initialize code generator
        code_gen = CodeGenerator(use_deepseek=True, load_in_4bit=use_4bit)

        # Load and preprocess dataset
        logger.info(f"Loading and preprocessing dataset with ALL language subsets...")

        # For all subsets, we'll need to handle this differently
        # We'll use the code_generator's built-in functionality for this

        # Fine-tune the model
        logger.info(f"Starting fine-tuning with all language subsets...")
        training_metrics = code_gen.fine_tune_deepseek(
            train_dataset=None,  # Let the function handle dataset loading for all subsets
            eval_dataset=None,   # Let the function handle dataset loading for all subsets
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            sequence_length=max_length,
            learning_rate=learning_rate,
            warmup_steps=100,
            subset=subset,
            all_subsets=True,    # This is the key parameter to use all subsets
            max_samples=max_samples
        )

        logger.info(f"Training completed with metrics: {training_metrics}")
        logger.info(f"Model saved to: {output_dir}")

        return True
    except Exception as e:
        logger.error(f"Error during code model training: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        logger.error("Missing required arguments: GPU type and VRAM size")
        logger.error("Usage: python train_code_model.py <gpu_type> <vram_size>")
        sys.exit(1)

    gpu_type = sys.argv[1]
    vram_size = sys.argv[2]

    success = train_code_model(gpu_type, vram_size)

    if not success:
        logger.error("Code model training failed")
        sys.exit(1)

    logger.info("Code model training completed successfully")
