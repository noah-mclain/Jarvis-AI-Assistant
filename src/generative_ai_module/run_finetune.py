#!/usr/bin/env python3
"""
DeepSeek-Coder Fine-tuning Runner

This script runs the fine-tuning process for the DeepSeek-Coder model
from the project root directory, avoiding any import issues.

Usage:
    python run_finetune.py 
    
Additional options:
    --max-samples=1000      # Limit the number of samples for memory constraints
    --batch-size=64         # Batch size for GPU training
    --load-in-4bit          # Use 4-bit quantization for extreme memory saving
    --subset=python         # Just use one language if memory is limited
    --all-subsets           # Only use specific subset if memory is limited
    --sequence-length=2048  # Maximum sequence length for training
"""

# Import unsloth first to ensure optimizations are applied
import unsloth

# Standard library imports
import sys
import os
import subprocess

# Import torch after unsloth
import torch

# Import Unsloth fine-tuning (main implementation)
from src.generative_ai_module.unsloth_deepseek import finetune_with_unsloth, create_text_dataset_from_tokenized
from src.generative_ai_module.code_preprocessing import load_and_preprocess_dataset

# Import the fine-tuning functionality
from src.generative_ai_module.finetune_deepseek import parse_args

def apply_memory_efficient_defaults():
    """Apply memory-efficient defaults for running on GPU with Unsloth"""
    # Detect if we're on Apple Silicon
    on_apple_silicon = hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

    # Set batch size based on hardware
    if on_apple_silicon:
        # Use smaller batch size for Apple Silicon
        batch_size = "2"
        max_samples = "100"  # Limit samples for Apple Silicon
        epochs = "3"         # Fewer epochs for quicker training
        seq_length = "1024"  # Shorter sequence length for Apple Silicon
    else:
        # With Unsloth, we can use larger batch sizes or more samples on NVIDIA GPUs
        batch_size = "8" 
        max_samples = "10000"  # More samples with Unsloth
        epochs = "50"
        seq_length = "2048"    # Longer sequences with Unsloth

    print(f"Applying memory-efficient defaults for {'Apple Silicon' if on_apple_silicon else 'NVIDIA GPU'}")
    print("Using Unsloth optimization (always enabled)")
    print(f"Batch size: {batch_size}, Max samples: {max_samples}, Epochs: {epochs}, Sequence length: {seq_length}")

    # Set output directory to models/deepseek_unsloth in the root directory
    output_dir = os.path.join(os.path.dirname(__file__), "models", "deepseek_unsloth")

    # Set appropriate command-line arguments based on the device
    args = [
        "--epochs", epochs,              # Training epochs
        "--batch-size", batch_size,      # Batch size
        "--max-samples", max_samples,    # Number of samples for training
        "--output-dir", output_dir,      # Output directory in the root/models folder
        "--sequence-length", seq_length, # Sequence length for training
    ]

    # Add boolean flags as standalone arguments
    if on_apple_silicon:
        args.extend(("--use-mini-dataset", "--force-gpu"))
        print("Using memory-efficient GPU settings for Apple Silicon")
    else:
        args.extend(("--all-subsets", "--force-gpu"))
    args.extend(["--subset", "python"])  # Python subset as fallback

    # Add quantization flags - 4-bit works on CUDA but not on MPS (Apple Silicon)
    if not on_apple_silicon:
        args.append("--load-in-4bit")    # Use 4-bit quantization for NVIDIA GPUs
    else:
        args.append("--load-in-8bit")    # Use 8-bit quantization for Apple Silicon

    # Extend sys.argv with the arguments
    sys.argv.extend(args)

def train_deepseek_and_text_models():
    """Run both DeepSeek fine-tuning and text model training"""
    # First run the DeepSeek fine-tuning
    apply_memory_efficient_defaults()
    args = parse_args()
    train_with_unsloth(args)

    # Then run the unified pipeline to train both text models
    pipeline_script = os.path.join(os.path.dirname(__file__), 
                                  "src", "generative_ai_module", 
                                  "unified_generation_pipeline.py")

    if os.path.exists(pipeline_script):
        training_text_and_deepseek_models(pipeline_script)
    else:
        print(f"Error: Could not find unified pipeline script at {pipeline_script}")


# TODO Rename this here and in `train_deepseek_and_text_models`
def training_text_and_deepseek_models(pipeline_script):
    print("\n\n===== Now training text generation models =====")

    if (
        on_apple_silicon := hasattr(torch, 'backends')
        and hasattr(torch.backends, 'mps')
        and torch.backends.mps.is_available()
    ):
        # Smaller settings for Apple Silicon
        max_samples = "100"
        epochs = "3"
        print(f"Configuring text model training for Apple Silicon (samples: {max_samples}, epochs: {epochs})")
        print("NOTE: Text models will still use GPU as they are small enough for Apple Silicon")
    else:
        # Larger settings for NVIDIA GPUs
        max_samples = "5000"
        epochs = "50"

    # Set output directory in the root models folder
    model_dir = os.path.join(os.path.dirname(__file__), "models")

    # Construct command to run the unified pipeline
    cmd = [sys.executable, pipeline_script, 
           "--mode", "train",
           "--train-type", "both",
           "--dataset", "all",
           "--max-samples", max_samples,
           "--epochs", epochs,
           "--output-dir", model_dir,  # Specify output directory
           "--save-model",   # Boolean flag as standalone argument
           "--force-gpu"]    # Still use GPU for text models

    # Call the script using the same interpreter
    import subprocess
    subprocess.run(cmd)

def train_with_unsloth(args):
    """Use Unsloth for optimized training (now the default and only training method)"""
    print("\n=== Training with Unsloth optimization ===")

    # Detect if we're on Apple Silicon
    on_apple_silicon = hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

    # Load dataset
    if args.use_mini_dataset:
        from src.generative_ai_module.finetune_deepseek import create_mini_dataset
        train_dataset, eval_dataset = create_mini_dataset(args.sequence_length)
        # We need to convert tokenized dataset to text format for Unsloth
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True)
        train_dataset = create_text_dataset_from_tokenized(train_dataset, tokenizer)
        eval_dataset = create_text_dataset_from_tokenized(eval_dataset, tokenizer)
    else:
        # For Unsloth, we need untokenized datasets with the 'text' field
        train_dataset, valid_dataset = load_and_preprocess_dataset(
            max_samples=args.max_samples,
            sequence_length=args.sequence_length,
            subset=args.subset,
            all_subsets=args.all_subsets,
            return_raw=True  # Get raw text instead of tokenized
        )

        # Set aside some validation data for testing
        if valid_dataset and len(valid_dataset) > 0:
            # Split validation into validation and test sets (80/20 split)
            val_size = int(0.8 * len(valid_dataset))
            test_size = len(valid_dataset) - val_size

            # Only split if we have enough data
            if val_size > 0 and test_size > 0:
                eval_dataset = valid_dataset.select(range(val_size))
                test_dataset = valid_dataset.select(range(val_size, len(valid_dataset)))
                print(f"Split validation data into {len(eval_dataset)} validation and {len(test_dataset)} test samples")
            else:
                eval_dataset = valid_dataset
                test_dataset = None
        else:
            eval_dataset = None
            test_dataset = None

    # Configure Unsloth parameters based on dataset size
    if train_dataset and len(train_dataset) > 0:
        steps_per_epoch = max(1, len(train_dataset) // args.batch_size)
        max_steps = args.epochs * steps_per_epoch
        warmup_steps = min(100, int(max_steps * 0.1))  # 10% of total steps, capped at 100
    else:
        print("Warning: No training data found. Using default training parameters.")
        max_steps = 100  # Fallback to default
        warmup_steps = 10

    # Run finetuning with Unsloth
    training_metrics = finetune_with_unsloth(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model_name="deepseek-ai/deepseek-coder-6.7b-base",
        output_dir=args.output_dir,
        max_seq_length=args.sequence_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        learning_rate=args.learning_rate,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        r=16,  # LoRA rank
    )

    # Run evaluation on test set if available
    if test_dataset and len(test_dataset) > 0:
        evaluate_models(args, test_dataset, training_metrics)
    print("\n=== Unsloth training completed successfully ===")


# TODO Rename this here and in `train_with_unsloth`
def evaluate_models(args, test_dataset, training_metrics):
    print("\n=== Evaluating model on test set ===")
    from src.generative_ai_module.unsloth_deepseek import evaluate_model

    test_metrics = evaluate_model(
        model_dir=args.output_dir,
        test_dataset=test_dataset
    )

    # Merge metrics
    training_metrics.update({
        'test_loss': test_metrics.get('eval_loss'),
        'test_perplexity': test_metrics.get('perplexity'),
    })

    print(f"Test loss: {test_metrics.get('eval_loss', 'N/A')}")
    print(f"Test perplexity: {test_metrics.get('perplexity', 'N/A')}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments provided, use memory-efficient defaults
        # Detect if we're on Apple Silicon
        on_apple_silicon = hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
        print("\n=== Using memory-efficient defaults with hardware-appropriate settings ===")
        if on_apple_silicon:
            print("Apple Silicon GPU detected")
            print("Epochs: 3 (reduced for Apple Silicon)")
            print("Batch size: 2 (small batch for memory constraints)")
            print("Max samples: 100 (limited samples for faster training)")
            print("Using mini dataset: Yes (reduced dataset for testing)")
            print("Using GPU: Yes (with memory-optimized settings)")
            print("NOTE: DeepSeek-Coder is a very large model. Training will use memory-optimized settings.")
        else:
            print("NVIDIA GPU detected")
            print("Epochs: 50")
            print("Batch size: 8")
            print("Max samples: 10000")
            print("Using 4-bit quantization: Yes (NVIDIA GPU only)")
                
        print("Training on python subset")
        print("Using Unsloth optimization (always enabled)")
        print("=== To see all options, run with --help ===\n")
        
        # Check if the user wants to train all models
        train_all = input("Do you want to train both DeepSeek code model and text models? (y/n): ").strip().lower()
        
        if train_all == 'y':
            train_deepseek_and_text_models()
        else:
            # Just train DeepSeek with Unsloth
            apply_memory_efficient_defaults()
            args = parse_args()
            train_with_unsloth(args)
    else:
        # Parse command line arguments and run DeepSeek fine-tuning with Unsloth
        args = parse_args()
        train_with_unsloth(args)