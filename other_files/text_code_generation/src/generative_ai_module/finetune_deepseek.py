#!/usr/bin/env python3
# ===== BEGIN JARVIS IMPORT FIX =====
# This block was added by the fix_jarvis_imports.py script
import sys
import os
import multiprocessing

# Add the project root to sys.path
_jarvis_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _jarvis_project_root not in sys.path:
    sys.path.insert(0, _jarvis_project_root)

# Import the necessary functions directly
try:
    from src.generative_ai_module.import_fix import calculate_metrics, save_metrics, EvaluationMetrics
except ImportError:
    # If that fails, define them locally
    import torch
    import numpy as np

    def calculate_metrics(model, data_batches, device):
        """Calculate metrics on a dataset (loss, perplexity, accuracy)"""
        model.eval()
        total_loss = 0.0
        total_batches = 0
        total_correct = 0
        total_samples = 0

        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for input_batch, target_batch in data_batches:
                input_batch = input_batch.to(device)
                target_batch = target_batch.to(device)
                output, _ = model(input_batch)
                loss = criterion(output.view(-1, output.size(-1)), target_batch.view(-1))
                total_loss += loss.item()
                predictions = output.argmax(dim=-1)
                correct = (predictions == target_batch).sum().item()
                total_correct += correct
                total_samples += target_batch.numel()
                total_batches += 1

        avg_loss = total_loss / max(1, total_batches)
        perplexity = np.exp(avg_loss)
        accuracy = total_correct / max(1, total_samples)

        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'accuracy': accuracy
        }

    class EvaluationMetrics:
        """Class for evaluating generative models"""
        def __init__(self, metrics_dir="evaluation_metrics", use_gpu=None):
            self.metrics_dir = metrics_dir
            os.makedirs(metrics_dir, exist_ok=True)
            self.use_gpu = torch.cuda.is_available() if use_gpu is None else use_gpu

        def evaluate_generation(self, prompt, generated_text, reference_text=None,
                              dataset_name="unknown", save_results=True):
            import json
            from datetime import datetime

            results = {
                "prompt": prompt,
                "generated_text": generated_text,
                "dataset": dataset_name,
                "timestamp": datetime.now().isoformat()
            }

            if reference_text:
                results["reference_text"] = reference_text

            if save_results:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_path = os.path.join(self.metrics_dir,
                                           f"evaluation_{dataset_name}_{timestamp}.json")
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2)

            return results

    def save_metrics(metrics, model_name, dataset_name, timestamp=None):
        """Save evaluation metrics to a JSON file"""
        import json
        from datetime import datetime

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        metrics_dir = "metrics"
        os.makedirs(metrics_dir, exist_ok=True)

        model_name_clean = model_name.replace('/', '_')
        dataset_name_clean = dataset_name.replace('/', '_')

        filename = f"{model_name_clean}_{dataset_name_clean}_{timestamp}.json"
        filepath = os.path.join(metrics_dir, filename)

        metrics_with_meta = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "timestamp": timestamp,
            "metrics": metrics
        }

        with open(filepath, 'w') as f:
            json.dump(metrics_with_meta, f, indent=2)

        print(f"Saved metrics to {filepath}")
        return filepath
# ===== END JARVIS IMPORT FIX =====
"""
Module for fine-tuning DeepSeek Coder models on custom code datasets.
"""

import argparse
import os
import sys
import time
import datetime
import logging
import torch
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments
from peft import LoraConfig

# Fix imports - If running as a script, use absolute imports, otherwise use relative
if __name__ == "__main__":
    # Add the parent directory to the path to make the module importable
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
    from src.generative_ai_module.code_generator import CodeGenerator
    from src.generative_ai_module.code_preprocessing import load_and_preprocess_dataset
    from src.generative_ai_module.utils import setup_logging, sync_logs, sync_to_gdrive, sync_from_gdrive, ensure_directory_exists
else:
    # When imported as a module, use relative imports
    from .code_generator import CodeGenerator
    from .code_preprocessing import load_and_preprocess_dataset
    from .utils import setup_logging, sync_logs, sync_to_gdrive, sync_from_gdrive, ensure_directory_exists

# Configure logging
logger = logging.getLogger(__name__)

# Add this near the top of the file after imports
__all__ = ['main', 'parse_args', 'create_mini_dataset', 'setup_environment']

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DeepSeek-Coder on code datasets")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=50,
                      help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                      help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                      help="Learning rate for fine-tuning")
    parser.add_argument("--warmup-steps", type=int, default=100,
                      help="Number of warmup steps")
    parser.add_argument("--sequence-length", type=int, default=512,
                      help="Maximum sequence length")

    # Dataset options
    parser.add_argument("--max-samples", type=int, default=5000,
                      help="Maximum number of samples to use (None for all, default 5000 for memory constraints)")
    parser.add_argument("--subset", type=str, default="python",
                      choices=["python", "java", "go", "php", "ruby", "javascript"],
                      help="Language subset for code dataset (only used if --all-subsets=False)")
    parser.add_argument("--all-subsets", action="store_true", default=True,
                      help="Whether to use all language subsets (default: True)")
    parser.add_argument("--eval-split", type=float, default=0.1,
                      help="Fraction of data to use for evaluation")
    parser.add_argument("--use-mini-dataset", action="store_true",
                      help="Use mini dataset for quick testing instead of code_search_net")

    # Get root directory (up two levels from the script)
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_output_dir = os.path.join(root_dir, "models", "deepseek_finetuned")

    # Output options
    parser.add_argument("--output-dir", type=str, default=default_output_dir,
                      help="Output directory for fine-tuned model")
    parser.add_argument("--save-steps", type=int, default=100,
                      help="Save checkpoint every X steps")
    parser.add_argument("--save-total-limit", type=int, default=3,
                      help="Maximum number of checkpoints to keep")

    # Hardware options
    parser.add_argument("--gpu", action="store_true", default=True,
                      help="Force GPU usage (will use GPU by default if available)")
    parser.add_argument("--cpu", action="store_true",
                      help="Force CPU usage even if GPU is available")
    parser.add_argument("--load-in-8bit", action="store_true", default=True,
                      help="Load model in 8-bit precision to reduce memory usage")
    parser.add_argument("--load-in-4bit", action="store_true",
                      help="Load model in 4-bit precision for extreme memory saving (overrides 8-bit)")
    parser.add_argument("--force-gpu", action="store_true", default=True,
                      help="Force the use of GPU (MPS for Apple Silicon, CUDA for NVIDIA)")

    # Reproducibility options
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")

    # Logging options
    parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose logging")

    # Unsloth optimization option
    parser.add_argument("--use-unsloth", action="store_true",
                      help="Use Unsloth for faster training and reduced memory usage")

    return parser.parse_args()

def setup_environment(args):
    """Set up the environment for fine-tuning, including GPU detection and memory optimizations"""
    import torch
    import datetime
    import logging
    import os

    # Import utilities for GPU configuration
    from src.generative_ai_module.utils import setup_gpu_for_training, force_cuda_device, is_paperspace_environment

    # Always Force GPU usage regardless of args
    args.gpu = True
    args.force_gpu = True
    args.cpu = False  # Override any CPU request

    print("⚡ Enforcing GPU usage for all fine-tuning operations")

    # Apply GPU configuration with maximum enforcement
    device, gpu_config = setup_gpu_for_training(force_gpu=True)

    # Apply RTX5000-specific configurations (will only affect RTX5000 on Paperspace)
    if is_paperspace_environment() and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        if "RTX5000" in gpu_name or "RTX 5000" in gpu_name:
            # Apply the recommended optimizations for RTX5000
            args.load_in_4bit = True  # Force 4-bit quantization for RTX5000

            # Set environment variables
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

            # Adjust other parameters for optimal RTX5000 performance
            if not hasattr(args, 'gradient_accumulation_steps'):
                args.gradient_accumulation_steps = gpu_config.get("gradient_accumulation_steps", 8)

            # Check for specific training optimizations for RTX5000
            if hasattr(torch.cuda, 'get_device_properties'):
                props = torch.cuda.get_device_properties(0)
                memory_gb = props.total_memory / (1024**3)

                # Memory-based optimizations
                if memory_gb < 17:  # RTX5000 has 16GB VRAM
                    print(f"Optimizing for RTX5000 with {memory_gb:.2f}GB VRAM")
                    # Ensure small batch size for 16GB VRAM
                    if args.batch_size > 2:
                        print(f"Reducing batch size from {args.batch_size} to 1 for RTX5000")
                        args.batch_size = 1

    # Force CUDA as default tensor type if available
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        # Set default device in PyTorch 2.0+
        if hasattr(torch, 'set_default_device'):
            torch.set_default_device('cuda')

        return torch.device("cuda")

    # For Apple Silicon, use MPS
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("Using Apple Silicon GPU via MPS backend (CUDA not available)")

        # Clear any existing MPS cache to free up memory
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()

        # Set appropriate tensor type for MPS
        torch.set_default_tensor_type(torch.FloatTensor)

        return torch.device("mps")

    # This should only happen if no GPU is available despite attempts to force it
    print("Warning: No GPU available despite GPU enforcement. Training will be VERY slow on CPU.")
    return torch.device("cpu")

def create_mini_dataset(sequence_length=512):
    """Create a small dataset for testing fine-tuning without external data"""
    print("Creating mini test dataset instead of using code_search_net...")

    # Sample code examples
    examples = [
        {
            "text": "### Instruction: Implement a function to calculate the Fibonacci sequence.\n\n### Response:\ndef fibonacci(n):\n    if n <= 0:\n        return []\n    elif n == 1:\n        return [0]\n    elif n == 2:\n        return [0, 1]\n    else:\n        fib = [0, 1]\n        for i in range(2, n):\n            fib.append(fib[i-1] + fib[i-2])\n        return fib"
        },
        {
            "text": "### Instruction: Write a function to check if a string is a palindrome.\n\n### Response:\ndef is_palindrome(s):\n    s = s.lower()\n    s = ''.join(c for c in s if c.isalnum())\n    return s == s[::-1]"
        },
        {
            "text": "### Instruction: Create a function to sort a list using bubble sort.\n\n### Response:\ndef bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr"
        },
        {
            "text": "### Instruction: Implement a function to find the greatest common divisor of two numbers.\n\n### Response:\ndef gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a"
        },
        {
            "text": "### Instruction: Write a function to check if a number is prime.\n\n### Response:\ndef is_prime(n):\n    if n <= 1:\n        return False\n    if n <= 3:\n        return True\n    if n % 2 == 0 or n % 3 == 0:\n        return False\n    i = 5\n    while i * i <= n:\n        if n % i == 0 or n % (i + 2) == 0:\n            return False\n        i += 6\n    return True"
        }
    ]

    # Create dataset
    dataset = Dataset.from_dict({"text": [ex["text"] for ex in examples]})

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=sequence_length,
            return_tensors="pt"
            # device_map parameter removed to fix the error
        ).to("cpu")

    # Process dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_dataset = tokenized_dataset.with_format("torch", device="cpu")


    # Split into train and validation
    train_size = int(0.8 * len(tokenized_dataset))
    train_dataset = tokenized_dataset.select(range(train_size))
    eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))

    print(f"Created mini dataset with {len(train_dataset)} training and {len(eval_dataset)} validation examples")

    return train_dataset, eval_dataset

def main(args=None):
    """Main entry point for fine-tuning"""

    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    multiprocessing.set_start_method('spawn', force=True)
    # Set up logging to file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"finetune_deepseek_{timestamp}.log"
    setup_logging(log_file)

    logger.info("Starting DeepSeek fine-tuning process")

    start_time = time.time()

    # Parse command line arguments
    if args is None:
        args = parse_args()

    # Setup environment
    device = setup_environment(args)

    # Ensure we have access to necessary data from Google Drive
    try:
        sync_from_gdrive("datasets")
        sync_from_gdrive("models")
        logger.info("Synced latest datasets and models from Google Drive")
    except Exception as e:
        logger.warning(f"Error syncing from Google Drive: {str(e)}")

    print("\n" + "="*80)
    print("DeepSeek-Coder Fine-tuning Script")
    print(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")

    # Either use mini dataset or load from code_search_net
    if args.use_mini_dataset:
        train_dataset, eval_dataset = create_mini_dataset(args.sequence_length)
    else:
        if args.all_subsets:
            print(
                "Loading and preprocessing ALL language subsets from code_search_net dataset..."
            )
        else:
            print(f"Loading and preprocessing {args.subset} subset from code_search_net dataset...")

        train_dataset, eval_dataset = load_and_preprocess_dataset(
            max_samples=args.max_samples,
            sequence_length=args.sequence_length,
            subset=args.subset,
            all_subsets=args.all_subsets
        )
        train_dataset = train_dataset.with_format("torch")  # Keep on CPU
        eval_dataset = eval_dataset.with_format("torch")    # Keep on CPU

        # After loading datasets:
        print("\n=== Dataset Device Verification ===")
        print(f"Train dataset device: {train_dataset['input_ids'].device}")  # Should show "cpu"
        print(f"Eval dataset device: {eval_dataset['input_ids'].device}")    # Should show "cpu"

    print(f"\nInitializing DeepSeek-Coder model...")
    code_gen = CodeGenerator(
        use_deepseek=True,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        force_gpu=args.force_gpu
    )

    # After model initialization:
    print("\n=== Model Type Verification ===")
    print(type(code_gen.model))  # Should show PeftModelForCausalLM

    print(f"\nStarting fine-tuning with the following parameters:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - Sequence length: {args.sequence_length}")
    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Evaluation samples: {len(eval_dataset)}")
    print(f"  - Using all language subsets: {args.all_subsets}")
    if not args.all_subsets:
        print(f"  - Language subset: {args.subset}")
    print(f"  - Output directory: {args.output_dir}")
    print(f"  - Device: {device}")

    # Fine-tune the model
    training_metrics = code_gen.fine_tune_deepseek(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        subset=args.subset,
        all_subsets=args.all_subsets
    )

    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print("\n" + "="*80)
    print("Fine-tuning completed!")
    print(f"Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Final training loss: {training_metrics.get('final_loss', 'N/A')}")
    print(f"Evaluation loss: {training_metrics.get('eval_loss', 'N/A')}")
    print(f"Perplexity: {training_metrics.get('perplexity', 'N/A')}")
    print(f"Model saved to: {args.output_dir}")
    print("="*80 + "\n")

    # Test the model with a simple prompt
    print("Generating sample code with fine-tuned model...")
    test_prompt = "Write a function to calculate the Fibonacci sequence"
    generated_code = code_gen.generate_code(test_prompt, length=200)

    print("\nSample generated code:")
    print("-" * 40)
    print(generated_code)
    print("-" * 40)

    print("\nFine-tuning process complete. You can now use the model for code generation!")

    # Sync to Google Drive after training
    try:
        sync_to_gdrive("models")
        sync_to_gdrive("metrics")
        sync_logs()
        logger.info("Synced models, metrics, and logs to Google Drive")
    except Exception as e:
        logger.warning(f"Error syncing to Google Drive: {str(e)}")

    logger.info("Fine-tuning complete!")

if __name__ == "__main__":
    main()