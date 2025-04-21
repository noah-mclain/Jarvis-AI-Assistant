#!/usr/bin/env python3
"""
DeepSeek-Coder Fine-tuning Script

This script provides a convenient way to fine-tune the deepseek-coder model
on code datasets for improved code generation capabilities.

Usage:
    python finetune_deepseek.py --epochs 3 --batch-size 4 --max-samples 1000
    
By default, this will use data from all programming languages in the code_search_net dataset.
To use just a specific language, use the --subset option with --all-subsets=False.
"""

import argparse
import os
import sys
import time
import datetime
import torch
from datasets import Dataset
from transformers import AutoTokenizer

# Add parent directory to path if needed
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, parent_dir)

# Import after adjusting path
from src.generative_ai_module.code_generator import CodeGenerator
from src.generative_ai_module.code_preprocessing import load_and_preprocess_dataset

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
    
    return parser.parse_args()

def setup_environment(args):
    """Setup training environment and reproducibility"""
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Set environment variables for MPS (Apple Silicon)
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Avoid offloading tensors to disk
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0" 
        # Increase memory allocation on MPS
        os.environ["PYTORCH_MPS_ALLOCATOR_MEMPROFILE"] = "1"
        # Better memory management for MPS
        os.environ["PYTORCH_MPS_ACTIVE_MEMORY_MANAGER"] = "1"
        # Disable distributed training for MPS
        os.environ["LOCAL_RANK"] = "-1"
        # Enable garbage collection more aggressively
        import gc
        gc.collect()
        
    # Configure device
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")
        print("Using CPU for training (forced by --cpu flag)")
    elif args.force_gpu:
        # Check for Apple Silicon MPS support
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple Silicon GPU via MPS backend")
            
            # Set Apple Silicon specific optimizations
            print("Applying Apple Silicon memory optimizations...")
            # Force garbage collection
            import gc
            gc.collect()
            # Empty MPS cache if possible
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            
            # Reduce default tensor size to save memory
            torch.set_default_tensor_type(torch.FloatTensor)
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("Warning: GPU requested but neither MPS nor CUDA available. Falling back to CPU.")
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU via MPS backend")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU for training (no GPU available)")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    return device

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
        )
    
    # Process dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    # Split into train and validation
    train_size = int(0.8 * len(tokenized_dataset))
    train_dataset = tokenized_dataset.select(range(train_size))
    eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
    
    print(f"Created mini dataset with {len(train_dataset)} training and {len(eval_dataset)} validation examples")
    
    return train_dataset, eval_dataset

def main(args=None):
    start_time = time.time()

    # Parse command line arguments
    if args is None:
        args = parse_args()

    # Setup environment
    device = setup_environment(args)

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

    print(f"\nInitializing DeepSeek-Coder model...")
    code_gen = CodeGenerator(
        use_deepseek=True,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        force_gpu=args.force_gpu
    )

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

if __name__ == "__main__":
    main() 