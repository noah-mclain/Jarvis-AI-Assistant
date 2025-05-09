"""
Optimize DeepSeek Storage for Fine-tuning

This script demonstrates how to use storage optimization techniques for fine-tuning
DeepSeek-Coder in environments with limited storage space (e.g., Gradient Pro's 15GB limit).

Usage:
    python optimize_deepseek_storage.py [--storage-type gdrive] [--quantize 4]
"""

import argparse
import os
import sys
import torch
from datasets import Dataset
from transformers import AutoTokenizer

# Add parent directory to path if needed
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, parent_dir)

# Import after adjusting path
from src.generative_ai_module.code_preprocessing import load_and_preprocess_dataset
from src.generative_ai_module.unsloth_deepseek import finetune_with_optimal_storage
from src.generative_ai_module.storage_optimization import (
    optimize_storage_for_model, 
    setup_streaming_dataset, 
    compress_dataset
)

def parse_args():
    parser = argparse.ArgumentParser(description="Optimize DeepSeek fine-tuning for limited storage environments")
    
    # Model quantization options
    parser.add_argument("--quantize", type=int, default=4, choices=[4, 8],
                      help="Quantization bits (4 or 8)")
    
    # External storage options
    parser.add_argument("--storage-type", type=str, default="local", 
                      choices=["local", "gdrive", "s3"],
                      help="Storage type to use (local, gdrive, or s3)")
    parser.add_argument("--remote-path", type=str, default="DeepSeek_Models",
                      help="Path in remote storage")
    
    # S3 options
    parser.add_argument("--s3-bucket", type=str, 
                      help="S3 bucket name")
    parser.add_argument("--aws-access-key-id", type=str,
                      help="AWS access key ID")
    parser.add_argument("--aws-secret-access-key", type=str, 
                      help="AWS secret access key")
    
    # Dataset options
    parser.add_argument("--max-samples", type=int, default=2000, 
                      help="Maximum number of samples to use (default: 2000)")
    parser.add_argument("--use-mini-dataset", action="store_true",
                      help="Use mini dataset for testing instead of code_search_net")
    parser.add_argument("--all-subsets", action="store_true", default=False,
                      help="Whether to use all language subsets")
    parser.add_argument("--subset", type=str, default="python", 
                      choices=["python", "java", "go", "php", "ruby", "javascript"],
                      help="Language subset for code dataset")
    
    # Training options
    parser.add_argument("--max-steps", type=int, default=200, 
                      help="Maximum training steps")
    parser.add_argument("--batch-size", type=int, default=2, 
                      help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=2e-5, 
                      help="Learning rate for fine-tuning")
    parser.add_argument("--sequence-length", type=int, default=512, 
                      help="Maximum sequence length")
    parser.add_argument("--warmup-steps", type=int, default=50, 
                      help="Number of warmup steps")
    
    # Checkpointing options
    parser.add_argument("--checkpoint-strategy", type=str, default="improvement", 
                      choices=["improvement", "regular", "hybrid"],
                      help="Checkpoint strategy")
    parser.add_argument("--max-checkpoints", type=int, default=2, 
                      help="Maximum number of checkpoints to keep")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="models/deepseek_optimized", 
                      help="Output directory")
    
    return parser.parse_args()

def create_mini_dataset(sequence_length=512):
    """Create a small dataset for quick testing of storage optimization"""
    print("Creating mini test dataset...")
    
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
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset.select(range(train_size))
    eval_dataset = dataset.select(range(train_size, len(dataset)))
    
    print(f"Created mini dataset with {len(train_dataset)} training and {len(eval_dataset)} validation examples")
    
    return train_dataset, eval_dataset

def print_storage_usage(dir_path=None):
    """Print current storage usage information"""
    import shutil
    
    if dir_path:
        dir_size = sum(os.path.getsize(os.path.join(dirpath, filename)) 
                    for dirpath, _, filenames in os.walk(dir_path) 
                    for filename in filenames)
        print(f"Directory {dir_path}: {dir_size / (1024*1024*1024):.2f} GB")
    
    # Get total disk usage
    total, used, free = shutil.disk_usage("/")
    print(f"Total disk space: {total / (1024*1024*1024):.2f} GB")
    print(f"Used disk space: {used / (1024*1024*1024):.2f} GB")
    print(f"Free disk space: {free / (1024*1024*1024):.2f} GB")
    print(f"Used percentage: {used / total * 100:.2f}%")

def main():
    # Parse arguments
    args = parse_args()
    
    print("\n" + "="*80)
    print("DeepSeek-Coder Storage-Optimized Fine-tuning")
    print("="*80)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print initial storage usage
    print("\nInitial storage status:")
    print_storage_usage()
    
    # Prepare datasets
    if args.use_mini_dataset:
        train_dataset, eval_dataset = create_mini_dataset(args.sequence_length)
    else:
        print(f"\nLoading and preprocessing dataset...")
        if args.all_subsets:
            print("Using all language subsets")
        else:
            print(f"Using {args.subset} subset")
            
        # Use streaming dataset if possible
        try:
            train_dataset, eval_dataset = load_and_preprocess_dataset(
                max_samples=args.max_samples,
                sequence_length=args.sequence_length,
                subset=args.subset,
                all_subsets=args.all_subsets
            )
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Falling back to mini dataset")
            train_dataset, eval_dataset = create_mini_dataset(args.sequence_length)
    
    # Print storage usage after dataset loading
    print("\nStorage status after dataset loading:")
    print_storage_usage()
    
    # Fine-tune with optimal storage settings
    print("\nStarting fine-tuning with storage optimization...")
    
    # Configure storage optimization
    storage_type = args.storage_type
    remote_path = args.remote_path
    
    # Print optimization settings
    print(f"\nStorage optimization settings:")
    print(f"- Quantization: {args.quantize}-bit")
    print(f"- Storage type: {storage_type}")
    print(f"- Remote path: {remote_path}")
    print(f"- Checkpoint strategy: {args.checkpoint_strategy} (max {args.max_checkpoints})")
    print(f"- Max training steps: {args.max_steps}")
    print(f"- Batch size: {args.batch_size}")
    
    try:
        # Run fine-tuning with optimal storage settings
        training_metrics = finetune_with_optimal_storage(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            model_name="deepseek-ai/deepseek-coder-6.7b-base",
            output_dir=args.output_dir,
            max_seq_length=args.sequence_length,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            # Storage optimization
            quantize_bits=args.quantize,
            storage_type=storage_type,
            remote_path=remote_path,
            checkpoint_strategy=args.checkpoint_strategy,
            max_checkpoints=args.max_checkpoints,
            # AWS credentials if using S3
            aws_access_key_id=args.aws_access_key_id,
            aws_secret_access_key=args.aws_secret_access_key,
            s3_bucket=args.s3_bucket
        )
        
        print("\nTraining completed successfully!")
        print(f"Training metrics: {training_metrics}")
        
        # Final storage usage
        print("\nFinal storage status:")
        print_storage_usage()
        print_storage_usage(args.output_dir)
        
        # Print storage savings
        print("\nStorage optimization summary:")
        original_model_size = 13.5  # GB, approximate size of deepseek-coder-6.7b
        adapter_size = sum(os.path.getsize(os.path.join(dirpath, filename)) 
                        for dirpath, _, filenames in os.walk(args.output_dir) 
                        for filename in filenames) / (1024*1024*1024)
        
        print(f"Original model size: ~{original_model_size:.2f} GB")
        print(f"LoRA adapter size: {adapter_size:.2f} GB")
        print(f"Storage savings: {(original_model_size - adapter_size) / original_model_size * 100:.2f}%")
        
        # Instructions for loading the model
        print("\nTo load this fine-tuned model:")
        print("```python")
        print("from unsloth import FastLanguageModel")
        print(f"adapter_path = '{args.output_dir}'")
        print("base_model = 'deepseek-ai/deepseek-coder-6.7b-base'")
        print("model, tokenizer = FastLanguageModel.from_pretrained(")
        print(f"    model_name=base_model,")
        print(f"    max_seq_length=2048,")
        print(f"    load_in_{args.quantize}bit=True")
        print(")")
        print("model = FastLanguageModel.get_peft_model(model)")
        print("model.load_adapter(adapter_path)")
        print("```")
        
    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 