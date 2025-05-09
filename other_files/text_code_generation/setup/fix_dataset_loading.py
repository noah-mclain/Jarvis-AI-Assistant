#!/usr/bin/env python3
"""
Fix for dataset loading issues in DeepSeek Coder training.
This script provides a robust dataset loading solution that works on Paperspace.
"""

import os
import sys
import torch
import gc
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets

# Force CPU for initial model loading
os.environ["FORCE_CPU_ONLY_FOR_INITIAL_LOAD"] = "1"

# Set PyTorch to use CPU as default device initially
if hasattr(torch, 'set_default_device'):
    torch.set_default_device('cpu')
torch.set_default_tensor_type('torch.FloatTensor')

# Clear any existing CUDA memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

# Set environment variables for optimal memory usage
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,garbage_collection_threshold:0.9"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/tmp/hf_cache"  # Use temporary directory for cache

# Add the current directory to the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

def create_code_dataset(max_samples=1000, sequence_length=512):
    """
    Create a code dataset from scratch using GitHub code and code_search_net.
    This function loads each subset separately and combines them.
    
    Args:
        max_samples: Maximum number of samples to include
        sequence_length: Maximum sequence length
        
    Returns:
        DatasetDict with train and validation splits
    """
    print("Creating robust code dataset...")
    
    datasets = []
    
    # Try to load Python subset
    try:
        print("Loading Python subset from code_search_net...")
        python_dataset = load_dataset("code_search_net", "python", split="train")
        if python_dataset:
            # Take a subset of the data
            python_samples = min(max_samples // 2, len(python_dataset))
            python_dataset = python_dataset.select(range(python_samples))
            datasets.append(python_dataset)
            print(f"Added {python_samples} Python samples")
    except Exception as e:
        print(f"Error loading Python subset: {e}")
    
    # Try to load JavaScript subset
    try:
        print("Loading JavaScript subset from code_search_net...")
        js_dataset = load_dataset("code_search_net", "javascript", split="train")
        if js_dataset:
            # Take a subset of the data
            js_samples = min(max_samples // 2, len(js_dataset))
            js_dataset = js_dataset.select(range(js_samples))
            datasets.append(js_dataset)
            print(f"Added {js_samples} JavaScript samples")
    except Exception as e:
        print(f"Error loading JavaScript subset: {e}")
    
    # If we couldn't load any datasets, create a minimal one
    if not datasets:
        print("Creating minimal code dataset...")
        
        # Create a minimal dataset with Python code examples
        code_examples = [
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)",
            "class Node:\n    def __init__(self, value):\n        self.value = value\n        self.left = None\n        self.right = None",
            "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",
            "import numpy as np\n\ndef matrix_multiply(a, b):\n    return np.dot(a, b)",
            "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1"
        ]
        
        # Create a dataset from the examples
        minimal_dataset = Dataset.from_dict({"code": code_examples})
        datasets.append(minimal_dataset)
        print(f"Added {len(code_examples)} minimal code examples")
    
    # Combine all datasets
    if len(datasets) > 1:
        combined_dataset = concatenate_datasets(datasets)
    else:
        combined_dataset = datasets[0]
    
    print(f"Combined dataset size: {len(combined_dataset)} samples")
    
    # Split into train and validation
    train_size = int(0.9 * len(combined_dataset))
    val_size = len(combined_dataset) - train_size
    
    # Shuffle the dataset
    combined_dataset = combined_dataset.shuffle(seed=42)
    
    # Create train and validation splits
    train_dataset = combined_dataset.select(range(train_size))
    val_dataset = combined_dataset.select(range(train_size, train_size + val_size))
    
    # Create a DatasetDict
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })
    
    print(f"Final dataset: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
    
    return dataset_dict

def tokenize_dataset(dataset_dict, tokenizer, sequence_length=512):
    """
    Tokenize the dataset without using device_map parameter.
    
    Args:
        dataset_dict: DatasetDict with train and validation splits
        tokenizer: Tokenizer to use
        sequence_length: Maximum sequence length
        
    Returns:
        DatasetDict with tokenized train and validation splits
    """
    print("Tokenizing dataset...")
    
    def tokenize_function(examples):
        # Extract the code field or use the first available text field
        if "code" in examples:
            texts = examples["code"]
        elif "func" in examples:
            texts = examples["func"]
        elif "content" in examples:
            texts = examples["content"]
        elif "text" in examples:
            texts = examples["text"]
        else:
            # Use the first field that contains strings
            for key, value in examples.items():
                if isinstance(value, list) and all(isinstance(item, str) for item in value):
                    texts = value
                    break
            else:
                raise ValueError("No suitable text field found in the dataset")
        
        # Tokenize without device_map parameter
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=sequence_length,
            return_tensors="pt"
        )
    
    # Tokenize train and validation splits
    tokenized_train = dataset_dict["train"].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset_dict["train"].column_names
    )
    
    tokenized_val = dataset_dict["validation"].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset_dict["validation"].column_names
    )
    
    # Create a new DatasetDict with tokenized datasets
    tokenized_dataset_dict = DatasetDict({
        "train": tokenized_train,
        "validation": tokenized_val
    })
    
    print("Dataset tokenization complete")
    
    return tokenized_dataset_dict

# This function will be imported and used in the training script
def get_fixed_datasets(tokenizer, max_samples=1000, sequence_length=512):
    """
    Get fixed datasets for training.
    
    Args:
        tokenizer: Tokenizer to use
        max_samples: Maximum number of samples to include
        sequence_length: Maximum sequence length
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    # Create the dataset
    dataset_dict = create_code_dataset(max_samples, sequence_length)
    
    # Tokenize the dataset
    tokenized_dataset_dict = tokenize_dataset(dataset_dict, tokenizer, sequence_length)
    
    return tokenized_dataset_dict["train"], tokenized_dataset_dict["validation"]

if __name__ == "__main__":
    # This script can be run standalone to test dataset creation
    from transformers import AutoTokenizer
    
    print("Testing dataset creation...")
    
    # Load a simple tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Create and tokenize the dataset
    train_dataset, eval_dataset = get_fixed_datasets(tokenizer, max_samples=10, sequence_length=128)
    
    print(f"Created datasets: {len(train_dataset)} training samples, {len(eval_dataset)} validation samples")
    print("Sample input_ids shape:", train_dataset[0]["input_ids"].shape)
    print("Test successful!")
