"""
DeepSeek-Coder Fine-tuning with Unsloth - Usage Examples

This script demonstrates how to use the Unsloth-optimized DeepSeek fine-tuning
for different scenarios and hardware configurations.

Examples:
    - Basic fine-tuning with default settings
    - Custom dataset fine-tuning
    - Hardware-specific optimizations
    - Evaluating fine-tuned models

Usage:
    python finetune_deepseek_examples.py --example [example_name]
"""

# Always import unsloth first before other dependencies
import unsloth

# Import standard libraries
import argparse
import os
import torch
from datasets import load_dataset

# Import Unsloth fine-tuning functionality
from unsloth_deepseek import (
    get_unsloth_model, 
    finetune_with_unsloth, 
    evaluate_model,
    preprocess_for_unsloth
)

from code_preprocessing import load_and_preprocess_dataset, split_dataset

def example_basic_finetuning():
    """Basic fine-tuning example with built-in dataset loading"""
    print("\n=== Example: Basic Fine-tuning ===")
    
    # Load dataset using the enhanced preprocessing function
    train_dataset, eval_dataset = load_and_preprocess_dataset(
        max_samples=1000,  # Limit samples for faster execution
        sequence_length=2048,
        subset="python",
        all_subsets=False,
        return_raw=True  # Return raw text for Unsloth
    )
    
    # Fine-tune with Unsloth
    finetune_with_unsloth(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir="models/deepseek_unsloth_example",
        max_seq_length=2048,
        per_device_train_batch_size=4,
        max_steps=100  # Short training for example purposes
    )

def example_custom_dataset():
    """Fine-tuning with a custom dataset"""
    print("\n=== Example: Custom Dataset Fine-tuning ===")
    
    # Create or load a custom dataset
    try:
        # Try to load a popular code dataset as an example
        dataset = load_dataset("codeparrot/github-code", streaming=False)
        
        # Take a small subset for this example
        train_data = dataset["train"].select(range(1000))
        
        # Create validation split
        splits = split_dataset(train_data, val_ratio=0.2, test_ratio=0.1)
        train_dataset = splits["train"]
        eval_dataset = splits["validation"]
        test_dataset = splits["test"]
        
        # Process the dataset for Unsloth
        train_dataset = preprocess_for_unsloth(train_dataset, format="instruction", subset="python")
        eval_dataset = preprocess_for_unsloth(eval_dataset, format="instruction", subset="python")
        
        # Fine-tune the model
        finetune_with_unsloth(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir="models/deepseek_custom_dataset",
            max_seq_length=1024,
            per_device_train_batch_size=4,
            max_steps=50  # Short training for example
        )
        
        # Evaluate on test set
        evaluate_model(
            model_dir="models/deepseek_custom_dataset",
            test_dataset=preprocess_for_unsloth(test_dataset, format="instruction", subset="python"),
            max_seq_length=1024
        )
        
    except Exception as e:
        print(f"Error loading custom dataset: {e}")
        print("Falling back to mini dataset example")
        
        # Fall back to mini dataset example
        from finetune_deepseek import create_mini_dataset
        from transformers import AutoTokenizer
        
        # Create a mini dataset
        train_data, eval_data = create_mini_dataset(sequence_length=512)
        
        # Convert tokenized dataset to text format for Unsloth
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True)
        
        from unsloth_deepseek import create_text_dataset_from_tokenized
        train_text = create_text_dataset_from_tokenized(train_data, tokenizer)
        eval_text = create_text_dataset_from_tokenized(eval_data, tokenizer)
        
        # Fine-tune with the mini dataset
        finetune_with_unsloth(
            train_dataset=train_text,
            eval_dataset=eval_text,
            output_dir="models/deepseek_mini_dataset",
            max_seq_length=512,
            per_device_train_batch_size=2,
            max_steps=10  # Very short training for this example
        )

def example_hardware_optimized():
    """Example showing hardware-specific optimizations"""
    print("\n=== Example: Hardware-Optimized Fine-tuning ===")
    
    # Detect hardware and apply appropriate settings
    if torch.cuda.is_available():
        print("NVIDIA GPU detected - using optimized settings for CUDA")
        device_type = "cuda"
        batch_size = 8
        max_seq_length = 2048
        load_in_4bit = True
        load_in_8bit = False
        max_samples = 5000
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("Apple Silicon detected - using optimized settings for MPS")
        device_type = "mps"
        batch_size = 2
        max_seq_length = 1024
        load_in_4bit = False
        load_in_8bit = True
        max_samples = 500
    else:
        print("CPU detected - using minimal settings")
        device_type = "cpu"
        batch_size = 1
        max_seq_length = 512
        load_in_4bit = False
        load_in_8bit = True
        max_samples = 100
    
    # Load dataset with hardware-appropriate settings
    train_dataset, eval_dataset = load_and_preprocess_dataset(
        max_samples=max_samples,
        sequence_length=max_seq_length,
        subset="python",
        all_subsets=False,
        return_raw=True
    )
    
    # Fine-tune with hardware-optimized settings
    finetune_with_unsloth(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=f"models/deepseek_optimized_{device_type}",
        max_seq_length=max_seq_length,
        per_device_train_batch_size=batch_size,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        max_steps=50  # Short training for example
    )

def example_evaluation():
    """Example demonstrating model evaluation"""
    print("\n=== Example: Fine-tuned Model Evaluation ===")
    
    # First, check if we have a fine-tuned model
    model_dir = "models/deepseek_unsloth_example"
    
    if not os.path.exists(model_dir):
        print(f"No fine-tuned model found at {model_dir}")
        print("Running a quick fine-tuning first...")
        
        # Run a quick fine-tuning to create a model
        from finetune_deepseek import create_mini_dataset
        from transformers import AutoTokenizer
        from unsloth_deepseek import create_text_dataset_from_tokenized
        
        train_data, eval_data = create_mini_dataset(sequence_length=512)
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True)
        
        train_text = create_text_dataset_from_tokenized(train_data, tokenizer)
        eval_text = create_text_dataset_from_tokenized(eval_data, tokenizer)
        
        finetune_with_unsloth(
            train_dataset=train_text,
            eval_dataset=eval_text,
            output_dir=model_dir,
            max_seq_length=512,
            per_device_train_batch_size=2,
            max_steps=10
        )
    
    # Load test dataset
    test_dataset, _ = load_and_preprocess_dataset(
        max_samples=100,
        sequence_length=512,
        subset="python",
        all_subsets=False,
        return_raw=True
    )
    
    # Evaluate the model
    eval_metrics = evaluate_model(
        model_dir=model_dir,
        test_dataset=test_dataset,
        max_seq_length=512,
        batch_size=2
    )
    
    # Print evaluation metrics
    print("\nEvaluation metrics:")
    for key, value in eval_metrics.items():
        print(f"  - {key}: {value}")

def example_generate_code():
    """Example demonstrating code generation using a fine-tuned model"""
    print("\n=== Example: Code Generation with Fine-tuned Model ===")
    
    # Check if we have a fine-tuned model
    model_dir = "models/deepseek_unsloth_example"
    
    if not os.path.exists(model_dir):
        print(f"No fine-tuned model found at {model_dir}")
        print("Please run example_basic_finetuning first")
        return
    
    # Load the fine-tuned model
    model, tokenizer = get_unsloth_model(
        model_name="deepseek-ai/deepseek-coder-6.7b-base",
        model_dir=model_dir,
        max_seq_length=2048,
        load_in_4bit=True
    )
    
    # Example prompts for code generation
    test_prompts = [
        "### Instruction: Write a Python function to find the largest number in a list.\n\n### Response:",
        "### Instruction: Create a function that calculates the factorial of a number recursively.\n\n### Response:",
        "### Instruction: Implement a binary search algorithm in Python.\n\n### Response:"
    ]
    
    # Generate code for each prompt
    for prompt in test_prompts:
        print("\nPrompt:", prompt)
        print("\nGenerating code...")
        
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=250,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
            )
        
        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Print the generated code (just the response part)
        if "### Response:" in generated_text:
            generated_code = generated_text.split("### Response:", 1)[1].strip()
            print("\nGenerated code:")
            print(generated_code)
        else:
            print("\nGenerated text:", generated_text)

def parse_args():
    parser = argparse.ArgumentParser(description="DeepSeek-Coder Fine-tuning Examples")
    parser.add_argument("--example", type=str, default="all",
                      choices=["basic", "custom", "hardware", "evaluate", "generate", "all"],
                      help="Which example to run")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    examples = {
        "basic": example_basic_finetuning,
        "custom": example_custom_dataset,
        "hardware": example_hardware_optimized,
        "evaluate": example_evaluation,
        "generate": example_generate_code,
        "all": None  # Special case to run all examples
    }
    
    if args.example == "all":
        print("Running all examples sequentially...")
        for name, func in examples.items():
            if name != "all" and callable(func):
                print(f"\n{'='*50}")
                print(f"Running example: {name}")
                print(f"{'='*50}")
                func()
    else:
        # Run the specified example
        examples[args.example]() 