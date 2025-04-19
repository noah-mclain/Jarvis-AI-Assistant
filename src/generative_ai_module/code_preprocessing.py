from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer
import time
import os
import json
import datetime
import torch
import datasets

def load_and_preprocess_dataset(max_samples=None, sequence_length=512, subset="python", all_subsets=False):
    """
    Load and preprocess the code dataset for fine-tuning a code generation model
    
    Args:
        max_samples: Maximum number of samples to include (None for all)
        sequence_length: Maximum sequence length for tokenization
        subset: Language subset of the code dataset (default: python)
        all_subsets: If True, load all language subsets instead of just one
        
    Returns:
        train_data, valid_data: Preprocessed datasets ready for fine-tuning
    """
    if all_subsets:
        return load_and_preprocess_all_subsets(max_samples, sequence_length)
        
    print(f"Loading code_search_net dataset ({subset} subset)...")
    start_time = time.time()
    
    # Load dataset
    try:
        dataset = load_dataset("code_search_net", subset, trust_remote_code=True)
        train_data = dataset["train"]
        valid_data = dataset["validation"]
        
        # Limit samples if specified
        if max_samples:
            train_data = train_data.select(range(min(max_samples, len(train_data))))
            valid_data = valid_data.select(range(min(max_samples // 10, len(valid_data))))
        
        print(f"Loaded dataset with {len(train_data)} training and {len(valid_data)} validation samples")
        
        # Preprocess code examples
        def code_preprocess(example):
            # Format as instruction-following format
            # Using documentation as instruction and code as completion
            docs = example.get('func_documentation_string', '') or "Write a function with the following name and signature."
                
            prompt = f"### Instruction: Implement the following function based on this description:\n{docs}\n\n### Response:\n"
            completion = example.get('func_code_string', example.get('whole_func_string', ''))
            return {"text": prompt + completion}
        
        print("Preprocessing code examples...")
        train_data = train_data.map(code_preprocess)
        valid_data = valid_data.map(code_preprocess)
        
        # Load tokenizer for deepseek-coder
        print("Loading deepseek-coder tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True)
        
        # Configure tokenizer
        tokenizer.pad_token = tokenizer.eos_token
        
        # Tokenize the data
        print(f"Tokenizing data with max length {sequence_length}...")
        def tokenize_function(examples):
            try:
                # Print a sample of the inputs for debugging
                if "text" in examples and len(examples["text"]) > 0:
                    print(f"Sample text to tokenize (truncated): {examples['text'][0][:100]}...")
                else:
                    print(f"Warning: No 'text' field found in examples. Keys: {examples.keys()}")
                    return {"input_ids": [], "attention_mask": []}
                
                # Perform tokenization with proper error handling
                tokenized = tokenizer(
                    examples["text"], 
                    truncation=True, 
                    padding="max_length", 
                    max_length=sequence_length,
                    return_tensors="pt"
                )
                
                return tokenized
            except Exception as e:
                print(f"Error during tokenization: {e}")
                # Return empty tensors as fallback
                batch_size = len(examples.get("text", []))
                return {
                    "input_ids": [[0] * sequence_length] * batch_size,
                    "attention_mask": [[0] * sequence_length] * batch_size
                }
        
        train_data = train_data.map(tokenize_function, batched=True, remove_columns=["text"])
        valid_data = valid_data.map(tokenize_function, batched=True, remove_columns=["text"])
        
        # Set format for PyTorch
        train_data.set_format(type="torch", columns=["input_ids", "attention_mask"])
        valid_data.set_format(type="torch", columns=["input_ids", "attention_mask"])
        
        # Collect metrics
        preprocessing_time = time.time() - start_time
        metrics = {
            'dataset': f"code_search_net_{subset}",
            'train_samples': len(train_data),
            'valid_samples': len(valid_data),
            'sequence_length': sequence_length,
            'preprocessing_time': preprocessing_time,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Save metrics
        save_preprocessing_metrics(metrics)
        
        print(f"Preprocessing completed in {preprocessing_time:.2f} seconds")
        return train_data, valid_data
        
    except Exception as e:
        print(f"Error preprocessing code dataset: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def load_and_preprocess_all_subsets(max_samples=None, sequence_length=512):
    """
    Load and preprocess all language subsets of the code_search_net dataset
    
    Args:
        max_samples: Maximum number of samples per language (None for all)
        sequence_length: Maximum sequence length for tokenization
        
    Returns:
        train_data, valid_data: Combined preprocessed datasets from all languages
    """
    all_subsets = ["python", "java", "go", "php", "ruby", "javascript"]
    start_time = time.time()

    print("Loading ALL language subsets from code_search_net dataset...")

    combined_train = None
    combined_valid = None
    total_train_samples = 0
    total_valid_samples = 0

    # Load tokenizer once for all datasets
    print("Loading deepseek-coder tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Preprocess code examples
    def code_preprocess(example):
        # Format as instruction-following format with language info
        # Using documentation as instruction and code as completion
        lang = example.get('language', 'code')
        docs = example.get('func_documentation_string', '') or "Write a function with the following name and signature."
            
        prompt = f"### Instruction: Implement the following {lang} function based on this description:\n{docs}\n\n### Response:\n"
        completion = example.get('func_code_string', example.get('whole_func_string', ''))
        return {"text": prompt + completion}

    # Tokenize the data
    def tokenize_function(examples):
        try:
            # Print a sample of the inputs for debugging
            if "text" in examples and len(examples["text"]) > 0:
                print(f"Sample text to tokenize (truncated): {examples['text'][0][:100]}...")
            else:
                print(f"Warning: No 'text' field found in examples. Keys: {examples.keys()}")
                return {"input_ids": [], "attention_mask": []}
            
            # Perform tokenization with proper error handling
            tokenized = tokenizer(
                examples["text"], 
                truncation=True, 
                padding="max_length", 
                max_length=sequence_length,
                return_tensors="pt"
            )
            
            return tokenized
        except Exception as e:
            print(f"Error during tokenization: {e}")
            # Return empty tensors as fallback
            batch_size = len(examples.get("text", []))
            return {
                "input_ids": [[0] * sequence_length] * batch_size,
                "attention_mask": [[0] * sequence_length] * batch_size
            }

    # Process each language subset
    for subset in all_subsets:
        try:
            print(f"\nProcessing {subset} subset...")
            dataset = load_dataset("code_search_net", subset, trust_remote_code=True)

            train_data = dataset["train"]
            valid_data = dataset["validation"]

            # Ensure language field is present
            train_data = train_data.map(lambda x: {"language": subset})
            valid_data = valid_data.map(lambda x: {"language": subset})

            # Limit samples if specified
            if max_samples:
                samples_per_lang = max_samples // len(all_subsets)
                train_data = train_data.select(range(min(samples_per_lang, len(train_data))))
                valid_data = valid_data.select(range(min(samples_per_lang // 10, len(valid_data))))

            print(f"  - Loaded {len(train_data)} training and {len(valid_data)} validation samples")
            total_train_samples += len(train_data)
            total_valid_samples += len(valid_data)

            # Preprocess code examples
            train_data = train_data.map(code_preprocess)
            valid_data = valid_data.map(code_preprocess)

            # Tokenize the data
            train_data = train_data.map(tokenize_function, batched=True, remove_columns=["text"])
            valid_data = valid_data.map(tokenize_function, batched=True, remove_columns=["text"])

            # Set format for PyTorch
            train_data.set_format(type="torch", columns=["input_ids", "attention_mask"])
            valid_data.set_format(type="torch", columns=["input_ids", "attention_mask"])

            # Skip empty datasets
            if len(train_data) == 0:
                print(f"Skipping empty train dataset for {subset}")
                continue
                
            # Combine with other subsets
            if combined_train is None:
                combined_train = train_data
                combined_valid = valid_data
            else:
                combined_train = concatenate_datasets([combined_train, train_data])
                if len(valid_data) > 0:
                    # Only add non-empty validation datasets
                    if combined_valid is None:
                        combined_valid = valid_data
                    else:
                        combined_valid = concatenate_datasets([combined_valid, valid_data])

        except Exception as e:
            print(f"Error processing {subset} subset: {e}")
            import traceback
            traceback.print_exc()
            
    # Check if we have any data to return
    if combined_train is None or len(combined_train) == 0:
        print("No valid training data was processed. Creating minimal dummy dataset.")
        # Create a minimal dummy dataset to avoid crashing
        dummy_data = {
            "input_ids": torch.zeros((1, sequence_length), dtype=torch.long),
            "attention_mask": torch.zeros((1, sequence_length), dtype=torch.long)
        }
        combined_train = combined_train or Dataset.from_dict(dummy_data)
        combined_valid = combined_valid or Dataset.from_dict(dummy_data)

    # Collect metrics
    preprocessing_time = time.time() - start_time
    metrics = {
        'dataset': "code_search_net_all_languages",
        'subsets': all_subsets,
        'train_samples': total_train_samples,
        'valid_samples': total_valid_samples,
        'sequence_length': sequence_length,
        'preprocessing_time': preprocessing_time,
        'timestamp': datetime.datetime.now().isoformat()
    }

    # Save metrics
    save_preprocessing_metrics(metrics)

    print(f"\nTotal samples processed: {total_train_samples} training, {total_valid_samples} validation")
    print(f"Combined preprocessing completed in {preprocessing_time:.2f} seconds")

    return combined_train, combined_valid

def save_preprocessing_metrics(metrics):
    """Save preprocessing metrics to a JSON file"""
    metrics_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = os.path.join(metrics_dir, f"preprocessing_code_{timestamp}.json")
    
    # Ensure all numeric values are converted to float for JSON serialization
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            serializable_metrics[key] = float(value)
        elif isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
            serializable_metrics[key] = [float(x) for x in value]
        elif isinstance(value, dict):
            serializable_metrics[key] = {
                k: float(v) if isinstance(v, (int, float)) else v
                for k, v in value.items()
            }
        else:
            serializable_metrics[key] = value
    
    with open(metrics_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=4)
    
    print(f"Preprocessing metrics saved to {metrics_path}")
