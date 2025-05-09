from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer
import time
import os
import json
import datetime
import torch
import datasets
import numpy as np
from tqdm import tqdm
from src.generative_ai_module.utils import get_storage_path, sync_to_gdrive

def load_and_preprocess_dataset(max_samples=None, sequence_length=512, subset="python", all_subsets=False, return_raw=False, test_split=0.1, val_split=0.1):
    """
    Load and preprocess the code dataset for fine-tuning a code generation model

    Args:
        max_samples: Maximum number of samples to include (None for all)
        sequence_length: Maximum sequence length for tokenization
        subset: Language subset of the code dataset (default: python)
        all_subsets: If True, load all language subsets instead of just one
        return_raw: If True, return raw text dataset instead of tokenized dataset (for Unsloth)
        test_split: Fraction of data to use for test set (default: 0.1)
        val_split: Fraction of data to use for validation set (default: 0.1)

    Returns:
        train_data, valid_data: Preprocessed datasets ready for fine-tuning
        If return_raw is True, datasets will have 'text' field
        Otherwise, they will be tokenized with 'input_ids' and 'attention_mask'
    """
    if all_subsets:
        return load_and_preprocess_all_subsets(
            max_samples=max_samples,
            sequence_length=sequence_length,
            return_raw=return_raw,
            test_split=test_split,
            val_split=val_split
        )

    print(f"Loading code_search_net dataset ({subset} subset)...")
    start_time = time.time()

    # Load dataset
    try:
        # Check if subset contains multiple languages (comma-separated)
        if ',' in subset:
            print(f"Multiple languages detected in subset: {subset}")
            languages = [lang.strip() for lang in subset.split(',')]
            print(f"Will load each language separately: {languages}")

            # Load each language separately and combine
            combined_datasets = []
            for lang in languages:
                try:
                    print(f"Loading {lang} subset...")
                    lang_dataset = load_dataset("code_search_net", lang)
                    combined_datasets.append(lang_dataset)
                    print(f"Successfully loaded {lang} subset")
                except Exception as e:
                    print(f"Error loading {lang} subset: {e}")

            if not combined_datasets:
                print("Failed to load any language subsets. Creating mini dataset as fallback")
                from src.generative_ai_module.finetune_deepseek import create_mini_dataset
                return create_mini_dataset(sequence_length)

            # Combine datasets
            dataset = combined_datasets[0]  # Start with the first dataset

            # Merge the rest if any
            for i in range(1, len(combined_datasets)):
                for split in dataset:
                    if split in combined_datasets[i]:
                        dataset[split] = concatenate_datasets([dataset[split], combined_datasets[i][split]])

            print(f"Combined dataset created with splits: {list(dataset.keys())}")
        else:
            # Single language subset
            try:
                dataset = load_dataset("code_search_net", subset)
            except Exception as e:
                print(f"Standard loading failed, trying alternate method: {e}")
                # If that fails, try with trust_remote_code
                try:
                    dataset = load_dataset("code_search_net", subset, trust_remote_code=True)
                except Exception as e2:
                    print(f"Alternative loading also failed: {e2}")
                    print("Creating mini dataset as fallback")
                    # Create a minimal dataset as a fallback
                    from src.generative_ai_module.finetune_deepseek import create_mini_dataset
                    return create_mini_dataset(sequence_length)

        # Verify we have the expected splits
        required_splits = ['train', 'validation']
        missing_splits = [split for split in required_splits if split not in dataset]
        if missing_splits:
            print(f"Warning: Dataset is missing expected splits: {missing_splits}")
            print("Creating splits from available data...")

            # If no validation set, create one from train
            if 'validation' not in dataset and 'train' in dataset:
                train_valid_test_split = split_dataset(
                    dataset['train'],
                    val_ratio=val_split,
                    test_ratio=test_split
                )
                train_data = train_valid_test_split['train']
                valid_data = train_valid_test_split['validation']
                test_data = train_valid_test_split['test']
            else:
                # Last resort: create a small dummy dataset
                print("Error: No suitable data found. Creating dummy dataset.")
                return create_mini_dataset(sequence_length)
        else:
            # Standard case: we have train and validation splits
            train_data = dataset['train']

            # If validation set exists, split it to get some test data
            if test_split > 0 and 'validation' in dataset:
                valid_test_split = split_dataset(
                    dataset['validation'],
                    val_ratio=1 - test_split,  # This gives us the validation portion
                    test_ratio=test_split      # This gives us the test portion
                )
                valid_data = valid_test_split['train']  # Renaming for clarity
                test_data = valid_test_split['validation']  # Renaming for clarity
            else:
                valid_data = dataset['validation']
                test_data = None

        # Limit samples if specified
        if max_samples:
            train_data = train_data.select(range(min(max_samples, len(train_data))))
            # Adjust validation and test sets proportionally
            val_size = min(max(int(max_samples * val_split), 50), len(valid_data))
            valid_data = valid_data.select(range(val_size))

            if test_data:
                test_size = min(max(int(max_samples * test_split), 50), len(test_data))
                test_data = test_data.select(range(test_size))

        print(f"Dataset loaded with: {len(train_data)} training, {len(valid_data)} validation, and {len(test_data) if test_data else 0} test samples")

        # Clean the data - remove examples with missing or empty content
        train_data = clean_code_dataset(train_data)
        valid_data = clean_code_dataset(valid_data)
        if test_data:
            test_data = clean_code_dataset(test_data)

        print(f"After cleaning: {len(train_data)} training, {len(valid_data)} validation, and {len(test_data) if test_data else 0} test samples")

        # Preprocess code examples
        def code_preprocess(example):
            try:
                # Format as instruction-following format with language info
                # Using documentation as instruction and code as completion
                docs = example.get('func_documentation_string', '') or "Write a function with the following name and signature."
                code = example.get('func_code_string', '') or example.get('whole_func_string', '')

                # Skip if missing critical information
                if not docs.strip() or not code.strip():
                    return {"text": ""}

                # Create properly formatted instruction-response pair
                prompt = f"### Instruction: Implement the following {subset} function based on this description:\n{docs}\n\n### Response:\n"
                return {"text": prompt + code}
            except Exception as e:
                print(f"Error preprocessing example: {e}")
                return {"text": ""}

        print("Preprocessing code examples...")
        train_data = train_data.map(code_preprocess)
        valid_data = valid_data.map(code_preprocess)
        if test_data:
            test_data = test_data.map(code_preprocess)

        # Remove empty examples after preprocessing
        train_data = train_data.filter(lambda x: bool(x.get('text', '').strip()))
        valid_data = valid_data.filter(lambda x: bool(x.get('text', '').strip()))
        if test_data:
            test_data = test_data.filter(lambda x: bool(x.get('text', '').strip()))

        # For Unsloth, we need to return the raw text dataset before tokenization
        if return_raw:
            print("Returning raw text dataset for Unsloth...")
            # Include test_data for Unsloth
            return train_data, valid_data

        # Load tokenizer for deepseek-coder
        print("Loading deepseek-coder tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True)

        # Configure tokenizer
        tokenizer.pad_token = tokenizer.eos_token

        # Tokenize the data
        print(f"Tokenizing data with max length {sequence_length}...")

        # More robust tokenization that handles errors
        def tokenize_function(examples):
            try:
                # Print a sample of the inputs for debugging
                if "text" in examples and len(examples["text"]) > 0:
                    print(f"Sample text to tokenize (truncated): {examples['text'][0][:100]}...")
                else:
                    print(f"Warning: No 'text' field found in examples. Keys: {examples.keys()}")
                    return {"input_ids": [], "attention_mask": []}

                # Filter out empty strings
                valid_texts = [text for text in examples["text"] if text.strip()]
                if not valid_texts:
                    return {"input_ids": [], "attention_mask": []}

                # Perform tokenization with proper error handling - REMOVE device_map parameter
                tokenized = tokenizer(
                    examples["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=sequence_length,
                    return_tensors="pt"
                    # device_map parameter removed to fix the error
                )

                # Ensure we have the right number of examples
                if len(valid_texts) != len(examples["text"]):
                    # Create placeholder tensors for skipped texts
                    n_examples = len(examples["text"])
                    input_ids = torch.zeros((n_examples, sequence_length), dtype=torch.long)
                    attention_mask = torch.zeros((n_examples, sequence_length), dtype=torch.long)

                    # Fill in valid values
                    valid_idx = 0
                    for i, text in enumerate(examples["text"]):
                        if text.strip():
                            input_ids[i] = tokenized["input_ids"][valid_idx]
                            attention_mask[i] = tokenized["attention_mask"][valid_idx]
                            valid_idx += 1

                    return {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask
                    }

                return tokenized
            except Exception as e:
                print(f"Error during tokenization: {e}")
                # Return empty tensors as fallback
                batch_size = len(examples.get("text", []))
                return {
                    "input_ids": torch.zeros((batch_size, sequence_length), dtype=torch.long),
                    "attention_mask": torch.zeros((batch_size, sequence_length), dtype=torch.long)
                }

        # Tokenize all datasets
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
            'test_samples': len(test_data) if test_data else 0,
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

def load_and_preprocess_all_subsets(max_samples=None, sequence_length=512, return_raw=False, test_split=0.1, val_split=0.1):
    """
    Load and preprocess all language subsets of the code_search_net dataset

    Args:
        max_samples: Maximum number of samples per language (None for all)
        sequence_length: Maximum sequence length for tokenization
        return_raw: If True, return raw text dataset instead of tokenized dataset (for Unsloth)
        test_split: Fraction of data to use for test set
        val_split: Fraction of data to use for validation set

    Returns:
        train_data, valid_data: Combined preprocessed datasets from all languages
    """
    # All available language subsets in CodeSearchNet
    all_subsets = ["python", "java", "go", "php", "ruby", "javascript"]
    start_time = time.time()

    # Print information about the dataset
    print("CodeSearchNet dataset information:")
    print("  - Available languages: python, java, go, php, ruby, javascript")
    print("  - Each language contains code functions with documentation")
    print("  - Dataset will be processed to create instruction-response pairs")
    print("  - All languages will be combined into a single dataset")

    print("Loading ALL language subsets from code_search_net dataset...")

    combined_train = None
    combined_valid = None
    combined_test = None
    total_train_samples = 0
    total_valid_samples = 0
    total_test_samples = 0

    # Load tokenizer once for all datasets
    print("Loading deepseek-coder tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Preprocess code examples
    def code_preprocess(example):
        try:
            # Format as instruction-following format with language info
            # Using documentation as instruction and code as completion
            lang = example.get('language', 'code')
            docs = example.get('func_documentation_string', '') or "Write a function with the following name and signature."
            code = example.get('func_code_string', '') or example.get('whole_func_string', '')

            # Skip if missing critical information
            if not docs.strip() or not code.strip():
                return {"text": ""}

            prompt = f"### Instruction: Implement the following {lang} function based on this description:\n{docs}\n\n### Response:\n"
            return {"text": prompt + code}
        except Exception as e:
            print(f"Error preprocessing example: {e}")
            return {"text": ""}

    # Tokenize the data
    def tokenize_function(examples):
        try:
            # Print a sample of the inputs for debugging
            if "text" in examples and len(examples["text"]) > 0:
                print(f"Sample text to tokenize (truncated): {examples['text'][0][:100]}...")
            else:
                print(f"Warning: No 'text' field found in examples. Keys: {examples.keys()}")
                return {"input_ids": [], "attention_mask": []}

            # Filter out empty strings
            valid_texts = [text for text in examples["text"] if text.strip()]
            if not valid_texts:
                return {"input_ids": [], "attention_mask": []}

            # Perform tokenization with proper error handling - REMOVE device_map parameter
            tokenized = tokenizer(
                valid_texts,
                truncation=True,
                padding="max_length",
                max_length=sequence_length,
                return_tensors="pt"
                # device_map parameter removed to fix the error
            )

            # Ensure we have the right number of examples
            if len(valid_texts) != len(examples["text"]):
                # Create placeholder tensors for skipped texts
                n_examples = len(examples["text"])
                input_ids = torch.zeros((n_examples, sequence_length), dtype=torch.long)
                attention_mask = torch.zeros((n_examples, sequence_length), dtype=torch.long)

                # Fill in valid values
                valid_idx = 0
                for i, text in enumerate(examples["text"]):
                    if text.strip():
                        input_ids[i] = tokenized["input_ids"][valid_idx]
                        attention_mask[i] = tokenized["attention_mask"][valid_idx]
                        valid_idx += 1

                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                }

            return tokenized
        except Exception as e:
            print(f"Error during tokenization: {e}")
            # Return empty tensors as fallback
            batch_size = len(examples.get("text", []))
            return {
                "input_ids": torch.zeros((batch_size, sequence_length), dtype=torch.long),
                "attention_mask": torch.zeros((batch_size, sequence_length), dtype=torch.long)
            }

    # Process each language subset
    successful_loads = 0
    for subset in all_subsets:
        try:
            print(f"\nProcessing {subset} subset...")

            # Try loading without trust_remote_code first, then fall back to with it
            try:
                dataset = load_dataset("code_search_net", subset)
            except Exception as e:
                print(f"Standard loading failed, trying alternate method: {e}")
                try:
                    dataset = load_dataset("code_search_net", subset, trust_remote_code=True)
                except Exception as e2:
                    print(f"Error processing {subset} subset: {e2}")
                    continue  # Skip this subset and try another

            # Verify we have the expected splits
            required_splits = ['train', 'validation']
            missing_splits = [split for split in required_splits if split not in dataset]
            if missing_splits:
                print(f"Warning: {subset} dataset is missing expected splits: {missing_splits}")
                if 'train' not in dataset:
                    print(f"Skipping {subset} subset (no training data)")
                    continue

                # If no validation set, create one from train
                if 'validation' not in dataset and 'train' in dataset:
                    train_valid_test_split = split_dataset(
                        dataset['train'],
                        val_ratio=val_split,
                        test_ratio=test_split
                    )
                    train_data = train_valid_test_split['train']
                    valid_data = train_valid_test_split['validation']
                    test_data = train_valid_test_split['test']
                else:
                    continue
            else:
                # Standard case: we have train and validation splits
                train_data = dataset['train']

                # If validation set exists, split it to get some test data
                if test_split > 0 and 'validation' in dataset:
                    valid_test_split = split_dataset(
                        dataset['validation'],
                        val_ratio=1 - test_split,  # This gives us the validation portion
                        test_ratio=test_split      # This gives us the test portion
                    )
                    valid_data = valid_test_split['train']  # Renaming for clarity
                    test_data = valid_test_split['validation']  # Renaming for clarity
                else:
                    valid_data = dataset['validation']
                    test_data = None

            # Ensure language field is present
            train_data = train_data.map(lambda x: {"language": subset})
            valid_data = valid_data.map(lambda x: {"language": subset})
            if test_data:
                test_data = test_data.map(lambda x: {"language": subset})

            # Limit samples if specified
            if max_samples:
                # Calculate per-language sample sizes based on total max_samples
                # Allow for more flexible distribution - allocate more to training
                train_ratio = 0.8  # Use 80% for training
                val_ratio = 0.1    # Use 10% for validation
                test_ratio = 0.1   # Use 10% for testing

                # Calculate per-language sample limits
                # We want the total (across all languages) to respect max_samples
                max_per_lang = max_samples // len(all_subsets)
                max_train = int(max_per_lang * train_ratio)
                max_val = int(max_per_lang * val_ratio)
                max_test = int(max_per_lang * test_ratio)

                # Ensure minimum sample sizes for validation and test
                max_train = max(1, max_train)
                max_val = max(1, max_val)
                max_test = max(1, max_test)

                # Limit training samples
                train_data = train_data.select(range(min(max_train, len(train_data))))

                # Limit validation samples
                valid_data = valid_data.select(range(min(max_val, len(valid_data))))

                # Limit test samples if present
                if test_data:
                    test_data = test_data.select(range(min(max_test, len(test_data))))

            print(f"  - Loaded {len(train_data)} training, {len(valid_data)} validation, and {len(test_data) if test_data else 0} test samples")

            # Clean the data - remove examples with missing or empty content
            train_data = clean_code_dataset(train_data)
            valid_data = clean_code_dataset(valid_data)
            if test_data:
                test_data = clean_code_dataset(test_data)

            print(f"  - After cleaning: {len(train_data)} training, {len(valid_data)} validation, and {len(test_data) if test_data else 0} test samples")

            total_train_samples += len(train_data)
            total_valid_samples += len(valid_data)
            total_test_samples += len(test_data) if test_data else 0

            # Preprocess code examples
            train_data = train_data.map(code_preprocess)
            valid_data = valid_data.map(code_preprocess)
            if test_data:
                test_data = test_data.map(code_preprocess)

            # Remove empty examples after preprocessing
            train_data = train_data.filter(lambda x: bool(x.get('text', '').strip()))
            valid_data = valid_data.filter(lambda x: bool(x.get('text', '').strip()))
            if test_data:
                test_data = test_data.filter(lambda x: bool(x.get('text', '').strip()))

            # Skip empty datasets
            if len(train_data) == 0:
                print(f"Skipping empty train dataset for {subset}")
                continue

            # Combine with other subsets
            if combined_train is None:
                combined_train = train_data
                combined_valid = valid_data
                if test_data:
                    combined_test = test_data
            else:
                combined_train = concatenate_datasets([combined_train, train_data])

                if len(valid_data) > 0:
                    # Only add non-empty validation datasets
                    if combined_valid is None:
                        combined_valid = valid_data
                    else:
                        combined_valid = concatenate_datasets([combined_valid, valid_data])

                if test_data and len(test_data) > 0:
                    # Only add non-empty test datasets
                    if combined_test is None:
                        combined_test = test_data
                    else:
                        combined_test = concatenate_datasets([combined_test, test_data])

            successful_loads += 1
            print(f"Successfully loaded {subset} subset")

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

    # For Unsloth, we return the raw text datasets before tokenization
    if return_raw:
        print("Returning raw text dataset for Unsloth...")
        return combined_train, combined_valid

    # Tokenize and format datasets for standard training
    if not return_raw:
        # Tokenize the data
        combined_train = combined_train.map(tokenize_function, batched=True, remove_columns=["text"])
        combined_valid = combined_valid.map(tokenize_function, batched=True, remove_columns=["text"])

        # Set format for PyTorch
        combined_train.set_format(type="torch", columns=["input_ids", "attention_mask"])
        combined_valid.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # Collect metrics
    preprocessing_time = time.time() - start_time
    metrics = {
        'dataset': "code_search_net_all_languages",
        'subsets': all_subsets,
        'train_samples': total_train_samples,
        'valid_samples': total_valid_samples,
        'test_samples': total_test_samples,
        'sequence_length': sequence_length,
        'preprocessing_time': preprocessing_time,
        'timestamp': datetime.datetime.now().isoformat()
    }

    # Save metrics
    save_preprocessing_metrics(metrics)

    print(f"\nTotal samples processed: {total_train_samples} training, {total_valid_samples} validation, {total_test_samples} test")
    print(f"Combined preprocessing completed in {preprocessing_time:.2f} seconds")

    return combined_train, combined_valid

def clean_code_dataset(dataset):
    """
    Clean a code dataset by removing examples with missing or empty content.

    Args:
        dataset: The dataset to clean

    Returns:
        Cleaned dataset
    """
    # Define a function to check if an example is valid
    def is_valid_example(example):
        # Check if documentation exists
        has_docs = bool(example.get('func_documentation_string', '').strip())

        # Check if code exists (in either field)
        has_code = (
            bool(example.get('func_code_string', '').strip()) or
            bool(example.get('whole_func_string', '').strip())
        )

        # Must have both documentation and code
        return has_docs and has_code

    # Apply the filter
    original_size = len(dataset)
    cleaned_dataset = dataset.filter(is_valid_example)
    new_size = len(cleaned_dataset)

    if original_size != new_size:
        removed = original_size - new_size
        percent_removed = (removed / original_size) * 100 if original_size > 0 else 0
        print(f"Removed {removed} examples ({percent_removed:.1f}%) with missing or empty content")

    return cleaned_dataset

def split_dataset(dataset, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Split a dataset into train, validation, and test sets.

    Args:
        dataset: The dataset to split
        val_ratio: Fraction of data to use for validation
        test_ratio: Fraction of data to use for test
        seed: Random seed for reproducibility

    Returns:
        Dictionary with 'train', 'validation', and 'test' splits
    """
    # Calculate split sizes
    train_ratio = 1.0 - (val_ratio + test_ratio)

    # Ensure ratios sum to 1.0
    if train_ratio <= 0:
        print("Warning: Invalid split ratios. Adjusting to 80/10/10 split.")
        train_ratio = 0.8
        val_ratio = 0.1
        test_ratio = 0.1

    # Shuffle and split the dataset
    shuffled = dataset.shuffle(seed=seed)

    # Calculate indices for splitting
    val_idx = int(len(shuffled) * train_ratio)
    test_idx = int(len(shuffled) * (train_ratio + val_ratio))

    # Create the splits
    train_data = shuffled.select(range(val_idx))
    valid_data = shuffled.select(range(val_idx, test_idx))
    test_data = shuffled.select(range(test_idx, len(shuffled)))

    print(f"Split dataset into {len(train_data)} training, {len(valid_data)} validation, and {len(test_data)} test examples")

    return {
        'train': train_data,
        'validation': valid_data,
        'test': test_data
    }

def save_preprocessing_metrics(metrics):
    """Save preprocessing metrics to a JSON file"""
    # Use the storage path utility to get the correct path
    metrics_dir = get_storage_path("metrics")
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

    # Sync metrics to Google Drive
    sync_to_gdrive("metrics")
