"""
DeepSeek-Coder Fine-tuning with Unsloth

This module implements fine-tuning for DeepSeek-Coder models using Unsloth for
faster training, reduced memory usage, and longer context length support.

Functions:
    - get_unsloth_model: Load a DeepSeek model optimized with Unsloth
    - finetune_with_unsloth: Fine-tune a DeepSeek model using Unsloth
    - evaluate_model: Evaluate a fine-tuned model on test data
    - create_text_dataset_from_tokenized: Convert tokenized dataset to text format
    - preprocess_for_unsloth: Preprocess code dataset for Unsloth compatibility
"""

# Import Unsloth for optimized training
from unsloth import FastLanguageModel
from unsloth.models import FastDeepseekV2ForCausalLM  # For DeepSeek support
from trl import SFTTrainer, SFTConfig

import os
import time
import datetime
import json
import torch
import numpy as np
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

def get_unsloth_model(
    model_name="deepseek-ai/deepseek-coder-6.7b-base",
    model_dir=None,  # Directory for loading a saved model
    max_seq_length=2048,
    load_in_4bit=True,
    load_in_8bit=False,
    full_finetuning=False,
    gradient_checkpointing="unsloth",
    r=16,  # LoRA rank
    lora_alpha=16,
    lora_dropout=0,
    use_rslora=False,
    token=None  # HF token if needed
):
    """
    Load a DeepSeek model optimized with Unsloth.
    
    Args:
        model_name: Name of the DeepSeek model to load
        model_dir: Directory for loading a saved model (if None, load from HF)
        max_seq_length: Maximum sequence length for model
        load_in_4bit: Whether to load in 4-bit precision (for less VRAM usage)
        load_in_8bit: Whether to load in 8-bit precision (for less VRAM usage but more accurate than 4-bit)
        full_finetuning: Whether to do full finetuning (more VRAM but potentially better results)
        gradient_checkpointing: Whether to use gradient checkpointing for long sequences ("unsloth" for best performance)
        r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        use_rslora: Whether to use rank-stabilized LoRA
        token: Hugging Face token for accessing gated models
        
    Returns:
        model, tokenizer: The optimized model and tokenizer
    """
    # Determine device
    if torch.cuda.is_available():
        device_type = "cuda"
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device_type = "mps"
        print("Using Apple Silicon GPU via MPS backend")
        # MPS (Apple Silicon) doesn't fully support 4-bit, so use 8-bit
        if load_in_4bit:
            print("Apple Silicon detected - switching from 4-bit to 8-bit quantization")
            load_in_4bit = False
            load_in_8bit = True
    else:
        device_type = "cpu"
        print("Using CPU (no GPU available)")
        # CPU doesn't work well with 4-bit, use 8-bit
        if load_in_4bit:
            load_in_4bit = False
            load_in_8bit = True

    if model_dir:
        print(f"Loading model from: {model_dir}")
    else:
        print(f"Loading model from Hugging Face: {model_name}")

    # Load tokenizer first
    try:
        # Try loading tokenizer from model_dir if specified
        if model_dir and os.path.exists(os.path.join(model_dir, "tokenizer_config.json")):
            tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, token=token)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=token)

        # Ensure PAD token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Falling back to base model tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=token)
        tokenizer.pad_token = tokenizer.eos_token

    # Load optimized model with Unsloth
    try:
        # Determine model path (local or HF)
        model_path = model_dir or model_name

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=None,  # Let Unsloth decide based on hardware
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            full_finetuning=full_finetuning,
            token=token,
        )

        # Target all linear layers in DeepSeek
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                         "gate_proj", "up_proj", "down_proj"]

        # Add fast LoRA weights
        model = FastLanguageModel.get_peft_model(
            model,
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",  # Optimized setting
            use_gradient_checkpointing=gradient_checkpointing,
            random_state=3407,  # For reproducibility
            max_seq_length=max_seq_length,
            use_rslora=use_rslora,
        )

        print("Model loaded with Unsloth optimization:")
        print(f"  - Using 4-bit quantization: {load_in_4bit}")
        print(f"  - Using 8-bit quantization: {load_in_8bit}")
        print(f"  - Sequence length: {max_seq_length}")
        print(f"  - Gradient checkpointing: {gradient_checkpointing}")
        print(f"  - LoRA rank: {r}")

    except Exception as e:
        print(f"Error loading model with Unsloth: {e}")
        raise ValueError(f"Failed to load model: {e}") from e

    return model, tokenizer

def finetune_with_unsloth(
    train_dataset,
    eval_dataset=None,
    model_name="deepseek-ai/deepseek-coder-6.7b-base",
    output_dir="models/deepseek_unsloth",
    max_seq_length=2048,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    max_steps=60,
    learning_rate=2e-5,
    load_in_4bit=True,
    load_in_8bit=False,
    r=16,
    token=None,
    save_total_limit=3,
    logging_steps=1,
    eval_strategy="steps",
    eval_steps=20,
):
    """
    Fine-tune a DeepSeek model using Unsloth optimization.
    
    Args:
        train_dataset: Dataset for training
        eval_dataset: Dataset for evaluation (optional)
        model_name: Name of the DeepSeek model to fine-tune
        output_dir: Directory to save fine-tuned model
        max_seq_length: Maximum sequence length for model
        per_device_train_batch_size: Batch size per device during training
        gradient_accumulation_steps: Number of updates steps to accumulate before performing a backward/update pass
        warmup_steps: Linear warmup over warmup_steps
        max_steps: Total number of training steps
        learning_rate: Initial learning rate
        load_in_4bit: Whether to load in 4-bit precision (for less VRAM usage)
        load_in_8bit: Whether to load in 8-bit precision (for less VRAM usage but more accurate than 4-bit)
        r: LoRA rank
        token: Hugging Face token for accessing gated models
        save_total_limit: Limit the total amount of checkpoints saved
        logging_steps: Log every x updates steps
        eval_strategy: Evaluation strategy to adopt during training
        eval_steps: Number of steps between evaluations
        
    Returns:
        Dictionary with training metrics
    """
    start_time = time.time()

    # Check if train_dataset is None or empty
    if not train_dataset or len(train_dataset) == 0:
        raise ValueError("Training dataset is empty or None. Cannot proceed with fine-tuning.")

    # Verify dataset format
    if "text" not in train_dataset.column_names and ("input_ids" in train_dataset.column_names and "attention_mask" in train_dataset.column_names):
        print("Dataset already tokenized, creating a new dataset with detokenized text")
        # We need to create a new dataset with the text field
        raise ValueError("Dataset already tokenized. Unsloth requires raw text dataset. Please provide untokenized dataset with 'text' field.")

    # Clean dataset by filtering out empty or None examples
    def is_valid_example(example):
        return bool(example.get("text", "").strip())

    print("Cleaning dataset...")
    original_train_size = len(train_dataset)
    train_dataset = train_dataset.filter(is_valid_example)
    new_train_size = len(train_dataset)
    if original_train_size != new_train_size:
        print(f"Removed {original_train_size - new_train_size} invalid examples from training dataset")

    if eval_dataset:
        original_eval_size = len(eval_dataset)
        eval_dataset = eval_dataset.filter(is_valid_example)
        new_eval_size = len(eval_dataset)
        if original_eval_size != new_eval_size:
            print(f"Removed {original_eval_size - new_eval_size} invalid examples from evaluation dataset")

    # Load model with Unsloth optimization
    model, tokenizer = get_unsloth_model(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        r=r,
        token=token,
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Check for CUDA compatibility
    if torch.cuda.is_available() and eval_strategy == "epoch" and len(train_dataset) > 10000:
        print("Warning: Using 'epoch' evaluation strategy with large dataset may cause CUDA OOM errors")
        print("Switching to 'steps' evaluation strategy")
        eval_strategy = "steps"

    # Configure SFT trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            output_dir=output_dir,
            optim="adamw_8bit",
            save_total_limit=save_total_limit,
            save_strategy=eval_strategy,
            evaluation_strategy=eval_strategy,
            eval_steps=eval_steps,
            seed=3407,  # For reproducibility
            report_to="none",  # Disable wandb logging
            load_best_model_at_end=True,  # Load the best model at the end of training
            metric_for_best_model="eval_loss",  # Use eval loss to determine the best model
        ),
        packing=False,  # Packing can make Unsloth slower
    )

    # Fine-tune the model
    print("Starting fine-tuning with Unsloth...")
    print(f"Training on {len(train_dataset)} examples for {max_steps} steps with batch size {per_device_train_batch_size}")

    # Gracefully handle training exceptions
    try:
        training_output = trainer.train()
        print("Training completed successfully")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

        # Save partial results if possible
        try:
            print("Attempting to save partial model...")
            trainer.save_model(os.path.join(output_dir, "partial_model"))
            print("Partial model saved")
        except Exception:
            print("Failed to save partial model")

        # Return partial metrics
        training_time = time.time() - start_time
        hours, remainder = divmod(training_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        return {
            'model_name': model_name,
            'dataset_size': len(train_dataset),
            'training_time': training_time,
            'training_time_formatted': f"{int(hours)}h {int(minutes)}m {int(seconds)}s",
            'error': str(e),
            'timestamp': datetime.datetime.now().isoformat(),
        }

    # Calculate training time
    training_time = time.time() - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")

    # Save the model and tokenizer
    print(f"Saving fine-tuned model to {output_dir}...")
    trainer.save_model(output_dir)

    # Evaluate if eval dataset is provided
    eval_metrics = {}
    if eval_dataset is not None:
        print("Evaluating fine-tuned model...")
        eval_metrics = trainer.evaluate()
        print(f"Evaluation loss: {eval_metrics.get('eval_loss', 'N/A')}")
        if 'eval_loss' in eval_metrics:
            perplexity = np.exp(eval_metrics['eval_loss'])
            print(f"Perplexity: {perplexity:.4f}")
            eval_metrics['perplexity'] = perplexity

    # Collect metrics
    training_metrics = {
        'model_name': model_name,
        'dataset_size': len(train_dataset),
        'eval_dataset_size': len(eval_dataset) if eval_dataset else 0,
        'batch_size': per_device_train_batch_size,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'effective_batch_size': per_device_train_batch_size * gradient_accumulation_steps,
        'training_time': training_time,
        'training_time_formatted': f"{int(hours)}h {int(minutes)}m {int(seconds)}s",
        'max_seq_length': max_seq_length,
        'lora_rank': r,
        'load_in_4bit': load_in_4bit,
        'load_in_8bit': load_in_8bit,
        'timestamp': datetime.datetime.now().isoformat(),
        **eval_metrics
    }

    # Save metrics
    metrics_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = os.path.join(metrics_dir, f"unsloth_training_{timestamp}.json")

    with open(metrics_path, 'w') as f:
        json.dump(training_metrics, f, indent=4)

    print(f"Training metrics saved to {metrics_path}")

    return training_metrics

def evaluate_model(
    model_dir,
    test_dataset,
    max_seq_length=2048,
    batch_size=4,
    load_in_4bit=True,
    load_in_8bit=False
):
    """
    Evaluate a fine-tuned model on test data.
    
    Args:
        model_dir: Directory containing the fine-tuned model
        test_dataset: Dataset for evaluation
        max_seq_length: Maximum sequence length for model
        batch_size: Batch size for evaluation
        load_in_4bit: Whether to load in 4-bit precision (for less VRAM usage)
        load_in_8bit: Whether to load in 8-bit precision (for less VRAM usage but more accurate than 4-bit)
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Evaluating model from {model_dir} on {len(test_dataset)} test examples")
    
    # Load model with Unsloth
    model, tokenizer = get_unsloth_model(
        model_name="deepseek-ai/deepseek-coder-6.7b-base",  # Base model
        model_dir=model_dir,  # Load fine-tuned weights from this directory
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )
    
    # Ensure we have text field in the dataset
    if "text" not in test_dataset.column_names:
        raise ValueError("Test dataset must have a 'text' field")
    
    # Clean dataset by filtering out empty or None examples
    def is_valid_example(example):
        return bool(example.get("text", "").strip())
    
    print("Cleaning test dataset...")
    original_size = len(test_dataset)
    test_dataset = test_dataset.filter(is_valid_example)
    new_size = len(test_dataset)
    if original_size != new_size:
        print(f"Removed {original_size - new_size} invalid examples from test dataset")
    
    # Configure trainer for evaluation
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            per_device_eval_batch_size=batch_size,
            output_dir="temp_eval_dir",  # Temporary directory for evaluation
        ),
        packing=False,
    )
    
    # Evaluate the model
    print("Starting evaluation...")
    eval_metrics = trainer.evaluate(eval_dataset=test_dataset)
    
    # Calculate perplexity
    if 'eval_loss' in eval_metrics:
        perplexity = np.exp(eval_metrics['eval_loss'])
        eval_metrics['perplexity'] = perplexity
        print(f"Evaluation loss: {eval_metrics['eval_loss']:.4f}")
        print(f"Perplexity: {perplexity:.4f}")
    
    return eval_metrics

def create_text_dataset_from_tokenized(tokenized_dataset, tokenizer):
    """
    Convert a tokenized dataset back to text format for Unsloth compatibility.
    
    Args:
        tokenized_dataset: The tokenized dataset with input_ids and attention_mask
        tokenizer: The tokenizer to decode the input_ids
        
    Returns:
        Dataset with a text field
    """
    if "input_ids" not in tokenized_dataset.column_names:
        raise ValueError("Dataset must have 'input_ids' field to be detokenized")
    
    texts = []
    print(f"Converting tokenized dataset with {len(tokenized_dataset)} examples to text format...")
    
    for item in tqdm(tokenized_dataset, desc="Converting to text dataset"):
        # Decode input_ids to text, handling potential errors
        try:
            # Handle different input_ids formats (list, tensor, etc.)
            if isinstance(item["input_ids"], torch.Tensor):
                input_ids = item["input_ids"].tolist()
            else:
                input_ids = item["input_ids"]
                
            # Skip empty or padding-only sequences
            if not input_ids or all(id == tokenizer.pad_token_id for id in input_ids):
                print("Warning: Found empty or padding-only sequence, skipping")
                continue
                
            # Decode to text
            text = tokenizer.decode(input_ids, skip_special_tokens=True)
            
            # Skip empty texts
            if not text.strip():
                print("Warning: Decoded to empty text, skipping")
                continue
                
            texts.append({"text": text})
        except Exception as e:
            print(f"Error decoding input_ids: {e}")
            continue
    
    print(f"Successfully converted {len(texts)} examples to text format")
    
    # If all examples were invalid, create a minimal dataset with a warning
    if not texts:
        print("WARNING: All examples were invalid. Creating a minimal dataset.")
        texts = [{"text": "WARNING: This is a dummy example because all real examples were invalid."}]
    
    return Dataset.from_list(texts)

def preprocess_for_unsloth(dataset, format="instruction", subset="python", clean_data=True):
    """
    Preprocesses code dataset for instruction format expected by Unsloth.
    
    Args:
        dataset: The raw dataset with code examples
        format: The format to convert to ("instruction" or "chat")
        subset: The programming language subset
        clean_data: Whether to clean the data (remove empty examples, etc.)
        
    Returns:
        Preprocessed dataset with text field formatted for Unsloth
    """
    # First, verify required fields are present
    required_fields = ['func_documentation_string', 'func_code_string']
    missing_fields = [field for field in required_fields if field not in dataset.column_names]
    
    if missing_fields:
        print(f"Warning: Dataset is missing required fields: {missing_fields}")
        # Check for alternative fields
        if 'whole_func_string' not in dataset.column_names and 'func_code_string' in missing_fields:
            print("Error: Dataset must have either 'func_code_string' or 'whole_func_string'")
            raise ValueError("Dataset missing critical code content fields")
    
    def code_preprocess(example):
        try:
            # Format as instruction-following format
            # Using documentation as instruction and code as completion
            docs = example.get('func_documentation_string', '') or "Write a function with the following name and signature."
            lang = example.get('language', subset)
            
            # Handle potentially missing code field
            if 'func_code_string' in example and example['func_code_string']:
                code = example['func_code_string']
            elif 'whole_func_string' in example and example['whole_func_string']:
                code = example['whole_func_string']
            else:
                # Skip examples without code
                return {"text": ""}
                
            # Skip if documentation is empty or code is empty
            if not docs.strip() or not code.strip():
                return {"text": ""}
                
            if format == "instruction":
                prompt = f"### Instruction: Implement the following {lang} function based on this description:\n{docs}\n\n### Response:\n"
                return {"text": prompt + code}
            elif format == "chat":
                # Chat format can be useful for some models
                system = "You are an expert coding assistant that helps implement functions based on descriptions."
                user_msg = f"Implement the following {lang} function based on this description:\n{docs}"
                return {
                    "text": f"<|system|>\n{system}\n<|user|>\n{user_msg}\n<|assistant|>\n{code}"
                }
            else:
                raise ValueError(f"Unknown format: {format}")
        except Exception as e:
            print(f"Error processing example: {e}")
            return {"text": ""}  # Return empty text for invalid examples
    
    print(f"Preprocessing dataset with {len(dataset)} examples...")
    processed_dataset = dataset.map(code_preprocess)
    
    # Clean dataset if requested
    if clean_data:
        # Filter out empty or invalid examples
        original_size = len(processed_dataset)
        processed_dataset = processed_dataset.filter(lambda example: bool(example.get("text", "").strip()))
        new_size = len(processed_dataset)
        if original_size != new_size:
            print(f"Removed {original_size - new_size} invalid examples during preprocessing")
    
    return processed_dataset

if __name__ == "__main__":
    # Simple test with mini dataset
    from finetune_deepseek import create_mini_dataset
    
    # Create minimal test dataset
    train_dataset, eval_dataset = create_mini_dataset(sequence_length=512)
    
    # We need to convert tokenized dataset to text format for Unsloth
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True)
    train_text_dataset = create_text_dataset_from_tokenized(train_dataset, tokenizer)
    eval_text_dataset = create_text_dataset_from_tokenized(eval_dataset, tokenizer)
    
    # Fine-tune with Unsloth
    finetune_with_unsloth(
        train_text_dataset,
        eval_text_dataset,
        output_dir="models/deepseek_unsloth_test",
        max_steps=10,  # Short test run
    ) 