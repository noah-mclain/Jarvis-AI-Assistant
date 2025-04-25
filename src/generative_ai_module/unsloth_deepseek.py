"""
DeepSeek-Coder Fine-tuning with Unsloth Optimization

This module provides functions to fine-tune DeepSeek-Coder models using Unsloth optimization.
Unsloth speeds up LLM training and inference while reducing memory usage.

Functions:
    - get_unsloth_model: Load a DeepSeek model optimized with Unsloth
    - finetune_with_unsloth: Fine-tune a DeepSeek model using Unsloth
    - evaluate_model: Evaluate a fine-tuned model on test data
    - create_text_dataset_from_tokenized: Convert tokenized dataset to text format
    - preprocess_for_unsloth: Preprocess code dataset for Unsloth compatibility
"""

# Import unsloth first, before transformers and other libraries
# This ensures all optimizations are properly applied
import unsloth
from unsloth import FastLanguageModel
from unsloth.models import FastDeepseekV2ForCausalLM  # For DeepSeek support

# Import other libraries after unsloth
import os
import time
import datetime
import json
import torch
import numpy as np
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, TrainingArguments
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

def get_unsloth_model(
    model_name: str = "deepseek-ai/deepseek-coder-6.7b-base",
    model_dir: Optional[str] = None,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    use_peft: bool = True,
    r: int = 16,
    target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    alpha: int = 16,
    dropout: float = 0.1
) -> Tuple:
    """
    Load a DeepSeek-Coder model with Unsloth optimization.
    
    Args:
        model_name: The model name or path to load
        model_dir: Directory containing fine-tuned weights (None for base model)
        max_seq_length: Maximum sequence length for the model
        load_in_4bit: Whether to load the model in 4-bit quantization
        load_in_8bit: Whether to load the model in 8-bit quantization
        use_peft: Whether to use PEFT/LoRA for fine-tuning
        r: Rank for LoRA fine-tuning
        target_modules: Which modules to apply LoRA to
        alpha: LoRA alpha parameter
        dropout: Dropout rate for LoRA
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Load model and tokenizer with Unsloth optimization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        # Set device_map to auto to let Unsloth handle device placement
        device_map="auto"
    )
    
    # Apply LoRA if using PEFT
    if use_peft:
        lora_config = LoraConfig(
            r=r,
            target_modules=target_modules,
            lora_alpha=alpha,
            lora_dropout=dropout,
            task_type="CAUSAL_LM",
        )
        
        # Apply LoRA config
        model = FastLanguageModel.get_peft_model(
            model, 
            lora_config,
            # For inference, we can disable gradient checkpointing to save memory
            inference_mode=(model_dir is not None)
        )
    
    # Load fine-tuned weights if provided
    if model_dir and os.path.exists(model_dir):
        model.load_adapter(model_dir, adapter_name="default")
        print(f"Loaded fine-tuned weights from {model_dir}")
    
    return model, tokenizer

def finetune_with_unsloth(
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    model_name: str = "deepseek-ai/deepseek-coder-6.7b-base",
    output_dir: str = "models/deepseek_unsloth",
    max_seq_length: int = 2048,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-5,
    max_steps: int = 500,
    logging_steps: int = 10,
    save_steps: int = 100,
    warmup_steps: int = 50,
    weight_decay: float = 0.01,
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    r: int = 16,
    target_modules: List[str] = None,
    save_total_limit: int = 3,
):
    """
    Fine-tune a DeepSeek-Coder model with Unsloth optimization.
    
    Args:
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        model_name: The model name or path to fine-tune
        output_dir: Directory to save fine-tuned model
        max_seq_length: Maximum sequence length for the model
        per_device_train_batch_size: Batch size per device during training
        gradient_accumulation_steps: Number of updates steps to accumulate before backward pass
        learning_rate: Learning rate for training
        max_steps: Maximum number of training steps
        logging_steps: Log metrics every X steps
        save_steps: Save model checkpoint every X steps
        warmup_steps: Number of steps for learning rate warm-up
        weight_decay: Weight decay for regularization
        load_in_4bit: Whether to load the model in 4-bit quantization
        load_in_8bit: Whether to load the model in 8-bit quantization
        r: Rank for LoRA fine-tuning
        target_modules: Which modules to apply LoRA to
        save_total_limit: Maximum number of checkpoints to save
        
    Returns:
        Dictionary with training metrics
    """
    start_time = time.time()
    
    # Set default target modules for DeepSeek-Coder if not specified
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # Load model and tokenizer with Unsloth optimization
    model, tokenizer = get_unsloth_model(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        use_peft=True,
        r=r,
        target_modules=target_modules
    )
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_steps=max_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        lr_scheduler_type="cosine",
        fp16=not load_in_4bit,  # Use fp16 if not using 4-bit quantization
        bf16=False,
        optim="adamw_torch",
        report_to="none",  # Disable reporting to wandb or other services by default
        group_by_length=True,  # More efficient batching by sequence length
        save_strategy="steps",
        remove_unused_columns=True,
        run_name="deepseek_unsloth"
    )
    
    # Create SFT trainer
    # We need to set tokenizer_name explicitly here for DeepSeek
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",  # Use 'text' field for training
        max_seq_length=max_seq_length,
        args=training_args,
        packing=True,  # Enable packing for more efficient training
        tokenizer_name=model_name  # Set tokenizer name explicitly
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the fine-tuned model
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")
    
    # Create metrics dictionary
    metrics = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        "training_time_minutes": round((time.time() - start_time) / 60, 2),
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "lora_rank": r,
    }
    
    # Save metrics
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

def evaluate_model(
    model_dir: str,
    test_dataset: Dataset,
    max_seq_length: int = 2048,
    batch_size: int = 4,
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
) -> Dict[str, float]:
    """
    Evaluate a fine-tuned model on test data.
    
    Args:
        model_dir: Directory containing fine-tuned model
        test_dataset: Test dataset
        max_seq_length: Maximum sequence length for the model
        batch_size: Batch size for evaluation
        load_in_4bit: Whether to load the model in 4-bit quantization
        load_in_8bit: Whether to load the model in 8-bit quantization
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Load the model and tokenizer
    base_model_name = "deepseek-ai/deepseek-coder-6.7b-base"  # Default base model
    
    # Try to load the model config to get the actual base model name
    try:
        with open(os.path.join(model_dir, "adapter_config.json"), 'r') as f:
            config = json.load(f)
            if "base_model_name_or_path" in config:
                base_model_name = config["base_model_name_or_path"]
    except:
        print(f"Could not load adapter_config.json, using default base model: {base_model_name}")
    
    # Load the fine-tuned model
    model, tokenizer = get_unsloth_model(
        model_name=base_model_name,
        model_dir=model_dir,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit
    )
    
    # Put model in evaluation mode
    model.eval()
    
    # Initialize metrics
    total_loss = 0.0
    num_samples = 0
    
    # Process dataset in batches
    for i in range(0, len(test_dataset), batch_size):
        batch = test_dataset[i:i+batch_size]
        batch_texts = batch["text"] if "text" in batch else batch
        
        # Tokenize inputs
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=max_seq_length
        ).to(model.device)
        
        # Calculate loss
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        
        # Update metrics
        total_loss += loss.item() * len(batch)
        num_samples += len(batch)
        
        if i % 10 == 0:
            print(f"Processed {num_samples} samples, current avg loss: {total_loss / num_samples:.4f}")
    
    # Calculate final metrics
    avg_loss = total_loss / num_samples
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    metrics = {
        "loss": avg_loss,
        "perplexity": perplexity,
        "num_samples": num_samples
    }
    
    # Save metrics
    metrics_path = os.path.join(model_dir, "evaluation_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

def create_text_dataset_from_tokenized(
    dataset: Dict, 
    tokenizer: Any
) -> Dataset:
    """
    Convert a tokenized dataset to text format for Unsloth.
    
    Args:
        dataset: Tokenized dataset
        tokenizer: The tokenizer for the model
        
    Returns:
        Text-based dataset
    """
    # Decode tokens to texts
    texts = [tokenizer.decode(ids) for ids in dataset["input_ids"]]
    
    # Create a new dataset with texts
    return Dataset.from_dict({"text": texts})

def preprocess_for_unsloth(
    examples: Dict, 
    tokenizer: Any, 
    max_seq_length: int = 2048
) -> Dict:
    """
    Preprocess dataset examples for Unsloth training.
    
    Args:
        examples: Dataset examples
        tokenizer: The tokenizer for the model
        max_seq_length: Maximum sequence length for the model
        
    Returns:
        Processed examples
    """
    # If we already have text data, just return it
    if "text" in examples:
        return examples
    
    # Convert input_ids back to text if needed
    if "input_ids" in examples:
        if isinstance(examples["input_ids"], list):
            # Handle batch of examples
            texts = [tokenizer.decode(ids) for ids in examples["input_ids"]]
            return {"text": texts}
        else:
            # Handle single example
            return {"text": tokenizer.decode(examples["input_ids"])}
    
    # If we have prompt and completion fields (instruction format)
    if all(key in examples for key in ["instruction", "response"]):
        texts = []
        for i in range(len(examples["instruction"])):
            # Format as instruction-response pair
            prompt = examples["instruction"][i]
            response = examples["response"][i]
            
            # Create the formatted text
            if "### Instruction:" not in prompt:
                text = f"### Instruction: {prompt}\n\n### Response: {response}"
            else:
                # Already formatted, just combine
                text = f"{prompt}\n\n### Response: {response}"
            
            texts.append(text)
        
        return {"text": texts}
    
    # If we have raw code samples
    if "code" in examples:
        return {"text": examples["code"]}
    
    # Default case: just return the examples as is
    return examples

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