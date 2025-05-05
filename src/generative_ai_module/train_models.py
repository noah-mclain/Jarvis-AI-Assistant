#!/usr/bin/env python3
# ===== BEGIN JARVIS IMPORT FIX =====
# This block was added by the fix_jarvis_imports.py script
import sys
import os

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
    
    def get_loss_function(task_type="generation"):
        """
        Return an appropriate loss function based on the task type.
        
        Args:
            task_type (str): Type of task: 'classification', 'regression', or 'generation'
            
        Returns:
            torch.nn.Module: The appropriate loss function
        """
        if task_type == "classification":
            return torch.nn.CrossEntropyLoss()
        elif task_type == "regression":
            return torch.nn.MSELoss()
        elif task_type == "generation":
            # For text generation with possible padding tokens
            return torch.nn.CrossEntropyLoss(ignore_index=-100)
        else:
            # Default to standard cross entropy
            return torch.nn.CrossEntropyLoss()
    
    def calculate_metrics(model, data_batches, device, task_type="generation"):
        """Calculate metrics on a dataset (loss, perplexity, accuracy)"""
        model.eval()
        total_loss = 0.0
        total_batches = 0
        total_correct = 0
        total_samples = 0
        
        # Get the appropriate loss function
        criterion = get_loss_function(task_type)
        
        with torch.no_grad():
            for input_batch, target_batch in data_batches:
                try:
                    # Move data to the model's device
                    input_batch = input_batch.to(device)
                    target_batch = target_batch.to(device)
                    
                    # Get vocabulary size (for safety checks)
                    vocab_size = model.embedding.num_embeddings
                    
                    # Safety check: Ensure target indices are within valid range
                    if target_batch.max() >= vocab_size:
                        target_batch = torch.clamp(target_batch, 0, vocab_size - 1)
                    
                    # Forward pass
                    output, _ = model(input_batch)
                    
                    # Handle different target shapes
                    if target_batch.dim() == 1:
                        # For 1D targets (just indices)
                        loss = criterion(output, target_batch)
                        
                        # Calculate accuracy
                        pred = output.argmax(dim=1)
                        total_correct += (pred == target_batch).sum().item()
                        total_samples += target_batch.size(0)
                    else:
                        # For 2D targets
                        loss = criterion(output.view(-1, output.size(-1)), target_batch.view(-1))
                        
                        # Calculate accuracy
                        pred = output.view(-1, output.size(-1)).argmax(dim=1)
                        # Create mask to ignore padding tokens (value 0)
                        mask = target_batch.view(-1) != 0  
                        total_correct += ((pred == target_batch.view(-1)) & mask).sum().item()
                        total_samples += mask.sum().item()
                    
                    total_loss += loss.item()
                    total_batches += 1
                
                except Exception as e:
                    print(f"Error calculating metrics for batch: {str(e)}")
                    continue
        
        # Calculate average metrics
        avg_loss = total_loss / max(1, total_batches)
        perplexity = math.exp(min(avg_loss, 20))  # Cap perplexity to prevent overflow
        accuracy = total_correct / max(1, total_samples)
        
        return {
            "loss": avg_loss,
            "perplexity": perplexity,
            "accuracy": accuracy
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
Consolidated Training Script for Jarvis AI

This script provides a unified interface for training all types of models:
1. Text generation models on various datasets:
   - writing_prompts
   - persona_chat
   - pile
   - openassistant
   - gpteacher
2. DeepSeek-Coder model for code generation

Features:
- Consistent API for all model types
- Improved dataset handling with validation
- Comprehensive evaluation metrics
- Visualization of training progress
- GPU acceleration with auto-detection
"""

import os
import sys
import argparse
import torch
import json
import logging
import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from transformers import TrainerCallback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from tqdm import tqdm
import math
import tempfile

# Force Paperspace environment to be True
# This replaces the is_paperspace_environment() function call with a hardcoded True
def is_paperspace_environment():
    return True

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to the Python path
if not __name__ == "__main__":
    # When imported as a module, use relative imports
    from .text_generator import TextGenerator
    from .unified_dataset_handler import UnifiedDatasetHandler
    from .evaluation_metrics import EvaluationMetrics
    from .utils import get_storage_path, sync_to_gdrive, sync_logs, setup_logging, ensure_directory_exists, sync_from_gdrive

class TrainingVisualizer:
    """Class to handle visualization of training metrics"""
    
    def __init__(self, output_dir="visualizations"):
        """Initialize the visualizer with an output directory"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up plot styling
        plt.style.use('ggplot')
        
    def visualize_training(self, dataset_name: str, metrics: Dict[str, List[float]]):
        """Create visualizations for training metrics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Loss curve
        if 'loss' in metrics and len(metrics['loss']) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(metrics['loss'], 'b-', label='Training Loss')
            
            if 'val_loss' in metrics and len(metrics['val_loss']) > 0:
                plt.plot(metrics['val_loss'], 'r-', label='Validation Loss')
                
            plt.title(f'Training and Validation Loss - {dataset_name}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            # Save the figure
            loss_path = os.path.join(self.output_dir, f"{dataset_name}_loss_{timestamp}.png")
            plt.savefig(loss_path, dpi=300)
            plt.close()
            print(f"Loss visualization saved to {loss_path}")
            
        # Accuracy if available
        if 'accuracy' in metrics and len(metrics['accuracy']) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(metrics['accuracy'], 'g-', label='Training Accuracy')
            
            if 'val_accuracy' in metrics and len(metrics['val_accuracy']) > 0:
                plt.plot(metrics['val_accuracy'], 'm-', label='Validation Accuracy')
                
            plt.title(f'Training and Validation Accuracy - {dataset_name}')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            
            # Save the figure
            acc_path = os.path.join(self.output_dir, f"{dataset_name}_accuracy_{timestamp}.png")
            plt.savefig(acc_path, dpi=300)
            plt.close()
            print(f"Accuracy visualization saved to {acc_path}")
            
        # Perplexity if available
        if 'perplexity' in metrics and len(metrics['perplexity']) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(metrics['perplexity'], 'c-', label='Training Perplexity')
            
            if 'val_perplexity' in metrics and len(metrics['val_perplexity']) > 0:
                plt.plot(metrics['val_perplexity'], 'y-', label='Validation Perplexity')
                
            plt.title(f'Training and Validation Perplexity - {dataset_name}')
            plt.xlabel('Epoch')
            plt.ylabel('Perplexity')
            plt.legend()
            plt.grid(True)
            
            # Save the figure
            perp_path = os.path.join(self.output_dir, f"{dataset_name}_perplexity_{timestamp}.png")
            plt.savefig(perp_path, dpi=300)
            plt.close()
            print(f"Perplexity visualization saved to {perp_path}")
            
    def create_comparison_plot(self, metrics_by_dataset: Dict[str, Dict[str, List[float]]], metric_name: str):
        """Create a comparison plot for a specific metric across datasets"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        plt.figure(figsize=(12, 8))
        
        for dataset_name, metrics in metrics_by_dataset.items():
            if metric_name in metrics and len(metrics[metric_name]) > 0:
                plt.plot(metrics[metric_name], label=dataset_name)
                
        plt.title(f'Comparison of {metric_name.capitalize()} Across Datasets')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name.capitalize())
        plt.legend()
        plt.grid(True)
        
        # Save the figure
        comparison_path = os.path.join(self.output_dir, f"comparison_{metric_name}_{timestamp}.png")
        plt.savefig(comparison_path, dpi=300)
        plt.close()
        print(f"Comparison visualization saved to {comparison_path}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train models for Jarvis AI Assistant")
    
    # Basic options
    parser.add_argument('--model_type', type=str, choices=['text', 'code'], 
                       help='Type of model to train')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset name, path, or comma-separated list of HuggingFace datasets')
    parser.add_argument('--dataset_path', type=str, default=None,
                       help='Path to the dataset')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for model and results')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Name of model to use for training')
    parser.add_argument('--model_size', type=str, choices=['small', 'medium', 'large', 'xl', 
                                                     'deepseek-small', 'deepseek-medium'],
                       default='medium', help='Size of model to use')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate for training')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Number of gradient accumulation steps')
    
    # Optimization options
    parser.add_argument('--no_fp16', action='store_true',
                       help='Disable FP16 training')
    parser.add_argument('--no_peft', action='store_true',
                       help='Disable parameter-efficient fine-tuning (PEFT)')
    parser.add_argument('--peft_type', type=str, choices=['lora', 'prefix', 'prompt'],
                       default='lora', help='Type of PEFT to use')
    parser.add_argument('--use_int8', action='store_true',
                       help='Use int8 quantization')
    
    # LoRA parameters
    parser.add_argument('--lora_r', type=int, default=16,
                       help='Rank for LoRA adapters')
    parser.add_argument('--lora_alpha', type=int, default=32,
                       help='Alpha for LoRA adapters')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                       help='Dropout for LoRA adapters')
    
    # Trainer parameters
    parser.add_argument('--evaluation_strategy', type=str, choices=['steps', 'epoch', 'no'],
                       default='epoch', help='Evaluation strategy')
    parser.add_argument('--save_strategy', type=str, choices=['steps', 'epoch', 'no'],
                       default='epoch', help='Save strategy')
    parser.add_argument('--logging_steps', type=int, default=10,
                       help='Logging steps')
    parser.add_argument('--run_name', type=str, default=None,
                       help='Run name for logging')
    
    # Execution options
    parser.add_argument('--cpu_only', action='store_true',
                       help='Force CPU usage even if GPU is available')
    parser.add_argument('--install_dependencies', action='store_true',
                       help='Install dependencies before running')
    
    # Evaluation parameters
    parser.add_argument('--no_enhanced_eval', action='store_true',
                       help='Disable enhanced evaluation with BERTScore, ROUGE, etc.')
    parser.add_argument('--eval_metrics_dir', type=str, default='evaluation_metrics',
                       help='Directory to store evaluation metrics')
    parser.add_argument('--visualize_metrics', action='store_true',
                       help='Create visualizations of training metrics')
    parser.add_argument('--eval_samples', type=int, default=10,
                       help='Number of samples to use for evaluation')
    parser.add_argument('--eval_bert_model', type=str, 
                       default='bert-base-uncased',
                       help='BERT model to use for evaluation metrics')
    
    # Feedback options
    parser.add_argument('--collect_human_feedback', action='store_true',
                       help='Collect human feedback on model outputs')
    parser.add_argument('--report_name', type=str, default=None,
                       help='Name for the evaluation report')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Process dataset if it contains comma-separated values
    if args.dataset and ',' in args.dataset:
        # Store the original comma-separated string for use with HuggingFace datasets
        args.dataset_list = [ds.strip() for ds in args.dataset.split(',')]
        logger.info(f"Using multiple datasets: {args.dataset_list}")
    
    return args


def install_dependencies():
    """Install required dependencies for training"""
    print("Checking and installing required dependencies...")
    
    # Try to import tensorboard, install if not available
    try:
        import importlib.util
        if importlib.util.find_spec("tensorboard") is not None:
            print("TensorBoard is already installed.")
        else:
            raise ImportError("Tensorboard not found")
    except ImportError:
        print("Installing TensorBoard...")
        subprocess.run([sys.executable, "-m", "pip", "install", "tensorboard"])
        print("TensorBoard installed successfully.")
    
    # Try to import evaluation metric dependencies
    try:
        # Use importlib to check if bert_score is available without importing it directly
        if importlib.util.find_spec("bert_score") is not None:
            print("BERTScore is already installed.")
        else:
            raise ImportError("bert_score not found")
    except ImportError:
        print("Installing evaluation metric dependencies...")
        # Install dependencies directly instead of using EvaluationMetrics.install_dependencies()
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "bert-score", "rouge-score", "nltk", "transformers"
            ])
            # Initialize NLTK
            try:
                import nltk
                nltk.download('punkt', quiet=True)
                print("NLTK punkt downloaded successfully.")
            except Exception as e:
                print(f"Warning: Failed to download NLTK data: {e}")
            
            print("Successfully installed all dependencies for evaluation metrics")
        except Exception as e:
            print(f"Warning: Error installing dependencies: {e}")
            print("You may need to install these packages manually: bert-score rouge-score nltk transformers")


def train_text_model(
    dataset: Union[str, List[str]],
    model_name_or_path: str = "distilgpt2",
    batch_size: int = 4,
    epochs: int = 3,
    learning_rate: float = 3e-5,
    weight_decay: float = 0.01,
    max_length: int = 512,
    output_dir: str = "models/text_gen",
    eval_metrics_dir: str = "metrics",
    dataset_subset: Optional[str] = None,
    max_samples: Optional[int] = None,
    evaluation_strategy: str = "epoch",
    save_strategy: str = "epoch",
    logging_steps: int = 100,
    eval_steps: Optional[int] = None,
    visualize_metrics: bool = False,
    use_deepspeed: bool = False,
    use_8bit: bool = False,
    use_4bit: bool = False,
    use_qlora: bool = False,
    gradient_accumulation_steps: int = 1,
    fp16: bool = False,
    bf16: bool = False,
    temperature: float = 1.0,
    resume_from_checkpoint: Union[bool, str] = False,
    use_mps: bool = False,
    use_flash_attn: bool = False,
    use_unsloth: bool = True,
    cache_dir: Optional[str] = None,
    trainer_cls=None,
) -> tuple:
    """
    Train a text generation model
    
    Args:
        dataset: Dataset name or list of dataset names
        model_name_or_path: Model name or path
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        max_length: Maximum sequence length
        output_dir: Directory to save the model
        eval_metrics_dir: Directory to save evaluation metrics
        dataset_subset: Optional subset for specific datasets (e.g., pile)
        max_samples: Maximum number of samples to load from each dataset
        evaluation_strategy: Evaluation strategy (epoch, steps)
        save_strategy: Save strategy (epoch, steps)
        logging_steps: Logging steps
        eval_steps: Evaluation steps (if evaluation_strategy is steps)
        visualize_metrics: Whether to visualize metrics
        use_deepspeed: Whether to use DeepSpeed
        use_8bit: Whether to use 8-bit quantization
        use_4bit: Whether to use 4-bit quantization
        use_qlora: Whether to use QLoRA
        gradient_accumulation_steps: Gradient accumulation steps
        fp16: Whether to use mixed precision (fp16)
        bf16: Whether to use mixed precision (bf16)
        temperature: Temperature for generation
        resume_from_checkpoint: Resume from checkpoint
        use_mps: Whether to use MPS
        use_flash_attn: Whether to use Flash Attention
        use_unsloth: Whether to use Unsloth optimizations
        cache_dir: Directory to cache models and datasets
        trainer_cls: Optional custom trainer class
        
    Returns:
        Trained model, tokenizer, training arguments
    """
    # Import statements inside the function to avoid issues with circular imports
    import os
    import torch
    import numpy as np
    from pathlib import Path
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
        DataCollatorForLanguageModeling, DataCollatorWithPadding,
        DataCollatorForSeq2Seq
    )
    from datasets import Dataset, DatasetDict
    
    # Configure GPU settings
    use_gpu = torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"
    
    if use_gpu:
        print(f"Using CUDA GPU for text generation: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU for text generation - this will be slow")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(eval_metrics_dir, exist_ok=True)
    
    # Create a unified dataset handler from the refactored architecture
    from .unified_dataset_handler import UnifiedDatasetHandler
    from transformers import AutoTokenizer
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    # Add special tokens if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"Vocabulary size: {len(tokenizer)}")
    special_tokens = [tokenizer.pad_token, tokenizer.unk_token, 
                      tokenizer.bos_token, tokenizer.eos_token]
    print(f"Special tokens: {special_tokens}")
    
    # Initialize dataset handler
    dataset_handler = UnifiedDatasetHandler(
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=batch_size
    )
    
    # Check if we have multiple datasets
    if isinstance(dataset, str) and ',' in dataset:
        dataset_names = dataset.split(',')
        print(f"Using multiple datasets: {dataset_names}")
        combined_dataset = None
        
        for dataset_name in dataset_names:
            dataset_name = dataset_name.strip()
            print(f"\n{'='*50}\nTraining text generation model on {dataset_name} dataset\n{'='*50}")
            
            # Load single dataset
            print(f"Processing HuggingFace dataset: {dataset_name}")
            current_dataset = dataset_handler.load_dataset(
                dataset_name=dataset_name,
                split="train",
                max_samples=max_samples,  # Apply sample limit to each dataset
                subset=dataset_subset,
                cache_dir=cache_dir,
                use_cache=True
            )
            
            # Combine datasets
            if combined_dataset is None:
                combined_dataset = current_dataset
            else:
                # Combine the datasets
                combined_texts = combined_dataset["train"]["text"] + current_dataset["train"]["text"]
                
                # Re-tokenize if needed (to ensure consistent formatting)
                if tokenizer:
                    # Process in manageable batches to avoid OOM
                    input_ids = []
                    attention_masks = []
                    batch_size = 1000
                    
                    for i in range(0, len(combined_texts), batch_size):
                        batch_texts = combined_texts[i:min(i+batch_size, len(combined_texts))]
                        encodings = tokenizer(
                            batch_texts,
                            truncation=True,
                            padding='max_length',
                            max_length=max_length,
                            return_tensors='pt'
                        )
                        input_ids.extend(encodings['input_ids'].tolist())
                        attention_masks.extend(encodings['attention_mask'].tolist())
                    
                    # Update combined dataset
                    from datasets import Dataset
                    combined_dataset = {
                        "train": Dataset.from_dict({
                            "text": combined_texts,
                            "input_ids": input_ids,
                            "attention_mask": attention_masks
                        })
                    }
        
        # Use the combined dataset for training
        train_dataset = combined_dataset["train"]
        
    else:
        # Single dataset case
        print(f"\n{'='*50}\nTraining Text Model on {dataset.upper()} dataset\n{'='*50}")
        
        dataset_dict = dataset_handler.load_dataset(
            dataset_name=dataset,
            split="train",
            max_samples=max_samples,
            subset=dataset_subset,
            cache_dir=cache_dir,
            use_cache=True
        )
        
        train_dataset = dataset_dict["train"]
    
    # Print effective batch size information
    print(f"Using batch size {batch_size} with gradient accumulation steps {gradient_accumulation_steps}")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=0.1,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        save_total_limit=3,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps if eval_steps else None,
        load_best_model_at_end=True,
        report_to="tensorboard",
        fp16=fp16,
        bf16=bf16,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )
    
    # Load the model
    try:
        # Try to use Unsloth for optimization if requested
        if use_unsloth:
            try:
                from unsloth import FastLanguageModel
                print("Using Unsloth optimizations for text model")
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name_or_path,
                    max_seq_length=max_length,
                    dtype=torch.bfloat16 if bf16 else torch.float16 if fp16 else None,
                    load_in_4bit=use_4bit,
                    load_in_8bit=use_8bit,
                    device_map="auto",
                )
                
                # Prepare the model for QLoRA training if requested
                if use_qlora:
                    print("Using QLoRA for parameter-efficient fine-tuning")
                    from peft import LoraConfig
                    
                    lora_config = LoraConfig(
                        r=16,
                        lora_alpha=32,
                        lora_dropout=0.05,
                        bias="none",
                        task_type="CAUSAL_LM",
                    )
                    
                    model = FastLanguageModel.get_peft_model(
                        model,
                        lora_config,
                        train_mode=True
                    )
            except ImportError:
                print("Unsloth not available, falling back to standard Transformers")
                use_unsloth = False
        
        # Standard Transformers loading if not using Unsloth or if Unsloth failed
        if not use_unsloth:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                load_in_8bit=use_8bit,
                load_in_4bit=use_4bit,
                torch_dtype=torch.bfloat16 if bf16 else torch.float16 if fp16 else None,
                device_map="auto" if use_gpu else None,
            )
            
            # Apply QLoRA if requested
            if use_qlora:
                print("Using QLoRA for parameter-efficient fine-tuning")
                from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
                
                model = prepare_model_for_kbit_training(model)
                lora_config = LoraConfig(
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                model = get_peft_model(model, lora_config)
    
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    # Setup data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Setup trainer
    if trainer_cls:
        trainer = trainer_cls(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,  # Using same dataset for eval for simplicity
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,  # Using same dataset for eval for simplicity
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
    
    # Train the model
    if resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()
    
    # Save model, tokenizer, and config
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Visualize metrics if requested
    if visualize_metrics:
        # Import visualization functions
        from .evaluation_metrics import visualize_training_metrics
        
        # Get training logs
        logs = trainer.state.log_history
        
        # Visualize metrics
        visualize_training_metrics(
            logs=logs,
            output_dir=eval_metrics_dir,
            metric_names=["loss", "eval_loss"],
            title=f"Training Metrics for {model_name_or_path} on {dataset}",
        )
    
    return model, tokenizer, training_args


def train_code_model(args, force_gpu: bool = True):
    """Train DeepSeek-Coder model for code generation"""
    print("\n===== Training DeepSeek-Coder Model =====")

    # On Apple Silicon (M1/M2/M3)
    on_apple_silicon = hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

    # Common arguments
    finetune_script = os.path.join(os.path.dirname(__file__), "finetune_deepseek.py")

    # Set default batch size if not specified
    deepseek_batch_size = args.deepseek_batch_size if hasattr(args, 'deepseek_batch_size') else args.batch_size

    cmd_args = [
        sys.executable,  # Use the current Python interpreter
        finetune_script,
        "--epochs", str(args.epochs),
        "--batch-size", str(deepseek_batch_size),
        "--max-samples", str(args.max_samples),
        "--sequence-length", str(args.sequence_length),
        "--learning-rate", str(args.learning_rate),
        "--warmup-steps", str(args.warmup_steps),
        "--force-gpu",    # Boolean flag (no value)
    ]

    # Add code subset if specified - use --subset instead of --code-subset
    if hasattr(args, 'code_subset') and args.code_subset:
        cmd_args.extend(["--subset", args.code_subset])

    # Use all code subsets if requested
    if hasattr(args, 'all_code_subsets') and args.all_code_subsets:
        cmd_args.append("--all-subsets")

    # Use mini dataset for faster testing if requested
    if hasattr(args, 'use_mini_dataset') and args.use_mini_dataset:
        cmd_args.append("--use-mini-dataset")
        print("Using mini dataset for quick testing")

    # Add quantization for NVIDIA GPUs only
    if hasattr(args, 'load_in_4bit') and args.load_in_4bit and not on_apple_silicon and torch.cuda.is_available():
        cmd_args.append("--load-in-4bit")
    
    # Add output directory
    if hasattr(args, 'model_dir') and args.model_dir:
        cmd_args.extend(["--output-dir", os.path.join(args.model_dir, "deepseek_finetuned")])

    # Run the training process
    print("Running DeepSeek-Coder training with command:")
    print(" ".join(cmd_args))
    result = subprocess.run(cmd_args)

    if result.returncode != 0:
        print("ERROR: DeepSeek-Coder training failed!")
        return False

    return True


def save_training_metrics(metrics, run_name, epoch=None):
    """
    Save training metrics to a file.
    
    Args:
        metrics (dict): Dictionary containing metrics
        run_name (str): Name of the training run
        epoch (int, optional): Current epoch number
    """
    epoch_str = f"_epoch_{epoch}" if epoch is not None else ""
    
    # Use the storage path utility to get the correct path
    metrics_dir = get_storage_path("metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    metrics_file = os.path.join(metrics_dir, f"{run_name}{epoch_str}_metrics.json")
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Sync metrics to Google Drive
    sync_to_gdrive("metrics")
    
    logger.info(f"Saved metrics to {metrics_file}")


def train_model(
    model_name,
    dataset_path,
    output_dir=None,
    epochs=3,
    batch_size=4,
    learning_rate=2e-5,
    max_length=512,
    use_peft=True,
    peft_type='lora',
    use_int8=False,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    gradient_accumulation_steps=1,
    warmup_steps=100,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_steps=10,
    seed=42,
    push_to_hub=False,
    hub_model_id=None,
    hub_private_repo=True,
    report_to='tensorboard',
    fp16=True,
    run_name=None,
    save_total_limit=3,
    use_enhanced_eval=True,  # New parameter to control enhanced evaluation
    eval_metrics_dir=None    # Directory for enhanced evaluation metrics
):
    """
    Unified function to train models with enhanced evaluation metrics.
    
    Args:
        model_name: Name of pretrained model to fine-tune
        dataset_path: Path to dataset files
        output_dir: Directory to save model and results
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        max_length: Maximum sequence length
        use_peft: Whether to use PEFT for efficient fine-tuning
        peft_type: Type of PEFT method ('lora', 'prefix', etc.)
        use_int8: Whether to use int8 quantization
        lora_r: LoRA attention dimension
        lora_alpha: LoRA alpha parameter
        lora_dropout: Dropout probability for LoRA layers
        gradient_accumulation_steps: Number of steps to accumulate gradients
        warmup_steps: Number of warmup steps for scheduler
        evaluation_strategy: When to evaluate ('steps', 'epoch', or 'no')
        save_strategy: When to save checkpoints ('steps', 'epoch', or 'no')
        logging_steps: Number of steps between logging
        seed: Random seed
        push_to_hub: Whether to push to Hugging Face Hub
        hub_model_id: Model ID for Hub (if pushing)
        hub_private_repo: Whether Hub repo should be private
        report_to: Where to report results ('tensorboard', 'wandb', etc.)
        fp16: Whether to use mixed precision training
        run_name: Name for this training run
        save_total_limit: Maximum number of checkpoints to keep
        use_enhanced_eval: Whether to use enhanced evaluation metrics
        eval_metrics_dir: Directory for enhanced evaluation metrics
    
    Returns:
        (Trainer, additional_results): Tuple with the trainer object and additional results dict
    """
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        default_data_collator,
    )
    
    try:
        from datasets import load_from_disk, Dataset
    except ImportError:
        print("Datasets library not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
        from datasets import load_from_disk, Dataset
    
    # Initialize enhanced evaluation metrics if requested
    enhanced_eval_metrics = None
    if use_enhanced_eval:
        try:
            from src.generative_ai_module.evaluation_metrics import EvaluationMetrics
            metrics_dir = eval_metrics_dir or os.path.join(output_dir, "evaluation_metrics")
            enhanced_eval_metrics = EvaluationMetrics(metrics_dir=metrics_dir)
            print(f"Enhanced evaluation metrics initialized at {metrics_dir}")
        except Exception as e:
            print(f"Error initializing enhanced evaluation metrics: {e}")
            use_enhanced_eval = False
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with appropriate quantization
    if use_int8:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Set up PEFT if requested
    if use_peft:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
            
            if peft_type.lower() == 'lora':
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias="none",
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
                )
                
                model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()
            else:
                print(f"PEFT type {peft_type} not implemented. Using base model.")
        except ImportError:
            print("PEFT library not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "peft"])
            from peft import LoraConfig, TaskType, get_peft_model
            
            # Retry with installed library
            if peft_type.lower() == 'lora':
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias="none"
                )
                
                model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()
            else:
                print(f"PEFT type {peft_type} not implemented. Using base model.")
    
    # Load dataset
    try:
        # Try to load as a Hugging Face dataset
        dataset = load_from_disk(dataset_path)
    except Exception as e:
        print(f"Error loading dataset from disk: {e}")
        print("Trying to load as a regular file...")
        
        # Fallback to loading as a regular file
        try:
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            
            # Convert to dataset
            dataset = Dataset.from_dict(data)
        except Exception as e2:
            print(f"Failed to load dataset: {e2}")
            return None, {"error": f"Failed to load dataset: {e2}"}
    
    # Split dataset if not already split
    if "train" not in dataset:
        dataset = dataset.train_test_split(test_size=0.1, seed=seed)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        evaluation_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        fp16=fp16,
        warmup_steps=warmup_steps,
        save_total_limit=save_total_limit,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        hub_private_repo=hub_private_repo,
        report_to=report_to,
        run_name=run_name,
        seed=seed,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # Create callback with enhanced evaluation
    callbacks = []
    if use_enhanced_eval and enhanced_eval_metrics:
        # Extract dataset name from path
        dataset_name = os.path.basename(dataset_path).split('.')[0]
        
        callback = CustomCallback(
            sync_interval=1,
            evaluation_metrics=enhanced_eval_metrics,
            eval_dataset=dataset["test"] if "test" in dataset else None,
            model_name=model_name,
            dataset_name=dataset_name
        )
        callbacks.append(callback)
    else:
        # Use basic callback for syncing
        callback = CustomCallback(sync_interval=1)
        callbacks.append(callback)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"] if "test" in dataset else None,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        callbacks=callbacks
    )
    
    # Train the model
    train_result = trainer.train()
    
    # Save the fine-tuned model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Get evaluation results if available
    eval_results = {}
    if use_enhanced_eval and enhanced_eval_metrics and hasattr(callback, "eval_results"):
        # Create a report from the evaluation results
        try:
            eval_results = enhanced_eval_metrics.create_evaluation_report(
                evaluation_results=callback.eval_results,
                report_name=f"{os.path.basename(model_name)}_{dataset_name}_final_report",
                include_samples=True
            )
            
            # Generate visualizations
            enhanced_eval_metrics.visualize_metrics(
                eval_results, 
                output_dir=os.path.join(output_dir, "evaluation_visualizations"),
                plot_type="comprehensive"
            )
            
            print(f"Created enhanced evaluation report with {len(callback.eval_results)} samples")
        except Exception as e:
            print(f"Error creating enhanced evaluation report: {e}")
    
    # Return the trainer and any additional results
    return trainer, {
        "train_result": train_result,
        "enhanced_eval_results": eval_results
    }


class CustomCallback(TrainerCallback):
    """
    Custom callback for training to sync checkpoints and log metrics.
    Also performs enhanced evaluation with the new metrics module.
    """
    
    def __init__(self, sync_interval=1, evaluation_metrics=None, 
                eval_dataset=None, model_name=None, dataset_name=None):
        """
        Initialize the callback.
        
        Args:
            sync_interval: How often to sync to Google Drive (in epochs)
            evaluation_metrics: EvaluationMetrics instance for enhanced evaluation
            eval_dataset: Dataset to use for generating evaluation examples
            model_name: Name of the model being trained
            dataset_name: Name of the dataset being used
        """
        self.sync_interval = sync_interval
        self.last_sync_epoch = 0
        self.evaluation_metrics = evaluation_metrics
        self.eval_dataset = eval_dataset
        self.model_name = model_name or "model"
        self.dataset_name = dataset_name or "dataset"
        self.eval_results = []
    
    def on_save(self, args, state, control, **kwargs):
        """Sync checkpoints to Google Drive when a model is saved"""
        # Skip if not configured for Paperspace or sync not needed
        if not is_paperspace_environment() or not args.output_dir:
            return
        
        # Check if it's time to sync
        current_epoch = state.epoch // self.sync_interval
        if current_epoch > self.last_sync_epoch:
            self.last_sync_epoch = current_epoch
            
            # Sync the output directory to Google Drive
            try:
                from src.generative_ai_module.utils import sync_to_gdrive
                sync_to_gdrive(args.output_dir)
                print(f"Synced checkpoints to Google Drive after epoch {state.epoch:.2f}")
            except Exception as e:
                print(f"Failed to sync checkpoints: {e}")
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Run enhanced evaluation when the model is evaluated"""
        if not self.evaluation_metrics or not self.eval_dataset:
            return
        
        # Get the model from kwargs
        model = kwargs.get("model")
        if not model:
            return
        
        try:
            # Generate text samples for evaluation
            examples = self._generate_evaluation_examples(model, max_samples=5)
            if not examples:
                return
            
            # Evaluate each example with our enhanced metrics
            step_results = []
            for example in examples:
                result = self.evaluation_metrics.evaluate_generation(
                    prompt=example["prompt"],
                    generated_text=example["generated_text"],
                    reference_text=example["reference_text"],
                    dataset_name=self.dataset_name,
                    task_type="generation",
                    save_results=False  # Don't save individual results
                )
                step_results.append(result)
            
            # Store the results for later reporting
            self.eval_results.extend(step_results)
            
            # Log summary metrics
            if metrics is not None and step_results:
                # Extract BERTScore and add to metrics
                bert_scores = []
                for result in step_results:
                    if "metrics" in result and "bert_score" in result["metrics"]:
                        bert_score = result["metrics"]["bert_score"]
                        if "aggregate" in bert_score and "f1_mean" in bert_score["aggregate"]:
                            bert_scores.append(bert_score["aggregate"]["f1_mean"])
                
                if bert_scores:
                    metrics["bert_score_f1"] = sum(bert_scores) / len(bert_scores)
                
                # Extract ROUGE scores
                rouge_scores = []
                for result in step_results:
                    if "metrics" in result and "rouge" in result["metrics"]:
                        rouge = result["metrics"]["rouge"]
                        if "aggregate" in rouge and "rougeL_fmeasure_mean" in rouge["aggregate"]:
                            rouge_scores.append(rouge["aggregate"]["rougeL_fmeasure_mean"])
                
                if rouge_scores:
                    metrics["rouge_l"] = sum(rouge_scores) / len(rouge_scores)
            
            print(f"Completed enhanced evaluation at step {state.global_step}")
        except Exception as e:
            print(f"Error during enhanced evaluation: {e}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Generate final evaluation report at the end of training"""
        if not self.evaluation_metrics or not self.eval_results:
            return
        
        try:
            # Create comprehensive evaluation report
            report = self.evaluation_metrics.create_evaluation_report(
                evaluation_results=self.eval_results,
                report_name=f"{self.model_name}_{self.dataset_name}_training_report",
                include_samples=True
            )
            
            # Generate visualizations
            vis_files = self.evaluation_metrics.visualize_metrics(
                report, 
                output_dir=os.path.join(args.output_dir, "evaluation_visualizations"),
                plot_type="comprehensive"
            )
            
            print(f"Created final evaluation report and visualizations")
            
            # Sync to Google Drive if in Paperspace
            if is_paperspace_environment():
                try:
                    from src.generative_ai_module.utils import sync_to_gdrive
                    sync_to_gdrive(args.output_dir)
                    print("Synced final evaluation report to Google Drive")
                except Exception as e:
                    print(f"Failed to sync final report: {e}")
        except Exception as e:
            print(f"Error creating final evaluation report: {e}")
    
    def _generate_evaluation_examples(self, model, max_samples=5):
        """Generate text examples for evaluation"""
        if not self.eval_dataset:
            return []
        
        try:
            from src.generative_ai_module.text_generator import TextGenerator
            
            # Initialize text generator with the current model
            generator = TextGenerator(model=model)
            
            # Get samples from evaluation dataset
            examples = []
            
            # Extract prompt-response pairs based on dataset format
            if hasattr(self.eval_dataset, "get_prompt_response_pairs"):
                # Use dataset helper method if available
                pairs = self.eval_dataset.get_prompt_response_pairs(max_pairs=max_samples)
            else:
                # Try to extract directly from dataset features
                pairs = []
                features = self.eval_dataset.features if hasattr(self.eval_dataset, "features") else {}
                
                # Check for common feature names
                prompt_keys = ["prompt", "input", "question", "instruction"]
                response_keys = ["response", "output", "answer", "completion"]
                
                # Find matching keys
                prompt_key = next((k for k in prompt_keys if k in features), None)
                response_key = next((k for k in response_keys if k in features), None)
                
                # Extract pairs if keys found
                if prompt_key and response_key:
                    for i in range(min(max_samples, len(self.eval_dataset))):
                        item = self.eval_dataset[i]
                        if prompt_key in item and response_key in item:
                            pairs.append({
                                "prompt": item[prompt_key],
                                "response": item[response_key]
                            })
                else:
                    # Last resort - try to infer from structure
                    for i in range(min(max_samples, len(self.eval_dataset))):
                        item = self.eval_dataset[i]
                        if len(item) >= 2:  # Assume first two items are prompt and response
                            pairs.append({
                                "prompt": str(item[0]),
                                "response": str(item[1])
                            })
            
            # Generate text for each prompt and prepare evaluation examples
            for pair in pairs:
                prompt = pair.get("prompt", "")
                reference = pair.get("response", "")
                
                if not prompt:
                    continue
                
                try:
                    # Generate text
                    generated = generator.generate_text(prompt, max_length=len(reference) + 50)
                    
                    # Add to examples
                    examples.append({
                        "prompt": prompt,
                        "generated_text": generated,
                        "reference_text": reference
                    })
                except Exception as e:
                    print(f"Error generating text for evaluation: {e}")
                    continue
            
            return examples
        except Exception as e:
            print(f"Error preparing evaluation examples: {e}")
            return []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Handle logging of metrics"""
        if not logs:
            return
        
        # Log the metrics
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                print(f"Step {state.global_step}: {k} = {v}")
        
        # Custom metric logging can be added here
        # For example, writing to TensorBoard, etc.


def main():
    """Main function to parse arguments and run the appropriate training"""
    import argparse
    import logging
    import glob
    import os
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a model on a specific dataset.')
    
    # General arguments
    parser.add_argument('--model_type', type=str, required=True, choices=['text', 'code', 'image', 'speech', 'multimodal'],
                      help='Type of model to train')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, required=True,
                      help='Dataset name(s) to use for training. Can be a single dataset or comma-separated list.')
    parser.add_argument('--max_samples', type=int, default=None,
                      help='Maximum number of samples to use from each dataset')
    parser.add_argument('--dataset_subset', type=str, default=None,
                      help='Specific subset of the dataset to use (for datasets with subsets)')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Training batch size')
    parser.add_argument('--epochs', type=int, default=3,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-5,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                      help='Weight decay')
    parser.add_argument('--max_length', type=int, default=512,
                      help='Maximum sequence length')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                      help='Number of gradient accumulation steps')
    
    # Model arguments
    parser.add_argument('--model_name_or_path', type=str, default="distilgpt2",
                      help='Base model name or path to use')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default="./models",
                      help='Directory to save the trained model')
    parser.add_argument('--eval_metrics_dir', type=str, default="./metrics",
                      help='Directory to save evaluation metrics')
    parser.add_argument('--save_strategy', type=str, default="epoch", choices=['epoch', 'steps'],
                      help='When to save model checkpoints')
    parser.add_argument('--evaluation_strategy', type=str, default="epoch", choices=['epoch', 'steps'],
                      help='When to run evaluation')
    parser.add_argument('--logging_steps', type=int, default=100,
                      help='Steps between logging')
    parser.add_argument('--eval_steps', type=int, default=None,
                      help='Steps between evaluation if evaluation_strategy is steps')
    
    # Optimization arguments
    parser.add_argument('--use_deepspeed', action='store_true',
                      help='Whether to use DeepSpeed')
    parser.add_argument('--use_8bit', action='store_true',
                      help='Whether to use 8-bit quantization')
    parser.add_argument('--use_4bit', action='store_true',
                      help='Whether to use 4-bit quantization')
    parser.add_argument('--use_qlora', action='store_true',
                      help='Whether to use QLoRA')
    parser.add_argument('--fp16', action='store_true',
                      help='Whether to use mixed precision (fp16)')
    parser.add_argument('--bf16', action='store_true',
                      help='Whether to use mixed precision (bf16)')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                      help='Path to checkpoint to resume from')
    parser.add_argument('--use_flash_attn', action='store_true',
                      help='Whether to use Flash Attention')
    parser.add_argument('--use_unsloth', action='store_true', default=True,
                      help='Whether to use Unsloth optimizations')
    
    # Miscellaneous
    parser.add_argument('--visualize_metrics', action='store_true',
                      help='Visualize training metrics')
    parser.add_argument('--cache_dir', type=str, default=None,
                      help='Directory to cache models and datasets')
    
    args = parser.parse_args()
    
    # Print all arguments for logging
    logger.info("Training with the following arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Handle model training based on model type
    if args.model_type == 'text':
        model, tokenizer, training_args = train_text_model(
            dataset=args.dataset,
            model_name_or_path=args.model_name_or_path,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_length=args.max_length,
            output_dir=args.output_dir,
            eval_metrics_dir=args.eval_metrics_dir,
            dataset_subset=args.dataset_subset,
            max_samples=args.max_samples,
            evaluation_strategy=args.evaluation_strategy,
            save_strategy=args.save_strategy,
            logging_steps=args.logging_steps,
            eval_steps=args.eval_steps,
            visualize_metrics=args.visualize_metrics,
            use_deepspeed=args.use_deepspeed,
            use_8bit=args.use_8bit,
            use_4bit=args.use_4bit,
            use_qlora=args.use_qlora,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            fp16=args.fp16,
            bf16=args.bf16,
            resume_from_checkpoint=args.resume_from_checkpoint,
            use_flash_attn=args.use_flash_attn,
            use_unsloth=args.use_unsloth,
            cache_dir=args.cache_dir
        )
        
        # Print success message
        print(f"\nSuccessfully trained text model!")
        print(f"Model saved to: {args.output_dir}")

    elif args.model_type == 'code':
        from .code_generator import train_code_model
        
        train_code_model(
            dataset=args.dataset,
            model_name_or_path=args.model_name_or_path,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_length=args.max_length,
            output_dir=args.output_dir,
            eval_metrics_dir=args.eval_metrics_dir,
            max_samples=args.max_samples
        )
        
    else:
        logger.error(f"Model type '{args.model_type}' not implemented yet.")
        return


if __name__ == "__main__":
    # Add the parent directory to the path to make the module importable
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
    # Import these directly rather than through relative imports
    from src.generative_ai_module.text_generator import TextGenerator
    from src.generative_ai_module.unified_dataset_handler import UnifiedDatasetHandler
    from src.generative_ai_module.evaluation_metrics import EvaluationMetrics
    from src.generative_ai_module.utils import get_storage_path, sync_to_gdrive, sync_logs, setup_logging, ensure_directory_exists, sync_from_gdrive
    # Import install_dependencies from evaluation_metrics if needed
    try:
        from src.generative_ai_module.evaluation_metrics import install_dependencies as eval_install_dependencies
    except ImportError:
        # It's okay if this fails, we have our own implementation
        pass
    main() 