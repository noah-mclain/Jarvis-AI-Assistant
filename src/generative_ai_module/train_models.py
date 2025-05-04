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


def train_text_model(dataset_name: str, args, force_gpu: bool = True):
    """Train a text generation model on a specific dataset with visualization"""
    print(f"\n{'='*50}")
    print(f"Training Text Model on {dataset_name.upper()} dataset")
    print(f"{'='*50}")

    # Check if model_dir exists
    if not hasattr(args, 'model_dir') or not args.model_dir:
        args.model_dir = os.path.join(args.output_dir, 'models')
        os.makedirs(args.model_dir, exist_ok=True)
        
    # Check for early stopping
    if not hasattr(args, 'early_stopping'):
        args.early_stopping = 0
        
    # Check for validation split
    if not hasattr(args, 'validation_split'):
        args.validation_split = 0.1
        
    # Check for test split
    if not hasattr(args, 'test_split'):
        args.test_split = 0.1
        
    # Check for max_samples
    if not hasattr(args, 'max_samples'):
        args.max_samples = None

    # Determine gradient accumulation steps based on available memory
    # This allows simulation of larger batch sizes on limited GPU memory
    gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 4)
    effective_batch_size = args.batch_size * gradient_accumulation_steps
    print(f"Using batch size {args.batch_size} with gradient accumulation steps {gradient_accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")

    # Clear CUDA cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Create text generator and dataset handler
    generator = TextGenerator(force_gpu=force_gpu)
    dataset_handler = UnifiedDatasetHandler()
    visualizer = TrainingVisualizer(args.eval_metrics_dir)

    # Get device
    device = generator.device
    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading data from {dataset_name}...")
    data = dataset_handler.load_dataset(
        dataset_name=dataset_name,
        max_samples=args.max_samples
    )

    # Check if we have valid data
    if not data or not data.get('batches'):
        print(f"ERROR: No batches found in preprocessed data for {dataset_name}")
        return None

    # Prepare data for training
    print("Preparing data for training...")
    splits = dataset_handler.prepare_for_training(
        dataset=data,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        test_split=args.test_split,
        max_target_idx=generator.model.embedding.num_embeddings - 1  # Ensure target indices stay in range
    )

    train_data = splits.get("train", {})
    val_data = splits.get("validation", {})
    test_data = splits.get("test", {})

    # Initialize metrics tracking
    metrics = {
        'loss': [],
        'val_loss': [],
        'test_loss': [],
        'perplexity': [],
        'val_perplexity': [],
        'test_perplexity': [],
        'accuracy': [],
        'val_accuracy': [],
        'test_accuracy': []
    }

    # Setup loss function
    criterion = get_loss_function(task_type="generation")

    # Training loop
    print(f"Starting training for {args.epochs} epochs...")

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        # Clear CUDA cache before each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Training phase
        generator.model.train()
        epoch_loss = 0.0
        epoch_batches = 0
        steps_since_update = 0

        # Training progress bar
        train_progress = tqdm(train_data['batches'], desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for i, (input_batch, target_batch) in enumerate(train_progress):
            try:
                # Move data to the model's device
                input_batch = input_batch.to(device)
                target_batch = target_batch.to(device)
                
                # Get model vocabulary size (n_classes)
                vocab_size = generator.model.embedding.num_embeddings
                
                # Safety check: Ensure target indices are within valid range
                if target_batch.max() >= vocab_size:
                    # Clip target indices to valid range
                    target_batch = torch.clamp(target_batch, 0, vocab_size - 1)
                
                # Zero gradients only at the start of accumulation steps
                if steps_since_update == 0:
                    generator.optimizer.zero_grad()
                
                # Forward pass
                output, _ = generator.model(input_batch)
                
                # Calculate loss - handle different shapes safely
                if target_batch.dim() == 1:
                    # For 1D targets, the shape should be [batch_size]
                    # and output should be [batch_size, vocab_size]
                    loss = criterion(output, target_batch)
                else:
                    # For 2D targets, the shape should be [batch_size, seq_len]
                    # Reshape as needed
                    loss = criterion(
                        output.view(-1, output.size(-1)), 
                        target_batch.view(-1)
                    )
                
                # Scale loss by gradient accumulation steps
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                    
                # Backward pass 
                loss.backward()
                
                steps_since_update += 1
                
                # Update weights after accumulating gradients
                if steps_since_update >= gradient_accumulation_steps:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(generator.model.parameters(), 5.0)
                    generator.optimizer.step()
                    generator.optimizer.zero_grad()
                    steps_since_update = 0
                
                # Track metrics (use unscaled loss for logging)
                batch_loss = loss.item() * (gradient_accumulation_steps if gradient_accumulation_steps > 1 else 1)
                epoch_loss += batch_loss
                epoch_batches += 1
                
                # Update progress bar
                train_progress.set_postfix(loss=epoch_loss/max(1, epoch_batches))
                
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print("\nCUDA out of memory, clearing cache and continuing...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    steps_since_update = 0  # Reset accumulation
                    continue
                elif "device-side assert triggered" in str(e):
                    print(f"\nCUDA Error in batch: {str(e)}")
                    print(f"Input shape: {input_batch.shape}, Target shape: {target_batch.shape}")
                    print(f"Target range: min={target_batch.min().item()}, max={target_batch.max().item()}")
                    steps_since_update = 0  # Reset accumulation
                    continue
                else:
                    # Log the error and continue with the next batch
                    print(f"\nError processing batch: {str(e)}")
                    print(f"Input shape: {input_batch.shape}, Target shape: {target_batch.shape}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    steps_since_update = 0  # Reset accumulation
                    continue
            
            except Exception as e:
                print(f"\nUnexpected error: {str(e)}")
                steps_since_update = 0  # Reset accumulation
                continue
        
        # Perform final optimization step if there are any remaining gradients
        if steps_since_update > 0:
            torch.nn.utils.clip_grad_norm_(generator.model.parameters(), 5.0)
            generator.optimizer.step()
            generator.optimizer.zero_grad()

        # Calculate average training loss
        train_loss = epoch_loss / max(1, epoch_batches)

        # Validation phase
        if val_data and val_data.get('batches'):
            val_metrics = calculate_metrics(generator.model, val_data['batches'], device, task_type="generation")
            val_loss = val_metrics['loss']
            val_perplexity = val_metrics['perplexity']
            val_accuracy = val_metrics['accuracy']

            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0

                # Save the best model
                best_model_path = os.path.join(args.model_dir, f"{dataset_name}_best_model.pt")
                generator.save_model(best_model_path)
                print(f"New best model saved to {best_model_path}")
            else:
                epochs_without_improvement += 1

            print(f"Epoch {epoch+1}/{args.epochs} - "
                  f"Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Val Perplexity: {val_perplexity:.2f}, Val Accuracy: {val_accuracy:.4f}")

            # Update metrics
            metrics['val_loss'].append(val_loss)
            metrics['val_perplexity'].append(val_perplexity)
            metrics['val_accuracy'].append(val_accuracy)
        else:
            print(f"Epoch {epoch+1}/{args.epochs} - Loss: {train_loss:.4f}")

        # Calculate and update training metrics
        train_metrics = calculate_metrics(generator.model, train_data['batches'], device, task_type="generation")
        metrics['loss'].append(train_loss)
        metrics['perplexity'].append(train_metrics['perplexity'])
        metrics['accuracy'].append(train_metrics['accuracy'])

        # Check early stopping
        if args.early_stopping > 0 and epochs_without_improvement >= args.early_stopping:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Final test evaluation
    if test_data and test_data.get('batches'):
        print("\nEvaluating on test set...")
        test_metrics = calculate_metrics(generator.model, test_data['batches'], device, task_type="generation")
        metrics['test_loss'] = [test_metrics['loss']]
        metrics['test_perplexity'] = [test_metrics['perplexity']]
        metrics['test_accuracy'] = [test_metrics['accuracy']]

        print(f"Test Loss: {test_metrics['loss']:.4f}")
        print(f"Test Perplexity: {test_metrics['perplexity']:.2f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")

    # Visualize training progress
    visualizer.visualize_training(dataset_name, metrics)

    # Save the final model
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, f"{dataset_name}_model.pt")
    generator.save_model(model_path)
    print(f"Final model saved to {model_path}")

    # Save the metrics for future reference
    metrics_dir = get_storage_path("metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, f"{dataset_name}_metrics.json")
    with open(metrics_path, 'w') as f:
        # Convert numpy values to Python types for JSON serialization
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, list):
                serializable_metrics[key] = [float(v) if isinstance(v, (np.float32, np.float64)) else v for v in value]
            else:
                serializable_metrics[key] = float(value) if isinstance(value, (np.float32, np.float64)) else value

        json.dump(serializable_metrics, f, indent=2)
    print(f"Training metrics saved to {metrics_path}")
    
    # Sync metrics to Google Drive
    sync_to_gdrive("metrics")

    return metrics


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
    """Main entry point for the training script"""
    args = parse_args()
    
    # Install dependencies if requested
    if args.install_dependencies:
        install_dependencies()
    
    # Initialize logging
    log_dir = os.path.join("/notebooks/Jarvis_AI_Assistant/logs" if os.path.exists("/notebooks") else os.path.expanduser("~/Jarvis_AI_Assistant/logs"))
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create output directory
    if not args.output_dir:
        dataset_name = args.dataset.replace('/', '_') if args.dataset else "unknown"
        args.output_dir = f"models/{args.model_type}_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up evaluation metrics options
    use_enhanced_eval = not args.no_enhanced_eval
    eval_metrics_dir = args.eval_metrics_dir or os.path.join(args.output_dir, "evaluation_metrics")
    os.makedirs(eval_metrics_dir, exist_ok=True)
    
    # Process datasets
    datasets_to_train = []
    
    # Handle comma-separated HuggingFace datasets
    if hasattr(args, 'dataset_list') and args.dataset_list:
        # Use the list of HuggingFace datasets
        for dataset in args.dataset_list:
            datasets_to_train.append({
                'name': dataset,
                'path': dataset,  # The dataset path is the same as the name for HuggingFace datasets
                'is_huggingface': True
            })
    # Handle single dataset (which could be a HuggingFace dataset with '/' in the name)
    elif args.dataset:
        is_huggingface = '/' in args.dataset
        dataset_path = args.dataset  # Use the dataset name as the path for HuggingFace datasets
        
        if not is_huggingface:
            # Local dataset - check if it exists
            default_path = f"data/{args.dataset}"
            if os.path.exists(default_path):
                dataset_path = default_path
            elif args.dataset_path:
                dataset_path = args.dataset_path
            elif not os.path.exists(args.dataset):
                print(f"Dataset not found at {default_path}")
                print("Please provide the dataset path with --dataset_path")
                return 1
        
        datasets_to_train.append({
            'name': args.dataset,
            'path': dataset_path,
            'is_huggingface': is_huggingface
        })
    else:
        print("No dataset specified. Please provide a dataset with --dataset.")
        return 1
    
    # Determine model name based on size if not provided
    model_name = args.model_name
    if not model_name:
        if args.model_size == "small":
            model_name = "gpt2"
        elif args.model_size == "medium":
            model_name = "gpt2-medium"
        elif args.model_size == "large":
            model_name = "gpt2-large"
        elif args.model_size == "xl":
            model_name = "gpt2-xl"
        elif args.model_size == "deepseek-small":
            model_name = "deepseek-ai/deepseek-coder-1.3b-base"
        elif args.model_size == "deepseek-medium":
            model_name = "deepseek-ai/deepseek-coder-6.7b-base"
        else:
            model_name = "gpt2"
    
    # Handle specific model types
    if args.model_type == "text":
        # Process each dataset
        all_results = {}
        
        for dataset_info in datasets_to_train:
            dataset_name = dataset_info['name']
            dataset_path = dataset_info['path']
            is_huggingface = dataset_info['is_huggingface']
            
            print(f"\n{'='*50}")
            print(f"Training text generation model on {dataset_name} dataset")
            print(f"{'='*50}")
            
            # For HuggingFace datasets, we'll use the dataset processor's ability to handle them
            if is_huggingface:
                print(f"Processing HuggingFace dataset: {dataset_name}")
                # dataset_path is already set to the HuggingFace dataset name
            else:
                print(f"Processing local dataset from: {dataset_path}")
            
            # Create a dataset-specific output directory
            dataset_output_dir = os.path.join(args.output_dir, dataset_name.replace('/', '_'))
            os.makedirs(dataset_output_dir, exist_ok=True)
            
            # Train on this dataset
            dataset_metrics_dir = os.path.join(eval_metrics_dir, dataset_name.replace('/', '_'))
            os.makedirs(dataset_metrics_dir, exist_ok=True)
            
            try:
                # Process the dataset and train the model
                result = train_text_model(
                    dataset_name=dataset_name,
                    args=args,
                    force_gpu=not args.cpu_only
                )
                
                if result:
                    all_results[dataset_name] = result
                    print(f"Successfully trained on {dataset_name}")
                else:
                    print(f"Failed to train on {dataset_name}")
            except Exception as e:
                print(f"Error training on {dataset_name}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Summarize results
        if all_results:
            print("\n" + "="*50)
            print("Training Complete!")
            print("="*50)
            print(f"\nTrained on {len(all_results)} dataset(s):")
            for dataset_name in all_results.keys():
                print(f"  - {dataset_name}")
            print(f"\nModels saved to: {args.output_dir}")
            
            # Sync to Google Drive if in Paperspace
            if is_paperspace_environment():
                try:
                    from src.generative_ai_module.utils import sync_to_gdrive
                    sync_to_gdrive(args.output_dir)
                    print(f"Models and evaluation results synced to Google Drive")
                except Exception as e:
                    print(f"Error syncing to Google Drive: {e}")
        else:
            print("No datasets were successfully trained. Check the logs for details.")
    
    elif args.model_type == "code":
        # Train code generation model
        print(f"Training code generation model")
        
        # Set up enhanced evaluation for code models
        train_code_model(args, force_gpu=not args.cpu_only)
    
    else:
        print(f"Unknown model type: {args.model_type}")
        return 1
    
    return 0


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