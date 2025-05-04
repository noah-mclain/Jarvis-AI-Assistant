#!/usr/bin/env python3
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
    parser = argparse.ArgumentParser(description="Consolidated training script for Jarvis AI")
    
    # Model type selection
    parser.add_argument('--model-type', type=str, default='all',
                      choices=['all', 'text', 'code'],
                      help='Type of model to train (all, text, or code)')
    
    # Dataset selection for text models
    parser.add_argument('--datasets', type=str, nargs='+', 
                      choices=['all', 'writing_prompts', 'persona_chat', 'pile', 'openassistant', 'gpteacher'],
                      default=['all'],
                      help='Datasets to train on (default: all)')
    
    # For The Pile dataset
    parser.add_argument('--pile-subset', type=str, default=None,
                      help='Specific subset of The Pile (e.g., "pubmed", "github", "europarl")')
    
    # For code model
    parser.add_argument('--code-subset', type=str, default=None,
                      help='Specific subset of the code dataset (e.g., "python", "java")')
    parser.add_argument('--all-code-subsets', action='store_true',
                      help='Use all code dataset subsets')
    parser.add_argument('--use-deepseek', action='store_true',
                      help='Use DeepSeek-Coder for code generation')
    parser.add_argument('--deepseek-batch-size', type=int, default=4,
                      help='Batch size for DeepSeek-Coder training')
    
    # General training options
    parser.add_argument('--max-samples', type=int, default=500,
                      help='Maximum number of samples to use per dataset')
    parser.add_argument('--validation-split', type=float, default=0.2,
                      help='Fraction of data to use for validation')
    parser.add_argument('--test-split', type=float, default=0.1,
                      help='Fraction of data to use for testing')
    parser.add_argument('--epochs', type=int, default=10,
                      help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.002,
                      help='Learning rate for training')
    parser.add_argument('--early-stopping', type=int, default=3,
                      help='Stop training if validation loss does not improve for this many epochs')
    parser.add_argument('--sequence-length', type=int, default=100,
                      help='Sequence length for training examples')
    parser.add_argument('--warmup-steps', type=int, default=100,
                      help='Warmup steps for learning rate scheduler')
    
    # Output options
    parser.add_argument('--model-dir', type=str, default='models',
                      help='Directory to save models')
    parser.add_argument('--visualization-dir', type=str, default='visualizations',
                      help='Directory to save visualizations')
    parser.add_argument('--evaluation-dir', type=str, default='evaluation_metrics',
                      help='Directory to save evaluation metrics')
    
    # Hardware options
    parser.add_argument('--no-force-gpu', action='store_true',
                      help='Do not force GPU usage (by default, GPU is used if available)')
    parser.add_argument('--load-in-4bit', action='store_true',
                      help='Load models in 4-bit precision (for NVIDIA GPUs)')
    
    # Demo/testing options
    parser.add_argument('--use-mini-dataset', action='store_true',
                      help='Use mini dataset for quick testing')
    
    return parser.parse_args()


def install_dependencies():
    """Install required dependencies for training"""
    print("Checking and installing required dependencies...")
    
    # Try to import tensorboard, install if not available
    try:
        import tensorboard
        print("TensorBoard is already installed.")
    except ImportError:
        print("Installing TensorBoard...")
        subprocess.run([sys.executable, "-m", "pip", "install", "tensorboard"])
        print("TensorBoard installed successfully.")
    
    # Try to import evaluation metric dependencies
    try:
        from bert_score import BERTScorer
        print("BERTScore is already installed.")
    except ImportError:
        print("Installing evaluation metric dependencies...")
        EvaluationMetrics.install_dependencies()


def calculate_metrics(model, data_batches, device):
    """Calculate metrics on a dataset (loss, perplexity, accuracy)"""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    total_correct = 0
    total_samples = 0
    
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for input_batch, target_batch in data_batches:
            # Move data to the model's device
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            
            # Forward pass
            output, _ = model(input_batch)
            
            # Calculate loss
            loss = criterion(output.view(-1, output.size(-1)), target_batch.view(-1))
            total_loss += loss.item()
            
            # Calculate accuracy
            predictions = output.argmax(dim=-1)
            correct = (predictions == target_batch).sum().item()
            total_correct += correct
            total_samples += target_batch.numel()
            
            total_batches += 1
    
    # Calculate metrics
    avg_loss = total_loss / max(1, total_batches)
    perplexity = np.exp(avg_loss)
    accuracy = total_correct / max(1, total_samples)
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'accuracy': accuracy
    }


def train_text_model(dataset_name: str, args, force_gpu: bool = True):
    """Train a text generation model on a specific dataset with visualization"""
    print(f"\n{'='*50}")
    print(f"Training Text Model on {dataset_name.upper()} dataset")
    print(f"{'='*50}")

    # Create text generator and dataset handler
    generator = TextGenerator(force_gpu=force_gpu)
    dataset_handler = UnifiedDatasetHandler(force_gpu=force_gpu)
    visualizer = TrainingVisualizer(args.visualization_dir)

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
        test_split=args.test_split
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

    # Training loop
    print(f"Starting training for {args.epochs} epochs...")

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        # Training phase
        generator.model.train()
        epoch_loss = 0.0
        epoch_batches = 0

        # Training progress bar
        train_progress = tqdm(train_data['batches'], desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for input_batch, target_batch in train_progress:
            # Move data to the model's device
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            # Forward pass
            generator.optimizer.zero_grad()
            output, _ = generator.model(input_batch)

            # Calculate loss
            loss = torch.nn.functional.cross_entropy(output.view(-1, output.size(-1)), target_batch.view(-1))

            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.model.parameters(), 5.0)  # Gradient clipping
            generator.optimizer.step()

            # Update metrics
            epoch_loss += loss.item()
            epoch_batches += 1

            # Update progress bar
            train_progress.set_postfix(loss=epoch_loss/epoch_batches)

        # Calculate average training loss
        train_loss = epoch_loss / max(1, epoch_batches)

        # Validation phase
        if val_data and val_data.get('batches'):
            val_metrics = calculate_metrics(generator.model, val_data['batches'], device)
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
        train_metrics = calculate_metrics(generator.model, train_data['batches'], device)
        metrics['loss'].append(train_loss)
        metrics['perplexity'].append(train_metrics['perplexity'])
        metrics['accuracy'].append(train_metrics['accuracy'])

        # Check early stopping
        if args.early_stopping > 0 and epochs_without_improvement >= args.early_stopping:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Final test evaluation
    if test_data and test_data.get('batches'):
        evaluating_train_text_model(generator, test_data, device, metrics)
    # Visualize training progress
    visualizer.visualize_training(dataset_name, metrics)

    # Save the final model
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, f"{dataset_name}_model.pt")
    generator.save_model(model_path)
    print(f"Final model saved to {model_path}")

    # Save the metrics for future reference
    metrics_path = os.path.join(args.visualization_dir, f"{dataset_name}_metrics.json")
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

    return metrics


def evaluating_train_text_model(generator, test_data, device, metrics):
    print("\nEvaluating on test set...")
    test_metrics = calculate_metrics(generator.model, test_data['batches'], device)
    metrics['test_loss'] = [test_metrics['loss']]
    metrics['test_perplexity'] = [test_metrics['perplexity']]
    metrics['test_accuracy'] = [test_metrics['accuracy']]

    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Perplexity: {test_metrics['perplexity']:.2f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")


def train_code_model(args, force_gpu: bool = True):
    """Train DeepSeek-Coder model for code generation"""
    print("\n===== Training DeepSeek-Coder Model =====")

    # On Apple Silicon (M1/M2/M3)
    on_apple_silicon = hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

    # Common arguments
    finetune_script = os.path.join(os.path.dirname(__file__), "finetune_deepseek.py")

    cmd_args = [
        sys.executable,  # Use the current Python interpreter
        finetune_script,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.deepseek_batch_size),
        "--max-samples", str(args.max_samples),
        "--sequence-length", str(args.sequence_length),
        "--learning-rate", str(args.learning_rate),
        "--warmup-steps", str(args.warmup_steps),
        "--force-gpu",    # Boolean flag (no value)
    ]

    # Add code subset if specified
    if args.code_subset:
        cmd_args.extend(["--code-subset", args.code_subset])

    # Use all code subsets if requested
    if args.all_code_subsets:
        cmd_args.append("--all-subsets")

    # Use mini dataset for faster testing if requested
    if args.use_mini_dataset:
        cmd_args.append("--use-mini-dataset")
        print("Using mini dataset for quick testing")

    # Add quantization for NVIDIA GPUs only
    if args.load_in_4bit and not on_apple_silicon and torch.cuda.is_available():
        cmd_args.append("--load-in-4bit")

    cmd_args.extend(
        [
            "--model-dir",
            args.model_dir,
            "--visualization-dir",
            args.visualization_dir,
        ]
    )
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
    save_total_limit=3
):
    """
    Train a transformer model using the HuggingFace Transformers library.
    """
    if output_dir is None:
        # Use the storage path utility to get the correct path
        output_dir = get_storage_path("models", run_name or model_name.split('/')[-1])
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ... existing code ...
    
    # Add this at the end of the function, just before returning
    # Sync models and checkpoints to Google Drive after training
    sync_to_gdrive("models")
    sync_to_gdrive("checkpoints")
    
    return trainer, model


class CustomCallback(TrainerCallback):
    """Custom callback for syncing checkpoints to Google Drive during training."""
    
    def __init__(self, sync_interval=1):
        self.sync_interval = sync_interval
        self.last_sync_step = 0
    
    def on_save(self, args, state, control, **kwargs):
        """Called when the trainer saves a checkpoint."""
        # Sync metrics and checkpoints to Google Drive
        sync_to_gdrive("metrics")
        sync_to_gdrive("checkpoints")
        logger.info("Synced metrics and checkpoints to Google Drive")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when the trainer logs metrics."""
        if (state.global_step - self.last_sync_step) >= self.sync_interval * args.logging_steps:
            # Sync metrics to Google Drive
            sync_to_gdrive("metrics")
            self.last_sync_step = state.global_step
            logger.info(f"Synced metrics to Google Drive at step {state.global_step}")


def main():
    """
    Main function to run the training process.
    """
    # Set up logging to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"training_{timestamp}.log"
    setup_logging(log_file)
    
    logger.info("Starting training process")
    
    args = parse_args()
    
    # If we're in Paperspace, sync from Google Drive first
    if is_paperspace_environment():
        logger.info("Running in Paperspace environment, syncing from Google Drive...")
        sync_from_gdrive("datasets")
        sync_from_gdrive("models")
        sync_from_gdrive("metrics")
        logger.info("Synced latest data from Google Drive")
    
    # Create output directories - ensure they exist
    ensure_directory_exists("models")
    ensure_directory_exists("metrics")
    
    # Check GPU availability
    force_gpu = not args.no_force_gpu
    if torch.cuda.is_available():
        print(f"NVIDIA GPU detected: {torch.cuda.get_device_name(0)}")
        print("Using CUDA for training")
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("Apple Silicon GPU detected")
        print("Using MPS for training")
    else:
        print("WARNING: No GPU detected. Training will be very slow on CPU.")
        proceed = input("Do you want to continue anyway? (y/n): ").strip().lower()
        if proceed != 'y':
            print("Training cancelled.")
            return

    # Install dependencies
    install_dependencies()

    # Use smaller values for quick testing with mini dataset
    if args.use_mini_dataset:
        args.max_samples = min(args.max_samples, 50)
        args.epochs = min(args.epochs, 3)
        print(f"\nUsing mini dataset settings: max_samples={args.max_samples}, epochs={args.epochs}")

    # Determine which text datasets to train on
    datasets_to_train = []
    if 'all' in args.datasets:
        datasets_to_train = ['writing_prompts', 'persona_chat', 'pile', 'openassistant', 'gpteacher']
    else:
        datasets_to_train = args.datasets

    # Train code model if requested
    if args.model_type in ['all', 'code'] and args.use_deepseek:
        code_success = train_code_model(args, force_gpu=force_gpu)
        print(f"DeepSeek-Coder training: {'SUCCESS' if code_success else 'FAILED'}")

    # Train text models if requested
    if args.model_type in ['all', 'text']:
        print(f"\nWill train text models on the following datasets: {', '.join(datasets_to_train)}")

        # Train on each dataset and collect metrics
        all_metrics = {}
        start_time = time.time()

        for dataset_name in datasets_to_train:
            if dataset_metrics := train_text_model(
                dataset_name, args, force_gpu=force_gpu
            ):
                all_metrics[dataset_name] = dataset_metrics

        # Create comparison visualizations if multiple datasets were trained
        if len(all_metrics) > 1:
            visualizer = TrainingVisualizer(args.visualization_dir)
            for metric in ['loss', 'perplexity', 'accuracy']:
                visualizer.create_comparison_plot(all_metrics, metric)

        # Calculate total training time
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print("\n==================================================")
        print("Text Model Training Complete!")
        print(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

    # After training, sync everything to Google Drive
    if is_paperspace_environment():
        sync_to_gdrive("models")
        sync_to_gdrive("metrics")
        sync_logs()
        logger.info("Training complete! Model and metrics synced to Google Drive.")
    else:
        logger.info("Training complete!")


if __name__ == "__main__":
    # Add the parent directory to the path to make the module importable
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
    from src.generative_ai_module.text_generator import TextGenerator
    from src.generative_ai_module.unified_dataset_handler import UnifiedDatasetHandler
    from src.generative_ai_module.evaluation_metrics import EvaluationMetrics
    from src.generative_ai_module.utils import get_storage_path, sync_to_gdrive, sync_logs, setup_logging
    main() 