"""
Unified Generation Pipeline

This script combines the functionality of multiple scripts to provide a streamlined
approach for training and generating text or code with the generative AI module.

Features:
1. Load and preprocess data from tokenized datasets
2. Train text and code generation models
3. Generate text or code from prompts
4. Use character or tokenized input formats
5. Save and load models
6. Evaluate model performance
7. Interactive prompt-based generation
"""

import os
import sys
import torch
import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict, Any
import random
import json
import datetime
import matplotlib.pyplot as plt
import time
import math
import functools

from .utils import get_storage_path

from .prompt_enhancer import analyze_prompt
# Use try/except for importing evaluate_model
try:
    from .unsloth_deepseek import evaluate_model
    UNSLOTH_AVAILABLE = True
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"UnifiedGenerationPipeline: Failed to import evaluate_model: {e}")
    # Define a fallback evaluate_model function
    def evaluate_model(*args, **kwargs):
        logger.warning("evaluate_model not available - using stub version")
        return {"error": "evaluate_model not available"}
    UNSLOTH_AVAILABLE = False

# Define infinity for use in the code
infinity = float('inf')

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .text_generator import TextGenerator, CombinedModel
from .code_generator import CodeGenerator
from .dataset_processor import DatasetProcessor
from .improved_preprocessing import ImprovedPreprocessor

# Try to import tokenizer from examples directory
try:
    from .basic_tokenizer import BasicTokenizer
except ImportError:
    print("Warning: BasicTokenizer not found. Character-level generation will be used.")
    BasicTokenizer = None

# Execution time measurement utility function
def print_execution_time(func):
    """Decorator to print the execution time of a function"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.2f} seconds to execute")
        return result
    return wrapper

class UnifiedGenerationPipeline:
    """
    Unified Generation Pipeline for text and code generation.
    
    This class provides a unified interface for training models, preprocessing data,
    and generating text/code using various approaches.
    """
    
    def __init__(self, model_dir="models", force_gpu=True):
        """
        Initialize the pipeline with specified configuration.
        
        Args:
            model_dir: Directory to save/load models
            force_gpu: Whether to force GPU usage if available
        """
        self.model_dir = model_dir
        self.force_gpu = force_gpu
        self.device = torch.device('cuda' if torch.cuda.is_available() and force_gpu else 'cpu')
        self.visualizer = TrainingVisualizer()
        
        # Create necessary directories
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize to None, will be set when models are loaded/trained
        self.text_model = None
        self.code_model = None
        self.tokenizer = None
        self.vocab_size = None
    
    def train(self, dataset_name, train_type="text", epochs=50, 
              learning_rate=0.002, batch_size=16, sequence_length=2048,
              gradient_accumulation_steps=4, save_model=True,
              validation_split=0.2, use_deepseek=False, deepseek_batch_size=1,
              code_subset=None, all_subsets=False, max_samples=None):
        """
        Train a model on the specified dataset.
        
        Args:
            dataset_name: Name of the dataset to train on
            train_type: Type of model to train ("text" or "code")
            epochs: Number of training epochs
            learning_rate: Learning rate for training
            batch_size: Batch size for training
            sequence_length: Maximum sequence length
            gradient_accumulation_steps: Steps for gradient accumulation
            save_model: Whether to save the model after training
            validation_split: Percentage of data to use for validation
            use_deepseek: Whether to use DeepSeek models
            deepseek_batch_size: Batch size for DeepSeek models
            code_subset: Subset of code data to use (python, java, etc.)
            all_subsets: Whether to use all code subsets
            max_samples: Maximum number of samples to use from dataset
            
        Returns:
            Trained model and training metrics
        """
        if train_type.lower() == "text":
            model, metrics = train_text_generator(
                dataset_name=dataset_name,
                epochs=epochs,
                learning_rate=learning_rate,
                force_gpu=self.force_gpu,
                validation_split=validation_split,
                create_visualizations=True,
                batch_size=batch_size,
                sequence_length=sequence_length,
                gradient_accumulation_steps=gradient_accumulation_steps
            )
            self.text_model = model
            
            # Save the model if requested
            if save_model:
                model_path = os.path.join(self.model_dir, f"text_model_{dataset_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
                torch.save(model.state_dict(), model_path)
                print(f"Model saved to {model_path}")
            
            return model, metrics
            
        elif train_type.lower() == "code":
            # Here we would call the code_generator training function
            # Currently using a placeholder that calls the existing function
            print(f"Training code generator on {dataset_name}")
            model, metrics = train_code_generator(
                dataset_name=dataset_name,
                epochs=epochs,
                model_path=None  # We'll save it ourselves if needed
            )
            self.code_model = model
            
            # Save the model if requested
            if save_model:
                model_path = os.path.join(self.model_dir, f"code_model_{dataset_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
                torch.save(model.state_dict(), model_path)
                print(f"Model saved to {model_path}")
            
            return model, metrics
        
        else:
            raise ValueError(f"Unknown train_type: {train_type}. Must be 'text' or 'code'.")
    
    def load_model(self, model_path, model_type="text"):
        """
        Load a model from disk.
        
        Args:
            model_path: Path to the model file
            model_type: Type of model ("text" or "code")
            
        Returns:
            Loaded model
        """
        model, vocab_size = load_model(model_path, model_type)
        
        if model_type.lower() == "text":
            self.text_model = model
        else:
            self.code_model = model
            
        self.vocab_size = vocab_size
        return model
    
    def generate(self, prompt, model_type="text", max_length=100, 
                 temperature=0.7, top_p=0.9, use_tokenizer=True):
        """
        Generate text or code from a prompt.
        
        Args:
            prompt: Input prompt for generation
            model_type: Type of model to use ("text" or "code")
            max_length: Maximum length of generated output
            temperature: Temperature for sampling
            top_p: Top-p probability for nucleus sampling
            use_tokenizer: Whether to use tokenizer-based generation
            
        Returns:
            Generated text
        """
        model = self.text_model if model_type.lower() == "text" else self.code_model
        
        if model is None:
            raise ValueError(f"No {model_type} model loaded. Please load or train a model first.")
        
        if use_tokenizer and self.tokenizer:
            return generate_with_tokenizer(
                model=model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                max_length=max_length,
                temperature=temperature
            )
        else:
            # Fall back to character-level generation
            if self.vocab_size is None:
                raise ValueError("Vocabulary size not set. Please load a model first.")
                
            return generate_with_char_model(
                model=model,
                prompt=prompt,
                vocab_size=self.vocab_size,
                max_length=max_length,
                temperature=temperature
            )
    
    def interactive_mode(self, model_type="text"):
        """
        Start an interactive generation session.
        
        Args:
            model_type: Type of model to use ("text" or "code")
        """
        model = self.text_model if model_type.lower() == "text" else self.code_model
        
        if model is None:
            raise ValueError(f"No {model_type} model loaded. Please load or train a model first.")
            
        interactive_generation(model, self.tokenizer, model_type, self.force_gpu)
    
    def preprocess_dataset(self, dataset_name, output_dir=None):
        """
        Preprocess a dataset for training.
        
        Args:
            dataset_name: Name of the dataset to preprocess
            output_dir: Directory to save preprocessed data
            
        Returns:
            Preprocessed dataset
        """
        args = argparse.Namespace()
        args.dataset = dataset_name
        args.output_dir = output_dir or os.path.join("datasets", "processed")
        
        return preprocess_data(args)


# The rest of the file remains unchanged
class TrainingVisualizer:
    """Class to handle visualization of training metrics"""
    
    def __init__(self, output_dir="visualizations"):
        """Initialize the visualizer with an output directory"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories for different visualizations
        self.plots_dir = os.path.join(output_dir, "plots")
        self.comparisons_dir = os.path.join(output_dir, "comparisons")
        self.checkpoints_dir = os.path.join(output_dir, "checkpoints")
        
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.comparisons_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        
        # Set up plot styling
        plt.style.use('ggplot')
        
        # Set higher DPI for better quality images
        self.dpi = 300
        
        # Define color palette
        self.colors = {
            'train': '#1f77b4',  # Blue
            'validation': '#d62728',  # Red
            'accuracy': '#2ca02c',  # Green
            'perplexity': '#ff7f0e',  # Orange
            'learning_rate': '#9467bd',  # Purple
            'checkpoint': '#8c564b'  # Brown
        }
        
    def visualize_training(self, dataset_name: str, metrics: Dict[str, List[float]]):
        """Create visualizations for training metrics"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a visualization directory for this run
        dataset_vis_dir = os.path.join(self.plots_dir, f"{dataset_name}_{timestamp}")
        os.makedirs(dataset_vis_dir, exist_ok=True)
        
        # Create an index file for the visualizations
        index_file = os.path.join(dataset_vis_dir, "index.html")
        with open(index_file, 'w') as f:
            f.write(f"<html><head><title>Training Visualizations for {dataset_name}</title>\n")
            f.write("<style>body{font-family:Arial,sans-serif; margin:20px;} img{max-width:800px; margin:10px;}</style>\n")
            f.write("</head><body>\n")
            f.write(f"<h1>Training Visualizations for {dataset_name}</h1>\n")
            f.write(f"<p>Generated on: {timestamp}</p>\n")
            f.write("<hr/>\n")
        
        # Loss curve
        if 'loss' in metrics and len(metrics['loss']) > 0:
            loss_path = self._plot_loss_curves(metrics, dataset_name, dataset_vis_dir, timestamp)
            with open(index_file, 'a') as f:
                f.write("<h2>Loss Curves</h2>\n")
                f.write(f"<img src='{os.path.basename(loss_path)}' alt='Loss Curves'/>\n")
                f.write("<hr/>\n")
            
        # Accuracy if available
        if 'accuracy' in metrics and len(metrics['accuracy']) > 0:
            acc_path = self._plot_accuracy_curves(metrics, dataset_name, dataset_vis_dir, timestamp)
            with open(index_file, 'a') as f:
                f.write("<h2>Accuracy Curves</h2>\n")
                f.write(f"<img src='{os.path.basename(acc_path)}' alt='Accuracy Curves'/>\n")
                f.write("<hr/>\n")
            
        # Perplexity if available
        if 'perplexity' in metrics and len(metrics['perplexity']) > 0:
            perp_path = self._plot_perplexity_curves(metrics, dataset_name, dataset_vis_dir, timestamp)
            with open(index_file, 'a') as f:
                f.write("<h2>Perplexity Curves</h2>\n")
                f.write(f"<img src='{os.path.basename(perp_path)}' alt='Perplexity Curves'/>\n")
                f.write("<hr/>\n")
                
        # Learning rate if available
        if 'learning_rates' in metrics and metrics['learning_rates'] and len(metrics['learning_rates']) > 0:
            lr_path = self._plot_learning_rate_curve(metrics, dataset_name, dataset_vis_dir, timestamp)
            with open(index_file, 'a') as f:
                f.write("<h2>Learning Rate</h2>\n")
                f.write(f"<img src='{os.path.basename(lr_path)}' alt='Learning Rate'/>\n")
                f.write("<hr/>\n")
            
        # Combined metrics
        combined_path = self._plot_combined_metrics(metrics, dataset_name, dataset_vis_dir, timestamp)
        with open(index_file, 'a') as f:
            f.write("<h2>Combined Metrics</h2>\n")
            f.write(f"<img src='{os.path.basename(combined_path)}' alt='Combined Metrics'/>\n")
            f.write("<hr/>\n")
            
        # Checkpoint visualization if available
        if 'checkpoints' in metrics and metrics['checkpoints']:
            checkpoint_path = self._plot_checkpoint_progress(metrics, dataset_name, dataset_vis_dir, timestamp)
            with open(index_file, 'a') as f:
                f.write("<h2>Checkpoint Progress</h2>\n")
                f.write(f"<img src='{os.path.basename(checkpoint_path)}' alt='Checkpoint Progress'/>\n")
                f.write("<hr/>\n")
                
        # Training summary
        summary_path = os.path.join(dataset_vis_dir, f"{dataset_name}_summary_{timestamp}.json")
        with open(summary_path, 'w') as f:
            summary = {
                'dataset': dataset_name,
                'timestamp': timestamp,
                'epochs': len(metrics.get('loss', [])),
                'final_loss': metrics.get('loss', [])[-1] if metrics.get('loss', []) else None,
                'best_val_loss': min(metrics.get('val_loss', [infinity])) if metrics.get('val_loss', []) else None,
                'final_accuracy': metrics.get('accuracy', [])[-1] if metrics.get('accuracy', []) else None,
                'final_perplexity': metrics.get('perplexity', [])[-1] if metrics.get('perplexity', []) else None,
                'training_time': metrics.get('training_time', None),
                'batch_size': metrics.get('batch_size', None),
                'sequence_length': metrics.get('sequence_length', None),
                'gradient_accumulation_steps': metrics.get('gradient_accumulation_steps', None),
                'effective_batch_size': metrics.get('effective_batch_size', None)
            }
            json.dump(summary, f, indent=2)
            
        with open(index_file, 'a') as f:
            f.write("<h2>Training Summary</h2>\n")
            f.write("<pre>\n")
            for k, v in summary.items():
                if v is not None:
                    if isinstance(v, float):
                        f.write(f"{k}: {v:.4f}\n")
                    else:
                        f.write(f"{k}: {v}\n")
            f.write("</pre>\n")
            f.write("</body></html>\n")
        
        print(f"All visualizations saved to {dataset_vis_dir}")
        print(f"View summary at {index_file}")
        
    def _plot_loss_curves(self, metrics, dataset_name, output_dir, timestamp):
        """Create and save loss curve visualization"""
        plt.figure(figsize=(12, 8))
        
        epochs = range(1, len(metrics['loss']) + 1)
        plt.plot(epochs, metrics['loss'], 'b-', linewidth=2, label='Training Loss', color=self.colors['train'])
            
        if 'val_loss' in metrics and len(metrics['val_loss']) > 0:
            val_epochs = range(1, len(metrics['val_loss']) + 1)
            plt.plot(val_epochs, metrics['val_loss'], 'r-', linewidth=2, label='Validation Loss', color=self.colors['validation'])
                
        plt.title(f'Training and Validation Loss - {dataset_name}', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add annotations for best validation loss
        if 'val_loss' in metrics and metrics['val_loss']:
            best_epoch = np.argmin(metrics['val_loss']) + 1
            best_loss = min(metrics['val_loss'])
            plt.annotate(f'Best: {best_loss:.4f}',
                xy=(best_epoch, best_loss), 
                xytext=(best_epoch + 0.5, best_loss * 1.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
        
        # Add text with final losses
        final_train_loss = metrics['loss'][-1] if metrics['loss'] else "N/A"
        final_val_loss = metrics['val_loss'][-1] if 'val_loss' in metrics and metrics['val_loss'] else "N/A"
        
        plt.figtext(0.15, 0.02, f"Final train loss: {final_train_loss:.4f}", fontsize=12)
        if final_val_loss != "N/A":
            plt.figtext(0.55, 0.02, f"Final validation loss: {final_val_loss:.4f}", fontsize=12)
            
        # Save the figure
        loss_path = os.path.join(output_dir, f"{dataset_name}_loss_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(loss_path, dpi=self.dpi)
        plt.close()
        
        return loss_path
    
    def _plot_accuracy_curves(self, metrics, dataset_name, output_dir, timestamp):
        """Create and save accuracy curve visualization"""
        plt.figure(figsize=(12, 8))
        
        epochs = range(1, len(metrics['accuracy']) + 1)
        plt.plot(epochs, metrics['accuracy'], 'g-', linewidth=2, label='Training Accuracy', color=self.colors['accuracy'])
        
        if 'val_accuracy' in metrics and len(metrics['val_accuracy']) > 0:
            val_epochs = range(1, len(metrics['val_accuracy']) + 1)
            plt.plot(val_epochs, metrics['val_accuracy'], 'm-', linewidth=2, label='Validation Accuracy', color=self.colors['validation'])
                
        plt.title(f'Training and Validation Accuracy - {dataset_name}', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add annotations for best validation accuracy
        if 'val_accuracy' in metrics and metrics['val_accuracy']:
            best_epoch = np.argmax(metrics['val_accuracy']) + 1
            best_acc = max(metrics['val_accuracy'])
            plt.annotate(f'Best: {best_acc:.4f}',
                xy=(best_epoch, best_acc), 
                xytext=(best_epoch + 0.5, best_acc * 0.95),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
        
        # Save the figure
        acc_path = os.path.join(output_dir, f"{dataset_name}_accuracy_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(acc_path, dpi=self.dpi)
        plt.close()
        
        return acc_path
    
    def _plot_perplexity_curves(self, metrics, dataset_name, output_dir, timestamp):
        """Create and save perplexity curve visualization"""
        plt.figure(figsize=(12, 8))
        
        epochs = range(1, len(metrics['perplexity']) + 1)
        plt.plot(epochs, metrics['perplexity'], 'c-', linewidth=2, label='Training Perplexity', color=self.colors['perplexity'])
        
        if 'val_perplexity' in metrics and len(metrics['val_perplexity']) > 0:
            val_epochs = range(1, len(metrics['val_perplexity']) + 1)
            plt.plot(val_epochs, metrics['val_perplexity'], 'y-', linewidth=2, label='Validation Perplexity', color=self.colors['validation'])
                
        plt.title(f'Training and Validation Perplexity - {dataset_name}', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Perplexity', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add annotations for best validation perplexity
        if 'val_perplexity' in metrics and metrics['val_perplexity']:
            best_epoch = np.argmin(metrics['val_perplexity']) + 1
            best_perp = min(metrics['val_perplexity'])
            plt.annotate(f'Best: {best_perp:.4f}',
                xy=(best_epoch, best_perp), 
                xytext=(best_epoch + 0.5, best_perp * 1.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
        
        # Save the figure
        perp_path = os.path.join(output_dir, f"{dataset_name}_perplexity_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(perp_path, dpi=self.dpi)
        plt.close()
        
        return perp_path
    
    def _plot_learning_rate_curve(self, metrics, dataset_name, output_dir, timestamp):
        """Create and save learning rate curve visualization"""
        plt.figure(figsize=(12, 8))
        
        if 'learning_rates' in metrics and metrics['learning_rates']:
            epochs = range(1, len(metrics['learning_rates']) + 1)
            plt.plot(epochs, metrics['learning_rates'], '-', linewidth=2, color=self.colors['learning_rate'])
                
            plt.title(f'Learning Rate Schedule - {dataset_name}', fontsize=16)
            plt.xlabel('Epoch', fontsize=14)
            plt.ylabel('Learning Rate', fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Use logarithmic scale for better visualization
            plt.yscale('log')
            
            # Save the figure
            lr_path = os.path.join(output_dir, f"{dataset_name}_learning_rate_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(lr_path, dpi=self.dpi)
            plt.close()
            
            return lr_path
        
        return None
    
    def _plot_combined_metrics(self, metrics, dataset_name, output_dir, timestamp):
        """Create a combined visualization of all metrics"""
        plt.figure(figsize=(14, 10))
        
        # Create a 2x2 subplot grid
        fig, axs = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'Training Metrics Overview - {dataset_name}', fontsize=20)
        
        # Plot 1: Loss
        ax1 = axs[0, 0]
        epochs = range(1, len(metrics['loss']) + 1)
        ax1.plot(epochs, metrics['loss'], 'b-', linewidth=2, label='Training Loss', color=self.colors['train'])
            
        if 'val_loss' in metrics and len(metrics['val_loss']) > 0:
            val_epochs = range(1, len(metrics['val_loss']) + 1)
            ax1.plot(val_epochs, metrics['val_loss'], 'r-', linewidth=2, label='Validation Loss', color=self.colors['validation'])
        
        ax1.set_title('Loss Curves', fontsize=16)
        ax1.set_xlabel('Epoch', fontsize=14)
        ax1.set_ylabel('Loss', fontsize=14)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy
        ax2 = axs[0, 1]
        if 'accuracy' in metrics and len(metrics['accuracy']) > 0:
            epochs = range(1, len(metrics['accuracy']) + 1)
            ax2.plot(epochs, metrics['accuracy'], 'g-', linewidth=2, label='Training Accuracy', color=self.colors['accuracy'])
            
            if 'val_accuracy' in metrics and len(metrics['val_accuracy']) > 0:
                val_epochs = range(1, len(metrics['val_accuracy']) + 1)
                ax2.plot(val_epochs, metrics['val_accuracy'], 'm-', linewidth=2, label='Validation Accuracy', color=self.colors['validation'])
        
        ax2.set_title('Accuracy Curves', fontsize=16)
        ax2.set_xlabel('Epoch', fontsize=14)
        ax2.set_ylabel('Accuracy', fontsize=14)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Perplexity
        ax3 = axs[1, 0]
        if 'perplexity' in metrics and len(metrics['perplexity']) > 0:
            epochs = range(1, len(metrics['perplexity']) + 1)
            ax3.plot(epochs, metrics['perplexity'], 'c-', linewidth=2, label='Training Perplexity', color=self.colors['perplexity'])
            
            if 'val_perplexity' in metrics and len(metrics['val_perplexity']) > 0:
                val_epochs = range(1, len(metrics['val_perplexity']) + 1)
                ax3.plot(val_epochs, metrics['val_perplexity'], 'y-', linewidth=2, label='Validation Perplexity', color=self.colors['validation'])
        
        ax3.set_title('Perplexity Curves', fontsize=16)
        ax3.set_xlabel('Epoch', fontsize=14)
        ax3.set_ylabel('Perplexity', fontsize=14)
        ax3.tick_params(axis='both', which='major', labelsize=12)
        ax3.legend(fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Learning Rate or Checkpoint Loss
        ax4 = axs[1, 1]
        if 'learning_rates' in metrics and metrics['learning_rates'] and len(metrics['learning_rates']) > 0:
            epochs = range(1, len(metrics['learning_rates']) + 1)
            ax4.plot(epochs, metrics['learning_rates'], '-', linewidth=2, color=self.colors['learning_rate'])
            ax4.set_title('Learning Rate Schedule', fontsize=16)
            ax4.set_xlabel('Epoch', fontsize=14)
            ax4.set_ylabel('Learning Rate', fontsize=14)
            ax4.set_yscale('log')
        elif 'checkpoints' in metrics and metrics['checkpoints']:
            steps = [ckpt['step'] for ckpt in metrics['checkpoints']]
            losses = [ckpt['loss'] for ckpt in metrics['checkpoints']]
            ax4.plot(steps, losses, '-', linewidth=2, color=self.colors['checkpoint'])
            ax4.set_title('Checkpoint Losses', fontsize=16)
            ax4.set_xlabel('Step', fontsize=14)
            ax4.set_ylabel('Loss', fontsize=14)
        else:
            ax4.set_title('No Learning Rate or Checkpoint Data Available', fontsize=16)
            
        ax4.tick_params(axis='both', which='major', labelsize=12)
        ax4.grid(True, alpha=0.3)
        
        # Add metadata as text
        metadata_text = [
            f"Dataset: {dataset_name}",
            f"Epochs: {len(metrics.get('loss', []))}",
            f"Batch Size: {metrics.get('batch_size', 'N/A')}",
            f"Gradient Accumulation: {metrics.get('gradient_accumulation_steps', 'N/A')}",
            f"Sequence Length: {metrics.get('sequence_length', 'N/A')}",
            f"Training Time: {metrics.get('training_time', 'N/A'):.2f} seconds" if metrics.get('training_time') else "Training Time: N/A"
        ]
        
        fig.text(0.5, 0.01, "\n".join(metadata_text), ha='center', fontsize=14, 
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=1'))
        
        # Adjust layout and save figure
        plt.tight_layout(rect=[0, 0.04, 1, 0.96])
        combined_path = os.path.join(output_dir, f"{dataset_name}_combined_{timestamp}.png")
        plt.savefig(combined_path, dpi=self.dpi)
        plt.close()
        
        return combined_path
    
    def _plot_checkpoint_progress(self, metrics, dataset_name, output_dir, timestamp):
        """Create visualization of checkpoint progress"""
        if not metrics.get('checkpoints'):
            return None
            
        plt.figure(figsize=(14, 8))
        
        steps = [ckpt['step'] for ckpt in metrics['checkpoints']]
        losses = [ckpt['loss'] for ckpt in metrics['checkpoints']]
        
        plt.plot(steps, losses, '-o', linewidth=2, markersize=8, color=self.colors['checkpoint'])
        
        plt.title(f'Training Progress by Checkpoint - {dataset_name}', fontsize=16)
        plt.xlabel('Training Step', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add annotations for best checkpoint
        if losses:
            best_idx = np.argmin(losses)
            best_step = steps[best_idx]
            best_loss = losses[best_idx]
            plt.annotate(f'Best: {best_loss:.4f} (Step {best_step})',
                xy=(best_step, best_loss), 
                xytext=(best_step, best_loss * 1.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
        
        # Save the figure
        checkpoint_path = os.path.join(output_dir, f"{dataset_name}_checkpoints_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(checkpoint_path, dpi=self.dpi)
        plt.close()
        
        return checkpoint_path
        
    def create_comparison_plot(self, metrics_by_dataset: Dict[str, Dict[str, List[float]]], metric_name: str):
        """Create a comparison plot for a specific metric across datasets"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        plt.figure(figsize=(14, 10))
        
        for dataset_name, metrics in metrics_by_dataset.items():
            if metric_name in metrics and len(metrics[metric_name]) > 0:
                epochs = range(1, len(metrics[metric_name]) + 1)
                plt.plot(epochs, metrics[metric_name], '-', linewidth=2, label=dataset_name)
                
        plt.title(f'Comparison of {metric_name.capitalize()} Across Datasets', fontsize=18)
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel(metric_name.capitalize(), fontsize=16)
        plt.legend(fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        
        # Add a table with final values
        table_data = []
        for dataset_name, metrics in metrics_by_dataset.items():
            if metric_name in metrics and len(metrics[metric_name]) > 0:
                final_value = metrics[metric_name][-1]
                best_value = min(metrics[metric_name]) if metric_name in ['loss', 'perplexity'] else max(metrics[metric_name])
                table_data.append([dataset_name, f"{final_value:.4f}", f"{best_value:.4f}"])
        
        if table_data:
            # Add table as a text box
            header = ["Dataset", f"Final {metric_name}", f"Best {metric_name}"]
            table_text = []
            
            # Format header
            header_line = " | ".join(header)
            table_text.append(header_line)
            table_text.append("-" * len(header_line))
            
            # Format data rows
            for row in table_data:
                table_text.append(" | ".join(row))
            
            full_text = "\n".join(table_text)
            plt.figtext(0.5, 0.01, full_text, fontsize=14, ha='center', va='bottom',
                       bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=1'))
        
        # Save the figure with higher resolution
        comparison_path = os.path.join(self.comparisons_dir, f"comparison_{metric_name}_{timestamp}.png")
        plt.tight_layout(rect=[0, 0.1, 1, 0.98])
        plt.savefig(comparison_path, dpi=self.dpi)
        plt.close()
        
        print(f"Comparison visualization saved to {comparison_path}")
        return comparison_path
        
    def create_final_report(self, metrics_by_dataset: Dict[str, Dict[str, Any]]):
        """Create a final HTML report comparing all datasets"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f"training_report_{timestamp}.html")
        
        with open(report_path, 'w') as f:
            f.write("<html><head><title>Training Report</title>\n")
            f.write("<style>body{font-family:Arial,sans-serif; margin:20px;} table{border-collapse:collapse; width:100%;} ")
            f.write("th,td{text-align:left; padding:8px; border:1px solid #ddd;} ")
            f.write("th{background-color:#f2f2f2;} tr:nth-child(even){background-color:#f9f9f9;} ")
            f.write("img{max-width:900px; margin:10px;}</style>\n")
            f.write("</head><body>\n")
            f.write("<h1>Training Report</h1>\n")
            f.write(f"<p>Generated on: {timestamp}</p>\n")
            
            # Add comparison metrics table
            f.write("<h2>Metrics Comparison</h2>\n")
            f.write("<table>\n")
            f.write("<tr><th>Dataset</th><th>Final Loss</th><th>Best Val Loss</th>")
            f.write("<th>Final Accuracy</th><th>Final Perplexity</th>")
            f.write("<th>Batch Size</th><th>Sequence Length</th><th>Training Time</th></tr>\n")
            
            for dataset_name, metrics in metrics_by_dataset.items():
                final_loss = metrics.get('loss', [])[-1] if metrics.get('loss', []) else "N/A"
                best_val_loss = min(metrics.get('val_loss', [float('inf')])) if metrics.get('val_loss', []) else "N/A"
                final_acc = metrics.get('accuracy', [])[-1] if metrics.get('accuracy', []) else "N/A"
                final_perp = metrics.get('perplexity', [])[-1] if metrics.get('perplexity', []) else "N/A"
                batch_size = metrics.get('batch_size', "N/A")
                seq_len = metrics.get('sequence_length', "N/A")
                train_time = metrics.get('training_time', "N/A")
                
                f.write("<tr>")
                f.write(f"<td>{dataset_name}</td>")
                f.write(f"<td>{final_loss:.4f}</td>" if isinstance(final_loss, (int, float)) else f"<td>{final_loss}</td>")
                f.write(f"<td>{best_val_loss:.4f}</td>" if isinstance(best_val_loss, (int, float)) else f"<td>{best_val_loss}</td>")
                f.write(f"<td>{final_acc:.4f}</td>" if isinstance(final_acc, (int, float)) else f"<td>{final_acc}</td>")
                f.write(f"<td>{final_perp:.4f}</td>" if isinstance(final_perp, (int, float)) else f"<td>{final_perp}</td>")
                f.write(f"<td>{batch_size}</td>")
                f.write(f"<td>{seq_len}</td>")
                f.write(f"<td>{train_time:.2f}s</td>" if isinstance(train_time, (int, float)) else f"<td>{train_time}</td>")
                f.write("</tr>\n")
                
            f.write("</table>\n")
            
            # Create comparison plots
            metrics_list = ['loss', 'accuracy', 'perplexity']
            for metric in metrics_list:
                valid_datasets = {k: v for k, v in metrics_by_dataset.items() if metric in v and v[metric]}
                if valid_datasets:
                    comparison_path = self.create_comparison_plot(valid_datasets, metric)
                    if comparison_path:
                        f.write(f"<h2>{metric.capitalize()} Comparison</h2>\n")
                        rel_path = os.path.relpath(comparison_path, self.output_dir)
                        f.write(f"<img src='{rel_path}' alt='{metric.capitalize()} Comparison'/>\n")
            
            # Add links to individual dataset reports
            f.write("<h2>Individual Dataset Reports</h2>\n")
            f.write("<ul>\n")
            for dataset_name in metrics_by_dataset.keys():
                # Find the latest visualization directory for this dataset
                vis_dirs = [d for d in os.listdir(self.plots_dir) if d.startswith(f"{dataset_name}_")]
                if vis_dirs:
                    latest_dir = sorted(vis_dirs)[-1]
                    index_path = os.path.join(self.plots_dir, latest_dir, "index.html")
                    if os.path.exists(index_path):
                        rel_path = os.path.relpath(index_path, self.output_dir)
                        f.write(f"<li><a href='{rel_path}'>{dataset_name}</a></li>\n")
            
            f.write("</ul>\n")
            f.write("</body></html>\n")
            
        print(f"Final training report saved to {report_path}")
        return report_path

def calculate_metrics(model, data_batches, device):
    """Calculate evaluation metrics for a model on the given data"""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    
    total_loss = 0
    total_correct = 0
    total_predictions = 0
    
    with torch.no_grad():
        for inputs, targets in data_batches:
            # Move to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Get predictions
            outputs, _ = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            # Track loss
            total_loss += loss.item() * inputs.size(0)
            
            # Track accuracy (top-1)
            predictions = outputs.view(-1, outputs.size(-1)).argmax(dim=1)
            target_indices = targets.view(-1)
            correct_predictions = (predictions == target_indices).sum().item()
            
            total_correct += correct_predictions
            total_predictions += target_indices.size(0)
    
    # Calculate metrics
    avg_loss = total_loss / total_predictions if total_predictions > 0 else 0
    accuracy = total_correct / total_predictions if total_predictions > 0 else 0
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'perplexity': perplexity
    }

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Unified Generation Pipeline")
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='generate', choices=['train', 'generate', 'evaluate', 'interactive'],
                        help='Mode to run the model in')
    
    # Dataset selection
    parser.add_argument('--dataset', type=str, default='writing_prompts',
                        choices=['writing_prompts', 'persona_chat', 'both', 'pile', 'openassistant', 'gpteacher'],
                        help='Dataset to use')
    
    # For The Pile dataset
    parser.add_argument('--pile-subset', type=str, default=None,
                       help='Specific subset of The Pile (e.g., "pubmed", "github", "europarl")')
                       
    # Training settings
    parser.add_argument('--train-type', type=str, default='text', choices=['text', 'code', 'both'],
                        help='Type of model to train (text, code, or both)')
    parser.add_argument("--epochs", type=int, default=50, 
                      help="Number of training epochs")
    parser.add_argument("--save-model", action="store_true",
                      help="Save the trained models")
    
    # Generation options
    parser.add_argument("--gen-type", choices=["text", "code"], default="text",
                      help="Type of content to generate")
    parser.add_argument("--prompt", default="Hello",
                      help="Text prompt to start generation")
    parser.add_argument("--length", type=int, default=100,
                      help="Number of tokens/characters to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                      help="Sampling temperature (higher = more random)")
    parser.add_argument("--use-tokenizer", action="store_true",
                      help="Use tokenizer for generation instead of character-level")
    
    # Preprocessing options
    parser.add_argument("--preprocess-output", default="preprocessed_data",
                      help="Directory to save preprocessed data")
    parser.add_argument("--analyze", action="store_true",
                      help="Analyze token distribution when preprocessing")
    parser.add_argument("--max-samples", type=int, default=100,
                      help="Maximum number of samples to process (for preprocessing)")
    parser.add_argument("--min-length", type=int, default=10,
                      help="Minimum sequence length for preprocessing")
    parser.add_argument("--max-length", type=int, default=100,
                      help="Maximum sequence length for preprocessing")
    
    # Evaluation options
    parser.add_argument("--eval-split", type=float, default=0.2,
                      help="Fraction of data to use for evaluation")
    parser.add_argument("--metrics", nargs="+", default=["perplexity", "accuracy"],
                      help="Metrics to compute during evaluation")
    
    # Model paths
    parser.add_argument("--model-dir", default="models",
                      help="Directory for model files")
    parser.add_argument("--text-model", default="text_gen_model.pt",
                      help="Filename for text generator model")
    parser.add_argument("--code-model", default="code_gen_model.pt",
                      help="Filename for code generator model")
    
    # Deepseek options
    parser.add_argument("--use-deepseek", action="store_true",
                      help="Use deepseek-coder for code generation and training")
    parser.add_argument("--deepseek-batch-size", type=int, default=4,
                      help="Batch size for deepseek training")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                      help="Learning rate for deepseek fine-tuning")
    parser.add_argument("--sequence-length", type=int, default=512,
                      help="Maximum sequence length for deepseek")
    parser.add_argument("--warmup-steps", type=int, default=100,
                      help="Number of warmup steps for deepseek")
    parser.add_argument("--code-subset", default="python",
                      help="Language subset for code dataset (for deepseek)")
    parser.add_argument("--all-subsets", type=lambda x: (str(x).lower() == 'true'), default=True,
                      help="Whether to use all language subsets for code (default: True)")
    parser.add_argument("--force-gpu", action="store_true", default=True,
                      help="Force the use of GPU (MPS for Apple Silicon, CUDA for NVIDIA)")
    
    return parser.parse_args()

def train_text_generator(dataset_name: str, epochs: int = 50, model_path: str = None,
                        learning_rate: float = 0.002, clip_value: float = 5.0,
                        use_scheduler: bool = False, force_gpu: bool = True,
                        validation_split: float = 0.2, create_visualizations: bool = True,
                        batch_size: int = 16, sequence_length: int = 2048, 
                        gradient_accumulation_steps: int = 4) -> Tuple[torch.nn.Module, Dict[str, List[float]]]:
    """Train the text generator model with enhanced configuration and evaluation"""
    print(f"\nTraining text generator on {dataset_name} dataset...")
    
    # Set up checkpoint directory and log
    checkpoint_dir = os.path.join(os.path.dirname(model_path) if model_path else "checkpoints", f"{dataset_name}_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_log = os.path.join(checkpoint_dir, f"{dataset_name}_training_log.txt")
    with open(checkpoint_log, 'w') as f:
        f.write(f"Training started at: {datetime.datetime.now().isoformat()}\n")
        f.write(f"Dataset: {dataset_name}, Batch size: {batch_size}, Sequence length: {sequence_length}\n")
        f.write(f"Gradient accumulation steps: {gradient_accumulation_steps}, Learning rate: {learning_rate}\n\n")
    
    def log_checkpoint(message):
        with open(checkpoint_log, 'a') as f:
            timestamp = datetime.datetime.now().isoformat()
            f.write(f"[{timestamp}] {message}\n")
        print(f"CHECKPOINT: {message}")
    
    log_checkpoint(f"Initializing training for {dataset_name} dataset")

    # Initialize dataset processor
    dataset_processor = DatasetProcessor()

    try:
        # Load preprocessed data
        log_checkpoint("Loading preprocessed data")
        data = dataset_processor.load_preprocessed_data(dataset_name)
        if not data:
            log_checkpoint(f"Error: No data loaded for {dataset_name}")
            return None, None

        # Get vocabulary size
        vocab_size = data.get('vocab_size', 0)
        if vocab_size == 0:
            log_checkpoint("Error: Invalid vocabulary size")
            return None, None
        
        log_checkpoint(f"Vocabulary size: {vocab_size}")

        # Create model with enhanced configuration for writing prompts
        if dataset_name == "writing_prompts":
            hidden_size = 256  # Increased hidden size
            num_layers = 3     # Increased number of layers
        else:
            hidden_size = 128
            num_layers = 2

        log_checkpoint(f"Creating model with hidden_size={hidden_size}, num_layers={num_layers}")
        model = CombinedModel(
            input_size=vocab_size,
            hidden_size=hidden_size,
            output_size=vocab_size,
            num_layers=num_layers
        )

        # Initialize optimizer with learning rate
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Add learning rate scheduler if requested
        if use_scheduler:
            log_checkpoint("Using learning rate scheduler with patience=2")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=2, verbose=True
            )

        # Check for GPU
        device = torch.device("cpu")
        if force_gpu:
            # Try to use MPS (Metal Performance Shaders) for Apple Silicon
            if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                log_checkpoint("Using MPS (Apple Silicon GPU)")
                device = torch.device("mps")
            # Fall back to CUDA if available
            elif torch.cuda.is_available():
                log_checkpoint(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
                device = torch.device("cuda")
            else:
                log_checkpoint("Warning: GPU requested but neither MPS nor CUDA is available. Falling back to CPU.")
        elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            log_checkpoint("Using MPS (Apple Silicon GPU)")
            device = torch.device("mps")
        elif torch.cuda.is_available():
            log_checkpoint(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            device = torch.device("cuda")
        else:
            log_checkpoint("Using CPU (no GPU available)")

        log_checkpoint(f"Moving model to device: {device}")
        model = model.to(device)

        # Get batches and split into train/validation
        batches = data.get('batches', [])
        if not batches:
            log_checkpoint("Error: No batches found in data")
            return None, None
            
        # Split data for validation
        if validation_split > 0:
            val_size = max(1, int(len(batches) * validation_split))
            train_batches = batches[:-val_size]
            val_batches = batches[-val_size:]
            log_checkpoint(f"Split data: {len(train_batches)} training batches, {len(val_batches)} validation batches")
        else:
            train_batches = batches
            val_batches = []
            log_checkpoint(f"Using all {len(train_batches)} batches for training (no validation)")

        # Determine steps for gradient accumulation
        total_steps = epochs * len(train_batches)
        steps_per_epoch = len(train_batches)
        effective_batch_size = batch_size * gradient_accumulation_steps
        log_checkpoint(f"Using gradient accumulation: {gradient_accumulation_steps} steps")
        log_checkpoint(f"Effective batch size: {effective_batch_size}")
        log_checkpoint(f"Total training steps: {total_steps}")
        
        # Create checkpoint savings schedule
        checkpoint_frequency = max(1, steps_per_epoch // 5)  # Save 5 checkpoints per epoch
        log_checkpoint(f"Will save checkpoints every {checkpoint_frequency} steps")

        # Training loop
        log_checkpoint(f"Starting training on {len(train_batches)} batches for {epochs} epochs")
        model.train()
        criterion = torch.nn.CrossEntropyLoss()

        # Enhanced metrics tracking
        metrics = {
            'epoch_losses': [],
            'final_loss': None,
            'training_time': None,
            'device_used': str(device),
            'vocab_size': vocab_size,
            'num_batches': len(batches),
            'batch_size': batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'effective_batch_size': effective_batch_size,
            'sequence_length': sequence_length,
            'dataset': dataset_name,
            'model_type': 'text',
            'timestamp': datetime.datetime.now().isoformat(),
            'training_progress': [],
            'learning_rates': [] if use_scheduler else None,
            'model_config': {
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'learning_rate': learning_rate,
                'clip_value': clip_value,
                'use_scheduler': use_scheduler
            },
            'loss': [],  # For visualization
            'val_loss': [],  # For visualization
            'accuracy': [],  # For visualization
            'val_accuracy': [],  # For visualization
            'perplexity': [],  # For visualization
            'val_perplexity': [],  # For visualization
            'checkpoints': []  # Store checkpoint information
        }

        import time
        start_time = time.time()
        global_step = 0
        best_val_loss = float('inf')
        early_stopping_patience = 5
        early_stopping_counter = 0

        for epoch in range(epochs):
            epoch_loss = 0
            batch_count = 0
            
            log_checkpoint(f"Starting epoch {epoch+1}/{epochs}")
            optimizer.zero_grad()  # Zero gradients at the beginning of epoch

            for batch_idx, (input_batch, target_batch) in enumerate(tqdm(train_batches, desc=f"Epoch {epoch+1}/{epochs}")):
                # Move data to device
                input_batch = input_batch.to(device)
                target_batch = target_batch.to(device)

                # Forward pass
                output, _ = model(input_batch)

                # Calculate loss
                loss = criterion(output.view(-1, output.size(-1)), target_batch.view(-1))
                
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
                
                # Backward pass
                loss.backward()

                # Clip gradients
                if clip_value > 0 and (batch_idx + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                
                # Update weights only after accumulating gradients
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # Increment global step
                    global_step += 1
                    
                    # Log every 50 steps
                    if global_step % 50 == 0:
                        log_checkpoint(f"Step {global_step}: Loss = {loss.item() * gradient_accumulation_steps:.4f}")
                    
                    # Save checkpoint periodically
                    if global_step % checkpoint_frequency == 0:
                        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{global_step}.pt")
                        torch.save({
                            'epoch': epoch,
                            'global_step': global_step,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss.item() * gradient_accumulation_steps,
                            'vocab_size': vocab_size
                        }, checkpoint_path)
                        log_checkpoint(f"Saved checkpoint to {checkpoint_path}")
                        
                        # Add checkpoint info to metrics
                        metrics['checkpoints'].append({
                            'step': global_step,
                            'path': checkpoint_path,
                            'loss': loss.item() * gradient_accumulation_steps,
                            'epoch': epoch
                        })

                # Track metrics (use the unnormalized loss for reporting)
                epoch_loss += loss.item() * gradient_accumulation_steps
                batch_count += 1

            # Handle any remaining gradients
            if len(train_batches) % gradient_accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Calculate epoch statistics
            avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
            metrics['epoch_losses'].append(avg_epoch_loss)
            
            # Calculate training metrics
            log_checkpoint("Calculating training metrics")
            train_metrics = calculate_metrics(model, train_batches[:min(10, len(train_batches))], device)
            metrics['loss'].append(train_metrics['loss'])
            metrics['accuracy'].append(train_metrics['accuracy'])
            metrics['perplexity'].append(train_metrics['perplexity'])
            
            # Calculate validation metrics if available
            val_loss = None
            if val_batches:
                log_checkpoint("Calculating validation metrics")
                val_metrics = calculate_metrics(model, val_batches, device)
                metrics['val_loss'].append(val_metrics['loss'])
                metrics['val_accuracy'].append(val_metrics['accuracy'])
                metrics['val_perplexity'].append(val_metrics['perplexity'])
                val_loss = val_metrics['loss']
                
                log_checkpoint(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}, "
                      f"Val Loss: {val_metrics['loss']:.4f}, "
                      f"Val Accuracy: {val_metrics['accuracy']:.4f}, "
                      f"Val Perplexity: {val_metrics['perplexity']:.4f}")
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stopping_counter = 0
                    
                    # Save best model
                    if model_path:
                        best_model_path = model_path.replace('.pt', '_best.pt')
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch,
                            'loss': best_val_loss,
                            'vocab_size': vocab_size,
                            'metrics': metrics
                        }, best_model_path)
                        log_checkpoint(f"Saved best model with val_loss={best_val_loss:.4f} to {best_model_path}")
                else:
                    early_stopping_counter += 1
                    log_checkpoint(f"Validation loss did not improve. Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
                    
                    if early_stopping_counter >= early_stopping_patience:
                        log_checkpoint(f"Early stopping triggered after {epoch+1} epochs")
                        break
            else:
                log_checkpoint(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")

            metrics['training_progress'].append({
                'epoch': epoch + 1,
                'loss': avg_epoch_loss,
                'val_loss': val_loss,
                'timestamp': datetime.datetime.now().isoformat()
            })

            # Update learning rate if using scheduler
            if use_scheduler:
                val_loss = val_metrics['loss'] if val_batches else avg_epoch_loss
                scheduler.step(val_loss)
                current_lr = optimizer.param_groups[0]['lr']
                metrics['learning_rates'].append(current_lr)
                log_checkpoint(f"Learning rate updated to: {current_lr}")
                
            # Save model after each epoch
            if model_path:
                epoch_model_path = model_path.replace('.pt', f'_epoch_{epoch+1}.pt')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': avg_epoch_loss,
                    'vocab_size': vocab_size,
                    'metrics': metrics
                }, epoch_model_path)
                log_checkpoint(f"Saved model at epoch {epoch+1} to {epoch_model_path}")

        # Final metrics
        training_time = time.time() - start_time
        metrics['final_loss'] = metrics['epoch_losses'][-1] if metrics['epoch_losses'] else None
        metrics['training_time'] = training_time
        log_checkpoint(f"Training completed in {training_time:.2f} seconds")

        # Create visualizations if requested
        if create_visualizations:
            log_checkpoint("Generating training visualizations")
            visualizer = TrainingVisualizer(output_dir="visualizations")
            visualizer.visualize_training(dataset_name, metrics)

        # Save final model if requested
        if model_path:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epochs,
                'loss': metrics['final_loss'],
                'vocab_size': vocab_size,
                'metrics': metrics
            }, model_path)
            log_checkpoint(f"Final model saved to {model_path}")
            
            # Save metrics
            metrics_path = os.path.join(os.path.dirname(model_path), f"{dataset_name}_metrics.json")
            with open(metrics_path, 'w') as f:
                # Filter out tensors and other non-serializable objects
                serializable_metrics = {}
                for k, v in metrics.items():
                    if isinstance(v, (list, dict, str, int, float, bool)) or v is None:
                        serializable_metrics[k] = v
                    elif isinstance(v, torch.Tensor):
                        serializable_metrics[k] = v.item() if v.numel() == 1 else v.tolist()
                json.dump(serializable_metrics, f, indent=2)
            log_checkpoint(f"Training metrics saved to {metrics_path}")
            
            # Create a completion marker
            with open(os.path.join(os.path.dirname(model_path), f"{dataset_name}_TRAINING_COMPLETE.txt"), 'w') as f:
                f.write(f"Training completed at: {datetime.datetime.now().isoformat()}\n")
                f.write(f"Final loss: {metrics['final_loss']}\n")
                if 'val_loss' in metrics and metrics['val_loss']:
                    f.write(f"Best validation loss: {best_val_loss}\n")
                f.write(f"Training time: {training_time:.2f} seconds\n")

        log_checkpoint("Training process completed successfully")
        return model, metrics

    except Exception as e:
        error_msg = f"Error during training: {e}"
        log_checkpoint(error_msg)
        import traceback
        traceback_str = traceback.format_exc()
        log_checkpoint(f"Traceback: {traceback_str}")
        
        # Save error information
        if model_path:
            error_path = os.path.join(os.path.dirname(model_path), f"{dataset_name}_error.json")
            with open(error_path, 'w') as f:
                json.dump({
                    'error': str(e),
                    'traceback': traceback_str,
                    'timestamp': datetime.datetime.now().isoformat()
                }, f, indent=2)
            log_checkpoint(f"Error information saved to {error_path}")
            
        return None, None

def train_code_generator(dataset_name="writing_prompts", epochs=50, model_path=None):
    """Train the CodeGenerator model with preprocessed data"""
    # For code generation, we use the writing prompts dataset as a proxy for code
    return train_text_generator(dataset_name, epochs, model_path)

def load_model(model_path: str, model_type: str = "text") -> Tuple[torch.nn.Module, int]:
    """Load a trained model"""
    # Define all possible model paths
    possible_paths = [
        model_path,  # Original path
        os.path.join("models", os.path.basename(model_path)),  # models directory
        os.path.join("src/generative_ai_module/models", os.path.basename(model_path)),  # module models directory
        os.path.join(os.path.dirname(__file__), "models", os.path.basename(model_path)),  # relative module models
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", os.path.basename(model_path))  # project models
    ]

    # Try each path
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loading model from {path}")
            try:
                checkpoint = torch.load(path)

                # Get model parameters
                if (
                    isinstance(checkpoint, dict)
                    and 'model_state_dict' in checkpoint
                ):
                    state_dict = checkpoint['model_state_dict']
                    vocab_size = checkpoint.get('vocab_size', 104)
                    metrics = checkpoint.get('metrics', {})
                else:
                    state_dict = checkpoint
                    vocab_size = 104
                    metrics = {}
                # Create model
                model = CombinedModel(
                    input_size=vocab_size,
                    hidden_size=128,
                    output_size=vocab_size,
                    num_layers=2
                )

                # Load state dict
                model.load_state_dict(state_dict)
                model.eval()

                print(f"Successfully loaded {model_type} model with vocabulary size {vocab_size}")
                return model, vocab_size

            except Exception as e:
                print(f"Error loading model from {path}: {e}")
                continue

    # If no model found, try to find any model file
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    if os.path.exists(model_dir):
        if model_files := [
            f for f in os.listdir(model_dir) if f.endswith('.pt')
        ]:
            print(f"Found existing model files: {model_files}")
            print("Please specify the correct model file using --model-path")

    raise FileNotFoundError(
        "Model file not found. Please train a model first or specify the correct path."
    )

def generate_with_tokenizer(model, tokenizer, prompt, max_length=50, temperature=0.7):
    """Generate text using the model and tokenizer"""
    # Encode the initial prompt
    prompt_tokens = tokenizer.encode(prompt)
    if not prompt_tokens:
        prompt_tokens = [0] * 5  # Default padding
    elif len(prompt_tokens) < 5:
        # Pad for context window
        prompt_tokens = prompt_tokens + [0] * (5 - len(prompt_tokens))
    
    print(f"Encoded prompt: {prompt_tokens}")
    
    # Generate text
    model.eval()
    generated = []
    input_tensor = torch.tensor([prompt_tokens], dtype=torch.long)
    hidden = None
    
    with torch.no_grad():
        for _ in range(max_length):
            output, hidden = model(input_tensor, hidden)
            
            # Apply temperature sampling
            logits = output.squeeze().div(temperature)
            probs = torch.softmax(logits, dim=0)
            next_token = torch.multinomial(probs, 1).item()
            
            generated.append(next_token)
            input_tensor = torch.tensor([[next_token]], dtype=torch.long)
    
    # Decode and return
    decoded_text = tokenizer.decode(generated)
    return decoded_text, generated

def generate_with_char_model(model, prompt, vocab_size, max_length=50, temperature=0.7):
    """Generate text using character-level model without tokenizer"""
    # Create a simple TextGenerator just for generation purposes
    gen = TextGenerator()
    gen.model = model  # Use the loaded model

    # If vocab size doesn't match the default, create appropriate mappings
    if vocab_size != len(gen.all_chars):
        print(f"Note: Model vocabulary size ({vocab_size}) differs from default character set")

    return gen.generate(
        initial_str=prompt, pred_len=max_length, temperature=temperature
    )

def save_evaluation_metrics(metrics: Dict[str, Any], dataset_name: str, model_type: str = "text") -> None:
    """
    Save evaluation metrics to a JSON file with timestamp.
    
    Args:
        metrics: Dictionary of evaluation metrics
        dataset_name: Name of the dataset the model was evaluated on
        model_type: Type of model (text or code)
    """
    # Use the storage path utility to get the correct path
    metrics_dir = get_storage_path("metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # Format timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create metrics file path
    metrics_file = os.path.join(metrics_dir, f"{model_type}_{dataset_name}_{timestamp}.json")

    # Prepare serializable metrics
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, str, bool)):
            serializable_metrics[key] = value
        elif isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
            serializable_metrics[key] = float(value)
        elif isinstance(value, list):
            serializable_metrics[key] = [float(v) if isinstance(v, (np.float32, np.float64, np.int32, np.int64)) else v for v in value]
        elif isinstance(value, dict):
            serializable_metrics[key] = {
                k: float(v) if isinstance(v, (np.float32, np.float64, np.int32, np.int64)) else v
                for k, v in value.items()
            }
        else:
            serializable_metrics[key] = str(value)

    # Add metadata
    serializable_metrics |= {
        "dataset": dataset_name,
        "model_type": model_type,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    # Save to file
    with open(metrics_file, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)

    print(f"Evaluation metrics saved to {metrics_file}")
    
    # Sync metrics to Google Drive
    sync_to_gdrive("metrics")

def preprocess_data(args: argparse.Namespace) -> Dict[str, Any]:
    """Preprocess data based on arguments"""
    from .dataset_processor import DatasetProcessor
    from .improved_preprocessing import ImprovedPreprocessor

    print("Preprocessing data...")
    
    # Create processors
    dataset_processor = DatasetProcessor()
    improved_processor = ImprovedPreprocessor()
    
    # Prepare based on dataset choice
    if args.dataset == 'writing_prompts':
        print("Loading Writing Prompts dataset")
        data = improved_processor.process_dataset(
            'writing_prompts',
            max_samples=args.max_samples
        )
    elif args.dataset == 'persona_chat':
        print("Loading Persona Chat dataset")
        data = improved_processor.process_dataset(
            'persona_chat',
            max_samples=args.max_samples
        )
    elif args.dataset == 'pile':
        print(f"Loading Pile dataset{f' (subset: {args.pile_subset})' if args.pile_subset else ''}")
        data = dataset_processor.prepare_dataset(
            'pile',
            split='train',
            max_samples=args.max_samples,
            subset=args.pile_subset,
            batch_size=32
        )
    elif args.dataset == 'openassistant':
        print("Loading OpenAssistant dataset")
        data = dataset_processor.prepare_dataset(
            'openassistant',
            split='train',
            max_samples=args.max_samples,
            batch_size=32
        )
    elif args.dataset == 'gpteacher':
        print("Loading GPTeacher dataset")
        data = dataset_processor.prepare_dataset(
            'gpteacher',
            split='train',
            max_samples=args.max_samples,
            batch_size=32
        )
    elif args.dataset == 'both':
        print("Loading both Writing Prompts and Persona Chat datasets")
        writing_data = improved_processor.process_dataset(
            'writing_prompts',
            max_samples=args.max_samples // 2
        )
        persona_data = improved_processor.process_dataset(
            'persona_chat',
            max_samples=args.max_samples // 2
        )
        
        # Combine datasets (simple approach - just append)
        combined_batches = writing_data.get('batches', []) + persona_data.get('batches', [])
        data = {
            'dataset_name': 'combined',
            'batches': combined_batches
        }
    else:
        print(f"Using dataset from path: {args.dataset}")
        data = dataset_processor.prepare_local_dataset(
            args.dataset,
            sequence_length=100,
            batch_size=32
        )
    
    # Ensure batches are available
    if 'batches' not in data or not data['batches']:
        print("Warning: No batches found in preprocessed data.")
        return {}
        
    # Return processed data
    return data
    
def preprocess_datasets(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    """Preprocess multiple datasets if needed"""
    datasets = {}

    if args.dataset in ['writing_prompts', 'both']:
        print("\nPreprocessing Writing Prompts dataset...")
        dataset_args = argparse.Namespace(**vars(args))
        dataset_args.dataset = 'writing_prompts'
        datasets['writing_prompts'] = preprocess_data(dataset_args)

    if args.dataset in ['persona_chat', 'both']:
        print("\nPreprocessing Persona Chat dataset...")
        dataset_args = argparse.Namespace(**vars(args))
        dataset_args.dataset = 'persona_chat'
        datasets['persona_chat'] = preprocess_data(dataset_args)

    if args.dataset == 'pile':
        print("\nPreprocessing Pile dataset...")
        datasets['pile'] = preprocess_data(args)

    if args.dataset == 'openassistant':
        print("\nPreprocessing OpenAssistant dataset...")
        datasets['openassistant'] = preprocess_data(args)

    if args.dataset == 'gpteacher':
        print("\nPreprocessing GPTeacher dataset...")
        datasets['gpteacher'] = preprocess_data(args)

    return datasets

def train_on_datasets(args: argparse.Namespace, datasets: Dict[str, Dict[str, Any]]):
    """Train models on multiple datasets simultaneously"""
    print("Starting training for all datasets...")

    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)

    # Handle code dataset with deepseek if deepseek is enabled
    if args.use_deepseek and args.train_type in ["code", "both"]:
        from generative_ai_module.code_generator import CodeGenerator

        print("\n===== Training deepseek-coder model on code dataset =====")
        # Get the code dataset
        code_data = datasets.get("code", {})
        train_dataset = code_data.get("train_dataset")
        eval_dataset = code_data.get("valid_dataset")

        if train_dataset is None or eval_dataset is None:
            print("Error: Missing code datasets for deepseek. Attempting to load directly.")
            from generative_ai_module.code_preprocessing import load_and_preprocess_dataset
            train_dataset, eval_dataset = load_and_preprocess_dataset(
                max_samples=args.max_samples,
                sequence_length=args.sequence_length,
                subset=args.code_subset,
                all_subsets=args.all_subsets
            )

        if train_dataset is None or eval_dataset is None:
            print("Error: Failed to load code datasets for deepseek training.")
        else:
            # Initialize CodeGenerator with deepseek
            code_gen = CodeGenerator(use_deepseek=True)

            # Set output directory
            output_dir = os.path.join(args.model_dir, "deepseek_finetuned")

            # Fine-tune the model
            training_metrics = code_gen.fine_tune_deepseek(
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                output_dir=output_dir,
                epochs=args.epochs,
                batch_size=args.deepseek_batch_size,
                sequence_length=args.sequence_length,
                learning_rate=args.learning_rate,
                warmup_steps=args.warmup_steps,
                subset=args.code_subset,
                all_subsets=args.all_subsets
            )

            print(f"Deepseek training completed with metrics: {training_metrics}")

    # Train text generator on both datasets
    # Always train on persona_chat and writing_prompts datasets
    for dataset_name, data in datasets.items():
        if dataset_name == "persona_chat" or dataset_name == "writing_prompts":
            print(f"\n===== Training on {dataset_name} dataset =====")

            # Split data for evaluation
            train_data, eval_data = split_data(data, args.eval_split)

            # Train text generator
            if args.train_type in ["text", "both"]:
                model_path = os.path.join(args.model_dir, f"text_gen_{dataset_name}.pt")

                # Enhanced training configuration for writing prompts
                if dataset_name == "writing_prompts":
                    # Increase epochs for better training
                    epochs = args.epochs * 2
                    # Use smaller learning rate for better convergence
                    learning_rate = 0.001
                    # Use gradient clipping to prevent exploding gradients
                    clip_value = 1.0
                    # Use learning rate scheduler
                    use_scheduler = True
                else:
                    epochs = args.epochs
                    learning_rate = 0.002
                    clip_value = 5.0
                    use_scheduler = False

                text_model, vocab_size = train_text_generator(
                    dataset_name=dataset_name,
                    epochs=epochs,
                    model_path=model_path if args.save_model else None,
                    learning_rate=learning_rate,
                    clip_value=clip_value,
                    use_scheduler=use_scheduler,
                    force_gpu=args.force_gpu
                )

                # Evaluate text model
                if text_model:
                    print(f"\nEvaluating text model on {dataset_name}:")
                    metrics = evaluate_model(text_model, eval_data, args.metrics)
                    save_evaluation_metrics(metrics, dataset_name, "text")

        # Train traditional code generator (only on writing_prompts)
        if args.train_type in ["code", "both"] and dataset_name == "writing_prompts" and not args.use_deepseek:
            model_path = os.path.join(args.model_dir, args.code_model)
            code_model, vocab_size = train_code_generator(
                dataset_name=dataset_name,
                epochs=args.epochs,
                model_path=model_path if args.save_model else None
            )

            # Evaluate code model
            if code_model:
                print(f"\nEvaluating code model on {dataset_name}:")
                metrics = evaluate_model(code_model, eval_data, args.metrics)
                save_evaluation_metrics(metrics, dataset_name, "code")

def interactive_generation(model: torch.nn.Module, 
                         tokenizer: Any = None,
                         model_type: str = "text",
                         force_gpu: bool = True,
                         args: argparse.Namespace = None) -> None:
    """
    Provides an interactive prompt for generating text with the model.
    
    Args:
        model: The trained model
        tokenizer: Tokenizer for the model
        model_type: Type of model (text or code)
        force_gpu: Force GPU usage for generation
        args: Optional arguments from command line
    """
    # Always force GPU usage regardless of the parameter value
    force_gpu = True
    
    # Apply GPU configuration upfront
    import torch
    import os
    from .utils import setup_gpu_for_training, force_cuda_device
    
    print("⚡ Setting up GPU for interactive generation")
    
    # Set environment variables for GPU visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Force CUDA device if available
    device = force_cuda_device()
    print(f"Using device: {device}")
    
    # Make sure the model is on the right device
    if hasattr(model, 'to') and device.type != 'cpu':
        model = model.to(device)
    
    # Ensure we have usable args
    if args is None:
        class DefaultArgs:
            model_dir = "models"
            length = 100
            temperature = 0.7
            top_p = 0.95
        args = DefaultArgs()
    
    # Prepare generation parameters with higher defaults for interactive use
    length = getattr(args, 'length', 100)
    temperature = getattr(args, 'temperature', 0.7)
    top_p = getattr(args, 'top_p', 0.95)
    
    print("\n" + "="*80)
    print("Interactive Generation Mode")
    print("="*80)
    print("Type 'exit', 'quit', or Ctrl+C to end the session")
    print("Type 'settings' to adjust generation parameters")
    print("Type 'gpu' to force GPU usage explicitly")
    print(f"Generation Settings: length={length}, temperature={temperature}, top_p={top_p}")
    print("="*80 + "\n")
    
    # Get the text or code generator based on model type
    if model_type == "code":
        # For code generation, we'll use different defaults
        generate_func = lambda prompt, max_tokens, temp: generate_code(
            model, tokenizer, prompt, max_length=max_tokens, temperature=temp, top_p=top_p
        )
    else:
        # For text generation, we'll use different defaults
        generate_func = lambda prompt, max_tokens, temp: generate_text(
            model, tokenizer, prompt, max_length=max_tokens, temperature=temp
        )
        
    # Start the interactive loop
    while True:
        try:
            prompt = input("\nPrompt: ")
            
            # Handle special commands
            if prompt.lower() in ["exit", "quit"]:
                print("Exiting interactive mode")
                break
            
            elif prompt.lower() == "settings":
                try:
                    length = int(input(f"Enter max tokens (current: {length}): ") or length)
                    temperature = float(input(f"Enter temperature (current: {temperature}): ") or temperature)
                    top_p = float(input(f"Enter top_p (current: {top_p}): ") or top_p)
                    print(f"Updated settings: length={length}, temperature={temperature}, top_p={top_p}")
                except ValueError:
                    print("Invalid value entered. Settings unchanged.")
                continue
                
            elif prompt.lower() == "gpu":
                print("Enforcing GPU usage...")
                # Force CUDA device
                device = force_cuda_device()
                
                # Move model to device explicitly
                if hasattr(model, 'to') and device.type != 'cpu':
                    model = model.to(device)
                    
                print(f"Device after GPU enforcement: {device}")
                continue
                
            # Generate text or code
            start_time = time.time()
            
            # Enhanced prompt analysis for better generation
            enhanced_prompt = analyze_prompt(prompt) if hasattr(args, 'enhance_prompts') and args.enhance_prompts else prompt
            
            if enhanced_prompt != prompt:
                print("\nUsing enhanced prompt:")
                print("-" * 40)
                print(enhanced_prompt)
                print("-" * 40)
            
            # Generate with progress indication
            print("Generating...", end="", flush=True)
            generated_text = generate_func(enhanced_prompt, length, temperature)
            end_time = time.time()
            
            # Print generation time
            print(f" Done (took {end_time - start_time:.2f}s)")
            
            # Print generated text
            print("\nGenerated Output:")
            print("-" * 80)
            print(generated_text)
            print("-" * 80)
            
            # Offer to save the generated text
            save_option = input("Save this generated text? (y/n): ")
            if save_option.lower() == 'y':
                save_name = input("Enter filename (or press Enter for auto-generated name): ")
                
                if not save_name:
                    # Use first few words of prompt as filename
                    save_name = "_".join(prompt.split()[:5]).replace("/", "_")
                    save_name = f"{save_name}_{int(time.time())}"
                
                # Ensure the filename has the right extension
                if model_type == "code" and not any(save_name.endswith(ext) for ext in ['.py', '.js', '.cpp', '.c', '.java']):
                    save_name += '.py'  # Default to Python for code
                elif not save_name.endswith('.txt'):
                    save_name += '.txt'
                
                # Create outputs directory if it doesn't exist
                os.makedirs("outputs", exist_ok=True)
                
                # Save the generated text
                with open(f"outputs/{save_name}", 'w') as f:
                    f.write(generated_text)
                
                print(f"Saved to outputs/{save_name}")
            
        except KeyboardInterrupt:
            print("\nExiting interactive mode")
            break
        except Exception as e:
            print(f"Error during generation: {str(e)}")

def split_data(data: Dict[str, Any], eval_split: float = 0.2) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split data into training and evaluation sets"""
    # Shuffle the batches
    batches = data['batches']
    random.shuffle(batches)
    
    # Calculate split point
    split_idx = int(len(batches) * (1 - eval_split))
    
    # Split the batches
    train_batches = batches[:split_idx]
    eval_batches = batches[split_idx:]
    
    # Create copies of the data dictionary
    train_data = data.copy()
    eval_data = data.copy()
    
    # Update the batches
    train_data['batches'] = train_batches
    eval_data['batches'] = eval_batches
    
    print(f"\nSplit data into {len(train_batches)} training batches and {len(eval_batches)} evaluation batches")
    return train_data, eval_data

def main():
    """Main function for running the unified generation pipeline"""
    args = parse_args()
    
    # Track execution time
    start_time = time.time()
    
    # Set GPU performance optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("✅ Enabled cuDNN benchmark for better GPU performance")
    
    # Create pipeline instance
    pipeline = UnifiedGenerationPipeline(
        model_dir=args.model_dir,
        force_gpu=args.force_gpu
    )
    
    # Check for interactive mode first
    if args.interactive:
        # Load the correct model for interactive generation
        model, tokenizer, model_type = load_correct_model(args)
        if model:
            pipeline.text_model = model  # Set the model in our pipeline
            pipeline.tokenizer = tokenizer  # Set the tokenizer in our pipeline
            print(f"\nEntering interactive mode with {model_type} model...")
            pipeline.interactive_mode(model_type)
        else:
            print("Failed to load model for interactive mode.")
        return
    
    # Set random seed for reproducibility if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)
        print(f"Random seed set to {args.seed}")
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("preprocessed_data", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Display hardware information
    print("\n=== Hardware Information ===")
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("MPS (Apple Silicon) available for acceleration")
    else:
        print("No GPU acceleration available, using CPU")
    
    # Preprocessing
    datasets = {}
    if args.preprocess:
        print("\n=== Preprocessing Datasets ===")
        with print_execution_time("Preprocessing"):
            datasets = preprocess_datasets(args)
    
    # Training
    if args.train:
        print("\n=== Training Models ===")
        if not datasets and not args.preprocess:
            print("Loading preprocessed datasets for training...")
            dataset_processor = DatasetProcessor()
            
            # Load specified datasets
            for dataset_name in args.datasets:
                if dataset_name == "all":
                    for name in ["pile", "openassistant", "gpteacher"]:
                        data = dataset_processor.load_preprocessed_data(name)
                        if data:
                            datasets[name] = data
                            print(f"Loaded preprocessed data for {name}")
                        else:
                            print(f"Failed to load preprocessed data for {name}. Run with --preprocess first.")
                    break
                else:
                    data = dataset_processor.load_preprocessed_data(dataset_name)
                    if data:
                        datasets[dataset_name] = data
                        print(f"Loaded preprocessed data for {dataset_name}")
                    else:
                        print(f"Failed to load preprocessed data for {dataset_name}. Run with --preprocess first.")
        
        # Initialize training visualizer for comparing results
        visualizer = pipeline.visualizer  # Use the visualizer from our pipeline
        all_metrics = {}
        
        with print_execution_time("Training"):
            for dataset_name, dataset in datasets.items():
                print(f"\n=== Training on {dataset_name} dataset ===")
                # Skip if this dataset is marked to be skipped
                if args.skip and dataset_name in args.skip:
                    print(f"Skipping training on {dataset_name} as requested")
                    continue
                
                # Train model using our pipeline
                model, metrics = pipeline.train(
                    dataset_name=dataset_name,
                    train_type=args.train_type,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    batch_size=args.batch_size,
                    sequence_length=args.sequence_length,
                    gradient_accumulation_steps=args.gradient_accumulation_steps or 4,
                    save_model=args.save_model,
                    validation_split=args.validation_split,
                    use_deepseek=args.use_deepseek,
                    deepseek_batch_size=args.deepseek_batch_size,
                    code_subset=args.code_subset,
                    all_subsets=args.all_subsets,
                    max_samples=args.max_samples
                )
                
                if metrics:
                    all_metrics[dataset_name] = metrics
            
            # Create comparison visualizations if we have metrics for multiple datasets
            if len(all_metrics) > 1:
                visualizer.create_final_report(all_metrics)
                for metric in ['loss', 'accuracy', 'perplexity']:
                    visualizer.create_comparison_plot(all_metrics, metric)
    
    # Generation
    if args.generate:
        print("\n=== Generating Text ===")
        if not args.model_path:
            print("Error: Model path is required for generation. Use --model_path to specify the model.")
            return
        
        # Load the model using our pipeline
        model_type = "deepseek" if args.use_deepseek else ("code" if args.gen_type == "code" else "text")
        model = pipeline.load_model(args.model_path, model_type=model_type)
        
        if model is None:
            print(f"Failed to load model from {args.model_path}")
            return
        
        # Generate text using our pipeline
        print(f"\nGenerating {'code' if model_type == 'code' else 'text'} from prompt: {args.prompt}")
        generated_content = pipeline.generate(
            prompt=args.prompt,
            model_type=model_type,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            use_tokenizer=args.tokenizer is not None
        )
        
        print(f"\nGenerated {'code' if model_type == 'code' else 'text'}:\n{generated_content}")
    
    # Total execution time
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    print("\nPipeline execution completed successfully!")
    
    return

def load_correct_model(args):
    # Generate with deepseek if requested
    if args.gen_type == "code" and args.use_deepseek:
        from generative_ai_module.code_generator import CodeGenerator
        code_gen = CodeGenerator(use_deepseek=True)
        response = code_gen.generate_code(
            prompt=args.prompt,
            length=args.length,
            temperature=args.temperature
        )
        print("\nGenerated code:")
        print(response)
        return

    # Load appropriate model based on dataset
    dataset_name = args.dataset
    model_path = (
        os.path.join(args.model_dir, f"text_gen_{dataset_name}.pt")
        if args.gen_type == "text"
        else os.path.join(args.model_dir, args.code_model)
    )
    model, vocab_size = load_model(model_path, args.gen_type)

    # Generate response
    if args.use_tokenizer and BasicTokenizer:
        tokenizer = BasicTokenizer()
        response, _ = generate_with_tokenizer(
            model, tokenizer, args.prompt,
            max_length=args.length,
            temperature=args.temperature
        )
    else:
        response = generate_with_char_model(
            model, args.prompt, vocab_size,
            max_length=args.length,
            temperature=args.temperature
        )

    print("\nGenerated response:")
    print(response)

if __name__ == "__main__":
    main() 