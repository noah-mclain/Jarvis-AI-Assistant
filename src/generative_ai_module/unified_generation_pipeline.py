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

from generative_ai_module.prompt_enhancer import analyze_prompt
from generative_ai_module.unsloth_deepseek import evaluate_model

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from generative_ai_module.text_generator import TextGenerator, CombinedModel
from generative_ai_module.code_generator import CodeGenerator
from generative_ai_module.dataset_processor import DatasetProcessor
from generative_ai_module.improved_preprocessing import ImprovedPreprocessor

# Try to import tokenizer from examples directory
try:
    from generative_ai_module.basic_tokenizer import BasicTokenizer
except ImportError:
    print("Warning: BasicTokenizer not found. Character-level generation will be used.")
    BasicTokenizer = None

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
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
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
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
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
    parser.add_argument("--all-code-subsets", type=lambda x: (str(x).lower() == 'true'), default=True,
                      help="Whether to use all language subsets for code (default: True)")
    parser.add_argument("--force-gpu", action="store_true", default=True,
                      help="Force the use of GPU (MPS for Apple Silicon, CUDA for NVIDIA)")
    
    return parser.parse_args()

def train_text_generator(dataset_name: str, epochs: int = 50, model_path: str = None,
                        learning_rate: float = 0.002, clip_value: float = 5.0,
                        use_scheduler: bool = False, force_gpu: bool = True,
                        validation_split: float = 0.2, create_visualizations: bool = True) -> Tuple[torch.nn.Module, Dict[str, List[float]]]:
    """Train the text generator model with enhanced configuration and evaluation"""
    print(f"\nTraining text generator on {dataset_name} dataset...")

    # Initialize dataset processor
    dataset_processor = DatasetProcessor()

    try:
        # Load preprocessed data
        data = dataset_processor.load_preprocessed_data(dataset_name)
        if not data:
            print(f"Error: No data loaded for {dataset_name}")
            return None, None

        # Get vocabulary size
        vocab_size = data.get('vocab_size', 0)
        if vocab_size == 0:
            print("Error: Invalid vocabulary size")
            return None, None

        # Create model with enhanced configuration for writing prompts
        if dataset_name == "writing_prompts":
            hidden_size = 256  # Increased hidden size
            num_layers = 3     # Increased number of layers
        else:
            hidden_size = 128
            num_layers = 2

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
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=2, verbose=True
            )

        # Check for GPU
        device = torch.device("cpu")
        if force_gpu:
            # Try to use MPS (Metal Performance Shaders) for Apple Silicon
            if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print("Using MPS (Apple Silicon GPU)")
                device = torch.device("mps")
            # Fall back to CUDA if available
            elif torch.cuda.is_available():
                print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
                device = torch.device("cuda")
            else:
                print("Warning: GPU requested but neither MPS nor CUDA is available. Falling back to CPU.")
        elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("Using MPS (Apple Silicon GPU)")
            device = torch.device("mps")
        elif torch.cuda.is_available():
            print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            device = torch.device("cuda")
        else:
            print("Using CPU (no GPU available)")

        print(f"Using device: {device}")
        model = model.to(device)

        # Get batches and split into train/validation
        batches = data.get('batches', [])
        if not batches:
            print("Error: No batches found in data")
            return None, None
            
        # Split data for validation
        if validation_split > 0:
            val_size = max(1, int(len(batches) * validation_split))
            train_batches = batches[:-val_size]
            val_batches = batches[-val_size:]
            print(f"Split data: {len(train_batches)} training batches, {len(val_batches)} validation batches")
        else:
            train_batches = batches
            val_batches = []
            print(f"Using all {len(train_batches)} batches for training (no validation)")

        # Training loop
        print(f"Training on {len(train_batches)} batches for {epochs} epochs")
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
            'batch_size': batches[0][0].size(0) if batches else 0,
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
            'val_perplexity': []  # For visualization
        }

        import time
        start_time = time.time()

        for epoch in range(epochs):
            epoch_loss = 0
            batch_count = 0

            for input_batch, target_batch in tqdm(train_batches, desc=f"Epoch {epoch+1}/{epochs}"):
                # Move data to device
                input_batch = input_batch.to(device)
                target_batch = target_batch.to(device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                output, _ = model(input_batch)

                # Calculate loss
                loss = criterion(output.view(-1, output.size(-1)), target_batch.view(-1))

                # Backward pass and optimize
                loss.backward()

                # Clip gradients
                if clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

                optimizer.step()

                # Track metrics
                epoch_loss += loss.item()
                batch_count += 1

            # Calculate epoch statistics
            avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
            metrics['epoch_losses'].append(avg_epoch_loss)
            
            # Calculate training metrics
            train_metrics = calculate_metrics(model, train_batches[:min(10, len(train_batches))], device)
            metrics['loss'].append(train_metrics['loss'])
            metrics['accuracy'].append(train_metrics['accuracy'])
            metrics['perplexity'].append(train_metrics['perplexity'])
            
            # Calculate validation metrics if available
            if val_batches:
                val_metrics = calculate_metrics(model, val_batches, device)
                metrics['val_loss'].append(val_metrics['loss'])
                metrics['val_accuracy'].append(val_metrics['accuracy'])
                metrics['val_perplexity'].append(val_metrics['perplexity'])
                
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}, "
                      f"Val Loss: {val_metrics['loss']:.4f}, "
                      f"Val Accuracy: {val_metrics['accuracy']:.4f}, "
                      f"Val Perplexity: {val_metrics['perplexity']:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")

            metrics['training_progress'].append({
                'epoch': epoch + 1,
                'loss': avg_epoch_loss,
                'timestamp': datetime.datetime.now().isoformat()
            })

            # Update learning rate if using scheduler
            if use_scheduler:
                val_loss = val_metrics['loss'] if val_batches else avg_epoch_loss
                scheduler.step(val_loss)
                metrics['learning_rates'].append(optimizer.param_groups[0]['lr'])

        # Final metrics
        metrics['final_loss'] = metrics['epoch_losses'][-1] if metrics['epoch_losses'] else None
        metrics['training_time'] = time.time() - start_time

        # Create visualizations if requested
        if create_visualizations:
            visualizer = TrainingVisualizer(output_dir="visualizations")
            visualizer.visualize_training(dataset_name, metrics)

        # Save model if requested
        if model_path:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
            
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
            print(f"Training metrics saved to {metrics_path}")

        return model, metrics

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
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
    # Create metrics directory if it doesn't exist
    metrics_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "metrics")
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
                all_subsets=args.all_code_subsets
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
                all_subsets=args.all_code_subsets
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
                         force_gpu: bool = True) -> None:
    """Interactive generation mode for user prompts"""
    print("\nStarting interactive generation mode...")
    print("Type 'quit' to exit")
    print("The system will automatically determine whether to generate a story or dialogue response.")
    
    while True:
        try:
            prompt = input("\nEnter your prompt: ")
            if prompt.lower() == 'quit':
                break
            
            # Analyze the prompt to determine the appropriate dataset
            dataset_name = analyze_prompt(prompt)
            print(f"\nDetected prompt type: {'Story generation' if dataset_name == 'writing_prompts' else 'Dialogue'}")
            
            # Load the appropriate model
            model_path = os.path.join(args.model_dir, f"text_gen_{dataset_name}.pt")
            
            # Check if model exists, if not, train it
            if not os.path.exists(model_path):
                print(f"Model not found at {model_path}. Training new model...")
                datasets = preprocess_datasets(args)
                train_on_datasets(args, datasets)
            
            model, vocab_size = load_model(model_path)
            
            # Get device
            device = torch.device("cpu")
            if force_gpu:
                if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    device = torch.device("mps")
                    print("Using Apple Silicon GPU (MPS) for generation")
                elif torch.cuda.is_available():
                    device = torch.device("cuda")
                    print("Using NVIDIA GPU for generation")
                else:
                    print("Warning: GPU requested but not available. Using CPU.")
            model = model.to(device)
            
            # Generate response
            if tokenizer:
                response, _ = generate_with_tokenizer(
                    model, tokenizer, prompt,
                    max_length=args.length,
                    temperature=args.temperature
                )
            else:
                # Create a text generator with the loaded model
                text_gen = TextGenerator(force_gpu=force_gpu)
                text_gen.model = model
                
                response = text_gen.generate(
                    initial_str=prompt, 
                    pred_len=args.length,
                    temperature=args.temperature
                )
            
            print("\nGenerated response:")
            print(response)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error generating response: {e}")

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
    args = parse_args()
    
    # Set dataset to "both" to process both persona_chat and writing_prompts by default for training
    if args.mode == "train" and args.dataset not in ["all", "both", "code"]:
        print("Setting dataset to 'both' to train on both persona_chat and writing_prompts")
        args.dataset = "both"
        args.save_model = True  # Always save models
    
    if args.mode == "preprocess":
        # Preprocess all datasets
        datasets = preprocess_datasets(args)
    
    elif args.mode == "train":
        # Preprocess and train on all datasets
        datasets = preprocess_datasets(args)
        train_on_datasets(args, datasets)
    
    elif args.mode == "generate":
        load_correct_model(args)
    
    elif args.mode == "evaluate":
        # Load and evaluate models for each dataset
        datasets = preprocess_datasets(args)
        
        # Special case for deepseek code model
        if args.use_deepseek:
            print("\n===== Evaluating deepseek code model =====")
            # This would typically be handled in the fine-tuning process
            print("Deepseek evaluation is performed during the fine-tuning process")
            print("To evaluate the model again, run in train mode")
        
        for dataset_name, data in datasets.items():
            if dataset_name in ["persona_chat", "writing_prompts"]:
                print(f"\n===== Evaluating {dataset_name} dataset =====")
                _, eval_data = split_data(data, args.eval_split)
                
                # Evaluate text model
                model_path = os.path.join(args.model_dir, f"text_gen_{dataset_name}.pt")
                if os.path.exists(model_path):
                    model, _ = load_model(model_path)
                    metrics = evaluate_model(model, eval_data, args.metrics)
                    save_evaluation_metrics(metrics, dataset_name, "text")
    
    elif args.mode == "interactive":
        # Special case for deepseek code generation
        if args.gen_type == "code" and args.use_deepseek:
            from generative_ai_module.code_generator import CodeGenerator
            print("\nStarting interactive code generation mode with deepseek...")
            print("Type 'quit' to exit")
            
            code_gen = CodeGenerator(use_deepseek=True)
            
            while True:
                try:
                    prompt = input("\nEnter code description: ")
                    if prompt.lower() == 'quit':
                        break
                    
                    print("\nGenerating code...")
                    response = code_gen.generate_code(
                        prompt=prompt,
                        length=args.length,
                        temperature=args.temperature
                    )
                    
                    print("\nGenerated code:")
                    print(response)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error generating code: {e}")
            return
        
        # Regular text generation
        dataset_name = args.dataset
        if dataset_name not in ["persona_chat", "writing_prompts"]:
            # Default to persona_chat for interactive mode
            dataset_name = "persona_chat"
            
        model_path = os.path.join(args.model_dir, f"text_gen_{dataset_name}.pt")
        
        # Check if model exists, if not, train it
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}. Training new model...")
            datasets = preprocess_datasets(args)
            train_on_datasets(args, datasets)
        
        model, vocab_size = load_model(model_path)
        
        # Initialize tokenizer if requested
        tokenizer = BasicTokenizer() if args.use_tokenizer and BasicTokenizer else None
        
        # Start interactive session
        interactive_generation(model, tokenizer, args.gen_type, force_gpu=args.force_gpu)

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