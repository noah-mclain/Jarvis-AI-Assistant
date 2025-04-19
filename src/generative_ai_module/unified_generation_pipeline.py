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

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Unified Generation Pipeline")
    
    # Main mode selection
    parser.add_argument("--mode", choices=["train", "generate", "preprocess", "evaluate", "interactive"], 
                      default="train", help="Mode of operation")
    
    # Training options
    parser.add_argument("--train-type", choices=["text", "code", "both"], default="both",
                      help="Which generator to train (text, code, or both)")
    parser.add_argument("--epochs", type=int, default=50, 
                      help="Number of training epochs")
    parser.add_argument("--save-model", action="store_true",
                      help="Save the trained models")
    parser.add_argument("--dataset", choices=["persona_chat", "writing_prompts", "code"], 
                      default="persona_chat", help="Which dataset to use for training")
    
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
                        use_scheduler: bool = False, force_gpu: bool = True) -> Tuple[torch.nn.Module, int]:
    """Train the text generator model with enhanced configuration"""
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

        # Get batches
        batches = data.get('batches', [])
        if not batches:
            print("Error: No batches found in data")
            return None, None

        # Training loop
        print(f"Training on {len(batches)} batches for {epochs} epochs")
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
            }
        }

        import time
        start_time = time.time()

        for epoch in range(epochs):
            epoch_loss = 0
            batch_count = 0

            for input_batch, target_batch in tqdm(batches, desc=f"Epoch {epoch+1}/{epochs}"):
                # Move data to device
                input_batch = input_batch.to(device)
                target_batch = target_batch.to(device)

                optimizer.zero_grad()
                output, _ = model(input_batch)
                loss = criterion(output.view(-1, output.size(-1)), target_batch.view(-1))
                loss.backward()

                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            if batch_count > 0:
                avg_loss = epoch_loss / batch_count
                metrics['epoch_losses'].append(avg_loss)

                # Update learning rate if using scheduler
                if use_scheduler:
                    scheduler.step(avg_loss)
                    metrics['learning_rates'].append(optimizer.param_groups[0]['lr'])

                metrics['training_progress'].append({
                    'epoch': epoch + 1,
                    'loss': avg_loss,
                    'time_elapsed': time.time() - start_time
                })
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Calculate final metrics
        metrics['final_loss'] = metrics['epoch_losses'][-1]
        metrics['training_time'] = time.time() - start_time

        # Save model and metrics
        if model_path:
            # Ensure the models directory exists
            models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
            os.makedirs(models_dir, exist_ok=True)

            # Save the model
            model_path = os.path.join(models_dir, os.path.basename(model_path))
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab_size': vocab_size,
                'metrics': metrics
            }, model_path)
            print(f"Model saved to {model_path}")

            # Save metrics to a separate file
            save_evaluation_metrics(metrics, dataset_name, "text")

        return model, vocab_size

    except Exception as e:
        print(f"Error training text generator: {e}")
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

def preprocess_data(args: argparse.Namespace) -> Dict[str, Any]:
    """Enhanced preprocessing pipeline"""
    print("Starting data preprocessing...")
    
    # Handle code dataset separately for deepseek model
    if args.dataset == "code" and args.use_deepseek:
        from generative_ai_module.code_preprocessing import load_and_preprocess_dataset
        train_data, valid_data = load_and_preprocess_dataset(
            max_samples=args.max_samples,
            sequence_length=args.sequence_length,
            subset=args.code_subset,
            all_subsets=args.all_code_subsets
        )
        # Return dummy data structure to match expected format
        return {
            'train_dataset': train_data,
            'valid_dataset': valid_data,
            'vocab_size': None,  # Not used for deepseek
            'batches': [],  # Not used for deepseek
            'preprocessing_time': 0
        }
    
    # Initialize preprocessor
    preprocessor = ImprovedPreprocessor(
        min_length=args.min_length,
        max_length=args.max_length,
        analyze=args.analyze
    )
    
    # Process the dataset
    processed_data = preprocessor.process_dataset(
        dataset_name=args.dataset,
        max_samples=args.max_samples
    )
    
    # Save preprocessed data in the correct location
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # Join with 'preprocessed_data'
    output_dir = os.path.join(root_dir, "preprocessed_data")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save tokenized data
    preprocessor.save_tokenized_data(
        processed_data,
        output_dir=output_dir,
        dataset_name=args.dataset
    )
    
    if args.analyze:
        # Save analysis results
        analysis_results = preprocessor.analyze_token_distribution(processed_data)
        preprocessor.save_analysis_results(
            analysis_results,
            output_dir=output_dir,
            dataset_name=args.dataset
        )
    
    print(f"Preprocessing complete. Data saved to {output_dir}")
    return processed_data

def evaluate_model(model: torch.nn.Module, test_data: Dict[str, Any], metrics: List[str] = None) -> Dict[str, float]:
    """Evaluate model performance on test data"""
    if metrics is None:
        metrics = ["perplexity", "accuracy"]
    print("Starting model evaluation...")

    results = {
        'perplexity': None,
        'accuracy': None,
        'loss': None,
        'num_samples': 0,
        'evaluation_time': None,
        'device_used': str(next(model.parameters()).device)
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0
    total_tokens = 0
    correct_predictions = 0

    import time
    start_time = time.time()

    with torch.no_grad():
        for input_batch, target_batch in tqdm(test_data.get('batches', []), desc="Evaluating"):
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            # Forward pass
            output, _ = model(input_batch)

            # Calculate loss
            loss = criterion(output.view(-1, output.size(-1)), target_batch.view(-1))
            total_loss += loss.item() * input_batch.size(0)
            total_tokens += input_batch.size(0)

            # Calculate accuracy
            if "accuracy" in metrics:
                predictions = output.argmax(dim=-1)
                correct_predictions += (predictions == target_batch).sum().item()

    # Calculate metrics
    if "perplexity" in metrics:
        results["perplexity"] = np.exp(total_loss / total_tokens)
    if "accuracy" in metrics:
        results["accuracy"] = correct_predictions / total_tokens
    results["loss"] = total_loss / total_tokens
    results["num_samples"] = total_tokens
    results["evaluation_time"] = time.time() - start_time

    print("\nEvaluation complete:")
    for metric, value in results.items():
        if value is not None:
            if isinstance(value, (int, float)):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")

    return results

def save_evaluation_metrics(metrics: Dict[str, Any], dataset_name: str, model_type: str):
    """Save evaluation metrics to a JSON file"""
    metrics_dir = "metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = os.path.join(metrics_dir, f"{model_type}_{dataset_name}_{timestamp}.json")
    
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
    
    print(f"Evaluation metrics saved to {metrics_path}")
    return metrics_path

def analyze_prompt(prompt: str) -> str:
    """Analyze the prompt to determine which dataset to use"""
    prompt = prompt.lower()

    # Keywords for writing prompts (story generation)
    story_keywords = ['story', 'write', 'narrative', 'tale', 'plot', 'character', 'scene', 'setting', 'beginning', 'end']

    # Keywords for persona chat (dialogue)
    chat_keywords = ['chat', 'talk', 'conversation', 'discuss', 'ask', 'answer', 'question', 'respond', 'reply']

    story_count = sum(word in prompt for word in story_keywords)
    chat_count = sum(word in prompt for word in chat_keywords)

    return "writing_prompts" if story_count > chat_count else "persona_chat"

def preprocess_datasets(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    """Preprocess multiple datasets simultaneously"""
    print("Starting preprocessing for all datasets...")
    datasets = {}
    
    # Handle 'code' dataset for deepseek separately
    if args.dataset == "code" and args.use_deepseek:
        print("\n===== Processing code dataset for deepseek =====")
        from generative_ai_module.code_preprocessing import load_and_preprocess_dataset
        
        train_dataset, valid_dataset = load_and_preprocess_dataset(
            max_samples=args.max_samples,
            sequence_length=args.sequence_length,
            subset=args.code_subset,
            all_subsets=args.all_code_subsets
        )
        
        datasets["code"] = {
            'train_dataset': train_dataset,
            'valid_dataset': valid_dataset
        }
        
        return datasets
    
    # Process standard datasets
    dataset_list = ["persona_chat", "writing_prompts"]
    if args.dataset != "all":
        dataset_list = [args.dataset]
        
    for dataset_name in dataset_list:
        print(f"\n===== Processing {dataset_name} dataset =====")
        args.dataset = dataset_name
        processed_data = preprocess_data(args)
        datasets[dataset_name] = processed_data
        
        # Save preprocessing metrics
        metrics = {
            'dataset': dataset_name,
            'vocab_size': processed_data.get('vocab_size', 0),
            'num_samples': len(processed_data.get('batches', [])),
            'preprocessing_time': processed_data.get('preprocessing_time', 0),
            'timestamp': datetime.datetime.now().isoformat()
        }
        save_evaluation_metrics(metrics, dataset_name, "preprocessing")
    
    return datasets

def train_on_datasets(args: argparse.Namespace, datasets: Dict[str, Dict[str, Any]]):
    """Train models on multiple datasets simultaneously"""
    print("Starting training for all datasets...")
    
    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Handle code dataset with deepseek separately
    if args.dataset == "code" and args.use_deepseek and args.train_type in ["code", "both"]:
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
            return
        
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
        return
    
    # Train text generator on each dataset
    for dataset_name, data in datasets.items():
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
        
        # Train code generator (only on writing_prompts)
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
        if args.use_deepseek and args.dataset == "code":
            print("\n===== Evaluating deepseek code model =====")
            # This would typically be handled in the fine-tuning process
            print("Deepseek evaluation is performed during the fine-tuning process")
            print("To evaluate the model again, run in train mode")
            return
        
        for dataset_name, data in datasets.items():
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