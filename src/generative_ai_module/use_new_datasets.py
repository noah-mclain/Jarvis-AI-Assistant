"""
Unified Dataset and Generation Script for Jarvis AI Assistant

This script provides a comprehensive interface for:
1. Managing datasets (The Pile, OpenAssistant, GPTeacher)
2. Training models on these datasets
3. Generating responses with context handling
4. Testing and evaluating trained models
"""

import os
import sys
import json
import torch
import argparse
import datetime
import subprocess
from typing import Dict, List, Any, Union, Optional
from collections import deque

# Ensure the parent directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import from the generative_ai_module
from src.generative_ai_module.dataset_processor import DatasetProcessor
from src.generative_ai_module.text_generator import TextGenerator
from src.generative_ai_module.prompt_enhancer import PromptEnhancer, analyze_prompt

class ConversationContext:
    """Class for managing conversation context and history"""
    
    def __init__(self, max_history: int = 5, max_tokens: int = 1000):
        """Initialize the conversation context"""
        self.history = deque(maxlen=max_history)
        self.max_tokens = max_tokens
        self.metadata = {}
    
    def add_exchange(self, user_input: str, assistant_response: str) -> None:
        """Add a conversation exchange to history"""
        self.history.append({
            'user': user_input,
            'assistant': assistant_response,
            'timestamp': datetime.datetime.now().isoformat()
        })
    
    def get_formatted_history(self, include_current: bool = False, current_input: str = None) -> str:
        """Format conversation history for use in generation"""
        formatted = ""
        
        for exchange in self.history:
            formatted += f"USER: {exchange['user']}\n"
            formatted += f"ASSISTANT: {exchange['assistant']}\n\n"
        
        if include_current and current_input:
            formatted += f"USER: {current_input}\n"
            formatted += "ASSISTANT: "
            
        return formatted
    
    def save_to_file(self, filepath: str) -> None:
        """Save conversation context to a file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump({
                'history': list(self.history),
                'metadata': self.metadata
            }, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'ConversationContext':
        """Load conversation context from a file"""
        if not os.path.exists(filepath):
            return cls()
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        context = cls()
        for exchange in data.get('history', []):
            # Ensure we have the required fields
            if 'user' in exchange and 'assistant' in exchange:
                context.history.append(exchange)
        
        context.metadata = data.get('metadata', {})
        return context
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the conversation context"""
        self.metadata[key] = value
    
    def clear(self) -> None:
        """Clear the conversation history"""
        self.history.clear()
        self.metadata = {}

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Unified Dataset and Generation Script")
    
    # Dataset selection
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'pile', 'openassistant', 'gpteacher'],
                        help='Dataset to use (default: all)')
    
    # For The Pile dataset
    parser.add_argument('--pile-subset', type=str, default=None,
                        help='Specific subset of The Pile (e.g., "pubmed", "github", "europarl")')
    
    # Action to perform
    parser.add_argument('--action', type=str, default='interactive',
                        choices=['sample', 'train', 'test', 'generate', 'unified-train', 'interactive'],
                        help='Action to perform with the dataset')
    
    # Training options
    parser.add_argument('--max-samples', type=int, default=100,
                        help='Maximum number of samples to use')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to save/load model')
    parser.add_argument('--validation-split', type=float, default=0.2,
                       help='Fraction of data to use for validation')
    parser.add_argument('--test-split', type=float, default=0.1,
                       help='Fraction of data to use for testing')
    
    # Generation options
    parser.add_argument('--prompt', type=str, default=None,
                        help='Prompt for text generation')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Directory to save outputs (models, visualizations)')
    parser.add_argument('--context-file', type=str, default=None,
                       help='File to save/load conversation context')
    parser.add_argument('--max-history', type=int, default=5,
                       help='Maximum number of exchanges to keep in history')
    
    # Model selection for generation
    parser.add_argument('--use-best-model', action='store_true',
                       help='Use the best model instead of the final model')
    
    # Add temperature option for generation
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Temperature for text generation (higher = more random)')
    
    return parser.parse_args()

def get_datasets(args):
    """Determine which datasets to use based on arguments"""
    if args.dataset == 'all':
        return ['pile', 'openassistant', 'gpteacher']
    return [args.dataset]

def sample_dataset(args):
    """Display sample data from the chosen dataset(s)"""
    datasets = get_datasets(args)

    for dataset_name in datasets:
        print(f"\n===== SAMPLING FROM {dataset_name.upper()} =====")
        processor = DatasetProcessor()

        # Get sample text based on dataset
        if dataset_name == 'pile':
            subset_info = f" (subset: {args.pile_subset})" if args.pile_subset else ""
            print(f"Loading sample from The Pile{subset_info}...")
            text = processor.load_pile_dataset(
                subset=args.pile_subset, 
                max_samples=5
            )
        elif dataset_name == 'openassistant':
            print("Loading sample from OpenAssistant dataset...")
            text = processor.load_openassistant_dataset(max_samples=5)
        elif dataset_name == 'gpteacher':
            print("Loading sample from GPTeacher dataset...")
            text = processor.load_gpteacher_dataset(max_samples=5)

        # Display a sample
        print("\n----- DATASET SAMPLE -----")
        # Show first 1000 characters
        print(f"{text[:1000]}...")
        print(f"\nTotal text length: {len(text)} characters")

def split_data(data: Dict[str, Any], validation_split: float = 0.2, test_split: float = 0.1):
    """Split dataset into training, validation, and test sets"""
    if 'batches' not in data or not data['batches']:
        return data, {}, {}
    
    batches = data['batches']
    
    # Calculate split indices
    total = len(batches)
    n_test = max(1, int(total * test_split))
    n_val = max(1, int(total * validation_split))
    n_train = total - n_val - n_test
    
    # Create train data
    train_data = data.copy()
    train_data['batches'] = batches[:n_train]
    
    # Create validation data
    val_data = data.copy()
    val_data['batches'] = batches[n_train:n_train+n_val]
    
    # Create test data
    test_data = data.copy()
    test_data['batches'] = batches[n_train+n_val:]
    
    print(f"Split data into {n_train} training, {n_val} validation, and {n_test} test batches")
    return train_data, val_data, test_data

def train_model(args):
    """Train a model on the selected dataset(s)"""
    datasets = get_datasets(args)
    
    for dataset_name in datasets:
        print(f"\n===== TRAINING ON {dataset_name.upper()} =====")
        
        # Create text generator and processor
        generator = TextGenerator(force_gpu=True)
        processor = DatasetProcessor(generator)
        
        # Prepare the dataset
        print(f"Preparing data from {dataset_name}...")
        data = processor.prepare_dataset(
            source=dataset_name,
            split='train',
            max_samples=args.max_samples,
            batch_size=32,
            subset=args.pile_subset if dataset_name == 'pile' else None
        )
        
        # Check if we have valid data
        if not data.get('batches'):
            print(f"ERROR: No batches found in preprocessed data for {dataset_name}")
            continue
        
        # Split data into training, validation, and test sets
        train_data, val_data, test_data = split_data(data, args.validation_split, args.test_split)
        
        # Train the model
        print(f"Starting training for {args.epochs} epochs...")
        
        # Initialize best validation loss tracking
        best_val_loss = float('inf')
        best_epoch = -1
        
        for epoch in range(args.epochs):
            # Training phase
            epoch_loss = 0.0
            batch_count = 0
            
            for input_batch, target_batch in train_data['batches']:
                # Forward pass
                generator.optimizer.zero_grad()
                outputs, _ = generator.model(input_batch)
                
                # Calculate loss
                loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), target_batch.view(-1))
                
                # Backward pass and optimization
                loss.backward()
                generator.optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                
            # Calculate average loss
            avg_loss = epoch_loss / max(1, batch_count)
            
            # Validation phase
            if val_data and val_data.get('batches'):
                val_loss = 0.0
                val_batch_count = 0
                
                with torch.no_grad():
                    for input_batch, target_batch in val_data['batches']:
                        # Forward pass
                        outputs, _ = generator.model(input_batch)
                        
                        # Calculate loss
                        loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), target_batch.view(-1))
                        
                        val_loss += loss.item()
                        val_batch_count += 1
                
                # Calculate average validation loss
                avg_val_loss = val_loss / max(1, val_batch_count)
                
                # Check if this is the best model so far
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch
                    
                    # Save the best model
                    if args.model_path:
                        model_dir = os.path.dirname(args.model_path) or args.output_dir
                        os.makedirs(model_dir, exist_ok=True)
                        
                        # Create a dataset-specific best model path
                        best_model_path = os.path.join(model_dir, f"{dataset_name}_best_model.pt")
                        generator.save_model(best_model_path)
                        print(f"New best model saved to {best_model_path} (epoch {epoch+1})")
                
                print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}")
        
        # Test phase
        if test_data and test_data.get('batches'):
            test_loss = 0.0
            test_batch_count = 0
            correct_predictions = 0
            total_tokens = 0
            
            with torch.no_grad():
                for input_batch, target_batch in test_data['batches']:
                    # Forward pass
                    outputs, _ = generator.model(input_batch)
                    
                    # Calculate loss
                    loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), target_batch.view(-1))
                    
                    # Calculate accuracy
                    predictions = outputs.argmax(dim=-1)
                    correct = (predictions == target_batch).sum().item()
                    
                    test_loss += loss.item()
                    test_batch_count += 1
                    correct_predictions += correct
                    total_tokens += target_batch.numel()
            
            # Calculate metrics
            avg_test_loss = test_loss / max(1, test_batch_count)
            accuracy = correct_predictions / max(1, total_tokens)
            perplexity = torch.exp(torch.tensor(avg_test_loss)).item()
            
            print("\nTest Results:")
            print(f"  Loss: {avg_test_loss:.4f}")
            print(f"  Perplexity: {perplexity:.2f}")
            print(f"  Accuracy: {accuracy:.4f}")
        
        # Save the final model
        if args.model_path:
            model_dir = os.path.dirname(args.model_path) or args.output_dir
            os.makedirs(model_dir, exist_ok=True)
            
            # Create a dataset-specific model path
            model_filename = f"{dataset_name}_model.pt"
            model_path = os.path.join(model_dir, model_filename)
            generator.save_model(model_path)
            print(f"Final model saved to {model_path}")
        
        # Show training results
        print("\nTraining completed for this dataset!")
        if best_epoch >= 0:
            print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")

def test_model(args):
    """Test a trained model on a dataset"""
    if not args.model_path:
        print("ERROR: Model path must be provided for testing")
        return
    
    datasets = get_datasets(args)
    
    for dataset_name in datasets:
        print(f"\n===== TESTING ON {dataset_name.upper()} =====")
        
        # Determine which model to load (best or final)
        model_dir = os.path.dirname(args.model_path) or args.output_dir
        if args.use_best_model:
            model_path = os.path.join(model_dir, f"{dataset_name}_best_model.pt")
            print(f"Using best model from {model_path}")
        else:
            model_path = os.path.join(model_dir, f"{dataset_name}_model.pt")
            print(f"Using final model from {model_path}")
        
        # Create text generator and processor
        generator = TextGenerator(force_gpu=True)
        try:
            generator.load_model(model_path)
        except FileNotFoundError:
            print(f"ERROR: Model file not found at {model_path}")
            continue
        
        processor = DatasetProcessor(generator)
        
        # Prepare test data
        print(f"Preparing test data from {dataset_name}...")
        data = processor.prepare_dataset(
            source=dataset_name,
            split='test',  # Use test split
            max_samples=args.max_samples,
            batch_size=32,
            subset=args.pile_subset if dataset_name == 'pile' else None
        )
        
        # Check if we have valid data
        if not data.get('batches'):
            print(f"ERROR: No batches found in test data for {dataset_name}")
            continue
        
        # Run evaluation
        total_loss = 0.0
        total_samples = 0
        total_correct = 0
        
        generator.model.eval()
        with torch.no_grad():
            for input_batch, target_batch in data['batches']:
                # Forward pass
                outputs, _ = generator.model(input_batch)
                
                # Calculate loss
                loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), target_batch.view(-1))
                total_loss += loss.item() * input_batch.size(0)
                
                # Calculate accuracy
                predictions = outputs.argmax(dim=-1)
                total_correct += (predictions == target_batch).sum().item()
                total_samples += target_batch.numel()
        
        # Calculate metrics
        avg_loss = total_loss / max(1, total_samples)
        accuracy = total_correct / max(1, total_samples)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        # Print results
        print(f"\nTest Results for {dataset_name}:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Perplexity: {perplexity:.2f}")
        print(f"  Accuracy: {accuracy:.4f}")
        
        # Save test results
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            results_file = os.path.join(args.output_dir, f"{dataset_name}_test_results.json")
            
            with open(results_file, 'w') as f:
                json.dump({
                    'dataset': dataset_name,
                    'model_path': model_path,
                    'loss': float(avg_loss),
                    'perplexity': float(perplexity),
                    'accuracy': float(accuracy),
                    'samples': total_samples,
                    'timestamp': datetime.datetime.now().isoformat()
                }, f, indent=2)
                
            print(f"Test results saved to {results_file}")

def generate_text_with_context(generator, prompt, context=None, temperature=0.7, max_length=200):
    """Generate text with context awareness"""
    enhancer = PromptEnhancer()
    
    # Use context if provided
    if context:
        # Get formatted conversation history
        history = context.get_formatted_history(include_current=True, current_input=prompt)
        enhanced_prompt = enhancer.enhance_prompt(history)
    else:
        # Just enhance the prompt alone
        enhanced_prompt = enhancer.enhance_prompt(prompt)
    
    # Generate text
    generated_text = generator.generate(
        initial_str=enhanced_prompt, 
        pred_len=max_length,
        temperature=temperature
    )
    
    # Extract just the assistant's response if we used context
    if context:
        # Skip past the prompt and get the generated part only
        response = generated_text[len(enhanced_prompt):]
        
        # Try to find where the assistant's response ends (if any additional USER: appears)
        end_marker = '\nUSER:'
        if end_marker in response:
            response = response.split(end_marker)[0].strip()
    else:
        response = generated_text
    
    return response, enhanced_prompt

def generate_text(args):
    """Generate text using a trained model"""
    if not args.model_path and not os.path.isdir(args.output_dir):
        print("ERROR: Either model path or output directory must be provided")
        return
    
    if not args.prompt:
        print("ERROR: Prompt must be provided for generation")
        return
    
    # Determine which dataset model to use based on prompt analysis
    if os.path.isdir(args.model_path) or not args.model_path:
        model_dir = args.model_path or args.output_dir
        
        # Analyze the prompt to determine which dataset model to use
        best_dataset = determine_best_dataset_for_prompt(args.prompt)
        print(f"Automatically selected {best_dataset} model based on prompt analysis")
        
        if args.use_best_model:
            model_path = os.path.join(model_dir, f"{best_dataset}_best_model.pt")
        else:
            model_path = os.path.join(model_dir, f"{best_dataset}_model.pt")
    else:
        model_path = args.model_path
    
    print(f"Generating text using model from {model_path}...")
    
    # Load conversation context if provided
    context = None
    if args.context_file:
        context = ConversationContext.load_from_file(args.context_file)
        print(f"Loaded conversation context with {len(context.history)} previous exchanges")
    
    # Create and load the model
    generator = TextGenerator(force_gpu=True)
    try:
        generator.load_model(model_path)
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}")
        return
    
    # Generate text with context
    response, enhanced_prompt = generate_text_with_context(
        generator, 
        args.prompt, 
        context, 
        temperature=args.temperature
    )
    
    # Display results
    print("\n===== GENERATED TEXT =====")
    if context:
        print(f"Using conversation context with {len(context.history)} previous exchanges")
    print(f"Enhanced prompt: {enhanced_prompt[:100]}...")
    print("\nGenerated response:")
    print(response)
    
    # Update and save context if needed
    if args.context_file:
        context.add_exchange(args.prompt, response)
        context.save_to_file(args.context_file)
        print(f"Updated conversation context saved to {args.context_file}")
    
    # Save generated text if output directory is provided
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(args.output_dir, f"generated_text_{timestamp}.txt")
        
        with open(output_file, 'w') as f:
            f.write(f"Original prompt: {args.prompt}\n\n")
            if context:
                f.write("Conversation history:\n")
                f.write(context.get_formatted_history(include_current=False))
                f.write("\n\n")
            f.write(f"Enhanced prompt: {enhanced_prompt}\n\n")
            f.write(f"Generated response:\n{response}")
            
        print(f"\nGenerated text saved to {output_file}")

def determine_best_dataset_for_prompt(prompt):
    """Determine which dataset model would be best for the given prompt"""
    # For instructional/how-to prompts, use GPTeacher
    instruction_keywords = [
        'how to', 'explain', 'what is', 'guide', 'steps', 'instructions',
        'teach me', 'show me', 'procedure', 'tutorial'
    ]
    
    # For conversational/assistant-like prompts, use OpenAssistant
    assistant_keywords = [
        'can you', 'help me', 'please', 'I need', 'assistant', 
        'could you', 'would you', 'advice', 'suggest'
    ]
    
    # For factual/knowledge-based prompts, use The Pile
    pile_keywords = [
        'fact', 'data', 'research', 'study', 'information', 'history',
        'science', 'literature', 'statistics', 'analysis', 'report'
    ]
    
    # Count keyword matches
    prompt_lower = prompt.lower()
    instruction_count = sum(keyword in prompt_lower for keyword in instruction_keywords)
    assistant_count = sum(keyword in prompt_lower for keyword in assistant_keywords)
    pile_count = sum(keyword in prompt_lower for keyword in pile_keywords)
    
    # Determine which has the most matches
    if instruction_count >= assistant_count and instruction_count >= pile_count:
        return 'gpteacher'
    elif assistant_count >= instruction_count and assistant_count >= pile_count:
        return 'openassistant'
    else:
        return 'pile'

def run_unified_training(args):
    """Run the unified training script for all datasets"""
    print("Running unified training on all datasets...")

    # Prepare command to run the unified training script
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "train_unified_models.py")
    ]

    # Add arguments
    if args.dataset != 'all':
        cmd.extend(['--datasets', args.dataset])

    if args.pile_subset:
        cmd.extend(['--pile-subset', args.pile_subset])

    cmd.extend(
        [
            '--max-samples',
            str(args.max_samples),
            '--epochs',
            str(args.epochs),
            '--validation-split',
            str(args.validation_split),
            '--test-split',
            str(args.test_split),
        ]
    )
    if args.output_dir:
        model_dir = os.path.join(args.output_dir, 'models')
        viz_dir = os.path.join(args.output_dir, 'visualizations')
        cmd.extend(['--model-dir', model_dir])
        cmd.extend(['--visualization-dir', viz_dir])

    # Print the command
    print("Running command:")
    print(" ".join(cmd))

    # Run the command
    try:
        subprocess.run(cmd, check=True)
        print("Unified training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error running unified training: {e}")
    except FileNotFoundError:
        print("ERROR: train_unified_models.py not found!")
        print("Please ensure the unified training script exists.")

def interactive_session(args):
    """Run an interactive session with the model"""
    # Determine model path or use models directory
    if not args.model_path and not os.path.isdir(args.output_dir):
        print("ERROR: Either model path or output directory must be provided")
        return
    
    model_dir = args.model_path if os.path.isdir(args.model_path) else args.output_dir
    
    # Initialize conversation context
    if args.context_file:
        context = ConversationContext.load_from_file(args.context_file)
        print(f"Loaded conversation context with {len(context.history)} previous exchanges")
    else:
        context = ConversationContext(max_history=args.max_history)
        print(f"Created new conversation context (max history: {args.max_history})")
    
    # Create generators for each dataset
    generators = {}
    available_models = []
    
    for dataset_name in ['pile', 'openassistant', 'gpteacher']:
        model_filename = f"{dataset_name}_{'best_model' if args.use_best_model else 'model'}.pt"
        model_path = os.path.join(model_dir, model_filename)
        
        if os.path.exists(model_path):
            try:
                generator = TextGenerator(force_gpu=True)
                generator.load_model(model_path)
                generators[dataset_name] = generator
                available_models.append(dataset_name)
                print(f"Loaded {dataset_name} model from {model_path}")
            except Exception as e:
                print(f"Error loading {dataset_name} model: {e}")
    
    if not generators:
        print("ERROR: No models found. Please train models first.")
        return
    
    print("\n===== INTERACTIVE MODE =====")
    print("Type 'exit' to end the session")
    print("Type 'clear' to clear the conversation history")
    print(f"Available models: {', '.join(available_models)}")
    print("Enter your prompt:")
    
    while True:
        # Get user input
        user_input = input("\nUSER: ").strip()
        
        # Check for exit command
        if user_input.lower() == 'exit':
            break
        
        # Check for clear command
        if user_input.lower() == 'clear':
            context.clear()
            print("Conversation history cleared")
            continue
        
        # Determine which model to use
        dataset = determine_best_dataset_for_prompt(user_input)
        
        # Fallback if the determined model isn't available
        if dataset not in generators:
            dataset = available_models[0]
        
        print(f"Using {dataset} model for this response...")
        
        # Generate response
        response, _ = generate_text_with_context(
            generators[dataset],
            user_input,
            context,
            temperature=args.temperature
        )
        
        # Display response
        print(f"\nASSISTANT: {response}")
        
        # Update conversation context
        context.add_exchange(user_input, response)
        
        # Save context if specified
        if args.context_file:
            context.save_to_file(args.context_file)
    
    print("\nEnding interactive session...")
    
    # Save final context
    if args.context_file:
        context.save_to_file(args.context_file)
        print(f"Final conversation context saved to {args.context_file}")

def main():
    """Main function"""
    args = parse_args()
    
    # Import here to avoid circular imports
    import datetime
    
    print("==================================================")
    print("Jarvis AI Assistant - Unified Dataset Utility")
    print("==================================================")
    
    # Check for GPU
    if torch.cuda.is_available():
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("Using Apple Silicon MPS")
    else:
        print("WARNING: No GPU detected. Processing will be slower.")
    
    # Create output directory if needed
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Perform the requested action
    if args.action == 'sample':
        sample_dataset(args)
    elif args.action == 'train':
        train_model(args)
    elif args.action == 'test':
        test_model(args)
    elif args.action == 'generate':
        generate_text(args)
    elif args.action == 'unified-train':
        run_unified_training(args)
    elif args.action == 'interactive':
        interactive_session(args)
    
    print("\nDone!")

if __name__ == "__main__":
    main() 