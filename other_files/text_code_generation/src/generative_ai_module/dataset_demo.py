#!/usr/bin/env python3
"""
Dataset Demo for Jarvis AI Assistant

This script demonstrates the different datasets available in the Jarvis AI system.
It allows you to:
1. See examples from each dataset
2. Test the prompt analyzer to determine which dataset would be used
3. Run a simple training demo on a specific dataset
4. Compare dataset characteristics
5. Test conversation context handling

Usage:
    python dataset_demo.py --action show_examples --dataset writing_prompts
    python dataset_demo.py --action test_analyzer --prompt "Write a story about dragons"
    python dataset_demo.py --action compare_all
    python dataset_demo.py --action test_context --prompt "Hello, how are you?"
"""

import os
import sys
import argparse
import torch
from tqdm import tqdm
from pprint import pprint
from typing import Dict, List, Any, Optional

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import required modules
from .unified_dataset_handler import UnifiedDatasetHandler
from .text_generator import TextGenerator
from .prompt_enhancer import analyze_prompt

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Dataset Demo for Jarvis AI Assistant")
    
    # Main action to perform
    parser.add_argument('--action', type=str, required=True,
                      choices=['show_examples', 'test_analyzer', 'train_demo', 'compare_all', 'test_context'],
                      help='Action to perform')
    
    # Dataset selection for show_examples and train_demo
    parser.add_argument('--dataset', type=str,
                      choices=['writing_prompts', 'persona_chat', 'pile', 'openassistant', 'gpteacher', 'all'],
                      default='all',
                      help='Dataset to use')
    
    # For testing the prompt analyzer or conversation context
    parser.add_argument('--prompt', type=str,
                      help='Prompt to analyze or use for conversation')
    
    # Demo options
    parser.add_argument('--max-samples', type=int, default=5,
                      help='Maximum number of samples to show (for show_examples)')
    
    # Training demo options
    parser.add_argument('--epochs', type=int, default=3,
                      help='Number of training epochs for train_demo')
    
    # Conversation context options
    parser.add_argument('--context-file', type=str,
                      help='File to save/load conversation context')
    
    # Temperature for generation
    parser.add_argument('--temperature', type=float, default=0.7,
                      help='Temperature for text generation (higher = more random)')
    
    return parser.parse_args()

def show_dataset_examples(dataset_name: str, max_samples: int = 5):
    """Show examples from the specified dataset"""
    print(f"\n{'='*80}")
    print(f"Examples from {dataset_name.upper()} dataset")
    print(f"{'='*80}")
    
    handler = UnifiedDatasetHandler()
    
    # Load dataset
    print(f"Loading up to {max_samples} examples...")
    data = handler.load_dataset(dataset_name, max_samples=max_samples)
    
    # Check if we have valid data
    if not data:
        print(f"Could not load data from {dataset_name} dataset")
        return
    
    # Get examples
    examples = handler.extract_prompt_response_pairs(dataset_name, data, max_samples)
    
    # Print examples
    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print("-" * 40)
        if 'prompt' in example:
            print(f"Prompt: {example['prompt'][:500]}...")
        if 'response' in example:
            print(f"Response: {example['response'][:500]}...")
        print("-" * 40)
    
    # Print dataset info
    print("\nDataset information:")
    pprint(handler.get_dataset_info(dataset_name))

def test_prompt_analyzer(prompt: str):
    """Test the prompt analyzer by analyzing a prompt and showing which dataset would be used"""
    print(f"\n{'='*80}")
    print(f"Analyzing prompt: \"{prompt}\"")
    print(f"{'='*80}")
    
    # Use the unified handler to determine the best dataset
    handler = UnifiedDatasetHandler()
    dataset = handler.get_best_dataset_for_prompt(prompt)
    
    print(f"\nBest matching dataset: {dataset}")
    print("\nDataset information:")
    pprint(handler.get_dataset_info(dataset))
    
    # Print all datasets info for comparison
    print("\nAll available datasets:")
    for info in handler.list_all_datasets():
        print(f"- {info['name']}: {info['description']}")

def run_training_demo(dataset_name: str, epochs: int = 3):
    """Run a small training demo on the specified dataset"""
    print(f"\n{'='*80}")
    print(f"Training demo on {dataset_name.upper()} dataset ({epochs} epochs)")
    print(f"{'='*80}")

    # Use a minimal number of samples for the demo
    max_samples = 50

    print(f"Loading {max_samples} samples from {dataset_name}...")

    # Initialize the handler and load the dataset
    handler = UnifiedDatasetHandler()
    data = handler.load_dataset(dataset_name, max_samples=max_samples)

    if not data or 'batches' not in data or not data['batches']:
        print(f"❌ Failed to load proper data from {dataset_name}")
        return

    print(f"✅ Successfully loaded {len(data['batches'])} batches")

    # Prepare data for training with train/validation splits
    splits = handler.prepare_for_training(data, batch_size=16, validation_split=0.2)

    if not splits or 'train' not in splits:
        print("❌ Failed to prepare data for training")
        return

    print(f"✅ Prepared data: {len(splits['train']['batches'])} train, "
          f"{len(splits['validation']['batches'])} validation batches")

    # Initialize the TextGenerator
    print("Initializing TextGenerator...")
    generator = TextGenerator(force_gpu=torch.cuda.is_available())

    # Train for a few epochs
    print(f"Training for {epochs} epochs...")

    vocab_size = data.get('vocab_size', 0)
    generator.n_chars = vocab_size
    generator.model.input_size = vocab_size
    generator.model.output_size = vocab_size

    try:
        test_generation(generator, splits, epochs, dataset_name)
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()


def test_generation(generator, splits, epochs, dataset_name):
    losses = generator.train(splits['train']['batches'], epochs=epochs)
    print("Training complete!")
    print(f"Final loss: {losses[-1] if losses else 'N/A'}")

    # Test generation
    print("\nGenerating sample text:")
    if dataset_name == "persona_chat":
        prompt = "USER: Hi, how are you today?\nASSISTANT: "
    elif dataset_name == "writing_prompts":
        prompt = "<PROMPT>\nYou discover a hidden door in your basement.\n<STORY>\n"
    else:
        prompt = "Once upon a time, "

    generated_text = generator.generate(initial_str=prompt, pred_len=100, temperature=0.7)
    print("-" * 40)
    print(generated_text)
    print("-" * 40)

def compare_all_datasets():
    """Compare characteristics of all available datasets"""
    print(f"\n{'='*80}")
    print("Comparing All Datasets")
    print(f"{'='*80}")

    handler = UnifiedDatasetHandler()

    # Get information about all datasets
    dataset_infos = handler.list_all_datasets()

    print("\n=== Dataset Comparison ===\n")
    print("{:<15} {:<10} {:<25} {:<30}".format(
        "Dataset", "Available", "Format", "Typical Usage"))
    print("-" * 80)

    for info in dataset_infos:
        print("{:<15} {:<10} {:<25} {:<30}".format(
            info['name'],
            "✅" if info['available'] else "❌",
            info['format'][:25],
            info['typical_usage'][:30]
        ))

    # Load small samples from each dataset for size comparison
    print("\n=== Dataset Size Comparison (5 samples) ===\n")
    print("{:<15} {:<12} {:<12} {:<12}".format(
        "Dataset", "Batch Count", "Vocab Size", "Raw Size (KB)"))
    print("-" * 60)

    for dataset_name in handler.SUPPORTED_DATASETS:
        try:
            if data := handler.load_dataset(dataset_name, max_samples=5):
                batch_count = len(data.get('batches', []))
                vocab_size = data.get('vocab_size', 0)

                # Get raw text size
                raw_size = 0
                if dataset_name == "writing_prompts":
                    raw_text = handler.processor.load_writing_prompts(max_samples=5)
                    raw_size = len(raw_text) / 1024  # KB
                elif dataset_name == "persona_chat":
                    raw_text = handler.processor.load_persona_chat(max_samples=5)
                    raw_size = len(raw_text) / 1024  # KB
                elif dataset_name == "pile":
                    raw_text = handler.processor.load_pile_dataset(max_samples=5)
                    raw_size = len(raw_text) / 1024  # KB
                elif dataset_name == "openassistant":
                    raw_text = handler.processor.load_openassistant_dataset(max_samples=5)
                    raw_size = len(raw_text) / 1024  # KB
                elif dataset_name == "gpteacher":
                    raw_text = handler.processor.load_gpteacher_dataset(max_samples=5)
                    raw_size = len(raw_text) / 1024  # KB

                print("{:<15} {:<12} {:<12} {:<12.2f}".format(
                    dataset_name, batch_count, vocab_size, raw_size))
            else:
                print("{:<15} {:<12} {:<12} {:<12}".format(
                    dataset_name, "Failed", "N/A", "N/A"))
        except Exception as e:
            print("{:<15} {:<12} {:<12} {:<12}".format(
                dataset_name, "Error", "N/A", "N/A"))

def test_conversation_context(prompt: str, context_file: str = None, temperature: float = 0.7):
    """Test conversation context handling with the unified dataset handler"""
    print(f"\n{'='*80}")
    print("Testing Conversation Context")
    print(f"{'='*80}")

    # Initialize handler and generator
    handler = UnifiedDatasetHandler()
    generator = TextGenerator(force_gpu=torch.cuda.is_available())

    # Load model from checkpoint if available
    model_path = os.path.join("models", "text_gen_model.pt")
    if os.path.exists(model_path):
        try:
            print(f"Loading model from {model_path}...")
            generator.load_model(model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")

    # Load conversation context if provided
    if context_file and os.path.exists(context_file):
        print(f"Loading conversation context from {context_file}...")
        handler.load_conversation(context_file)

        if history := handler.conversation_context.get_formatted_history():
            load_save_conversation_history(
                "\nConversation history:", history
            )
        else:
            print("No conversation history found")

    # Generate response with context
    print(f"\nUser prompt: {prompt}")
    print("Generating response...")

    response = handler.generate_with_context(
        generator=generator,
        prompt=prompt,
        temperature=temperature
    )

    load_save_conversation_history("\nGenerated response:", response)
    # Save conversation if requested
    if context_file:
        print(f"Saving conversation to {context_file}...")
        handler.save_conversation(context_file)
        print("Conversation saved!")

    return response


def load_save_conversation_history(arg0, arg1):
    print(arg0)
    print("-" * 40)
    print(arg1)
    print("-" * 40)

def main():
    """Main function to run the dataset demo"""
    args = parse_args()
    
    # Handle different actions
    if args.action == 'show_examples':
        if args.dataset == 'all':
            # Show examples from all datasets
            for dataset in UnifiedDatasetHandler.SUPPORTED_DATASETS:
                show_dataset_examples(dataset, args.max_samples)
        else:
            # Show examples from a specific dataset
            show_dataset_examples(args.dataset, args.max_samples)
    
    elif args.action == 'test_analyzer':
        if not args.prompt:
            print("Error: --prompt is required for test_analyzer action")
            return
        test_prompt_analyzer(args.prompt)
    
    elif args.action == 'train_demo':
        if args.dataset == 'all':
            print("Please specify a single dataset for training demo")
            return
        run_training_demo(args.dataset, args.epochs)
    
    elif args.action == 'compare_all':
        compare_all_datasets()
        
    elif args.action == 'test_context':
        if not args.prompt:
            print("Error: --prompt is required for test_context action")
            return
        test_conversation_context(args.prompt, args.context_file, args.temperature)
    
if __name__ == "__main__":
    main() 