#!/usr/bin/env python3
"""
Evaluate Generation

This script evaluates generated text using comprehensive metrics including:
- BERTScore for semantic similarity
- ROUGE/BLEU for n-gram overlap
- Perplexity for fluency
- Hallucination detection for factual consistency
- Human feedback collection (optional)

Usage:
    # Evaluate single generation
    python evaluate_generation.py --reference-file ref.txt --generated-file gen.txt --dataset-name writing_prompts

    # Batch evaluation
    python evaluate_generation.py --batch-evaluate --dataset-name pile
    
    # Human feedback collection
    python evaluate_generation.py --generated-file gen.txt --collect-human-feedback
"""

import os
import sys
import json
import argparse
import torch
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import necessary modules
from src.generative_ai_module.evaluation_metrics import EvaluationMetrics
from src.generative_ai_module.text_generator import TextGenerator
from src.generative_ai_module.unified_dataset_handler import UnifiedDatasetHandler
from src.generative_ai_module.basic_tokenizer import BasicTokenizer

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate generated text with comprehensive metrics")
    
    # Input files
    parser.add_argument('--generated-file', type=str,
                      help='File containing generated text')
    parser.add_argument('--reference-file', type=str,
                      help='File containing reference text')
    parser.add_argument('--prompt-file', type=str,
                      help='File containing prompts')
    
    # Batch evaluation
    parser.add_argument('--batch-evaluate', action='store_true',
                      help='Perform batch evaluation on test sets')
    
    # Dataset selection
    parser.add_argument('--dataset-name', type=str, default='unknown',
                      help='Name of the dataset used for generation')
    
    # Evaluation options
    parser.add_argument('--collect-human-feedback', action='store_true',
                      help='Collect human feedback for evaluation')
    parser.add_argument('--facts-file', type=str,
                      help='File containing reference facts for hallucination detection')
    parser.add_argument('--metrics-dir', type=str, default='evaluation_metrics',
                      help='Directory to save evaluation metrics')
    parser.add_argument('--use-gpu', action='store_true',
                      help='Use GPU for computation if available')
    
    # Model loading for perplexity calculation
    parser.add_argument('--model-path', type=str,
                      help='Path to model for perplexity calculation')
    
    return parser.parse_args()

def read_text_file(file_path: str) -> str:
    """Read text from a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

def read_lines_file(file_path: str) -> List[str]:
    """Read lines from a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []

def load_json_file(file_path: str) -> Any:
    """Load JSON data from a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON from {file_path}: {e}")
        return None

def load_model_for_perplexity(model_path: str) -> tuple:
    """Load model and tokenizer for perplexity calculation"""
    try:
        # Initialize text generator
        generator = TextGenerator(force_gpu=torch.cuda.is_available())
        
        # Load the model
        generator.load_model(model_path)
        
        # Create a basic tokenizer
        tokenizer = BasicTokenizer()
        
        return generator.model, tokenizer
    except Exception as e:
        print(f"Error loading model for perplexity: {e}")
        return None, None

def evaluate_single_generation(args):
    """Evaluate a single generation against a reference"""
    # Check required files
    if not args.generated_file:
        print("Error: --generated-file is required")
        return

    # Initialize evaluator
    metrics = EvaluationMetrics(metrics_dir=args.metrics_dir, use_gpu=args.use_gpu)

    # Read files
    generated_text = read_text_file(args.generated_file)

    if not generated_text:
        print("Error: Generated text is empty")
        return

    # Prepare evaluation parameters
    eval_params = {
        "prompt": read_text_file(args.prompt_file) if args.prompt_file else "Unknown prompt",
        "generated_text": generated_text,
        "dataset_name": args.dataset_name,
        "collect_human_feedback": args.collect_human_feedback,
    }

    # Add reference text if available
    if args.reference_file:
        if reference_text := read_text_file(args.reference_file):
            eval_params["reference_text"] = reference_text

    # Add reference facts if available
    if args.facts_file:
        if reference_facts := read_lines_file(args.facts_file):
            eval_params["reference_facts"] = reference_facts

    # Add model for perplexity if available
    if args.model_path:
        model, tokenizer = load_model_for_perplexity(args.model_path)
        if model and tokenizer:
            eval_params["model"] = model
            eval_params["tokenizer"] = tokenizer

    # Run evaluation
    print("\nEvaluating generated text...")
    results = metrics.evaluate_generation(**eval_params)

    # Print summary
    print("\nEvaluation Summary:")
    print(f"Dataset: {args.dataset_name}")
    print(f"Generated text length: {len(generated_text)} characters")

    if "reference_text" in eval_params:
        print(f"Reference text length: {len(eval_params['reference_text'])} characters")

    # Print metrics
    if "metrics" in results:
        print("\nMetrics:")
        for category, values in results["metrics"].items():
            if isinstance(values, dict):
                print(f"  {category.upper()}:")
                for key, value in values.items():
                    if isinstance(value, list):
                        # Average the values if it's a list
                        avg_value = sum(value) / len(value) if value else 0
                        print(f"    {key}: {avg_value:.4f}")
                    else:
                        print(f"    {key}: {value:.4f}")
            else:
                print(f"  {category}: {values:.4f}")

    # Print human feedback if collected
    if "human_feedback" in results:
        print("\nHuman Feedback:")
        if "ratings" in results["human_feedback"]:
            for criterion, rating in results["human_feedback"]["ratings"].items():
                print(f"  {criterion}: {rating}/5")

        if results["human_feedback"].get("has_hallucination"):
            print("  Hallucination detected: Yes")
            print(f"  Details: {results['human_feedback'].get('hallucination_details', 'None')}")

    print(f"\nFull evaluation results saved to {args.metrics_dir}")
    return results

def batch_evaluate_dataset(args):
    """Perform batch evaluation on test sets from a dataset"""
    print(f"Performing batch evaluation on {args.dataset_name} dataset")

    # Initialize handlers
    metrics = EvaluationMetrics(metrics_dir=args.metrics_dir, use_gpu=args.use_gpu)
    handler = UnifiedDatasetHandler()

    # Load dataset
    try:
        # Load a small sample of the dataset
        dataset = handler.load_dataset(args.dataset_name, max_samples=100)

        # Prepare for training to get test split
        splits = handler.prepare_for_training(dataset)

        # Get test data
        test_data = splits.get("test", {})

        if not test_data or 'batches' not in test_data or not test_data['batches']:
            print(f"Error: No test data available for {args.dataset_name}")
            return

        # Extract prompt-response pairs
        pairs = handler.extract_prompt_response_pairs(args.dataset_name, test_data, max_pairs=20)

        if not pairs:
            print(f"Error: Could not extract prompt-response pairs from {args.dataset_name}")
            return

        # Prepare lists for batch evaluation
        prompts = [pair["prompt"] for pair in pairs]
        references = [pair["response"] for pair in pairs]

        # Load model for generation
        generator = TextGenerator(force_gpu=torch.cuda.is_available())
        try:
            # Try to load a model specific to this dataset
            model_path = f"models/{args.dataset_name}_model.pt"
            generator.load_model(model_path)
            print(f"Loaded model from {model_path}")
        except Exception:
            # Use a default model if specific one not available
            try:
                model_path = "models/text_generator_model.pt"
                generator.load_model(model_path)
                print(f"Using default model from {model_path}")
            except Exception:
                print("Warning: No model available for generation, skipping perplexity calculation")

        # Generate responses
        generated_texts = []
        print("\nGenerating responses for evaluation...")
        for prompt in prompts:
            generated = generator.generate(initial_str=prompt, pred_len=150, temperature=0.7)
            # Remove the prompt from the generated text
            if generated.startswith(prompt):
                generated = generated[len(prompt):].strip()
            generated_texts.append(generated)

        # Run batch evaluation
        results = metrics.batch_evaluate(
            prompts=prompts,
            generated_texts=generated_texts,
            reference_texts=references,
            dataset_name=args.dataset_name
        )

        # Print summary
        print("\nBatch Evaluation Summary:")
        print(f"Dataset: {args.dataset_name}")
        print(f"Number of samples: {len(prompts)}")

        # Print aggregate metrics
        if "metrics" in results:
            print("\nAggregate Metrics:")
            for metric, value in results["metrics"].items():
                print(f"  {metric}: {value:.4f}")

        print(f"\nFull batch evaluation results saved to {args.metrics_dir}")
        return results

    except Exception as e:
        print(f"Error during batch evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function"""
    args = parse_args()
    
    print("=" * 80)
    print("Jarvis AI - Generation Evaluation")
    print("=" * 80)
    
    # Ensure metrics directory exists
    os.makedirs(args.metrics_dir, exist_ok=True)
    
    # Perform evaluation based on arguments
    if args.batch_evaluate:
        batch_evaluate_dataset(args)
    else:
        evaluate_single_generation(args)

if __name__ == "__main__":
    main() 