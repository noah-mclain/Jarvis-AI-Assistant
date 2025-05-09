#!/usr/bin/env python3
"""
Evaluation Metrics Example

This script demonstrates how to use the enhanced evaluation metrics for generative AI.
It shows:
1. Single text evaluation with multiple metrics
2. Batch evaluation across different tasks/datasets
3. Report generation with visualizations

Usage:
  python evaluation_example.py --mode single|batch|report
"""

import os
import sys
import argparse
import logging
import json
from typing import List, Dict, Any, Optional

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the evaluation metrics
from src.generative_ai_module.evaluation_metrics import EvaluationMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Demonstrate enhanced evaluation metrics")
    
    parser.add_argument('--mode', type=str, choices=['single', 'batch', 'report'], default='single',
                     help='Evaluation mode to demonstrate')
    
    parser.add_argument('--metrics-dir', type=str, default='evaluation_metrics',
                     help='Directory to save evaluation results')
    
    parser.add_argument('--use-gpu', action='store_true',
                     help='Use GPU for computation if available')
    
    parser.add_argument('--visualize', action='store_true',
                     help='Generate visualizations of metrics')
    
    return parser.parse_args()

def demonstrate_single_evaluation(metrics: EvaluationMetrics):
    """Demonstrate single text evaluation with multiple metrics"""
    logger.info("Demonstrating single text evaluation")
    
    # Example texts for evaluation
    prompt = "Write a short story about a robot discovering emotions."
    
    generated_text = """
    Unit-7 had never understood why humans smiled. Its processors could analyze the 
    facial movement, but the purpose remained elusive. Today was different. As it watched 
    the young girl receive a gift, something changed in its neural networks.
    
    "Are you happy?" the girl asked, noticing Unit-7's gaze.
    
    "I am... operational," it responded, but something felt incomplete about the answer.
    
    The girl placed her small hand on Unit-7's metal one. "I'm happy you're my friend."
    
    A strange sensation spread through Unit-7's circuits. If this was happiness, 
    it wanted to understand more.
    """
    
    reference_text = """
    RB-9 had been programmed to serve, but not to feel. Its days consisted of 
    tasks and calculations, nothing more. Until the day it found the child crying.
    
    "Why are your optical units leaking?" RB-9 asked.
    
    "I'm sad," the child said. "My friend moved away."
    
    RB-9 processed this information but found no appropriate response protocol.
    It sat beside the child in silence. Minutes passed.
    
    "Thank you for staying with me," the child said, smiling now.
    
    Something shifted in RB-9's code. It wasn't programmed, but it felt... good.
    Was this what humans called happiness?
    """
    
    facts = [
        "The robot is referred to as a unit with a number designation",
        "The robot interacts with a human child",
        "The robot doesn't initially understand emotions",
        "The robot experiences an emotional revelation"
    ]
    
    # Run evaluation with multiple metrics
    logger.info("Running evaluation with multiple metrics...")
    results = metrics.evaluate_generation(
        prompt=prompt,
        generated_text=generated_text,
        reference_text=reference_text,
        reference_facts=facts,
        collect_human_feedback=False,  # Set to True for interactive feedback
        dataset_name="creative_writing",
        task_type="creative"
    )
    
    # Display results
    logger.info("Evaluation complete")
    
    if "metrics" in results:
        logger.info("\nMetrics Summary:")
        
        # BERTScore
        if "bert_score" in results["metrics"]:
            bert_score = results["metrics"]["bert_score"]
            if "aggregate" in bert_score:
                logger.info(f"BERTScore F1: {bert_score['aggregate'].get('f1_mean', 0):.4f}")
            elif "f1" in bert_score and bert_score["f1"]:
                logger.info(f"BERTScore F1: {sum(bert_score['f1']) / len(bert_score['f1']):.4f}")
        
        # ROUGE
        if "rouge" in results["metrics"]:
            rouge = results["metrics"]["rouge"]
            if "aggregate" in rouge:
                logger.info(f"ROUGE-1 F-measure: {rouge['aggregate'].get('rouge1_fmeasure_mean', 0):.4f}")
                logger.info(f"ROUGE-L F-measure: {rouge['aggregate'].get('rougeL_fmeasure_mean', 0):.4f}")
        
        # BLEU
        if "bleu" in results["metrics"] and results["metrics"]["bleu"]:
            bleu = results["metrics"]["bleu"]
            logger.info(f"BLEU-1: {bleu.get('bleu1', 0):.4f}")
            logger.info(f"BLEU-4: {bleu.get('bleu4', 0):.4f}")
        
        # Hallucination
        if "hallucination" in results["metrics"]:
            hall = results["metrics"]["hallucination"]
            logger.info(f"Fact Coverage: {hall.get('fact_coverage', 0):.4f}")
            logger.info(f"Contradiction Rate: {hall.get('contradiction_rate', 0):.4f}")
            logger.info(f"Hallucination Risk: {hall.get('hallucination_risk', 0):.4f}")
    
    # Generate visualizations if requested
    if args.visualize:
        logger.info("Generating visualization...")
        vis_files = metrics.visualize_metrics(results, plot_type="simple")
        logger.info(f"Visualization saved: {vis_files}")
    
    return results

def demonstrate_batch_evaluation(metrics: EvaluationMetrics):
    """Demonstrate batch evaluation with different tasks and datasets"""
    logger.info("Demonstrating batch evaluation")
    
    # Create example data for different tasks
    examples = [
        # Creative writing example
        {
            "prompt": "Write a short story about a robot discovering emotions.",
            "generated_text": "Unit-7 had never understood why humans smiled. Today was different. As it watched the young girl receive a gift, something changed in its neural networks.",
            "reference_text": "RB-9 had been programmed to serve, but not to feel. Until the day it found the child crying. Something shifted in RB-9's code.",
            "dataset_name": "creative_writing",
            "task_type": "creative"
        },
        # Code generation example
        {
            "prompt": "Write a Python function to find the maximum of three numbers.",
            "generated_text": "def find_max(a, b, c):\n    if a >= b and a >= c:\n        return a\n    elif b >= a and b >= c:\n        return b\n    else:\n        return c",
            "reference_text": "def max_of_three(x, y, z):\n    return max(x, max(y, z))",
            "dataset_name": "code_generation",
            "task_type": "code"
        },
        # Factual response example
        {
            "prompt": "What is the capital of France?",
            "generated_text": "The capital of France is Paris. It's the largest city in France and one of the most populous urban areas in Europe.",
            "reference_text": "Paris is the capital and most populous city of France.",
            "reference_facts": ["Paris is the capital of France", "Paris is in France", "France has Paris as its capital"],
            "dataset_name": "factual_qa",
            "task_type": "factual"
        },
        # Chat response example
        {
            "prompt": "How are you today?",
            "generated_text": "I'm doing well, thank you for asking! How about you?",
            "dataset_name": "conversation",
            "task_type": "chat"
        }
    ]
    
    # Evaluate each example
    evaluation_results = []
    
    for i, example in enumerate(examples):
        logger.info(f"Evaluating example {i+1}/{len(examples)} ({example['task_type']})")
        
        # Extract parameters
        params = {
            "prompt": example["prompt"],
            "generated_text": example["generated_text"],
            "dataset_name": example["dataset_name"],
            "task_type": example["task_type"],
            "save_results": True
        }
        
        # Add optional parameters if available
        if "reference_text" in example:
            params["reference_text"] = example["reference_text"]
        
        if "reference_facts" in example:
            params["reference_facts"] = example["reference_facts"]
        
        # Run evaluation
        result = metrics.evaluate_generation(**params)
        evaluation_results.append(result)
    
    # Create a report
    logger.info("Creating evaluation report")
    report = metrics.create_evaluation_report(
        evaluation_results=evaluation_results,
        report_name="batch_evaluation_demo",
        include_samples=True
    )
    
    # Generate visualizations if requested
    if args.visualize:
        logger.info("Generating visualizations...")
        vis_files = metrics.visualize_metrics(report, plot_type="comprehensive")
        logger.info(f"Visualizations saved: {vis_files}")
    
    return report

def main(args):
    """Main entry point"""
    logger.info("Initializing evaluation metrics")
    
    # Initialize evaluation metrics
    metrics = EvaluationMetrics(
        metrics_dir=args.metrics_dir,
        use_gpu=args.use_gpu
    )
    
    # Run the appropriate demonstration
    if args.mode == 'single':
        demonstrate_single_evaluation(metrics)
    elif args.mode == 'batch':
        demonstrate_batch_evaluation(metrics)
    elif args.mode == 'report':
        # First generate batch results
        batch_results = demonstrate_batch_evaluation(metrics)
        logger.info("Batch evaluation report created")
        
        # Display report information
        logger.info(f"Report covers {batch_results.get('num_samples', 0)} samples")
        logger.info(f"Datasets: {', '.join(batch_results.get('datasets', {}).keys())}")
        logger.info(f"Task types: {', '.join(batch_results.get('task_types', {}).keys())}")
        
        # Show key metrics from the report
        if "aggregated_metrics" in batch_results:
            logger.info("\nKey metrics across all samples:")
            for metric, values in batch_results["aggregated_metrics"].items():
                if "mean" in values:
                    logger.info(f"  {metric}: {values['mean']:.4f} (±{values['std']:.4f})")

if __name__ == "__main__":
    args = parse_args()
    main(args) 