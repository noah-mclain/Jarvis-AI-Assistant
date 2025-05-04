"""
Evaluation Metrics for Generative AI

This module provides comprehensive evaluation metrics for generative AI outputs:
1. BERTScore for semantic similarity evaluation
2. ROUGE/BLEU for text generation evaluation
3. Perplexity for language model quality assessment
4. Human feedback collection framework
5. Hallucination detection

These metrics help ensure the quality of generated text across different datasets and models.
"""

import os
import json
import torch
import logging
import numpy as np
from typing import Dict, List, Union, Optional, Any, Tuple
from datetime import datetime
from collections import defaultdict, Counter
from .utils import get_storage_path, sync_to_gdrive, ensure_directory_exists, sync_from_gdrive, is_paperspace_environment

# Configure logging
logger = logging.getLogger(__name__)

# Try to import optional dependencies, with graceful fallbacks
try:
    from bert_score import BERTScorer
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
    # Download required NLTK data if not already available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
except ImportError:
    NLTK_AVAILABLE = False

class EvaluationMetrics:
    """Class for comprehensive evaluation of generative text outputs"""
    
    def __init__(self, metrics_dir: str = "evaluation_metrics", use_gpu: bool = None):
        """
        Initialize the evaluation metrics module.
        
        Args:
            metrics_dir: Directory to save evaluation results
            use_gpu: Whether to use GPU for computations (will auto-detect if None)
        """
        self.metrics_dir = metrics_dir
        os.makedirs(metrics_dir, exist_ok=True)

        # Auto-detect GPU if not specified
        self.use_gpu = torch.cuda.is_available() if use_gpu is None else use_gpu
        # Initialize metric calculators
        self.bert_scorer = None
        self.rouge_scorer = None

        # Initialize BERTScore if available
        if BERT_SCORE_AVAILABLE:
            try:
                self.bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True, 
                                             device="cuda" if self.use_gpu else "cpu")
                print("BERTScore initialized successfully")
            except Exception as e:
                print(f"Failed to initialize BERTScore: {e}")
                self.bert_scorer = None
        else:
            print("BERTScore is not available. Install with: pip install bert-score")

        # Initialize ROUGE if available
        if ROUGE_AVAILABLE:
            try:
                self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                print("ROUGE initialized successfully")
            except Exception as e:
                print(f"Failed to initialize ROUGE: {e}")
                self.rouge_scorer = None
        else:
            print("ROUGE is not available. Install with: pip install rouge-score")

        # Initialize NLTK for BLEU if available
        if NLTK_AVAILABLE:
            self.smoothing = SmoothingFunction().method1
            print("NLTK initialized successfully for BLEU score")
        else:
            print("NLTK is not available. Install with: pip install nltk")
    
    def compute_perplexity(self, model: Any, tokenizer: Any, text: str, stride: int = 512) -> float:
        """
        Compute perplexity of text using a language model.
        
        Args:
            model: Language model (must have forward method)
            tokenizer: Tokenizer for the model
            text: Text to evaluate
            stride: Stride for long texts
            
        Returns:
            Perplexity score (lower is better)
        """
        try:
            # Tokenize input
            if hasattr(tokenizer, 'encode'):
                encodings = tokenizer.encode(text, return_tensors='pt')
            else:
                # Handle character tokenizer case
                encodings = torch.tensor([tokenizer.char_to_index.get(c, 0) for c in text]).unsqueeze(0)
            
            # Move to appropriate device
            device = next(model.parameters()).device
            encodings = encodings.to(device)
            
            # For long texts, use strided approach
            max_length = min(encodings.size(1), 1024)  # Cap at 1024 tokens to avoid OOM
            nlls = []
            
            for i in range(0, encodings.size(1), stride):
                end_idx = min(i + max_length, encodings.size(1))
                input_ids = encodings[:, i:end_idx]
                target_ids = input_ids.clone()
                
                with torch.no_grad():
                    outputs, _ = model(input_ids)
                    
                    # Compute loss
                    shift_logits = outputs[..., :-1, :].contiguous()
                    shift_labels = target_ids[..., 1:].contiguous()
                    
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                                   shift_labels.view(-1))
                    
                    # Store negative log likelihood
                    nlls.append(loss.mean().item())
            
            # Compute perplexity
            perplexity = np.exp(np.mean(nlls))
            return perplexity
        except Exception as e:
            print(f"Error computing perplexity: {e}")
            return float('inf')  # Return infinity on error
    
    def compute_bert_score(self, references: List[str], candidates: List[str]) -> Dict[str, List[float]]:
        """
        Compute BERTScore for candidate texts against references.
        
        Args:
            references: List of reference texts
            candidates: List of candidate (generated) texts
            
        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        if not BERT_SCORE_AVAILABLE or self.bert_scorer is None:
            print("BERTScore is not available")
            return {"precision": [], "recall": [], "f1": []}
        
        try:
            P, R, F1 = self.bert_scorer.score(candidates, references)
            return {
                "precision": P.tolist(),
                "recall": R.tolist(),
                "f1": F1.tolist()
            }
        except Exception as e:
            print(f"Error computing BERTScore: {e}")
            return {"precision": [], "recall": [], "f1": []}
    
    def compute_rouge(self, references: List[str], candidates: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Compute ROUGE scores for candidate texts against references.
        
        Args:
            references: List of reference texts
            candidates: List of candidate (generated) texts
            
        Returns:
            Dictionary with ROUGE scores (rouge1, rouge2, rougeL)
        """
        if not ROUGE_AVAILABLE or self.rouge_scorer is None:
            print("ROUGE is not available")
            return {}
        
        try:
            scores = defaultdict(list)
            
            for ref, cand in zip(references, candidates):
                rouge_scores = self.rouge_scorer.score(ref, cand)
                for key, value in rouge_scores.items():
                    scores[f"{key}_precision"].append(value.precision)
                    scores[f"{key}_recall"].append(value.recall)
                    scores[f"{key}_fmeasure"].append(value.fmeasure)
            
            # Average scores
            avg_scores = {k: np.mean(v) for k, v in scores.items()}
            
            return avg_scores
        except Exception as e:
            print(f"Error computing ROUGE: {e}")
            return {}
    
    def compute_bleu(self, references: List[str], candidates: List[str]) -> Dict[str, float]:
        """
        Compute BLEU scores for candidate texts against references.
        
        Args:
            references: List of reference texts
            candidates: List of candidate (generated) texts
            
        Returns:
            Dictionary with BLEU scores (BLEU-1, BLEU-2, BLEU-3, BLEU-4)
        """
        if not NLTK_AVAILABLE:
            print("NLTK is not available")
            return {}

        try:
            bleu_scores = {
                "bleu1": [],
                "bleu2": [],
                "bleu3": [],
                "bleu4": []
            }

            for ref, cand in zip(references, candidates):
                ref_tokens = nltk.word_tokenize(ref.lower())
                if cand_tokens := nltk.word_tokenize(cand.lower()):
                    bleu1 = sentence_bleu([ref_tokens], cand_tokens, 
                                        weights=(1, 0, 0, 0), smoothing_function=self.smoothing)
                    bleu2 = sentence_bleu([ref_tokens], cand_tokens, 
                                        weights=(0.5, 0.5, 0, 0), smoothing_function=self.smoothing)
                    bleu3 = sentence_bleu([ref_tokens], cand_tokens, 
                                        weights=(0.33, 0.33, 0.33, 0), smoothing_function=self.smoothing)
                    bleu4 = sentence_bleu([ref_tokens], cand_tokens, 
                                        weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self.smoothing)

                    bleu_scores["bleu1"].append(bleu1)
                    bleu_scores["bleu2"].append(bleu2)
                    bleu_scores["bleu3"].append(bleu3)
                    bleu_scores["bleu4"].append(bleu4)

            # Average scores
            avg_scores = {k: np.mean(v) if v else 0.0 for k, v in bleu_scores.items()}

            return avg_scores
        except Exception as e:
            print(f"Error computing BLEU: {e}")
            return {}
    
    def detect_hallucination(self, reference_facts: List[str], generated_text: str) -> Dict[str, Any]:
        """
        Detect potential hallucinations in generated text by comparing against reference facts.
        
        Args:
            reference_facts: List of factual statements that should be present/consistent
            generated_text: Generated text to evaluate for hallucinations
            
        Returns:
            Dictionary with hallucination metrics
        """
        try:
            # Simple approach: check what percentage of facts are reflected in the text
            fact_presence_count = 0
            fact_contradictions = []
            
            for fact in reference_facts:
                # Check for presence of fact (simple substring check)
                fact_tokens = set(nltk.word_tokenize(fact.lower()))
                gen_tokens = set(nltk.word_tokenize(generated_text.lower()))
                
                # Calculate word overlap
                overlap = len(fact_tokens.intersection(gen_tokens)) / len(fact_tokens) if fact_tokens else 0
                
                if overlap > 0.7:  # If 70% of fact words are present
                    fact_presence_count += 1
                else:
                    # Check for contradictions - this is a simple heuristic
                    # A more sophisticated approach would use an entailment model
                    contradiction_words = ['not', 'never', 'no', 'none', 'nothing', 'neither']
                    fact_has_neg = any(word in fact_tokens for word in contradiction_words)
                    gen_has_neg = any(word in gen_tokens for word in contradiction_words)
                    
                    # If negation differs and there's some overlap, might be a contradiction
                    if fact_has_neg != gen_has_neg and overlap > 0.3:
                        fact_contradictions.append(fact)
            
            # Calculate metrics
            fact_coverage = fact_presence_count / len(reference_facts) if reference_facts else 1.0
            contradiction_rate = len(fact_contradictions) / len(reference_facts) if reference_facts else 0.0
            
            return {
                "fact_coverage": fact_coverage,
                "contradiction_rate": contradiction_rate,
                "contradicted_facts": fact_contradictions,
                "hallucination_risk": 1.0 - fact_coverage if fact_coverage < 0.5 else contradiction_rate,
            }
        except Exception as e:
            print(f"Error detecting hallucinations: {e}")
            return {
                "fact_coverage": 0.0,
                "contradiction_rate": 1.0,
                "contradicted_facts": [],
                "hallucination_risk": 1.0,
                "error": str(e)
            }
    
    def collect_human_feedback(self, generated_text: str, prompt: str, 
                             save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Interactive tool to collect human feedback on generated text.
        
        Args:
            generated_text: Text generated by the model
            prompt: Original prompt used to generate the text
            save_path: Path to save feedback results (optional)
            
        Returns:
            Dictionary with feedback metrics
        """
        print("\n" + "="*80)
        print("Human Feedback Collection")
        print("="*80)

        print(f"\nOriginal prompt: {prompt}")
        print("\nGenerated text:")
        print("-"*40)
        print(generated_text)
        print("-"*40)

        # Collect ratings
        try:
            return self.collect_ratings(
                prompt, generated_text, save_path
            )
        except Exception as e:
            print(f"Error collecting feedback: {e}")
            return {"error": str(e)}

    def collect_ratings(self, prompt, generated_text, save_path):
        print("\nPlease rate the generated text on the following criteria (1-5, where 5 is best):")
        ratings = {"relevance": int(input("Relevance to prompt (1-5): "))}
        ratings["factuality"] = int(input("Factual accuracy (1-5): "))
        ratings["coherence"] = int(input("Coherence and fluency (1-5): "))
        ratings["quality"] = int(input("Overall quality (1-5): "))

        # Collect qualitative feedback
        print("\nPlease provide any additional feedback:")
        feedback = input("> ")

        # Collect hallucination assessment
        print("\nDid you notice any factual errors or hallucinations? (y/n)")
        has_hallucination = input("> ").lower().startswith('y')

        if has_hallucination:
            print("Please describe the hallucinations you observed:")
            hallucination_details = input("> ")
        else:
            hallucination_details = ""

        # Prepare results
        result = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "generated_text": generated_text,
            "ratings": ratings,
            "qualitative_feedback": feedback,
            "has_hallucination": has_hallucination,
            "hallucination_details": hallucination_details
        }

        # Save feedback if requested
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Append to existing file if it exists
            if os.path.exists(save_path):
                with open(save_path, 'r') as f:
                    try:
                        existing_data = json.load(f)
                        if isinstance(existing_data, list):
                            existing_data.append(result)
                        else:
                            existing_data = [existing_data, result]
                    except json.JSONDecodeError:
                        existing_data = [result]

                with open(save_path, 'w') as f:
                    json.dump(existing_data, f, indent=2)
            else:
                with open(save_path, 'w') as f:
                    json.dump([result], f, indent=2)

            print(f"\nFeedback saved to {save_path}")

        return result
    
    def evaluate_generation(self, prompt: str, generated_text: str, 
                           reference_text: Optional[str] = None,
                           reference_facts: Optional[List[str]] = None,
                           collect_human_feedback: bool = False,
                           model: Optional[Any] = None,
                           tokenizer: Optional[Any] = None,
                           dataset_name: str = "unknown",
                           save_results: bool = True) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a generated text.
        
        Args:
            prompt: Original prompt
            generated_text: Text generated by the model
            reference_text: Reference/ground truth text if available
            reference_facts: List of factual statements for hallucination detection
            collect_human_feedback: Whether to collect human feedback
            model: Language model for perplexity (optional)
            tokenizer: Tokenizer for the model (optional)
            dataset_name: Name of the dataset being evaluated
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary with all evaluation metrics
        """
        results = {
            "prompt": prompt,
            "generated_text": generated_text,
            "dataset": dataset_name,
            "timestamp": datetime.now().isoformat()
        }
        
        # Compute all available metrics
        metrics = {}
        
        # BERTScore (if reference is available)
        if reference_text:
            results["reference_text"] = reference_text
            metrics["bert_score"] = self.compute_bert_score([reference_text], [generated_text])
            metrics["rouge"] = self.compute_rouge([reference_text], [generated_text])
            metrics["bleu"] = self.compute_bleu([reference_text], [generated_text])
        
        # Hallucination detection (if reference facts are available)
        if reference_facts:
            metrics["hallucination"] = self.detect_hallucination(reference_facts, generated_text)
        
        # Perplexity (if model and tokenizer are available)
        if model and tokenizer:
            metrics["perplexity"] = self.compute_perplexity(model, tokenizer, generated_text)
        
        # Add metrics to results
        results["metrics"] = metrics
        
        # Collect human feedback if requested
        if collect_human_feedback:
            results["human_feedback"] = self.collect_human_feedback(
                generated_text, prompt, 
                save_path=os.path.join(self.metrics_dir, f"human_feedback_{dataset_name}.json")
            )
        
        # Save results if requested
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = os.path.join(self.metrics_dir, f"evaluation_{dataset_name}_{timestamp}.json")
            
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Evaluation results saved to {results_path}")
        
        return results
    
    def batch_evaluate(self, prompts: List[str], generated_texts: List[str],
                      reference_texts: Optional[List[str]] = None,
                      dataset_name: str = "unknown") -> Dict[str, Any]:
        """
        Batch evaluation of multiple generated texts.
        
        Args:
            prompts: List of prompts
            generated_texts: List of generated texts
            reference_texts: List of reference texts (optional)
            dataset_name: Name of the dataset being evaluated
            
        Returns:
            Dictionary with aggregated evaluation metrics
        """
        individual_results = []
        
        # Process each text
        for i, (prompt, generated) in enumerate(zip(prompts, generated_texts)):
            reference = reference_texts[i] if reference_texts else None
            
            print(f"Evaluating sample {i+1}/{len(prompts)}...")
            result = self.evaluate_generation(
                prompt=prompt,
                generated_text=generated,
                reference_text=reference,
                dataset_name=dataset_name,
                save_results=False  # Don't save individual results
            )
            
            individual_results.append(result)
        
        # Aggregate results
        aggregated = {
            "dataset": dataset_name,
            "num_samples": len(prompts),
            "timestamp": datetime.now().isoformat()
        }
        
        # Combine metrics
        metrics = defaultdict(list)
        
        for result in individual_results:
            for metric_name, metric_value in result.get("metrics", {}).items():
                if isinstance(metric_value, dict):
                    # For nested metrics like BERT score
                    for k, v in metric_value.items():
                        if isinstance(v, (int, float)):
                            metrics[f"{metric_name}_{k}"].append(v)
                        elif isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
                            metrics[f"{metric_name}_{k}"].extend(v)
                elif isinstance(metric_value, (int, float)):
                    metrics[metric_name].append(metric_value)
        
        # Calculate averages
        aggregated["metrics"] = {k: np.mean(v) for k, v in metrics.items() if v}
        
        # Save aggregated results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(self.metrics_dir, f"batch_evaluation_{dataset_name}_{timestamp}.json")
        
        with open(results_path, 'w') as f:
            json.dump(aggregated, f, indent=2)
        
        print(f"Batch evaluation results saved to {results_path}")
        return aggregated

    @staticmethod
    def install_dependencies():
        """Install required dependencies for evaluation metrics"""
        try:
            import subprocess
            import sys
            
            print("Installing evaluation metric dependencies...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "bert-score", "rouge-score", "nltk", "transformers"
            ])
            
            # Initialize NLTK
            import nltk
            nltk.download('punkt')
            
            print("Successfully installed all dependencies for evaluation metrics")
            return True
        except Exception as e:
            print(f"Error installing dependencies: {e}")
            return False

def save_metrics(metrics, model_name, dataset_name, timestamp=None):
    """
    Save evaluation metrics to a JSON file and sync to Google Drive.
    
    Args:
        metrics (dict): Dictionary of metrics
        model_name (str): Name of the model
        dataset_name (str): Name of the dataset
        timestamp (str, optional): Timestamp to use in the filename
    
    Returns:
        str: Path to the saved metrics file
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use storage path utility to get correct path and ensure it exists
    metrics_dir = ensure_directory_exists("metrics")
    
    # Create a clean filename
    model_name_clean = model_name.replace('/', '_')
    dataset_name_clean = dataset_name.replace('/', '_')
    
    filename = f"{model_name_clean}_{dataset_name_clean}_{timestamp}.json"
    filepath = os.path.join(metrics_dir, filename)
    
    # Add metadata
    metrics_with_meta = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "timestamp": timestamp,
        "metrics": metrics
    }
    
    # Save the metrics
    with open(filepath, 'w') as f:
        json.dump(metrics_with_meta, f, indent=2)
    
    # Sync to Google Drive if in Paperspace
    if is_paperspace_environment():
        sync_to_gdrive("metrics")
        logger.info(f"Saved metrics to {filepath} and synced to Google Drive")
    else:
        logger.info(f"Saved metrics to {filepath}")
    
    return filepath

class MetricLogger:
    """
    Logger for tracking and saving metrics during model evaluation.
    """
    
    def __init__(self, model_name, dataset_name):
        """
        Initialize the metric logger.
        
        Args:
            model_name (str): Name of the model being evaluated
            dataset_name (str): Name of the dataset being used
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.metrics = {}
        self.start_time = datetime.now()
        
        # Create timestamp for this evaluation run
        self.timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        
        # Use storage path utility to get metrics directory and ensure it exists
        self.metrics_dir = ensure_directory_exists("metrics")
    
    def log_metric(self, metric_name, value, step=None):
        """Log a single metric value."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        entry = {"value": value, "timestamp": datetime.now().isoformat()}
        if step is not None:
            entry["step"] = step
        
        self.metrics[metric_name].append(entry)
        logger.info(f"Metric {metric_name}: {value}")
    
    def log_metrics(self, metrics_dict, step=None):
        """Log multiple metrics at once."""
        for metric_name, value in metrics_dict.items():
            self.log_metric(metric_name, value, step)
    
    def save(self):
        """Save all metrics to a file and sync to Google Drive."""
        return save_metrics(self.metrics, self.model_name, self.dataset_name, self.timestamp)
    
    def save_and_sync(self, include_samples=True):
        """
        Save metrics and sync to Google Drive, with option to include sample outputs.
        
        Args:
            include_samples (bool): Whether to include sample outputs in the metrics
            
        Returns:
            str: Path to the saved metrics file
        """
        # Add run summary metrics
        self.metrics["run_summary"] = {
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds()
        }
        
        # Save the metrics
        filepath = self.save()
        
        # Sync to Google Drive if in Paperspace
        if is_paperspace_environment():
            sync_to_gdrive("metrics")
            logger.info(f"Metrics synced to Google Drive: {filepath}")
        
        return filepath

# Example usage
if __name__ == "__main__":
    # Install dependencies if needed
    if not all([BERT_SCORE_AVAILABLE, ROUGE_AVAILABLE, NLTK_AVAILABLE]):
        EvaluationMetrics.install_dependencies()
    
    # Initialize metrics
    metrics = EvaluationMetrics()
    
    # Example evaluation
    prompt = "Write a short story about a robot who discovers emotions."
    generated = """
    Unit-7 had never understood why humans smiled. Its processors could analyze the 
    facial movement, but the purpose remained elusive. Today was different. As it watched 
    the young girl receive a gift, something changed in its neural networks.
    
    "Are you happy?" the girl asked, noticing Unit-7's gaze.
    
    "I am... operational," it responded, but something felt incomplete about the answer.
    
    The girl placed her small hand on Unit-7's metal one. "I'm happy you're my friend."
    
    A strange sensation spread through Unit-7's circuits. If this was happiness, 
    it wanted to understand more.
    """
    
    reference = """
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
    
    # Run evaluation
    results = metrics.evaluate_generation(
        prompt=prompt,
        generated_text=generated,
        reference_text=reference,
        reference_facts=facts,
        collect_human_feedback=True,
        dataset_name="story_generation"
    )
    
    print("\nEvaluation complete!") 