"""
Consolidated Evaluation Module

This module provides a unified interface for evaluating text and code generation models.
It consolidates functionality from:
- evaluate_generation.py
- evaluation_example.py
- evaluation_metrics.py

Features:
- Comprehensive evaluation metrics for text generation
- Code-specific evaluation metrics
- Visualization of evaluation results
- Comparison of different models
"""

import os
import sys
import json
import numpy as np
import logging
import time
import re
import string
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import Counter
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - some evaluation metrics will be limited")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available - BLEU score calculation will be limited")

try:
    from rouge import Rouge
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logger.warning("Rouge not available - ROUGE score calculation will be limited")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logger.warning("Matplotlib/Seaborn not available - visualization will be limited")

# Utility functions
def normalize_text(text):
    """Normalize text for evaluation"""
    if not text:
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def tokenize_text(text):
    """Tokenize text for evaluation"""
    if NLTK_AVAILABLE:
        return word_tokenize(text)
    else:
        # Simple fallback tokenization
        return text.split()

class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for text and code generation.

    This class provides methods for:
    - Calculating BLEU, ROUGE, and other text similarity metrics
    - Evaluating code correctness and functionality
    - Visualizing evaluation results
    - Comparing different models
    """

    def __init__(self):
        """Initialize the evaluation metrics"""
        self.metrics = {}

    def calculate_bleu(self, reference, candidate):
        """
        Calculate BLEU score between reference and candidate texts.

        Args:
            reference: Reference text (ground truth)
            candidate: Candidate text (generated)

        Returns:
            BLEU score (0-1)
        """
        if not NLTK_AVAILABLE:
            logger.warning("NLTK not available - using simplified BLEU calculation")
            return self._simplified_bleu(reference, candidate)

        # Normalize and tokenize
        reference = normalize_text(reference)
        candidate = normalize_text(candidate)

        reference_tokens = tokenize_text(reference)
        candidate_tokens = tokenize_text(candidate)

        # Calculate BLEU score with smoothing
        smoothing = SmoothingFunction().method1

        # Handle empty sequences
        if not reference_tokens or not candidate_tokens:
            return 0.0

        try:
            # Calculate BLEU-1, BLEU-2, BLEU-3, and BLEU-4
            bleu1 = sentence_bleu([reference_tokens], candidate_tokens,
                                weights=(1, 0, 0, 0), smoothing_function=smoothing)
            bleu2 = sentence_bleu([reference_tokens], candidate_tokens,
                                weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
            bleu3 = sentence_bleu([reference_tokens], candidate_tokens,
                                weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
            bleu4 = sentence_bleu([reference_tokens], candidate_tokens,
                                weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)

            return {
                "bleu1": bleu1,
                "bleu2": bleu2,
                "bleu3": bleu3,
                "bleu4": bleu4,
                "bleu": bleu4  # Use BLEU-4 as the main BLEU score
            }
        except Exception as e:
            logger.error(f"Error calculating BLEU score: {str(e)}")
            return {
                "bleu1": 0.0,
                "bleu2": 0.0,
                "bleu3": 0.0,
                "bleu4": 0.0,
                "bleu": 0.0
            }

    def _simplified_bleu(self, reference, candidate):
        """Simplified BLEU calculation for when NLTK is not available"""
        # Normalize and tokenize
        reference = normalize_text(reference)
        candidate = normalize_text(candidate)

        reference_tokens = reference.split()
        candidate_tokens = candidate.split()

        # Handle empty sequences
        if not reference_tokens or not candidate_tokens:
            return {
                "bleu1": 0.0,
                "bleu2": 0.0,
                "bleu3": 0.0,
                "bleu4": 0.0,
                "bleu": 0.0
            }

        # Calculate n-gram precision
        def ngram_precision(n):
            ref_ngrams = Counter()
            for i in range(len(reference_tokens) - n + 1):
                ngram = tuple(reference_tokens[i:i+n])
                ref_ngrams[ngram] += 1

            cand_ngrams = Counter()
            for i in range(len(candidate_tokens) - n + 1):
                ngram = tuple(candidate_tokens[i:i+n])
                cand_ngrams[ngram] += 1

            # Count matching n-grams
            matches = 0
            for ngram, count in cand_ngrams.items():
                matches += min(count, ref_ngrams.get(ngram, 0))

            # Calculate precision
            if sum(cand_ngrams.values()) == 0:
                return 0.0

            return matches / sum(cand_ngrams.values())

        # Calculate brevity penalty
        bp = min(1.0, len(candidate_tokens) / max(1, len(reference_tokens)))

        # Calculate BLEU-1, BLEU-2, BLEU-3, and BLEU-4
        p1 = ngram_precision(1)
        p2 = ngram_precision(2)
        p3 = ngram_precision(3)
        p4 = ngram_precision(4)

        # Avoid log(0)
        p1 = max(p1, 1e-10)
        p2 = max(p2, 1e-10)
        p3 = max(p3, 1e-10)
        p4 = max(p4, 1e-10)

        bleu1 = bp * p1
        bleu2 = bp * (p1 * p2) ** 0.5
        bleu3 = bp * (p1 * p2 * p3) ** (1/3)
        bleu4 = bp * (p1 * p2 * p3 * p4) ** 0.25

        return {
            "bleu1": bleu1,
            "bleu2": bleu2,
            "bleu3": bleu3,
            "bleu4": bleu4,
            "bleu": bleu4  # Use BLEU-4 as the main BLEU score
        }

    def calculate_rouge(self, reference, candidate):
        """
        Calculate ROUGE scores between reference and candidate texts.

        Args:
            reference: Reference text (ground truth)
            candidate: Candidate text (generated)

        Returns:
            Dictionary with ROUGE scores
        """
        if not ROUGE_AVAILABLE:
            logger.warning("Rouge not available - using simplified ROUGE calculation")
            return self._simplified_rouge(reference, candidate)

        # Normalize texts
        reference = normalize_text(reference)
        candidate = normalize_text(candidate)

        # Handle empty texts
        if not reference or not candidate:
            return {
                "rouge1": {"f": 0.0, "p": 0.0, "r": 0.0},
                "rouge2": {"f": 0.0, "p": 0.0, "r": 0.0},
                "rougeL": {"f": 0.0, "p": 0.0, "r": 0.0}
            }

        try:
            # Calculate ROUGE scores
            rouge = Rouge()
            scores = rouge.get_scores(candidate, reference)[0]

            return scores
        except Exception as e:
            logger.error(f"Error calculating ROUGE score: {str(e)}")
            return {
                "rouge1": {"f": 0.0, "p": 0.0, "r": 0.0},
                "rouge2": {"f": 0.0, "p": 0.0, "r": 0.0},
                "rougeL": {"f": 0.0, "p": 0.0, "r": 0.0}
            }

    def _simplified_rouge(self, reference, candidate):
        """Simplified ROUGE calculation for when Rouge is not available"""
        # Normalize and tokenize
        reference = normalize_text(reference)
        candidate = normalize_text(candidate)

        reference_tokens = reference.split()
        candidate_tokens = candidate.split()

        # Handle empty sequences
        if not reference_tokens or not candidate_tokens:
            return {
                "rouge1": {"f": 0.0, "p": 0.0, "r": 0.0},
                "rouge2": {"f": 0.0, "p": 0.0, "r": 0.0},
                "rougeL": {"f": 0.0, "p": 0.0, "r": 0.0}
            }

        # Calculate n-gram overlap
        def ngram_overlap(n):
            ref_ngrams = Counter()
            for i in range(len(reference_tokens) - n + 1):
                ngram = tuple(reference_tokens[i:i+n])
                ref_ngrams[ngram] += 1

            cand_ngrams = Counter()
            for i in range(len(candidate_tokens) - n + 1):
                ngram = tuple(candidate_tokens[i:i+n])
                cand_ngrams[ngram] += 1

            # Count matching n-grams
            matches = 0
            for ngram, count in cand_ngrams.items():
                matches += min(count, ref_ngrams.get(ngram, 0))

            # Calculate precision and recall
            precision = matches / max(1, sum(cand_ngrams.values()))
            recall = matches / max(1, sum(ref_ngrams.values()))

            # Calculate F1 score
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)

            return {"f": f1, "p": precision, "r": recall}

        # Calculate ROUGE-1 and ROUGE-2
        rouge1 = ngram_overlap(1)
        rouge2 = ngram_overlap(2)

        # Calculate ROUGE-L (longest common subsequence)
        lcs_length = self._longest_common_subsequence(reference_tokens, candidate_tokens)

        precision = lcs_length / max(1, len(candidate_tokens))
        recall = lcs_length / max(1, len(reference_tokens))

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        rougeL = {"f": f1, "p": precision, "r": recall}

        return {
            "rouge1": rouge1,
            "rouge2": rouge2,
            "rougeL": rougeL
        }

    def _longest_common_subsequence(self, seq1, seq2):
        """Calculate the length of the longest common subsequence"""
        m, n = len(seq1), len(seq2)

        # Create a table to store the LCS lengths
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Fill the table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]

    def evaluate_text_generation(self, references, candidates):
        """
        Evaluate text generation with multiple metrics.

        Args:
            references: List of reference texts
            candidates: List of candidate texts

        Returns:
            Dictionary with evaluation metrics
        """
        if len(references) != len(candidates):
            raise ValueError("Number of references and candidates must match")

        # Initialize metrics
        metrics = {
            "bleu": 0.0,
            "rouge1_f": 0.0,
            "rouge2_f": 0.0,
            "rougeL_f": 0.0,
            "exact_match": 0,
            "token_overlap": 0.0
        }

        # Calculate metrics for each pair
        for reference, candidate in zip(references, candidates):
            # BLEU score
            bleu_scores = self.calculate_bleu(reference, candidate)
            metrics["bleu"] += bleu_scores["bleu"]

            # ROUGE scores
            rouge_scores = self.calculate_rouge(reference, candidate)
            metrics["rouge1_f"] += rouge_scores["rouge1"]["f"]
            metrics["rouge2_f"] += rouge_scores["rouge2"]["f"]
            metrics["rougeL_f"] += rouge_scores["rougeL"]["f"]

            # Exact match
            if normalize_text(reference) == normalize_text(candidate):
                metrics["exact_match"] += 1

            # Token overlap
            ref_tokens = set(tokenize_text(normalize_text(reference)))
            cand_tokens = set(tokenize_text(normalize_text(candidate)))

            if ref_tokens and cand_tokens:
                overlap = len(ref_tokens.intersection(cand_tokens)) / len(ref_tokens.union(cand_tokens))
                metrics["token_overlap"] += overlap

        # Calculate averages
        num_samples = len(references)
        metrics["bleu"] /= num_samples
        metrics["rouge1_f"] /= num_samples
        metrics["rouge2_f"] /= num_samples
        metrics["rougeL_f"] /= num_samples
        metrics["exact_match"] /= num_samples  # Convert to ratio
        metrics["token_overlap"] /= num_samples

        return metrics

    def visualize_metrics(self, metrics, output_path=None):
        """
        Visualize evaluation metrics.

        Args:
            metrics: Dictionary with evaluation metrics
            output_path: Path to save the visualization (optional)
        """
        if not VISUALIZATION_AVAILABLE:
            logger.warning("Visualization libraries not available")
            return

        # Create a bar chart
        plt.figure(figsize=(10, 6))

        # Filter metrics for visualization
        viz_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}

        # Create bar chart
        sns.barplot(x=list(viz_metrics.keys()), y=list(viz_metrics.values()))

        # Add labels and title
        plt.xlabel("Metric")
        plt.ylabel("Score")
        plt.title("Evaluation Metrics")

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)

        # Add values on top of bars
        for i, v in enumerate(viz_metrics.values()):
            plt.text(i, v + 0.01, f"{v:.4f}", ha="center")

        plt.tight_layout()

        # Save or show the plot
        if output_path:
            plt.savefig(output_path)
            logger.info(f"Visualization saved to {output_path}")
        else:
            plt.show()

    def evaluate_code_generation(self, references, candidates, execute_code=False):
        """
        Evaluate code generation with multiple metrics.

        Args:
            references: List of reference code snippets
            candidates: List of generated code snippets
            execute_code: Whether to execute the code for functional testing

        Returns:
            Dictionary with evaluation metrics
        """
        if len(references) != len(candidates):
            raise ValueError("Number of references and candidates must match")

        # Initialize metrics
        metrics = {
            "bleu": 0.0,
            "exact_match": 0,
            "syntax_correct": 0,
            "functional_correct": 0,
            "token_overlap": 0.0
        }

        # Calculate metrics for each pair
        for reference, candidate in zip(references, candidates):
            # BLEU score
            bleu_scores = self.calculate_bleu(reference, candidate)
            metrics["bleu"] += bleu_scores["bleu"]

            # Exact match
            if reference.strip() == candidate.strip():
                metrics["exact_match"] += 1

            # Token overlap
            ref_tokens = set(tokenize_text(reference))
            cand_tokens = set(tokenize_text(candidate))

            if ref_tokens and cand_tokens:
                overlap = len(ref_tokens.intersection(cand_tokens)) / len(ref_tokens.union(cand_tokens))
                metrics["token_overlap"] += overlap

            # Syntax correctness
            if self._check_syntax(candidate):
                metrics["syntax_correct"] += 1

            # Functional correctness (if requested)
            if execute_code and self._check_functionality(reference, candidate):
                metrics["functional_correct"] += 1

        # Calculate averages
        num_samples = len(references)
        metrics["bleu"] /= num_samples
        metrics["exact_match"] /= num_samples
        metrics["syntax_correct"] /= num_samples
        metrics["token_overlap"] /= num_samples

        if execute_code:
            metrics["functional_correct"] /= num_samples

        return metrics

    def _check_syntax(self, code):
        """Check if code has correct syntax"""
        try:
            # Try to parse the code
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError:
            return False

    def _check_functionality(self, reference, candidate):
        """
        Check if candidate code has the same functionality as reference code.
        This is a simplified implementation that would need to be expanded for real use.
        """
        # This is a placeholder for actual functionality testing
        # In a real implementation, you would:
        # 1. Execute both code snippets in a sandbox
        # 2. Compare outputs for a set of test inputs
        # 3. Return True if outputs match for all test inputs

        # For now, just return True if syntax is correct
        return self._check_syntax(candidate)

    def compare_models(self, model_results, output_path=None):
        """
        Compare evaluation results from multiple models.

        Args:
            model_results: Dictionary mapping model names to their evaluation metrics
            output_path: Path to save the visualization (optional)

        Returns:
            Dictionary with comparison results
        """
        if not VISUALIZATION_AVAILABLE:
            logger.warning("Visualization libraries not available")
            return model_results

        # Create a comparison dataframe
        import pandas as pd

        # Extract common metrics
        common_metrics = set()
        for metrics in model_results.values():
            common_metrics.update(metrics.keys())

        # Create a dataframe
        data = []
        for model_name, metrics in model_results.items():
            row = {"model": model_name}
            for metric in common_metrics:
                row[metric] = metrics.get(metric, 0.0)
            data.append(row)

        df = pd.DataFrame(data)

        # Create a comparison visualization
        plt.figure(figsize=(12, 8))

        # Create a grouped bar chart
        metrics_to_plot = [m for m in common_metrics if m != "model"]
        df_melted = pd.melt(df, id_vars=["model"], value_vars=metrics_to_plot, var_name="metric", value_name="value")

        sns.barplot(x="model", y="value", hue="metric", data=df_melted)

        # Add labels and title
        plt.xlabel("Model")
        plt.ylabel("Score")
        plt.title("Model Comparison")

        # Adjust legend
        plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()

        # Save or show the plot
        if output_path:
            plt.savefig(output_path)
            logger.info(f"Comparison visualization saved to {output_path}")
        else:
            plt.show()

        return model_results

# Create a singleton instance for easy import
evaluation_metrics = EvaluationMetrics()

# Expose key functions for backward compatibility
def evaluate_generation(references, candidates):
    """Evaluate text generation"""
    return evaluation_metrics.evaluate_text_generation(references, candidates)

def calculate_bleu(reference, candidate):
    """Calculate BLEU score"""
    return evaluation_metrics.calculate_bleu(reference, candidate)

def calculate_rouge(reference, candidate):
    """Calculate ROUGE scores"""
    return evaluation_metrics.calculate_rouge(reference, candidate)
