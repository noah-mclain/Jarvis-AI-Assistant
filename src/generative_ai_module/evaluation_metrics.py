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
    
    def __init__(self, metrics_dir: str = "evaluation_metrics", use_gpu: bool = None, 
                bert_model: str = "microsoft/deberta-xlarge-mnli"):
        """
        Initialize the evaluation metrics module.
        
        Args:
            metrics_dir: Directory to save evaluation results
            use_gpu: Whether to use GPU for computations (will auto-detect if None)
            bert_model: Model to use for BERTScore computation
        """
        self.metrics_dir = metrics_dir
        os.makedirs(metrics_dir, exist_ok=True)

        # Auto-detect GPU if not specified
        self.use_gpu = torch.cuda.is_available() if use_gpu is None else use_gpu
        # Initialize metric calculators
        self.bert_scorer = None
        self.rouge_scorer = None
        self.bert_model = bert_model

        # Initialize BERTScore if available
        if BERT_SCORE_AVAILABLE:
            try:
                # Use a more advanced model for better semantic evaluation
                self.bert_scorer = BERTScorer(
                    model_type=self.bert_model,
                    lang="en", 
                    rescale_with_baseline=True, 
                    device="cuda" if self.use_gpu else "cpu",
                    batch_size=8  # Increase batch size for faster evaluation
                )
                logger.info(f"BERTScore initialized successfully with model {self.bert_model}")
            except Exception as e:
                logger.error(f"Failed to initialize BERTScore with {self.bert_model}: {e}")
                # Fallback to default model
                try:
                    self.bert_scorer = BERTScorer(
                        lang="en", 
                        rescale_with_baseline=True, 
                        device="cuda" if self.use_gpu else "cpu"
                    )
                    logger.info("BERTScore initialized with fallback model")
                except Exception as e2:
                    logger.error(f"Failed to initialize BERTScore fallback: {e2}")
                    self.bert_scorer = None
        else:
            logger.warning("BERTScore is not available. Install with: pip install bert-score")

        # Initialize ROUGE if available
        if ROUGE_AVAILABLE:
            try:
                self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                logger.info("ROUGE initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize ROUGE: {e}")
                self.rouge_scorer = None
        else:
            logger.warning("ROUGE is not available. Install with: pip install rouge-score")

        # Initialize NLTK for BLEU if available
        if NLTK_AVAILABLE:
            self.smoothing = SmoothingFunction().method1
            logger.info("NLTK initialized successfully for BLEU score")
        else:
            logger.warning("NLTK is not available. Install with: pip install nltk")
    
    def compute_perplexity(self, model: Any, tokenizer: Any, text: str, 
                           stride: int = 512, max_length: int = 1024) -> Dict[str, float]:
        """
        Compute perplexity of text using a language model.
        
        Args:
            model: Language model (must have forward method)
            tokenizer: Tokenizer for the model
            text: Text to evaluate
            stride: Stride for sliding window in long texts
            max_length: Maximum sequence length to process at once
            
        Returns:
            Dictionary with perplexity metrics (lower is better)
        """
        if not text.strip():
            logger.warning("Empty text provided for perplexity calculation")
            return {"perplexity": float('inf'), "error": "Empty text"}
            
        try:
            # Detect if this is a HuggingFace tokenizer
            is_hf_tokenizer = hasattr(tokenizer, 'encode') and hasattr(tokenizer, 'batch_encode_plus')
            
            # Tokenize input based on tokenizer type
            if is_hf_tokenizer:
                # Use HuggingFace tokenizer
                encoding = tokenizer.encode(text, return_tensors='pt', truncation=False)
            elif hasattr(tokenizer, 'encode'):
                # Basic encode method
                encoding = tokenizer.encode(text, return_tensors='pt')
            elif hasattr(tokenizer, 'tokenize'):
                # Handle tokenize method
                tokens = tokenizer.tokenize(text)
                encoding = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)], dtype=torch.long)
            else:
                # Fall back to character-level tokenization
                char_to_idx = getattr(tokenizer, 'char_to_index', {})
                if not char_to_idx and hasattr(tokenizer, 'get_vocab'):
                    char_to_idx = tokenizer.get_vocab()
                
                if char_to_idx:
                    encoding = torch.tensor([[char_to_idx.get(c, 0) for c in text]], dtype=torch.long)
                else:
                    logger.error("Could not determine tokenization method")
                    return {"perplexity": float('inf'), "error": "Incompatible tokenizer"}
            
            # Get model device
            device = next(model.parameters()).device
            encoding = encoding.to(device)
            
            # For long texts, use strided approach to avoid OOM
            seq_len = encoding.size(1)
            nlls = []
            tokens_processed = 0
            
            # Process in strided windows
            for i in range(0, seq_len, stride):
                end_idx = min(i + max_length, seq_len)
                chunk_len = end_idx - i
                
                if chunk_len < 2:  # Need at least 2 tokens for loss computation
                    continue
                
                input_ids = encoding[:, i:end_idx]
                target_ids = input_ids.clone()
                
                with torch.no_grad():
                    # Handle different model return types
                    try:
                        # Try standard HF-style model first
                        outputs = model(input_ids)
                        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                    except (AttributeError, TypeError, ValueError):
                        try:
                            # Try tuple output (logits, hidden_states)
                            outputs, _ = model(input_ids)
                            logits = outputs
                        except (ValueError, TypeError):
                            # Last fallback - just run the model
                            logits = model(input_ids)
                    
                    # Compute loss (shift logits and labels for causal LM)
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = target_ids[..., 1:].contiguous()
                    
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    loss_vals = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                                      shift_labels.view(-1))
                    
                    # Only consider loss on actual tokens (exclude padding)
                    num_active_tokens = (shift_labels != getattr(tokenizer, 'pad_token_id', -100)).sum()
                    if num_active_tokens > 0:
                        effective_loss = loss_vals.sum() / num_active_tokens
                    else:
                        effective_loss = loss_vals.mean()
                    
                    nlls.append(effective_loss.item())
                    tokens_processed += chunk_len - 1  # Exclude first token in each window except first
            
            # Compute metrics
            if not nlls:
                return {"perplexity": float('inf'), "error": "No valid segments for perplexity calculation"}
                
            avg_nll = np.mean(nlls)
            ppl = np.exp(min(avg_nll, 20))  # Cap at exp(20) to prevent overflow
            
            return {
                "perplexity": float(ppl),
                "avg_nll": float(avg_nll),
                "tokens_processed": tokens_processed
            }
            
        except Exception as e:
            logger.error(f"Error computing perplexity: {e}")
            return {"perplexity": float('inf'), "error": str(e)}
    
    def compute_bert_score(self, references: List[str], candidates: List[str], 
                         batch_size: int = 8) -> Dict[str, Any]:
        """
        Compute BERTScore for candidate texts against references.
        
        Args:
            references: List of reference texts
            candidates: List of candidate (generated) texts
            batch_size: Batch size for processing (larger values are faster but use more memory)
            
        Returns:
            Dictionary with precision, recall, and F1 scores, plus per_sample scores
        """
        if not BERT_SCORE_AVAILABLE or self.bert_scorer is None:
            logger.warning("BERTScore is not available")
            return {"precision": [], "recall": [], "f1": []}
        
        try:
            # Process in batches to avoid OOM errors
            results = {"precision": [], "recall": [], "f1": [], "per_sample": []}
            
            for i in range(0, len(candidates), batch_size):
                batch_refs = references[i:i+batch_size]
                batch_cands = candidates[i:i+batch_size]
                
                if not batch_refs or not batch_cands:
                    continue
                    
                P, R, F1 = self.bert_scorer.score(batch_cands, batch_refs)
                
                # Append batch results
                results["precision"].extend(P.tolist())
                results["recall"].extend(R.tolist())
                results["f1"].extend(F1.tolist())
                
                # Store per-sample scores
                for j in range(len(batch_refs)):
                    results["per_sample"].append({
                        "precision": P[j].item(),
                        "recall": R[j].item(),
                        "f1": F1[j].item()
                    })
            
            # Add aggregate scores
            if results["f1"]:
                results["aggregate"] = {
                    "precision_mean": np.mean(results["precision"]),
                    "precision_std": np.std(results["precision"]),
                    "recall_mean": np.mean(results["recall"]),
                    "recall_std": np.std(results["recall"]),
                    "f1_mean": np.mean(results["f1"]),
                    "f1_std": np.std(results["f1"])
                }
            
            return results
        except Exception as e:
            logger.error(f"Error computing BERTScore: {e}")
            return {"precision": [], "recall": [], "f1": [], "error": str(e)}
    
    def compute_rouge(self, references: List[str], candidates: List[str]) -> Dict[str, Any]:
        """
        Compute ROUGE scores for candidate texts against references.
        
        Args:
            references: List of reference texts
            candidates: List of candidate (generated) texts
            
        Returns:
            Dictionary with ROUGE scores (rouge1, rouge2, rougeL) and per-sample details
        """
        if not ROUGE_AVAILABLE or self.rouge_scorer is None:
            logger.warning("ROUGE is not available")
            return {}
        
        try:
            # Create detailed scores dictionary
            scores = defaultdict(list)
            per_sample_scores = []
            
            for i, (ref, cand) in enumerate(zip(references, candidates)):
                # Skip empty texts
                if not ref.strip() or not cand.strip():
                    logger.warning(f"Skipping empty text in ROUGE computation at index {i}")
                    continue
                    
                try:
                    rouge_scores = self.rouge_scorer.score(ref, cand)
                    
                    # Record per-metric scores
                    sample_scores = {}
                    for key, value in rouge_scores.items():
                        scores[f"{key}_precision"].append(value.precision)
                        scores[f"{key}_recall"].append(value.recall)
                        scores[f"{key}_fmeasure"].append(value.fmeasure)
                        
                        sample_scores[key] = {
                            "precision": value.precision,
                            "recall": value.recall,
                            "fmeasure": value.fmeasure
                        }
                    
                    per_sample_scores.append(sample_scores)
                except Exception as e:
                    logger.error(f"Error computing ROUGE for sample {i}: {e}")
                    continue
            
            # Average scores
            avg_scores = {k: np.mean(v) for k, v in scores.items() if v}
            std_scores = {f"{k}_std": np.std(v) for k, v in scores.items() if v}
            
            # Add standard deviations
            avg_scores.update(std_scores)
            
            # Create final output
            result = {
                "aggregate": avg_scores,
                "per_sample": per_sample_scores
            }
            
            return result
        except Exception as e:
            logger.error(f"Error computing ROUGE: {e}")
            return {"error": str(e)}
    
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
        Uses multiple techniques including token overlap, contradiction detection, and semantic similarity.
        
        Args:
            reference_facts: List of factual statements that should be present/consistent
            generated_text: Generated text to evaluate for hallucinations
            
        Returns:
            Dictionary with hallucination metrics
        """
        if not reference_facts or not generated_text.strip():
            logger.warning("Empty reference facts or generated text provided for hallucination detection")
            return {
                "fact_coverage": 0.0,
                "contradiction_rate": 0.0,
                "hallucination_risk": 1.0,
                "error": "Empty input"
            }
            
        try:
            fact_presence = []
            fact_contradictions = []
            fact_details = []
            
            # Process the generated text once
            gen_tokens = set(nltk.word_tokenize(generated_text.lower()))
            gen_text_lower = generated_text.lower()
            
            # Process each fact
            for fact in reference_facts:
                fact_lower = fact.lower()
                fact_tokens = set(nltk.word_tokenize(fact_lower))
                
                if not fact_tokens:
                    continue
                
                # 1. Exact phrase matching
                exact_match = fact_lower in gen_text_lower
                
                # 2. Word overlap ratio
                overlap_ratio = len(fact_tokens.intersection(gen_tokens)) / len(fact_tokens)
                
                # 3. Contradiction indicators
                contradiction_terms = ['not', 'never', 'no', 'none', 'nothing', 'neither', 'contrary', 'false', 'untrue']
                fact_has_neg = any(neg in fact_tokens for neg in contradiction_terms)
                gen_has_neg_related_to_fact = False
                
                # Check if negation appears near fact keywords in the generated text
                fact_keywords = [token for token in fact_tokens if len(token) > 3 and token not in contradiction_terms]
                for keyword in fact_keywords:
                    # Find position of keyword in generated text
                    if keyword in gen_text_lower:
                        # Check nearby tokens for negation terms (simple approach)
                        context_start = max(0, gen_text_lower.find(keyword) - 30)
                        context_end = min(len(gen_text_lower), gen_text_lower.find(keyword) + len(keyword) + 30)
                        context = gen_text_lower[context_start:context_end]
                        
                        if any(neg in context for neg in contradiction_terms):
                            gen_has_neg_related_to_fact = True
                            break
                
                # Semantic similarity using BERTScore if available
                semantic_similarity = 0.0
                if self.bert_scorer:
                    try:
                        P, R, F1 = self.bert_scorer.score([generated_text], [fact])
                        semantic_similarity = F1.item()
                    except Exception as e:
                        logger.error(f"Error computing semantic similarity for hallucination detection: {e}")
                
                # Determine if the fact is present
                is_present = exact_match or overlap_ratio > 0.7 or semantic_similarity > 0.8
                
                # Determine if there's a contradiction
                is_contradiction = False
                if fact_has_neg != gen_has_neg_related_to_fact and overlap_ratio > 0.3:
                    is_contradiction = True
                
                fact_presence.append(is_present)
                
                if is_contradiction:
                    fact_contradictions.append(fact)
                
                # Record details for this fact
                fact_details.append({
                    "fact": fact,
                    "is_present": is_present,
                    "exact_match": exact_match,
                    "overlap_ratio": overlap_ratio,
                    "semantic_similarity": semantic_similarity,
                    "has_contradiction": is_contradiction
                })
            
            # Calculate metrics
            if fact_presence:
                fact_coverage = sum(fact_presence) / len(fact_presence)
            else:
                fact_coverage = 0.0
                
            contradiction_rate = len(fact_contradictions) / len(reference_facts) if reference_facts else 0.0
            
            # Compute overall hallucination risk
            # Higher weight on contradictions than missing facts
            hallucination_risk = 0.7 * contradiction_rate + 0.3 * (1.0 - fact_coverage)
            
            return {
                "fact_coverage": fact_coverage,
                "contradiction_rate": contradiction_rate,
                "contradicted_facts": fact_contradictions,
                "hallucination_risk": hallucination_risk,
                "fact_details": fact_details,
                "summary": {
                    "total_facts": len(reference_facts),
                    "facts_present": sum(fact_presence),
                    "contradictions": len(fact_contradictions)
                }
            }
        except Exception as e:
            logger.error(f"Error detecting hallucinations: {e}")
            return {
                "fact_coverage": 0.0,
                "contradiction_rate": 0.0,
                "contradicted_facts": [],
                "hallucination_risk": 1.0,
                "error": str(e)
            }
    
    def collect_human_feedback(self, generated_text: str, prompt: str, 
                             save_path: Optional[str] = None,
                             api_mode: bool = False,
                             api_feedback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Interactive tool to collect human feedback on generated text.
        
        Args:
            generated_text: Text generated by the model
            prompt: Original prompt used to generate the text
            save_path: Path to save feedback results (optional)
            api_mode: Whether to use provided API feedback instead of interactive collection
            api_feedback: Pre-provided feedback when in API mode
            
        Returns:
            Dictionary with feedback metrics
        """
        timestamp = datetime.now().isoformat()
        
        # Handle API-provided feedback
        if api_mode and api_feedback:
            result = {
                "timestamp": timestamp,
                "prompt": prompt,
                "generated_text": generated_text,
                "api_provided": True,
            }
            
            # Copy valid fields from api_feedback
            valid_fields = [
                "ratings", "qualitative_feedback", "has_hallucination", 
                "hallucination_details", "tags", "comparison_score"
            ]
            
            for field in valid_fields:
                if field in api_feedback:
                    result[field] = api_feedback[field]
            
            # Ensure required fields exist
            if "ratings" not in result:
                result["ratings"] = {}
                
            # Save feedback if requested
            if save_path:
                self._save_feedback(result, save_path)
                logger.info(f"API feedback saved to {save_path}")
                
            return result
        
        # Interactive collection mode
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
            return self._collect_interactive_ratings(prompt, generated_text, save_path)
        except Exception as e:
            logger.error(f"Error collecting feedback: {e}")
            return {"error": str(e), "timestamp": timestamp}

    def _collect_interactive_ratings(self, prompt, generated_text, save_path):
        """Interactive collection of human feedback ratings"""
        print("\nPlease rate the generated text on the following criteria (1-5, where 5 is best):")
        
        # Core quality metrics
        ratings = {}
        rating_criteria = {
            "relevance": "Relevance to prompt",
            "factuality": "Factual accuracy",
            "coherence": "Coherence and fluency",
            "creativity": "Creativity and originality",
            "usefulness": "Usefulness of information",
            "quality": "Overall quality"
        }
        
        for key, description in rating_criteria.items():
            try:
                value = int(input(f"{description} (1-5): "))
                # Validate input
                ratings[key] = max(1, min(5, value))
            except ValueError:
                print(f"Invalid input for {description}. Using default value 3.")
                ratings[key] = 3

        # Task-specific ratings (optional)
        print("\nDoes this response involve any of these tasks? (y/n for each)")
        task_types = {
            "code_generation": "Code generation",
            "creative_writing": "Creative writing",
            "factual_information": "Providing factual information",
            "conversation": "Conversational response",
            "problem_solving": "Problem solving"
        }
        
        task_ratings = {}
        for key, description in task_types.items():
            if input(f"{description}? (y/n): ").lower().startswith('y'):
                task_value = int(input(f"Rate quality for {description} (1-5): "))
                task_ratings[key] = max(1, min(5, task_value))
        
        # Collect qualitative feedback
        print("\nPlease provide any additional feedback:")
        feedback = input("> ")

        # Collect hallucination assessment
        print("\nDid you notice any factual errors or hallucinations? (y/n)")
        has_hallucination = input("> ").lower().startswith('y')

        hallucination_details = ""
        if has_hallucination:
            print("Please describe the hallucinations you observed:")
            hallucination_details = input("> ")
        
        # Collect content tags
        print("\nPlease add any relevant tags for this content (comma-separated):")
        tags_input = input("> ")
        tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()]

        # Prepare results
        result = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "generated_text": generated_text,
            "ratings": ratings,
            "task_ratings": task_ratings,
            "qualitative_feedback": feedback,
            "has_hallucination": has_hallucination,
            "hallucination_details": hallucination_details,
            "tags": tags
        }

        # Save feedback if requested
        if save_path:
            self._save_feedback(result, save_path)
        
        return result
    
    def _save_feedback(self, feedback_data, save_path):
        """Save feedback data to the specified path"""
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        # Append to existing file if it exists
        if os.path.exists(save_path):
            with open(save_path, 'r') as f:
                try:
                    existing_data = json.load(f)
                    if isinstance(existing_data, list):
                        existing_data.append(feedback_data)
                    else:
                        existing_data = [existing_data, feedback_data]
                except json.JSONDecodeError:
                    existing_data = [feedback_data]

            with open(save_path, 'w') as f:
                json.dump(existing_data, f, indent=2)
        else:
            with open(save_path, 'w') as f:
                json.dump([feedback_data], f, indent=2)

        logger.info(f"Feedback saved to {save_path}")
        if is_paperspace_environment():
            # Sync to Google Drive if in Paperspace environment
            sync_to_gdrive(os.path.dirname(save_path))

    def evaluate_generation(self, prompt: str, generated_text: str, 
                           reference_text: Optional[str] = None,
                           reference_facts: Optional[List[str]] = None,
                           collect_human_feedback: bool = False,
                           model: Optional[Any] = None,
                           tokenizer: Optional[Any] = None,
                           dataset_name: str = "unknown",
                           save_results: bool = True,
                           task_type: str = "general",
                           api_feedback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
            task_type: Type of generation task (general, creative, code, factual, chat)
            api_feedback: Pre-provided feedback when collecting human feedback in API mode
            
        Returns:
            Dictionary with all evaluation metrics
        """
        start_time = datetime.now()
        
        # Input validation
        if not generated_text or not generated_text.strip():
            logger.warning("Empty generated text provided for evaluation")
            return {"error": "Empty generated text", "prompt": prompt}
            
        results = {
            "prompt": prompt,
            "generated_text": generated_text,
            "dataset": dataset_name,
            "task_type": task_type,
            "timestamp": start_time.isoformat(),
            "metrics": {}
        }
        
        # Basic statistics
        results["statistics"] = {
            "prompt_length": len(prompt),
            "prompt_tokens": len(nltk.word_tokenize(prompt)) if NLTK_AVAILABLE else None,
            "generated_length": len(generated_text),
            "generated_tokens": len(nltk.word_tokenize(generated_text)) if NLTK_AVAILABLE else None,
        }
        
        # Compute all available metrics
        metrics = {}
        
        # Reference-based metrics (if reference is available)
        if reference_text:
            results["reference_text"] = reference_text
            results["statistics"]["reference_length"] = len(reference_text)
            results["statistics"]["reference_tokens"] = len(nltk.word_tokenize(reference_text)) if NLTK_AVAILABLE else None
            
            # BERTScore for semantic similarity
            bert_result = self.compute_bert_score([reference_text], [generated_text])
            metrics["bert_score"] = bert_result
            
            # ROUGE for n-gram overlap
            rouge_result = self.compute_rouge([reference_text], [generated_text])
            metrics["rouge"] = rouge_result
            
            # BLEU for machine translation style evaluation
            bleu_result = self.compute_bleu([reference_text], [generated_text])
            metrics["bleu"] = bleu_result
            
            # Length ratio (generated text length / reference text length)
            if results["statistics"]["reference_tokens"]:
                metrics["length_ratio"] = results["statistics"]["generated_tokens"] / results["statistics"]["reference_tokens"]
        
        # Hallucination detection (if reference facts are available)
        if reference_facts:
            results["reference_facts"] = reference_facts
            metrics["hallucination"] = self.detect_hallucination(reference_facts, generated_text)
        
        # Perplexity (if model and tokenizer are available)
        if model and tokenizer:
            metrics["perplexity_metrics"] = self.compute_perplexity(model, tokenizer, generated_text)
        
        # Task-specific metrics
        if task_type == "code":
            # Simple code quality metrics (just placeholders for now)
            import re
            metrics["code_metrics"] = {
                "has_comments": bool(re.search(r'#|//|/\*|\*/', generated_text)),
                "line_count": len(generated_text.split('\n')),
                "indentation_consistency": bool(re.search(r'^\s+', generated_text, re.MULTILINE))
            }
        elif task_type == "creative":
            # Rough creativity metrics
            if NLTK_AVAILABLE:
                try:
                    tokens = nltk.word_tokenize(generated_text.lower())
                    vocab = set(tokens)
                    metrics["creative_metrics"] = {
                        "vocabulary_diversity": len(vocab) / len(tokens) if tokens else 0,
                        "avg_word_length": sum(len(w) for w in vocab) / len(vocab) if vocab else 0
                    }
                except Exception as e:
                    logger.error(f"Error computing creative metrics: {e}")
        
        # Add metrics to results
        results["metrics"] = metrics
        
        # Collect human feedback if requested
        if collect_human_feedback:
            results["human_feedback"] = self.collect_human_feedback(
                generated_text=generated_text, 
                prompt=prompt, 
                save_path=os.path.join(self.metrics_dir, f"human_feedback_{dataset_name}.json"),
                api_mode=api_feedback is not None,
                api_feedback=api_feedback
            )
        
        # Add timing information
        end_time = datetime.now()
        results["processing_time"] = (end_time - start_time).total_seconds()
        
        # Save results if requested
        if save_results:
            timestamp = end_time.strftime("%Y%m%d_%H%M%S")
            results_path = os.path.join(self.metrics_dir, f"evaluation_{dataset_name}_{timestamp}.json")
            
            try:
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2)
                
                logger.info(f"Evaluation results saved to {results_path}")
                
                # Sync to Google Drive if in Paperspace environment
                if is_paperspace_environment():
                    sync_to_gdrive(self.metrics_dir)
            except Exception as e:
                logger.error(f"Error saving evaluation results: {e}")
        
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

    def create_evaluation_report(self, evaluation_results: List[Dict[str, Any]], 
                             report_name: str = "evaluation_report",
                             include_samples: bool = True,
                             max_samples: int = 5) -> Dict[str, Any]:
        """
        Create a comprehensive evaluation report from multiple evaluation results.
        
        Args:
            evaluation_results: List of evaluation result dictionaries from evaluate_generation
            report_name: Name to use for the report file
            include_samples: Whether to include sample texts in the report
            max_samples: Maximum number of samples to include in the report
            
        Returns:
            Dictionary with the aggregated report data
        """
        if not evaluation_results:
            logger.warning("No evaluation results provided for report creation")
            return {"error": "No data provided"}
            
        # Extract dataset and task information
        datasets = Counter([result.get("dataset", "unknown") for result in evaluation_results])
        task_types = Counter([result.get("task_type", "general") for result in evaluation_results])
        
        # Create report structure
        report = {
            "report_name": report_name,
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(evaluation_results),
            "datasets": dict(datasets),
            "task_types": dict(task_types),
            "aggregated_metrics": {},
            "metrics_by_dataset": {},
            "metrics_by_task": {}
        }
        
        # Extract and aggregate metrics
        all_metrics = defaultdict(list)
        dataset_metrics = defaultdict(lambda: defaultdict(list))
        task_metrics = defaultdict(lambda: defaultdict(list))
        
        # Helper function for metric extraction
        def extract_metrics(metrics_dict, prefix=""):
            flat_metrics = {}
            for key, value in metrics_dict.items():
                if isinstance(value, dict) and "aggregate" in value:
                    # Handle metrics with aggregate section
                    for subkey, subvalue in value["aggregate"].items():
                        if isinstance(subvalue, (int, float)):
                            flat_name = f"{prefix}{key}_{subkey}"
                            flat_metrics[flat_name] = subvalue
                elif isinstance(value, dict) and all(isinstance(v, (int, float)) for v in value.values()):
                    # Handle simple nested metrics
                    for subkey, subvalue in value.items():
                        flat_name = f"{prefix}{key}_{subkey}"
                        flat_metrics[flat_name] = subvalue
                elif isinstance(value, (int, float)):
                    # Handle simple metrics
                    flat_metrics[f"{prefix}{key}"] = value
                elif isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
                    # Handle list of values - use average
                    flat_metrics[f"{prefix}{key}"] = sum(value) / len(value) if value else 0
            return flat_metrics
        
        # Process each result
        for result in evaluation_results:
            dataset = result.get("dataset", "unknown")
            task_type = result.get("task_type", "general")
            
            # Extract flat metrics
            if "metrics" in result:
                flat_metrics = extract_metrics(result["metrics"])
                
                # Add to overall metrics
                for key, value in flat_metrics.items():
                    all_metrics[key].append(value)
                
                # Add to dataset-specific metrics
                for key, value in flat_metrics.items():
                    dataset_metrics[dataset][key].append(value)
                
                # Add to task-specific metrics
                for key, value in flat_metrics.items():
                    task_metrics[task_type][key].append(value)
        
        # Calculate aggregated statistics
        for key, values in all_metrics.items():
            if values:
                report["aggregated_metrics"][key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "count": len(values)
                }
        
        # Calculate dataset-specific statistics
        for dataset, metrics in dataset_metrics.items():
            report["metrics_by_dataset"][dataset] = {}
            for key, values in metrics.items():
                if values:
                    report["metrics_by_dataset"][dataset][key] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "count": len(values)
                    }
        
        # Calculate task-specific statistics
        for task, metrics in task_metrics.items():
            report["metrics_by_task"][task] = {}
            for key, values in metrics.items():
                if values:
                    report["metrics_by_task"][task][key] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "count": len(values)
                    }
        
        # Add samples if requested
        if include_samples:
            # Select diverse samples
            samples = []
            selected_indices = set()
            
            # Try to get samples from each dataset
            for dataset in datasets:
                dataset_results = [r for i, r in enumerate(evaluation_results) 
                                 if r.get("dataset") == dataset and i not in selected_indices]
                if dataset_results:
                    # Take a representative sample
                    sample = dataset_results[0]
                    samples.append({
                        "prompt": sample.get("prompt", ""),
                        "generated_text": sample.get("generated_text", ""),
                        "reference_text": sample.get("reference_text", ""),
                        "dataset": sample.get("dataset", "unknown"),
                        "task_type": sample.get("task_type", "general")
                    })
                    selected_indices.add(evaluation_results.index(sample))
                
                if len(samples) >= max_samples:
                    break
            
            # If we need more samples, add from different task types
            if len(samples) < max_samples:
                for task in task_types:
                    task_results = [r for i, r in enumerate(evaluation_results) 
                                   if r.get("task_type") == task and i not in selected_indices]
                    if task_results:
                        sample = task_results[0]
                        samples.append({
                            "prompt": sample.get("prompt", ""),
                            "generated_text": sample.get("generated_text", ""),
                            "reference_text": sample.get("reference_text", ""),
                            "dataset": sample.get("dataset", "unknown"),
                            "task_type": sample.get("task_type", "general")
                        })
                        selected_indices.add(evaluation_results.index(sample))
                    
                    if len(samples) >= max_samples:
                        break
            
            # If we still need more samples, add randomly
            if len(samples) < max_samples:
                remaining = [r for i, r in enumerate(evaluation_results) if i not in selected_indices]
                for sample in remaining[:max_samples - len(samples)]:
                    samples.append({
                        "prompt": sample.get("prompt", ""),
                        "generated_text": sample.get("generated_text", ""),
                        "reference_text": sample.get("reference_text", ""),
                        "dataset": sample.get("dataset", "unknown"),
                        "task_type": sample.get("task_type", "general")
                    })
            
            report["samples"] = samples
        
        # Save the report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.metrics_dir, f"{report_name}_{timestamp}.json")
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Evaluation report saved to {report_path}")
            
            # Sync to Google Drive if in Paperspace environment
            if is_paperspace_environment():
                sync_to_gdrive(self.metrics_dir)
        except Exception as e:
            logger.error(f"Error saving evaluation report: {e}")
        
        return report

    def visualize_metrics(self, evaluation_results: Union[Dict[str, Any], List[Dict[str, Any]]],
                            output_dir: Optional[str] = None,
                            plot_type: str = "comprehensive") -> Dict[str, str]:
        """
        Visualize evaluation metrics with plots.
        
        Args:
            evaluation_results: Evaluation results from evaluate_generation or create_evaluation_report
            output_dir: Directory to save visualization files (defaults to metrics_dir)
            plot_type: Type of visualization ("comprehensive", "comparison", "simple")
            
        Returns:
            Dictionary mapping plot types to file paths
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            from matplotlib.ticker import MaxNLocator
            import seaborn as sns
            
            # Make sure the directory exists
            if output_dir is None:
                output_dir = os.path.join(self.metrics_dir, "visualizations")
            os.makedirs(output_dir, exist_ok=True)
            
            # Convert single result to list for consistent handling
            if isinstance(evaluation_results, dict) and "metrics" in evaluation_results:
                evaluation_results = [evaluation_results]
            
            # Handle report format
            if isinstance(evaluation_results, dict) and "aggregated_metrics" in evaluation_results:
                return self._visualize_report(evaluation_results, output_dir, plot_type)
            
            # For single or multiple evaluation results
            return self._visualize_evaluation_results(evaluation_results, output_dir, plot_type)
            
        except ImportError as e:
            logger.error(f"Visualization requires matplotlib and seaborn: {e}")
            return {"error": f"Missing dependencies: {e}"}
        except Exception as e:
            logger.error(f"Error visualizing metrics: {e}")
            return {"error": str(e)}
    
    def _visualize_evaluation_results(self, results: List[Dict[str, Any]], 
                                     output_dir: str, plot_type: str) -> Dict[str, str]:
        """Visualize individual evaluation results"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plot_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Extract and flatten metrics for plotting
        all_flat_metrics = []
        for result in results:
            # Skip results without metrics
            if "metrics" not in result:
                continue
                
            # Get basic information
            dataset = result.get("dataset", "unknown")
            task_type = result.get("task_type", "general")
            
            # Extract flat metrics from nested structure
            flat_metrics = {}
            
            # Process bert_score
            if "bert_score" in result["metrics"]:
                bert_score = result["metrics"]["bert_score"]
                # Handle aggregate section if present
                if "aggregate" in bert_score:
                    for key, value in bert_score["aggregate"].items():
                        flat_metrics[f"bert_score_{key}"] = value
                # Handle f1/precision/recall lists
                for key in ["precision", "recall", "f1"]:
                    if key in bert_score and bert_score[key]:
                        if isinstance(bert_score[key], list) and all(isinstance(v, (int, float)) for v in bert_score[key]):
                            flat_metrics[f"bert_score_{key}"] = sum(bert_score[key]) / len(bert_score[key])
            
            # Process ROUGE scores
            if "rouge" in result["metrics"]:
                rouge = result["metrics"]["rouge"]
                if "aggregate" in rouge:
                    for key, value in rouge["aggregate"].items():
                        flat_metrics[f"rouge_{key}"] = value
            
            # Process BLEU scores
            if "bleu" in result["metrics"]:
                bleu = result["metrics"]["bleu"]
                for key, value in bleu.items():
                    flat_metrics[f"bleu_{key}"] = value
            
            # Process hallucination metrics
            if "hallucination" in result["metrics"]:
                hallucination = result["metrics"]["hallucination"]
                for key in ["fact_coverage", "contradiction_rate", "hallucination_risk"]:
                    if key in hallucination:
                        flat_metrics[f"hallucination_{key}"] = hallucination[key]
            
            # Process perplexity
            if "perplexity_metrics" in result["metrics"]:
                perplexity = result["metrics"]["perplexity_metrics"]
                if "perplexity" in perplexity:
                    flat_metrics["perplexity"] = perplexity["perplexity"]
            
            # Add metadata
            flat_metrics["dataset"] = dataset
            flat_metrics["task_type"] = task_type
            
            all_flat_metrics.append(flat_metrics)
        
        if not all_flat_metrics:
            logger.warning("No metrics available for visualization")
            return {"error": "No metrics available for visualization"}
        
        # Simple plot: overview of key metrics
        if plot_type in ["simple", "comprehensive"]:
            plt.figure(figsize=(10, 6))
            
            # Create a DataFrame for easier plotting
            import pandas as pd
            metrics_df = pd.DataFrame(all_flat_metrics)
            
            # Select key metrics to visualize
            key_metrics = [
                'bert_score_f1_mean', 'rouge_rouge1_fmeasure_mean', 
                'bleu_bleu1', 'hallucination_fact_coverage'
            ]
            
            # Filter to metrics that exist in our data
            plot_metrics = [m for m in key_metrics if m in metrics_df.columns]
            
            if not plot_metrics:
                logger.warning("No key metrics available for simple plot")
            else:
                # Plot boxplots for each metric
                plt.figure(figsize=(12, 6))
                sns.boxplot(data=metrics_df[plot_metrics])
                plt.title("Key Metrics Overview")
                plt.ylabel("Score")
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Save the plot
                simple_plot_file = os.path.join(output_dir, f"simple_metrics_{timestamp}.png")
                plt.savefig(simple_plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                plot_files["simple"] = simple_plot_file
                logger.info(f"Simple metrics plot saved to {simple_plot_file}")
        
        # Comprehensive visualization
        if plot_type == "comprehensive":
            # Group metrics by dataset for comparison
            if len(set(m["dataset"] for m in all_flat_metrics)) > 1:
                import pandas as pd
                metrics_df = pd.DataFrame(all_flat_metrics)
                
                # Metrics by dataset
                plt.figure(figsize=(14, 8))
                metric_cols = [c for c in metrics_df.columns if any(
                    c.startswith(p) for p in ['bert_score_', 'rouge_', 'bleu_', 'hallucination_'])]
                
                if metric_cols and 'dataset' in metrics_df.columns:
                    # Create a long-form dataframe for seaborn
                    plot_data = pd.melt(metrics_df, 
                                      id_vars=['dataset'], 
                                      value_vars=metric_cols,
                                      var_name='Metric', 
                                      value_name='Score')
                    
                    # Plot
                    g = sns.catplot(
                        data=plot_data, 
                        x='Metric', 
                        y='Score', 
                        hue='dataset',
                        kind='bar', 
                        height=6, 
                        aspect=2
                    )
                    g.set_xticklabels(rotation=45, ha='right')
                    plt.title("Metrics by Dataset")
                    plt.tight_layout()
                    
                    # Save
                    dataset_plot_file = os.path.join(output_dir, f"metrics_by_dataset_{timestamp}.png")
                    plt.savefig(dataset_plot_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    plot_files["dataset_comparison"] = dataset_plot_file
                    logger.info(f"Dataset comparison plot saved to {dataset_plot_file}")
            
            # Correlation heatmap
            try:
                import pandas as pd
                metrics_df = pd.DataFrame(all_flat_metrics)
                
                # Select numeric columns only
                numeric_cols = metrics_df.select_dtypes(include=['float64', 'int64']).columns
                if len(numeric_cols) > 1:  # Need at least 2 metrics for correlation
                    plt.figure(figsize=(10, 8))
                    correlation = metrics_df[numeric_cols].corr()
                    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
                    plt.title("Metric Correlation Matrix")
                    plt.tight_layout()
                    
                    # Save
                    corr_plot_file = os.path.join(output_dir, f"metric_correlation_{timestamp}.png")
                    plt.savefig(corr_plot_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    plot_files["correlation"] = corr_plot_file
                    logger.info(f"Correlation plot saved to {corr_plot_file}")
            except Exception as e:
                logger.error(f"Error creating correlation plot: {e}")
        
        return plot_files
    
    def _visualize_report(self, report: Dict[str, Any], output_dir: str, plot_type: str) -> Dict[str, str]:
        """Visualize a report with aggregated metrics"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plot_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Plot aggregated metrics
        if "aggregated_metrics" in report:
            plt.figure(figsize=(12, 8))
            
            # Extract metric means for plotting
            metrics = {}
            for metric, values in report["aggregated_metrics"].items():
                if isinstance(values, dict) and "mean" in values:
                    metrics[metric] = values["mean"]
            
            if not metrics:
                logger.warning("No mean values found in aggregated metrics")
            else:
                # Create bar chart of means
                plt.bar(metrics.keys(), metrics.values())
                plt.xticks(rotation=45, ha='right')
                plt.title(f"Aggregated Metrics: {report.get('report_name', 'Evaluation Report')}")
                plt.tight_layout()
                
                # Save
                agg_plot_file = os.path.join(output_dir, f"aggregated_metrics_{timestamp}.png")
                plt.savefig(agg_plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                plot_files["aggregated"] = agg_plot_file
                logger.info(f"Aggregated metrics plot saved to {agg_plot_file}")
        
        # Comparison between datasets
        if "metrics_by_dataset" in report and len(report["metrics_by_dataset"]) > 1:
            # Create a unified data structure for plotting
            import pandas as pd
            
            comparison_data = []
            for dataset, metrics in report["metrics_by_dataset"].items():
                for metric, values in metrics.items():
                    if isinstance(values, dict) and "mean" in values:
                        comparison_data.append({
                            "dataset": dataset,
                            "metric": metric,
                            "value": values["mean"]
                        })
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                
                # Create faceted bar chart
                g = sns.catplot(
                    data=comparison_df,
                    x="dataset",
                    y="value",
                    col="metric",
                    kind="bar",
                    col_wrap=3,
                    height=4,
                    aspect=1.2,
                    sharey=False
                )
                g.set_xticklabels(rotation=45)
                g.set_titles("{col_name}")
                plt.tight_layout()
                
                # Save
                comparison_file = os.path.join(output_dir, f"dataset_comparison_{timestamp}.png")
                plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                plot_files["dataset_comparison"] = comparison_file
                logger.info(f"Dataset comparison plot saved to {comparison_file}")
        
        # Similar comparison for task types
        if "metrics_by_task" in report and len(report["metrics_by_task"]) > 1:
            # Similar approach as above for tasks
            import pandas as pd
            
            task_data = []
            for task, metrics in report["metrics_by_task"].items():
                for metric, values in metrics.items():
                    if isinstance(values, dict) and "mean" in values:
                        task_data.append({
                            "task": task,
                            "metric": metric,
                            "value": values["mean"]
                        })
            
            if task_data:
                task_df = pd.DataFrame(task_data)
                
                # Create task comparison plot
                g = sns.catplot(
                    data=task_df,
                    x="task",
                    y="value",
                    col="metric",
                    kind="bar",
                    col_wrap=3,
                    height=4,
                    aspect=1.2,
                    sharey=False
                )
                g.set_xticklabels(rotation=45)
                g.set_titles("{col_name}")
                plt.tight_layout()
                
                # Save
                task_file = os.path.join(output_dir, f"task_comparison_{timestamp}.png")
                plt.savefig(task_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                plot_files["task_comparison"] = task_file
                logger.info(f"Task comparison plot saved to {task_file}")
        
        return plot_files

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

# Add command-line evaluation functionality from evaluate_model.py
def load_model_and_tokenizer(model_path: str, use_gpu: bool) -> tuple:
    """
    Load model and tokenizer from the specified path
    
    Args:
        model_path: Path to the model directory
        use_gpu: Whether to use GPU for inference
        
    Returns:
        tuple: (model, tokenizer, generator)
    """
    try:
        # First try loading as a Hugging Face model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Check if we have a PEFT model
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            from peft import PeftModel, PeftConfig
            
            logger.info(f"Loading PEFT model from {model_path}")
            config = PeftConfig.from_pretrained(model_path)
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                torch_dtype=torch.float16 if use_gpu else torch.float32,
                device_map="auto" if use_gpu else None
            )
            
            # Load PEFT adapter
            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            # Load regular model
            logger.info(f"Loading model from {model_path}")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if use_gpu else torch.float32,
                device_map="auto" if use_gpu else None
            )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Initialize text generator
        # We'll import dynamically to avoid circular imports
        from src.generative_ai_module.text_generator import TextGenerator
        generator = TextGenerator(
            model=model,
            tokenizer=tokenizer,
            force_gpu=use_gpu
        )
        
        return model, tokenizer, generator
    
    except Exception as e:
        logger.error(f"Error loading model and tokenizer: {e}")
        
        # Fallback to loading using TextGenerator
        try:
            logger.info("Trying to load with TextGenerator...")
            from src.generative_ai_module.text_generator import TextGenerator
            generator = TextGenerator(force_gpu=use_gpu)
            generator.load_model(model_path)
            
            return generator.model, generator.tokenizer, generator
        except Exception as e2:
            logger.error(f"Failed to load model with TextGenerator: {e2}")
            return None, None, None

def load_evaluation_dataset(dataset_path: str, max_samples: int = 50) -> List[Dict[str, str]]:
    """
    Load evaluation dataset from file
    
    Args:
        dataset_path: Path to the dataset file
        max_samples: Maximum number of samples to load
        
    Returns:
        List of prompt-response pairs
    """
    try:
        # Try different loading methods based on file extension
        if dataset_path.endswith('.json'):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON formats
            if isinstance(data, list):
                # List of samples
                pairs = []
                for item in data[:max_samples]:
                    # Extract prompt and reference based on common field names
                    prompt = None
                    reference = None
                    
                    # Check for common field names
                    for prompt_key in ['prompt', 'input', 'question', 'instruction']:
                        if prompt_key in item:
                            prompt = item[prompt_key]
                            break
                    
                    for ref_key in ['response', 'output', 'answer', 'completion', 'reference']:
                        if ref_key in item:
                            reference = item[ref_key]
                            break
                    
                    # Only add if we found both prompt and reference
                    if prompt is not None:
                        pair = {"prompt": prompt}
                        if reference is not None:
                            pair["reference"] = reference
                        
                        # Add any facts for hallucination detection
                        if "facts" in item:
                            pair["facts"] = item["facts"]
                        
                        pairs.append(pair)
            
            elif isinstance(data, dict):
                # Handle dictionary format with examples key
                if "examples" in data and isinstance(data["examples"], list):
                    return load_evaluation_dataset(data["examples"], max_samples)
                
                # Handle dictionary format with data key
                if "data" in data and isinstance(data["data"], list):
                    return load_evaluation_dataset(data["data"], max_samples)
                
                # Otherwise, try to create examples from the dict itself
                pairs = []
                for key, value in list(data.items())[:max_samples]:
                    pairs.append({"prompt": key, "reference": value})
            
            else:
                logger.error(f"Unsupported JSON format in {dataset_path}")
                return []
        
        elif dataset_path.endswith('.txt'):
            # Simple text file, assume alternating prompt/response
            with open(dataset_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            
            pairs = []
            for i in range(0, min(len(lines), max_samples * 2), 2):
                if i + 1 < len(lines):
                    pairs.append({"prompt": lines[i], "reference": lines[i+1]})
                else:
                    pairs.append({"prompt": lines[i]})
        
        elif dataset_path.endswith('.csv'):
            # CSV file
            import csv
            
            pairs = []
            with open(dataset_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader)
                
                # Identify prompt and reference columns
                prompt_idx = -1
                ref_idx = -1
                
                for i, header in enumerate(headers):
                    lower_header = header.lower()
                    if lower_header in ['prompt', 'input', 'question', 'instruction']:
                        prompt_idx = i
                    elif lower_header in ['response', 'output', 'answer', 'completion', 'reference']:
                        ref_idx = i
                
                if prompt_idx == -1:
                    logger.error(f"Could not identify prompt column in CSV file {dataset_path}")
                    return []
                
                # Extract pairs
                for i, row in enumerate(reader):
                    if i >= max_samples:
                        break
                    
                    if len(row) <= prompt_idx:
                        continue
                    
                    pair = {"prompt": row[prompt_idx]}
                    
                    if ref_idx != -1 and len(row) > ref_idx:
                        pair["reference"] = row[ref_idx]
                    
                    pairs.append(pair)
        
        elif os.path.isdir(dataset_path):
            # Try to load as a HuggingFace dataset
            try:
                from datasets import load_from_disk, Dataset
                
                dataset = load_from_disk(dataset_path)
                if isinstance(dataset, Dataset):
                    pairs = []
                    for i, item in enumerate(dataset):
                        if i >= max_samples:
                            break
                        
                        # Extract prompt and reference based on common field names
                        prompt = None
                        reference = None
                        
                        for prompt_key in ['prompt', 'input', 'question', 'instruction']:
                            if prompt_key in item:
                                prompt = item[prompt_key]
                                break
                        
                        for ref_key in ['response', 'output', 'answer', 'completion', 'reference']:
                            if ref_key in item:
                                reference = item[ref_key]
                                break
                        
                        if prompt is not None:
                            pair = {"prompt": prompt}
                            if reference is not None:
                                pair["reference"] = reference
                            
                            # Add any facts for hallucination detection
                            if "facts" in item:
                                pair["facts"] = item["facts"]
                            
                            pairs.append(pair)
            except Exception as e:
                logger.error(f"Error loading directory as dataset: {e}")
                return []
        
        else:
            logger.error(f"Unsupported file format: {dataset_path}")
            return []
        
        logger.info(f"Loaded {len(pairs)} evaluation examples from {dataset_path}")
        return pairs
    
    except Exception as e:
        logger.error(f"Error loading evaluation dataset: {e}")
        return []

def evaluate_model_with_dataset(args):
    """Main evaluation function for evaluating a model on a dataset"""
    # Set up output directory
    if hasattr(args, 'output_dir') and args.output_dir:
        output_dir = args.output_dir
    else:
        model_name = os.path.basename(os.path.normpath(args.model_path))
        dataset_name = args.dataset_name or os.path.basename(args.dataset).split('.')[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"evaluation_{model_name}_{dataset_name}_{timestamp}"
    
    ensure_directory_exists(output_dir)
    logger.info(f"Evaluation results will be saved to {output_dir}")
    
    # Initialize evaluation metrics
    metrics = EvaluationMetrics(
        metrics_dir=output_dir,
        use_gpu=args.use_gpu,
        bert_model=args.bert_model if hasattr(args, 'bert_model') else "microsoft/deberta-xlarge-mnli"
    )
    
    # Load model and tokenizer
    logger.info(f"Loading model from {args.model_path}")
    model, tokenizer, generator = load_model_and_tokenizer(args.model_path, args.use_gpu)
    
    if not model or not tokenizer or not generator:
        logger.error("Failed to load model. Exiting.")
        return
    
    # Load evaluation dataset
    logger.info(f"Loading evaluation dataset from {args.dataset}")
    dataset_name = args.dataset_name or os.path.basename(args.dataset).split('.')[0]
    examples = load_evaluation_dataset(args.dataset, args.max_samples if hasattr(args, 'max_samples') else 50)
    
    if not examples:
        logger.error("No valid examples found in the dataset. Exiting.")
        return
    
    # Generate and evaluate text for each example
    logger.info(f"Evaluating model on {len(examples)} examples...")
    all_results = []
    
    for i, example in enumerate(examples):
        logger.info(f"Processing example {i+1}/{len(examples)}")
        
        prompt = example["prompt"]
        
        # Generate text
        try:
            generated_text = generator.generate_text(
                prompt=prompt,
                max_new_tokens=args.max_new_tokens if hasattr(args, 'max_new_tokens') else 200,
                temperature=args.temperature if hasattr(args, 'temperature') else 0.7,
                top_p=args.top_p if hasattr(args, 'top_p') else 0.9
            )
            
            # Prepare evaluation parameters
            eval_params = {
                "prompt": prompt,
                "generated_text": generated_text,
                "dataset_name": dataset_name,
                "task_type": args.task_type if hasattr(args, 'task_type') else "general",
                "collect_human_feedback": args.collect_human_feedback if hasattr(args, 'collect_human_feedback') else False,
                "model": model,
                "tokenizer": tokenizer
            }
            
            # Add reference text if available
            if "reference" in example:
                eval_params["reference_text"] = example["reference"]
            
            # Add reference facts if available
            if "facts" in example:
                eval_params["reference_facts"] = example["facts"]
            
            # Run evaluation
            result = metrics.evaluate_generation(**eval_params)
            all_results.append(result)
            
            # Log progress
            if (i+1) % 10 == 0 or i+1 == len(examples):
                logger.info(f"Evaluated {i+1}/{len(examples)} examples")
        
        except Exception as e:
            logger.error(f"Error evaluating example {i+1}: {e}")
            continue
    
    # Create comprehensive evaluation report
    logger.info("Creating evaluation report...")
    report = metrics.create_evaluation_report(
        evaluation_results=all_results,
        report_name=f"{os.path.basename(args.model_path)}_{dataset_name}_evaluation",
        include_samples=True
    )
    
    # Generate visualizations
    if hasattr(args, 'visualize') and args.visualize:
        logger.info("Generating visualizations...")
        vis_dir = os.path.join(output_dir, "visualizations")
        ensure_directory_exists(vis_dir)
        
        vis_files = metrics.visualize_metrics(
            report,
            output_dir=vis_dir,
            plot_type=args.plot_type if hasattr(args, 'plot_type') else "comprehensive"
        )
        
        if vis_files and (not isinstance(vis_files, dict) or "error" not in vis_files):
            logger.info(f"Visualizations saved to {vis_dir}")
    
    # Sync to Google Drive if in Paperspace
    if is_paperspace_environment():
        try:
            sync_to_gdrive(output_dir)
            logger.info(f"Evaluation results synced to Google Drive")
        except Exception as e:
            logger.error(f"Error syncing to Google Drive: {e}")
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("Evaluation Summary")
    logger.info("="*50)
    
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Dataset: {dataset_name} ({len(examples)} examples)")
    logger.info(f"Task type: {args.task_type if hasattr(args, 'task_type') else 'general'}")
    
    if "aggregated_metrics" in report:
        logger.info("\nKey Metrics:")
        
        # Semantic similarity (BERTScore)
        bert_metrics = {k: v for k, v in report["aggregated_metrics"].items() if "bert_score" in k and "f1" in k}
        if bert_metrics:
            key, value = list(bert_metrics.items())[0]
            logger.info(f"  Semantic Similarity (BERTScore): {value['mean']:.4f} (±{value['std']:.4f})")
        
        # Text overlap (ROUGE-L)
        rouge_metrics = {k: v for k, v in report["aggregated_metrics"].items() 
                       if "rouge" in k and "rougeL_fmeasure" in k}
        if rouge_metrics:
            key, value = list(rouge_metrics.items())[0]
            logger.info(f"  Text Overlap (ROUGE-L): {value['mean']:.4f} (±{value['std']:.4f})")
        
        # Factual accuracy
        hall_metrics = {k: v for k, v in report["aggregated_metrics"].items() 
                      if "hallucination" in k and "fact_coverage" in k}
        if hall_metrics:
            key, value = list(hall_metrics.items())[0]
            logger.info(f"  Factual Coverage: {value['mean']:.4f} (±{value['std']:.4f})")
        
        # Perplexity
        perplexity_metrics = {k: v for k, v in report["aggregated_metrics"].items() if "perplexity" in k}
        if perplexity_metrics:
            key, value = list(perplexity_metrics.items())[0]
            logger.info(f"  Perplexity: {value['mean']:.4f} (±{value['std']:.4f})")
    
    logger.info(f"\nFull evaluation results saved to: {output_dir}")
    if hasattr(args, 'visualize') and args.visualize:
        logger.info(f"Visualizations available in: {os.path.join(output_dir, 'visualizations')}")
    
    return report

def run_evaluation_from_command_line():
    """Run evaluation from command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate models with enhanced metrics")
    
    # Model and dataset paths
    parser.add_argument("--model_path", type=str, required=True,
                      help="Path to the trained model directory")
    parser.add_argument("--dataset", type=str, required=True,
                      help="Path to evaluation dataset (JSON format)")
    
    # Evaluation options
    parser.add_argument("--output_dir", type=str, default=None,
                      help="Directory to save evaluation results")
    parser.add_argument("--max_samples", type=int, default=50,
                      help="Maximum number of samples to evaluate")
    parser.add_argument("--bert_model", type=str, default="microsoft/deberta-xlarge-mnli",
                      help="Model to use for BERTScore computation")
    parser.add_argument("--use_gpu", action="store_true",
                      help="Use GPU for evaluation if available")
    parser.add_argument("--collect_human_feedback", action="store_true",
                      help="Collect human feedback for evaluation")
    
    # Visualization options
    parser.add_argument("--visualize", action="store_true",
                      help="Generate visualizations of evaluation results")
    parser.add_argument("--plot_type", type=str, choices=["simple", "comprehensive", "comparison"],
                      default="comprehensive", help="Type of visualization to generate")
    
    # Text generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=200,
                      help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                      help="Temperature for text generation")
    parser.add_argument("--top_p", type=float, default=0.9,
                      help="Top-p (nucleus sampling) value")
    parser.add_argument("--batch_size", type=int, default=4,
                      help="Batch size for evaluation")
    
    # Task and dataset information
    parser.add_argument("--task_type", type=str, 
                      choices=["general", "creative", "code", "factual", "chat"],
                      default="general", help="Type of generation task")
    parser.add_argument("--dataset_name", type=str, default=None,
                      help="Name of the dataset (defaults to filename if not specified)")
    
    args = parser.parse_args()
    return evaluate_model_with_dataset(args)

if __name__ == "__main__":
    # If run directly with arguments, use command line evaluation
    import sys
    if len(sys.argv) > 1:
        run_evaluation_from_command_line() 

def visualize_training_metrics(training_logs, output_dir="./metrics"):
    """
    Visualize training metrics 
    
    Args:
        training_logs (dict): Dictionary containing training logs
        output_dir (str): Directory to save visualizations
    """
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import os
        import numpy as np
        
        logger.info(f"Visualizing training metrics to {output_dir}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert training logs to DataFrame
        if isinstance(training_logs, list):
            # List of dictionaries format
            df = pd.DataFrame(training_logs)
        elif isinstance(training_logs, dict):
            # Dictionary of lists format
            df = pd.DataFrame(training_logs)
        else:
            logger.error(f"Unsupported training logs format: {type(training_logs)}")
            return
            
        # Check if DataFrame is empty
        if df.empty:
            logger.warning("No training metrics to visualize")
            return
            
        # Plot loss
        if 'loss' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(df['loss'], label='Training Loss')
            if 'eval_loss' in df.columns:
                plt.plot(df['eval_loss'], label='Validation Loss')
            plt.title('Loss During Training')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'loss.png'))
            plt.close()
            
        # Plot learning rate
        if 'learning_rate' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(df['learning_rate'])
            plt.title('Learning Rate During Training')
            plt.xlabel('Step')
            plt.ylabel('Learning Rate')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'learning_rate.png'))
            plt.close()
            
        # Plot other metrics
        metrics_to_plot = [col for col in df.columns if col not in ['loss', 'eval_loss', 'learning_rate', 'epoch', 'step']]
        for metric in metrics_to_plot:
            if df[metric].dtype == np.float64 or df[metric].dtype == np.int64:
                plt.figure(figsize=(10, 6))
                plt.plot(df[metric])
                plt.title(f'{metric.capitalize()} During Training')
                plt.xlabel('Step')
                plt.ylabel(metric.capitalize())
                plt.grid(True)
                plt.savefig(os.path.join(output_dir, f'{metric}.png'))
                plt.close()
                
        # Combined plot
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        if 'loss' in df.columns:
            plt.plot(df['loss'], label='Training Loss')
        if 'eval_loss' in df.columns:
            plt.plot(df['eval_loss'], label='Validation Loss')
        plt.title('Training Metrics')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        if 'learning_rate' in df.columns:
            plt.plot(df['learning_rate'], color='green')
            plt.ylabel('Learning Rate')
        plt.xlabel('Step')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_summary.png'))
        plt.close()
        
        logger.info(f"Training visualizations saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error visualizing training metrics: {str(e)}")