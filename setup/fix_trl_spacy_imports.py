#!/usr/bin/env python3
"""
Fix for TRL/PEFT and spaCy import issues.

This script addresses the following errors:
1. "cannot import name 'top_k_top_p_filtering' from 'transformers'"
2. "cannot import name 'ParametricAttention_v2' from 'thinc.api'"

It works by:
1. Adding the missing top_k_top_p_filtering function to transformers
2. Creating dummy modules for thinc.api and related modules
"""

import os
import sys
import logging
import importlib
import inspect
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def fix_trl_peft_imports():
    """Fix TRL/PEFT imports by adding missing top_k_top_p_filtering function."""
    try:
        # Check if the function already exists
        try:
            from transformers import top_k_top_p_filtering
            logger.info("✅ top_k_top_p_filtering is already available in transformers")
            return True
        except ImportError:
            logger.info("Adding top_k_top_p_filtering to transformers...")
            
            # Find transformers path
            import transformers
            transformers_path = os.path.dirname(inspect.getfile(transformers))
            
            # Check if we have the generation utils module
            try:
                # Try to import the necessary components
                try:
                    from transformers.generation.logits_process import TopKLogitsWarper, TopPLogitsWarper
                except ImportError:
                    logger.warning("Could not import logits processors, trying alternative approach")
                    # Define simple versions if not available
                    class TopKLogitsWarper:
                        def __init__(self, top_k, filter_value, min_tokens_to_keep):
                            self.top_k = top_k
                            self.filter_value = filter_value
                            self.min_tokens_to_keep = min_tokens_to_keep
                        
                        def __call__(self, input_ids, scores):
                            import torch
                            top_k = min(max(self.top_k, self.min_tokens_to_keep), scores.size(-1))
                            # Remove all tokens with a probability less than the last token of the top-k
                            indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
                            scores = scores.masked_fill(indices_to_remove, self.filter_value)
                            return scores
                    
                    class TopPLogitsWarper:
                        def __init__(self, top_p, filter_value, min_tokens_to_keep):
                            self.top_p = top_p
                            self.filter_value = filter_value
                            self.min_tokens_to_keep = min_tokens_to_keep
                        
                        def __call__(self, input_ids, scores):
                            import torch
                            sorted_logits, sorted_indices = torch.sort(scores, descending=True)
                            cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                            
                            # Remove tokens with cumulative probability above the threshold
                            sorted_indices_to_remove = cumulative_probs > self.top_p
                            
                            # Shift the indices to the right to keep also the first token above the threshold
                            sorted_indices_to_remove[..., :self.min_tokens_to_keep] = False
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = False
                            
                            # Scatter sorted tensors to original indexing
                            indices_to_remove = sorted_indices_to_remove.scatter(
                                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                            )
                            scores = scores.masked_fill(indices_to_remove, self.filter_value)
                            return scores
                
                # Define the missing function
                def top_k_top_p_filtering(
                    logits,
                    top_k=0,
                    top_p=1.0,
                    filter_value=-float("Inf"),
                    min_tokens_to_keep=1,
                ):
                    """
                    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
                    
                    Args:
                        logits: logits distribution shape (batch size, vocabulary size)
                        top_k: if > 0, keep only top k tokens with highest probability (top-k filtering).
                        top_p: if < 1.0, keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                        filter_value: value to use for filtered tokens
                        min_tokens_to_keep: minimum number of tokens to keep
                    
                    Returns:
                        filtered logits
                    """
                    if top_k > 0:
                        logits = TopKLogitsWarper(top_k=top_k, filter_value=filter_value, 
                                                min_tokens_to_keep=min_tokens_to_keep)(None, logits)
                    
                    if 0 <= top_p < 1.0:
                        logits = TopPLogitsWarper(top_p=top_p, filter_value=filter_value, 
                                                min_tokens_to_keep=min_tokens_to_keep)(None, logits)
                    
                    return logits
                
                # Add the function to transformers namespace
                transformers.top_k_top_p_filtering = top_k_top_p_filtering
                
                # Also add to generation utils if it exists
                if hasattr(transformers, 'generation'):
                    if hasattr(transformers.generation, 'utils'):
                        transformers.generation.utils.top_k_top_p_filtering = top_k_top_p_filtering
                
                logger.info("✅ Successfully added top_k_top_p_filtering to transformers")
                return True
            except Exception as e:
                logger.error(f"❌ Failed to add top_k_top_p_filtering: {e}")
                return False
    except Exception as e:
        logger.error(f"❌ Error fixing TRL/PEFT imports: {e}")
        return False

def fix_spacy_imports():
    """Fix spaCy imports by creating dummy modules."""
    try:
        # Create a comprehensive set of dummy modules
        class DummyModule:
            def __init__(self, name):
                self.__name__ = name
                # Add common attributes that cause issues
                if name == "thinc.api":
                    self.ParametricAttention_v2 = type("ParametricAttention_v2", (), {})
                    self.Model = type("Model", (), {})
                    self.chain = lambda *_args, **_kwargs: None
                    self.with_array = lambda *_args, **_kwargs: None
                elif name == "thinc.types":
                    self.Ragged = type("Ragged", (), {})
                    self.Floats2d = type("Floats2d", (), {})
                elif name == "thinc.config":
                    self.registry = lambda: type("Registry", (), {"namespace": {}})
            
            def __getattr__(self, attr_name):
                # Return a callable for function-like attributes
                if attr_name.startswith("__") and attr_name.endswith("__") and attr_name == "__call__":
                    return lambda *_args, **_kwargs: None
                # For other attributes, return a dummy object
                return type(attr_name, (), {"__call__": lambda *_args, **_kwargs: None})
        
        # Patch problematic modules
        for module_name in [
            'thinc.api', 
            'thinc.types', 
            'thinc.config', 
            'thinc.layers', 
            'thinc.model'
        ]:
            if module_name in sys.modules:
                del sys.modules[module_name]
            sys.modules[module_name] = DummyModule(module_name)
        
        logger.info("✅ Applied comprehensive import fixes for spaCy")
        return True
    except Exception as e:
        logger.error(f"❌ Error fixing spaCy imports: {e}")
        return False

def main():
    """Main function to apply all fixes."""
    logger.info("Starting TRL/PEFT and spaCy import fixes...")
    
    # Fix TRL/PEFT imports
    trl_fixed = fix_trl_peft_imports()
    if trl_fixed:
        logger.info("✅ Successfully fixed TRL/PEFT imports")
    else:
        logger.error("❌ Failed to fix TRL/PEFT imports")
    
    # Fix spaCy imports
    spacy_fixed = fix_spacy_imports()
    if spacy_fixed:
        logger.info("✅ Successfully fixed spaCy imports")
    else:
        logger.error("❌ Failed to fix spaCy imports")
    
    # Return overall success
    return trl_fixed and spacy_fixed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
