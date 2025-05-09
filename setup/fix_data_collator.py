#!/usr/bin/env python3
"""
Fix for data collator issues in transformer models.

This module provides a patched data collator that ensures:
1. Attention masks have the correct 4D shape
2. All tensors have consistent dtypes
3. All tensors are on the correct device
"""

import sys
import logging
import inspect
from typing import Dict, List, Union, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def apply_data_collator_fix():
    """
    Apply a fix for data collator issues in transformer models.
    
    Returns:
        bool: True if the fix was applied successfully, False otherwise
    """
    try:
        import torch
        import transformers
        from transformers.data.data_collator import DataCollatorForLanguageModeling
        
        logger.info(f"Applying data collator fix for transformers {transformers.__version__}")
        
        # Store the original __call__ method
        original_call = DataCollatorForLanguageModeling.__call__
        
        # Define a patched __call__ method
        def patched_call(self, features, return_tensors=None):
            """
            Patched __call__ method that ensures:
            1. Attention masks have the correct 4D shape
            2. All tensors have consistent dtypes
            3. All tensors are on the correct device
            """
            try:
                # Call the original method to get the batch
                batch = original_call(self, features, return_tensors)
                
                # Check if we have an attention mask
                if "attention_mask" in batch:
                    # Get the device and dtype from input_ids
                    device = batch["input_ids"].device
                    dtype = batch["input_ids"].dtype
                    
                    # Get batch size and sequence length
                    batch_size, seq_length = batch["input_ids"].shape
                    
                    # Check if attention_mask has the wrong shape
                    attention_mask = batch["attention_mask"]
                    
                    # Ensure attention_mask is on the correct device and has the correct dtype
                    attention_mask = attention_mask.to(device=device, dtype=dtype)
                    
                    # If attention_mask is 2D, expand it to 4D
                    if attention_mask.dim() == 2:
                        # First, expand to [batch_size, 1, 1, seq_length]
                        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                        
                        # Then, expand to [batch_size, 1, seq_length, seq_length]
                        # Create a causal mask
                        causal_mask = torch.triu(
                            torch.ones((seq_length, seq_length), device=device, dtype=torch.bool),
                            diagonal=1
                        )
                        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]
                        causal_mask = causal_mask.expand(batch_size, 1, seq_length, seq_length)
                        
                        # Apply the attention mask
                        expanded_mask = attention_mask.expand(-1, -1, seq_length, -1)
                        combined_mask = causal_mask | ~expanded_mask.bool()
                        
                        # Convert to the expected format
                        attention_mask = ~combined_mask
                        
                        # Convert to the correct dtype
                        if dtype != torch.bool:
                            attention_mask = attention_mask.to(dtype=dtype)
                        
                        logger.info(f"Expanded attention mask to 4D: {attention_mask.shape}")
                    
                    # Update the batch with the fixed attention mask
                    batch["attention_mask"] = attention_mask
                
                # Ensure all tensors have the same dtype and are on the same device
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor) and key != "labels":
                        if value.dtype != dtype:
                            logger.info(f"Converting {key} from {value.dtype} to {dtype}")
                            batch[key] = value.to(dtype=dtype)
                        if value.device != device:
                            logger.info(f"Moving {key} from {value.device} to {device}")
                            batch[key] = value.to(device=device)
                
                return batch
            except Exception as e:
                logger.warning(f"Error in patched data collator: {e}")
                # Fall back to the original method
                return original_call(self, features, return_tensors)
        
        # Apply the patch
        DataCollatorForLanguageModeling.__call__ = patched_call
        logger.info("✅ Successfully patched DataCollatorForLanguageModeling.__call__")
        
        # Also patch the SimpleDataCollator if it exists
        try:
            # Try to find SimpleDataCollator in the current environment
            simple_data_collator_found = False
            
            # Check if it's in the transformers module
            if hasattr(transformers, "SimpleDataCollator"):
                SimpleDataCollator = transformers.SimpleDataCollator
                simple_data_collator_found = True
            else:
                # Try to find it in the current modules
                for name, module in sys.modules.items():
                    if hasattr(module, "SimpleDataCollator"):
                        SimpleDataCollator = module.SimpleDataCollator
                        simple_data_collator_found = True
                        break
            
            if simple_data_collator_found:
                # Store the original __call__ method
                original_simple_call = SimpleDataCollator.__call__
                
                # Define a patched __call__ method
                def patched_simple_call(self, features):
                    """
                    Patched __call__ method for SimpleDataCollator that ensures:
                    1. Attention masks have the correct 4D shape
                    2. All tensors have consistent dtypes
                    3. All tensors are on the correct device
                    """
                    try:
                        # Call the original method to get the batch
                        batch = original_simple_call(self, features)
                        
                        # Apply the same fixes as for DataCollatorForLanguageModeling
                        if "attention_mask" in batch:
                            # Get the device and dtype from input_ids
                            device = batch["input_ids"].device
                            dtype = batch["input_ids"].dtype
                            
                            # Get batch size and sequence length
                            batch_size, seq_length = batch["input_ids"].shape
                            
                            # Check if attention_mask has the wrong shape
                            attention_mask = batch["attention_mask"]
                            
                            # Ensure attention_mask is on the correct device and has the correct dtype
                            attention_mask = attention_mask.to(device=device, dtype=dtype)
                            
                            # If attention_mask is 2D, expand it to 4D
                            if attention_mask.dim() == 2:
                                # First, expand to [batch_size, 1, 1, seq_length]
                                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                                
                                # Then, expand to [batch_size, 1, seq_length, seq_length]
                                # Create a causal mask
                                causal_mask = torch.triu(
                                    torch.ones((seq_length, seq_length), device=device, dtype=torch.bool),
                                    diagonal=1
                                )
                                causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]
                                causal_mask = causal_mask.expand(batch_size, 1, seq_length, seq_length)
                                
                                # Apply the attention mask
                                expanded_mask = attention_mask.expand(-1, -1, seq_length, -1)
                                combined_mask = causal_mask | ~expanded_mask.bool()
                                
                                # Convert to the expected format
                                attention_mask = ~combined_mask
                                
                                # Convert to the correct dtype
                                if dtype != torch.bool:
                                    attention_mask = attention_mask.to(dtype=dtype)
                                
                                logger.info(f"Expanded attention mask to 4D: {attention_mask.shape}")
                            
                            # Update the batch with the fixed attention mask
                            batch["attention_mask"] = attention_mask
                        
                        # Ensure all tensors have the same dtype and are on the same device
                        for key, value in batch.items():
                            if isinstance(value, torch.Tensor) and key != "labels":
                                if value.dtype != dtype:
                                    logger.info(f"Converting {key} from {value.dtype} to {dtype}")
                                    batch[key] = value.to(dtype=dtype)
                                if value.device != device:
                                    logger.info(f"Moving {key} from {value.device} to {device}")
                                    batch[key] = value.to(device=device)
                        
                        return batch
                    except Exception as e:
                        logger.warning(f"Error in patched SimpleDataCollator: {e}")
                        # Fall back to the original method
                        return original_simple_call(self, features)
                
                # Apply the patch
                SimpleDataCollator.__call__ = patched_simple_call
                logger.info("✅ Successfully patched SimpleDataCollator.__call__")
            else:
                logger.info("SimpleDataCollator not found, skipping patch")
        except Exception as e:
            logger.warning(f"Error patching SimpleDataCollator: {e}")
        
        logger.info("✅ Data collator fix applied successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to apply data collator fix: {e}")
        return False

if __name__ == "__main__":
    # Apply the fix when the script is run directly
    success = apply_data_collator_fix()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
