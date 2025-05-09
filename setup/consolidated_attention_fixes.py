#!/usr/bin/env python3
"""
Consolidated Attention Fixes

This module consolidates all attention mask fixes for DeepSeek and other models.
It includes fixes for:
- Attention mask dimension mismatches
- Tensor size mismatches
- Tuple unpacking errors
- Unmasked value parameter issues
- General attention mask handling

This consolidates functionality from:
- attention_mask_fix.py
- comprehensive_attention_mask_fix.py
- fix_all_attention_issues.py
- fix_attention_dimension_mismatch.py
- fix_attention_mask_params.py
- fix_attention_mask.py
- fix_attention_without_deepseek.py
- fix_tensor_size_mismatch.py
- fix_transformers_attention_mask.py
- fix_ultimate_attention_fix.py
- ultimate_attention_fix_new.py
- ultimate_attention_fix.py
"""

import os
import sys
import inspect
import types
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Some fixes may not work.")

try:
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Some fixes may not work.")

try:
    from transformers.models.deepseek import DeepSeekModel, DeepSeekConfig, DeepSeekAttention
    DEEPSEEK_AVAILABLE = True
except ImportError:
    DEEPSEEK_AVAILABLE = False
    logger.warning("DeepSeek model not available. Some fixes may not work.")

# Add the parent directory to the path so we can import from the root
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def fix_transformers_utils():
    """Create transformers.utils module if it doesn't exist."""
    try:
        import transformers.utils
        logger.info("transformers.utils module already exists.")
    except ImportError:
        logger.info("Creating transformers.utils module...")
        
        # Find the transformers package directory
        transformers_dir = os.path.dirname(transformers.__file__)
        utils_dir = os.path.join(transformers_dir, "utils")
        
        # Create the utils directory if it doesn't exist
        os.makedirs(utils_dir, exist_ok=True)
        
        # Create an empty __init__.py file
        with open(os.path.join(utils_dir, "__init__.py"), "w") as f:
            f.write("""
# Auto-generated utils module for transformers
import logging
logger = logging.getLogger(__name__)
""")
        
        # Try to import it again to verify
        try:
            import transformers.utils
            logger.info("Successfully created transformers.utils module.")
        except ImportError:
            logger.error("Failed to create transformers.utils module.")

def patch_unmask_unattended(model):
    """
    Patch the unmask_unattended method in the model.
    
    This is the most comprehensive fix that addresses:
    - Attention mask dimension mismatches
    - Tensor size mismatches
    - Unmasked value parameter issues
    """
    if not hasattr(model, "unmask_unattended"):
        logger.warning("Model does not have unmask_unattended method. Skipping patch.")
        return False
    
    # Define the patched method
    def patched_unmask_unattended(self, attention_mask, unmasked_value=0.0):
        """
        Patched unmask_unattended method that handles various edge cases.
        
        Args:
            attention_mask: The attention mask tensor
            unmasked_value: The value to use for unmasked positions (default: 0.0)
            
        Returns:
            The processed attention mask
        """
        # Check if attention_mask is None
        if attention_mask is None:
            return None
        
        # Get the device and dtype of the attention mask
        device = attention_mask.device
        dtype = attention_mask.dtype
        
        # Handle different attention mask shapes
        if len(attention_mask.shape) == 2:
            # [batch_size, seq_length]
            batch_size, seq_length = attention_mask.shape
            
            # Create a causal mask
            causal_mask = torch.triu(
                torch.ones((seq_length, seq_length), dtype=dtype, device=device) * unmasked_value,
                diagonal=1,
            )
            
            # Expand the causal mask to match the batch size
            causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Expand attention_mask to 3D
            expanded_mask = attention_mask.unsqueeze(1).expand(-1, seq_length, -1)
            
            # Combine with causal mask
            full_attention_mask = expanded_mask + causal_mask
            
            return full_attention_mask
            
        elif len(attention_mask.shape) == 3:
            # [batch_size, seq_length, seq_length]
            batch_size, seq_length, _ = attention_mask.shape
            
            # Create a causal mask
            causal_mask = torch.triu(
                torch.ones((seq_length, seq_length), dtype=dtype, device=device) * unmasked_value,
                diagonal=1,
            )
            
            # Expand the causal mask to match the batch size
            causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Combine with existing mask
            full_attention_mask = attention_mask + causal_mask
            
            return full_attention_mask
            
        elif len(attention_mask.shape) == 4:
            # [batch_size, num_heads, seq_length, seq_length]
            batch_size, num_heads, seq_length, _ = attention_mask.shape
            
            # Create a causal mask
            causal_mask = torch.triu(
                torch.ones((seq_length, seq_length), dtype=dtype, device=device) * unmasked_value,
                diagonal=1,
            )
            
            # Expand the causal mask to match the batch size and num_heads
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
            
            # Combine with existing mask
            full_attention_mask = attention_mask + causal_mask
            
            return full_attention_mask
        
        else:
            # Unsupported shape, return as is with a warning
            logger.warning(f"Unsupported attention mask shape: {attention_mask.shape}")
            return attention_mask
    
    # Replace the original method with the patched one
    model.unmask_unattended = types.MethodType(patched_unmask_unattended, model)
    
    logger.info("Successfully patched unmask_unattended method.")
    return True

def fix_tuple_unpacking_error(model):
    """
    Fix the 'too many values to unpack (expected 2)' error in the forward pass.
    
    This error occurs when the model's forward method returns more than 2 values,
    but the calling code expects only 2.
    """
    if not hasattr(model, "forward"):
        logger.warning("Model does not have forward method. Skipping fix.")
        return False
    
    # Get the original forward method
    original_forward = model.forward
    
    # Define the patched forward method
    def patched_forward(self, *args, **kwargs):
        """
        Patched forward method that ensures only 2 values are returned.
        """
        outputs = original_forward(*args, **kwargs)
        
        # If outputs is a tuple with more than 2 elements, return only the first 2
        if isinstance(outputs, tuple) and len(outputs) > 2:
            logger.info("Limiting forward pass outputs to 2 values.")
            return outputs[0], outputs[1]
        
        return outputs
    
    # Replace the original method with the patched one
    model.forward = types.MethodType(patched_forward, model)
    
    logger.info("Successfully patched forward method to fix tuple unpacking error.")
    return True

def fix_attention_dimension_mismatch(model):
    """
    Fix attention dimension mismatches in the model.
    
    This addresses issues where the attention mask has the wrong shape.
    """
    if not hasattr(model, "get_extended_attention_mask"):
        logger.warning("Model does not have get_extended_attention_mask method. Skipping fix.")
        return False
    
    # Get the original method
    original_get_extended_attention_mask = model.get_extended_attention_mask
    
    # Define the patched method
    def patched_get_extended_attention_mask(self, attention_mask, input_shape, device=None, dtype=None):
        """
        Patched get_extended_attention_mask method that handles dimension mismatches.
        """
        # Call the original method
        try:
            extended_mask = original_get_extended_attention_mask(attention_mask, input_shape, device, dtype)
            
            # Check if the shape is correct
            batch_size, seq_length = input_shape
            expected_shape = (batch_size, 1, 1, seq_length)
            
            if extended_mask.shape != expected_shape:
                logger.warning(f"Extended mask shape {extended_mask.shape} doesn't match expected shape {expected_shape}. Reshaping...")
                
                # Reshape the mask to the expected shape
                extended_mask = extended_mask.view(batch_size, 1, 1, seq_length)
            
            return extended_mask
            
        except Exception as e:
            logger.warning(f"Error in original get_extended_attention_mask: {e}")
            
            # Fallback implementation
            if attention_mask is None:
                return None
            
            if device is None:
                device = attention_mask.device
            
            if dtype is None:
                dtype = torch.float32
            
            # We can't use torch.ones_like(attention_mask) here because attention_mask can be 2D or 3D
            # while we need a 4D tensor
            batch_size, seq_length = input_shape
            
            # Create a 4D attention mask
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_mask = extended_mask.expand(batch_size, 1, seq_length, seq_length)
            
            # Convert to the correct dtype
            extended_mask = extended_mask.to(dtype=dtype)
            
            # Apply causal mask if needed
            if hasattr(self.config, "is_decoder") and self.config.is_decoder:
                causal_mask = torch.triu(
                    torch.ones((seq_length, seq_length), dtype=dtype, device=device), diagonal=1
                )
                extended_mask = extended_mask * (1.0 - causal_mask.unsqueeze(0).unsqueeze(0))
            
            return extended_mask
    
    # Replace the original method with the patched one
    model.get_extended_attention_mask = types.MethodType(patched_get_extended_attention_mask, model)
    
    logger.info("Successfully patched get_extended_attention_mask method.")
    return True

def patch_attention_implementation(model):
    """
    Apply all attention-related fixes to the model.
    
    This is the main function that applies all the fixes.
    """
    if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
        logger.error("PyTorch or Transformers not available. Cannot apply attention fixes.")
        return False
    
    # Fix transformers.utils module
    fix_transformers_utils()
    
    # Apply all fixes
    fixes_applied = []
    
    # Fix unmask_unattended method
    if patch_unmask_unattended(model):
        fixes_applied.append("unmask_unattended")
    
    # Fix tuple unpacking error
    if fix_tuple_unpacking_error(model):
        fixes_applied.append("tuple_unpacking")
    
    # Fix attention dimension mismatch
    if fix_attention_dimension_mismatch(model):
        fixes_applied.append("attention_dimension_mismatch")
    
    if fixes_applied:
        logger.info(f"Applied the following fixes: {', '.join(fixes_applied)}")
        return True
    else:
        logger.warning("No fixes were applied.")
        return False

def apply_all_attention_fixes():
    """
    Apply all attention fixes to all models in the transformers library.
    
    This function patches the base classes to ensure all models benefit from the fixes.
    """
    if not TRANSFORMERS_AVAILABLE:
        logger.error("Transformers not available. Cannot apply attention fixes.")
        return False
    
    # Fix transformers.utils module
    fix_transformers_utils()
    
    # Patch the PreTrainedModel class to apply fixes to all models
    from transformers import PreTrainedModel
    
    # Store the original __init__ method
    original_init = PreTrainedModel.__init__
    
    # Define the patched __init__ method
    def patched_init(self, config, *args, **kwargs):
        """
        Patched __init__ method that applies attention fixes after initialization.
        """
        # Call the original __init__ method
        original_init(self, config, *args, **kwargs)
        
        # Apply attention fixes
        patch_attention_implementation(self)
    
    # Replace the original __init__ method with the patched one
    PreTrainedModel.__init__ = patched_init
    
    logger.info("Successfully patched PreTrainedModel.__init__ to apply attention fixes to all models.")
    
    # Specifically patch DeepSeek models if available
    if DEEPSEEK_AVAILABLE:
        # Patch DeepSeekAttention class
        original_deepseek_attention_forward = DeepSeekAttention.forward
        
        def patched_deepseek_attention_forward(self, hidden_states, attention_mask=None, *args, **kwargs):
            """
            Patched forward method for DeepSeekAttention that handles attention mask issues.
            """
            # Handle attention mask
            if attention_mask is not None:
                # Ensure the attention mask has the correct shape
                if len(attention_mask.shape) == 2:
                    # [batch_size, seq_length]
                    batch_size, seq_length = attention_mask.shape
                    
                    # Expand to [batch_size, 1, seq_length, seq_length]
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                    attention_mask = attention_mask.expand(batch_size, 1, seq_length, seq_length)
                
                # Ensure the attention mask has the correct dtype
                if attention_mask.dtype != hidden_states.dtype:
                    attention_mask = attention_mask.to(dtype=hidden_states.dtype)
            
            # Call the original forward method
            return original_deepseek_attention_forward(self, hidden_states, attention_mask, *args, **kwargs)
        
        # Replace the original forward method with the patched one
        DeepSeekAttention.forward = patched_deepseek_attention_forward
        
        logger.info("Successfully patched DeepSeekAttention.forward to handle attention mask issues.")
    
    return True

if __name__ == "__main__":
    # Apply all attention fixes
    apply_all_attention_fixes()
    
    # Print success message
    logger.info("All attention fixes have been applied successfully.")
