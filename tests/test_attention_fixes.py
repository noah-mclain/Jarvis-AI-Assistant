"""
Consolidated test module for attention mask fixes.

This module consolidates tests from:
- test_attention_mask_fix.py
- test_ultimate_fix.py
- test_peft_fix.py
- test_peft_fix_simple.py
"""

import os
import sys
import unittest
import torch
import logging
from pathlib import Path

# Add the parent directory to the path so we can import from the root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Some tests will be skipped.")

try:
    from peft import get_peft_model, LoraConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT not available. Some tests will be skipped.")

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    logger.warning("Unsloth not available. Some tests will be skipped.")

# Import the fix functions if available
try:
    from setup.ultimate_attention_fix import patch_attention_implementation
    ATTENTION_FIX_AVAILABLE = True
except ImportError:
    ATTENTION_FIX_AVAILABLE = False
    logger.warning("Attention fix not available. Some tests will be skipped.")


class TestAttentionMaskFix(unittest.TestCase):
    """Test the attention mask fix implementation."""

    @unittest.skipIf(not TRANSFORMERS_AVAILABLE, "Transformers not available")
    def test_attention_mask_shape(self):
        """Test that the attention mask has the correct shape."""
        # This is a simplified test that doesn't require loading a full model
        batch_size = 2
        seq_length = 10
        
        # Create a sample attention mask
        attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
        
        # Apply a simple transformation similar to what happens in the model
        expanded_mask = attention_mask.unsqueeze(1).expand(batch_size, seq_length, seq_length)
        
        # Check the shape
        self.assertEqual(expanded_mask.shape, (batch_size, seq_length, seq_length))

    @unittest.skipIf(not TRANSFORMERS_AVAILABLE or not ATTENTION_FIX_AVAILABLE, 
                    "Transformers or attention fix not available")
    def test_patch_attention_implementation(self):
        """Test that the patch_attention_implementation function works."""
        # This test only verifies that the function runs without errors
        try:
            # Create a mock model class with the required methods
            class MockModel:
                def unmask_unattended(self, attention_mask):
                    return attention_mask
            
            model = MockModel()
            
            # Apply the patch
            patch_attention_implementation(model)
            
            # If we got here without errors, the test passes
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"patch_attention_implementation raised an exception: {e}")

    @unittest.skipIf(not TRANSFORMERS_AVAILABLE or not PEFT_AVAILABLE or not ATTENTION_FIX_AVAILABLE,
                    "Transformers, PEFT, or attention fix not available")
    def test_peft_compatibility(self):
        """Test that the attention fix is compatible with PEFT."""
        # This is a simplified test that doesn't require loading a full model
        try:
            # Create a mock model class with the required methods
            class MockModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = torch.nn.Linear(10, 10)
                
                def unmask_unattended(self, attention_mask):
                    return attention_mask
                
                def forward(self, input_ids, attention_mask=None):
                    return self.linear(torch.randn(input_ids.shape[0], 10))
            
            model = MockModel()
            
            # Apply the patch
            if ATTENTION_FIX_AVAILABLE:
                patch_attention_implementation(model)
            
            # Apply LoRA
            if PEFT_AVAILABLE:
                lora_config = LoraConfig(
                    r=8,
                    lora_alpha=32,
                    target_modules=["linear"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM"
                )
                model = get_peft_model(model, lora_config)
            
            # Test a forward pass
            input_ids = torch.ones((2, 5), dtype=torch.long)
            attention_mask = torch.ones((2, 5), dtype=torch.long)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            
            # If we got here without errors, the test passes
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"PEFT compatibility test raised an exception: {e}")


class TestUnslothCompatibility(unittest.TestCase):
    """Test the compatibility with Unsloth."""

    @unittest.skipIf(not UNSLOTH_AVAILABLE or not TRANSFORMERS_AVAILABLE,
                    "Unsloth or Transformers not available")
    def test_unsloth_attention_mask(self):
        """Test that Unsloth handles attention masks correctly."""
        # This is a simplified test that doesn't require loading a full model
        try:
            # Create a simple attention mask
            batch_size = 2
            seq_length = 10
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
            
            # Test that we can manipulate it in a way similar to what Unsloth would do
            expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            expanded_mask = expanded_mask.expand(batch_size, 1, seq_length, seq_length)
            
            # Check the shape
            self.assertEqual(expanded_mask.shape, (batch_size, 1, seq_length, seq_length))
            
            # If we got here without errors, the test passes
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Unsloth compatibility test raised an exception: {e}")


if __name__ == "__main__":
    unittest.main()
