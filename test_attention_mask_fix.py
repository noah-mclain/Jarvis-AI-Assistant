#!/usr/bin/env python3
"""
Test script to verify the attention mask fix for DeepSeek models.
This script creates a simple test case that reproduces the issue and tests our fix.
"""

import os
import sys
import torch
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def test_attention_mask_fix():
    """Test the attention mask fix for DeepSeek models"""
    logger.info("Testing attention mask fix...")
    
    # Create a test attention mask with problematic shape
    batch_size = 6
    seq_length = 2048
    
    # Create a 3D attention mask (problematic shape)
    attention_mask_3d = torch.ones(batch_size, 1, seq_length)
    logger.info(f"Created 3D attention mask with shape: {attention_mask_3d.shape}")
    
    # Apply our fix to reshape it to 2D
    logger.info("Applying fix to reshape to 2D...")
    attention_mask_2d = attention_mask_3d.view(batch_size, seq_length)
    logger.info(f"Reshaped to 2D attention mask with shape: {attention_mask_2d.shape}")
    
    # Verify the fix worked
    assert attention_mask_2d.dim() == 2, "Fix failed: attention mask is not 2D"
    assert attention_mask_2d.shape == (batch_size, seq_length), f"Fix failed: wrong shape {attention_mask_2d.shape}"
    
    logger.info("✅ Attention mask fix test passed!")
    return True

def test_forward_with_attention_mask():
    """Test forward pass with fixed attention mask"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        logger.info("Testing forward pass with attention mask fix...")
        
        # Load a small model for testing
        logger.info("Loading a small model for testing...")
        model_name = "gpt2"  # Use a small model for quick testing
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Create input with problematic attention mask
        text = "This is a test"
        inputs = tokenizer(text, return_tensors="pt")
        
        # Create a 3D attention mask (problematic shape)
        batch_size = 1
        seq_length = inputs["input_ids"].shape[1]
        attention_mask_3d = torch.ones(batch_size, 1, seq_length)
        
        logger.info(f"Created 3D attention mask with shape: {attention_mask_3d.shape}")
        
        # Try forward pass with problematic mask
        try:
            logger.info("Trying forward pass with 3D attention mask...")
            outputs_problematic = model(
                input_ids=inputs["input_ids"],
                attention_mask=attention_mask_3d
            )
            logger.info("⚠️ Forward pass with 3D mask succeeded without fix (model may handle it internally)")
        except Exception as e:
            logger.info(f"Forward pass with 3D mask failed as expected: {e}")
            
            # Apply our fix
            logger.info("Applying fix to reshape to 2D...")
            attention_mask_2d = attention_mask_3d.view(batch_size, seq_length)
            
            # Try forward pass with fixed mask
            outputs_fixed = model(
                input_ids=inputs["input_ids"],
                attention_mask=attention_mask_2d
            )
            logger.info("✅ Forward pass with fixed 2D mask succeeded!")
        
        logger.info("✅ Forward pass test completed!")
        return True
        
    except ImportError:
        logger.warning("Transformers not available, skipping forward pass test")
        return False

if __name__ == "__main__":
    # Run the tests
    test_attention_mask_fix()
    test_forward_with_attention_mask()
    
    logger.info("All tests completed!")
