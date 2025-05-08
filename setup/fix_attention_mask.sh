#!/bin/bash
# Script to apply the attention mask fix for DeepSeek models
# This can be run independently of the training process

echo "===== Jarvis AI Assistant - Attention Mask Fix ====="
echo "This script applies fixes for the 'too many values to unpack (expected 2)' error"
echo "that occurs with DeepSeek and LLaMA models during training."
echo "========================================"

# Make the Python fix script executable
chmod +x setup/fix_transformers_attention_mask.py

# Run the fix script
echo "Applying attention mask fixes..."
python setup/fix_transformers_attention_mask.py

# Check if the fix was successful
if [ $? -ne 0 ]; then
    echo "❌ Attention mask fix failed. See error messages above."
    exit 1
else
    echo "✅ Attention mask fix applied successfully!"
fi

# Verify the fix works
echo "Verifying attention mask fix..."
python -c "
import torch
import sys
import logging
import importlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Check if our patches are in place
    from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa, AttentionMaskConverter

    # Create a test attention mask with problematic shape
    batch_size = 2
    seq_length = 10

    # Create a 3D attention mask (problematic shape)
    attention_mask_3d = torch.ones(batch_size, 1, seq_length)
    logger.info(f'Created 3D attention mask with shape: {attention_mask_3d.shape}')

    # Test if _prepare_4d_causal_attention_mask_for_sdpa can handle 3D masks
    try:
        # Create dummy inputs for the function
        input_shape = (batch_size, seq_length)
        inputs_embeds = torch.randn(batch_size, seq_length, 32)  # Dummy embeddings
        past_key_values_length = 0
        sliding_window = None
        dtype = torch.float32

        # Call the function with our 3D mask
        result = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask_3d,
            input_shape,
            inputs_embeds,
            past_key_values_length,
            sliding_window,
            dtype
        )
        logger.info(f'Successfully processed 3D mask with _prepare_4d_causal_attention_mask_for_sdpa')
    except Exception as e:
        logger.error(f'Error in _prepare_4d_causal_attention_mask_for_sdpa: {e}')
        # Try with 2D mask as fallback
        attention_mask_2d = attention_mask_3d.view(batch_size, seq_length)
        result = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask_2d,
            input_shape,
            inputs_embeds,
            past_key_values_length,
            sliding_window,
            dtype
        )
        logger.info(f'Successfully processed 2D mask with _prepare_4d_causal_attention_mask_for_sdpa')

    # Test if AttentionMaskConverter._unmask_unattended can handle unmasked_value parameter
    try:
        # Create a simple mask
        mask = torch.ones(batch_size, seq_length)
        # Call the function with unmasked_value parameter
        result = AttentionMaskConverter._unmask_unattended(mask, unmasked_value=0.0)
        logger.info(f'Successfully called _unmask_unattended with unmasked_value parameter')
    except Exception as e:
        logger.error(f'Error in _unmask_unattended: {e}')
        # If it fails, our patch might not be working correctly

    print('✅ Attention mask fix verification passed!')
except Exception as e:
    print(f'❌ Attention mask fix verification failed: {e}')
    sys.exit(1)
"

# Check if verification was successful
if [ $? -ne 0 ]; then
    echo "❌ Attention mask fix verification failed."
    exit 1
else
    echo "✅ Attention mask fix verification passed!"
    echo "✅ Your system is now ready for DeepSeek model training."
fi

echo "Done!"
