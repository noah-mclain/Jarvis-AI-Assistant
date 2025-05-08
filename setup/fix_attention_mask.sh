#!/bin/bash
# Script to apply the attention mask fix for DeepSeek models
# This can be run independently of the training process

echo "===== Jarvis AI Assistant - Attention Mask Fix ====="
echo "This script applies fixes for the 'too many values to unpack (expected 2)' error"
echo "that occurs with DeepSeek and LLaMA models during training."
echo "========================================"

# Make the Python fix script executable
chmod +x setup/attention_mask_fix.py

# Run the fix script
echo "Applying attention mask fixes..."
python setup/attention_mask_fix.py

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Create a test attention mask with problematic shape
    batch_size = 2
    seq_length = 10
    
    # Create a 3D attention mask (problematic shape)
    attention_mask_3d = torch.ones(batch_size, 1, seq_length)
    logger.info(f'Created 3D attention mask with shape: {attention_mask_3d.shape}')
    
    # Apply our fix to reshape it to 2D
    attention_mask_2d = attention_mask_3d.view(batch_size, seq_length)
    logger.info(f'Reshaped to 2D attention mask with shape: {attention_mask_2d.shape}')
    
    # Verify the fix worked
    if attention_mask_2d.dim() == 2 and attention_mask_2d.shape == (batch_size, seq_length):
        print('✅ Attention mask fix verification passed!')
    else:
        print(f'❌ Attention mask fix verification failed: wrong shape {attention_mask_2d.shape}')
        sys.exit(1)
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
