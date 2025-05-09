#!/usr/bin/env python3
"""
Fix for the CustomEncoderDecoderModel class in unified_deepseek_training.py.
This script patches the forward method to handle both tuple and non-tuple returns.
"""

import sys
import logging
import re
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def fix_custom_encoder_decoder_model():
    """Fix the CustomEncoderDecoderModel class in unified_deepseek_training.py"""
    try:
        # Find the unified_deepseek_training.py file
        file_path = Path("src/generative_ai_module/unified_deepseek_training.py")
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False

        # Read the file content
        with open(file_path, "r") as f:
            content = f.read()

        # Create a backup of the original file
        backup_path = file_path.with_suffix(".py.bak3")
        with open(backup_path, "w") as f:
            f.write(content)
        logger.info(f"Created backup of original file: {backup_path}")

        # Find the CustomEncoderDecoderModel class
        class_pattern = r"class CustomEncoderDecoderModel\([^)]*\):.*?def forward\(self,[^)]*\):[^}]*?return logits"
        class_match = re.search(class_pattern, content, re.DOTALL)

        if not class_match:
            logger.warning("Could not find the CustomEncoderDecoderModel class")
            return False

        # Get the class code
        class_code = class_match.group(0)

        # Check if the class already handles tuple returns
        if "isinstance" in class_code and "tuple" in class_code:
            logger.info("CustomEncoderDecoderModel already handles tuple returns")
            return True

        # Find the forward method
        forward_pattern = r"def forward\(self,[^)]*\):[^}]*?return logits"
        forward_match = re.search(forward_pattern, class_code, re.DOTALL)

        if not forward_match:
            logger.warning("Could not find the forward method in the CustomEncoderDecoderModel class")
            return False

        # Get the forward method code
        forward_method = forward_match.group(0)

        # Modify the forward method to handle tuple returns
        modified_forward_method = forward_method.replace(
            "        # Get embeddings from CNN model\n        with torch.no_grad():",
            "        # Get embeddings from CNN model\n        with torch.no_grad():\n            # Handle both tuple and non-tuple returns"
        )

        modified_forward_method = modified_forward_method.replace(
            "            encoder_hidden_states = self.cnn_model.model.encoder(",
            "            encoder_output = self.cnn_model.model.encoder("
        )

        modified_forward_method = modified_forward_method.replace(
            "        # Encode with transformer encoder\n        encoder_output = self.encoder(encoder_hidden_states, src_key_padding_mask=~input_attention_mask)",
            "            # Handle tuple returns from encoder\n            if isinstance(encoder_output, tuple):\n                encoder_hidden_states = encoder_output[0]\n            else:\n                encoder_hidden_states = encoder_output\n\n        # Encode with transformer encoder\n        encoder_output = self.encoder(encoder_hidden_states, src_key_padding_mask=~input_attention_mask)"
        )

        # Replace the original forward method with the modified one
        modified_class_code = class_code.replace(forward_method, modified_forward_method)

        # Replace the original class with the modified one
        content = content.replace(class_code, modified_class_code)

        # Write the modified content back to the file
        with open(file_path, "w") as f:
            f.write(content)
        logger.info(f"Successfully updated file: {file_path}")

        return True
    except Exception as e:
        logger.error(f"Error fixing CustomEncoderDecoderModel: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    # Fix CustomEncoderDecoderModel
    success = fix_custom_encoder_decoder_model()

    # Exit with appropriate code
    sys.exit(0 if success else 1)
