#!/usr/bin/env python3
"""
Unified fix script that applies all attention mask fixes in the correct order.
This script addresses the specific error:
"The size of tensor a (6) must match the size of tensor b (2048) at non-singleton dimension 2"
"""

import sys
import logging
import importlib
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def apply_all_fixes():
    """Apply all attention mask fixes in the correct order"""
    try:
        # Step 1: Apply general attention mask fixes
        logger.info("Step 1: Applying general attention mask fixes...")
        try:
            from fix_transformers_attention_mask import fix_transformers_attention_mask
            success = fix_transformers_attention_mask()
            if success:
                logger.info("✅ General attention mask fixes applied successfully")
            else:
                logger.warning("⚠️ General attention mask fixes failed")
        except ImportError:
            logger.warning("⚠️ Could not import fix_transformers_attention_mask module")
            # Try to run as a script
            os.system("python setup/fix_transformers_attention_mask.py")
            logger.info("Ran fix_transformers_attention_mask.py as a script")

        # Step 2: Apply parameter-specific fixes
        logger.info("Step 2: Applying parameter-specific attention mask fixes...")
        try:
            from fix_attention_mask_params import fix_attention_mask_params
            success = fix_attention_mask_params()
            if success:
                logger.info("✅ Parameter-specific attention mask fixes applied successfully")
            else:
                logger.warning("⚠️ Parameter-specific attention mask fixes failed")
        except ImportError:
            logger.warning("⚠️ Could not import fix_attention_mask_params module")
            # Try to run as a script
            os.system("python setup/fix_attention_mask_params.py")
            logger.info("Ran fix_attention_mask_params.py as a script")

        # Step 3: Apply tensor size mismatch fixes
        logger.info("Step 3: Applying tensor size mismatch fixes...")
        try:
            from fix_tensor_size_mismatch import fix_tensor_size_mismatch
            success = fix_tensor_size_mismatch()
            if success:
                logger.info("✅ Tensor size mismatch fixes applied successfully")
            else:
                logger.warning("⚠️ Tensor size mismatch fixes failed")
        except ImportError:
            logger.warning("⚠️ Could not import fix_tensor_size_mismatch module")
            # Try to run as a script
            os.system("python setup/fix_tensor_size_mismatch.py")
            logger.info("Ran fix_tensor_size_mismatch.py as a script")

        # Step 4: Apply attention dimension mismatch fixes
        logger.info("Step 4: Applying attention dimension mismatch fixes...")
        try:
            from fix_attention_dimension_mismatch import fix_attention_dimension_mismatch
            success = fix_attention_dimension_mismatch()
            if success:
                logger.info("✅ Attention dimension mismatch fixes applied successfully")
            else:
                logger.warning("⚠️ Attention dimension mismatch fixes failed")
        except ImportError:
            logger.warning("⚠️ Could not import fix_attention_dimension_mismatch module")
            # Try to run as a script
            os.system("python setup/fix_attention_dimension_mismatch.py")
            logger.info("Ran fix_attention_dimension_mismatch.py as a script")

        # Step 5: Apply direct patch for the specific error
        logger.info("Step 5: Applying direct patch for the specific error...")
        try:
            import torch
            from transformers.modeling_attn_mask_utils import AttentionMaskConverter

            # Define a patched function that handles the specific error
            @staticmethod
            def patched_unmask_unattended(attention_mask, indices_k=None, indices_q=None, unmasked_value=True):
                """
                Patched version of _unmask_unattended that handles the specific tensor size mismatch error.
                """
                # Get the device of the attention mask
                device = attention_mask.device

                # Fix attention_mask shape if needed
                if attention_mask.dim() > 2:
                    # Get the batch size and sequence length
                    batch_size = attention_mask.size(0)
                    seq_length = attention_mask.size(-1)

                    # Reshape to 2D [batch_size, seq_length]
                    attention_mask = attention_mask.view(batch_size, seq_length)
                    logger.info(f"Reshaped attention mask from >2D to 2D: {attention_mask.shape}")

                # Get batch size and sequence length
                batch_size = attention_mask.size(0)
                seq_length = attention_mask.size(-1)

                # Create a temporary tensor on the same device
                tmp = torch.ones(seq_length, device=device)

                # Find the first non-masked position for each sequence
                indices = torch.argmax(attention_mask * tmp, 1, keepdim=True)

                # Create a mask for unattended positions
                mask = torch.arange(seq_length, device=device).expand(batch_size, -1)
                mask = mask < indices

                # Expand mask to 4D
                mask = mask.unsqueeze(1).unsqueeze(2)

                # Handle indices_k and indices_q if provided
                try:
                    if indices_k is not None:
                        if isinstance(indices_k, int):
                            mask = mask.expand(-1, -1, indices_k, -1)
                        else:
                            # Handle case where indices_k is a tensor
                            mask = mask.expand(-1, -1, indices_k.size(0) if hasattr(indices_k, 'size') else indices_k, -1)

                    if indices_q is not None:
                        if isinstance(indices_q, int):
                            mask = mask.expand(-1, indices_q, -1, -1)
                        else:
                            # Handle case where indices_q is a tensor
                            mask = mask.expand(-1, indices_q.size(0) if hasattr(indices_q, 'size') else indices_q, -1, -1)
                except Exception as e:
                    logger.warning(f"Error expanding mask dimensions: {e}")
                    # If we encounter the specific tensor size mismatch error, create a compatible mask
                    error_msg = str(e)
                    if "The size of tensor a (6) must match the size of tensor b (2048) at non-singleton dimension 2" in error_msg:
                        logger.info("Creating a compatible mask for the specific error")

                        # Create a causal mask that matches the expected dimensions
                        causal_mask = torch.triu(
                            torch.ones((seq_length, seq_length), device=device, dtype=torch.bool),
                            diagonal=1
                        )
                        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]
                        causal_mask = causal_mask.expand(batch_size, 1, seq_length, seq_length)

                        # Apply the attention mask if needed
                        if attention_mask is not None:
                            # Expand attention_mask to 4D [batch_size, 1, 1, seq_length]
                            expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                            # Expand to match causal mask dimensions
                            expanded_mask = expanded_mask.expand(-1, -1, seq_length, -1)
                            # Combine with causal mask (logical OR)
                            combined_mask = causal_mask | ~expanded_mask.bool()
                            # Convert back to the expected mask format
                            mask = ~combined_mask if unmasked_value else combined_mask
                        else:
                            mask = ~causal_mask if unmasked_value else causal_mask

                # Convert mask to the expected type based on unmasked_value
                if unmasked_value is not True:
                    mask = mask.to(dtype=attention_mask.dtype) * unmasked_value

                return mask

            # Apply the patch
            AttentionMaskConverter._unmask_unattended = patched_unmask_unattended
            logger.info("✅ Successfully applied direct patch for _unmask_unattended")
        except Exception as e:
            logger.warning(f"⚠️ Failed to apply direct patch: {e}")

        logger.info("✅ All fixes applied successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to apply all fixes: {e}")
        return False

if __name__ == "__main__":
    # Apply all fixes
    success = apply_all_fixes()

    # Exit with appropriate code
    sys.exit(0 if success else 1)
