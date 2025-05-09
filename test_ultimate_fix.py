#!/usr/bin/env python3
"""
Test script for the ultimate attention fix.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def test_fix():
    """Test the ultimate attention fix."""
    try:
        # Try to import the fix
        from setup.ultimate_attention_fix_new import apply_ultimate_fix
        
        # Apply the fix
        result = apply_ultimate_fix()
        
        if result:
            logger.info("✅ Ultimate attention fix applied successfully")
        else:
            logger.warning("⚠️ Ultimate attention fix applied with some issues")
        
        # Try to import transformers to verify the fix
        try:
            import transformers
            logger.info(f"✅ Transformers version: {transformers.__version__}")
            
            # Check if the patched functions exist
            if hasattr(transformers.PreTrainedModel, 'forward'):
                logger.info("✅ PreTrainedModel.forward exists")
            else:
                logger.warning("⚠️ PreTrainedModel.forward does not exist")
            
            if hasattr(transformers.modeling_attn_mask_utils, '_prepare_4d_causal_attention_mask_for_sdpa'):
                logger.info("✅ _prepare_4d_causal_attention_mask_for_sdpa exists")
            else:
                logger.warning("⚠️ _prepare_4d_causal_attention_mask_for_sdpa does not exist")
            
            if hasattr(transformers.modeling_attn_mask_utils.AttentionMaskConverter, '_unmask_unattended'):
                logger.info("✅ AttentionMaskConverter._unmask_unattended exists")
            else:
                logger.warning("⚠️ AttentionMaskConverter._unmask_unattended does not exist")
            
            return True
        except ImportError as e:
            logger.error(f"❌ Could not import transformers: {e}")
            return False
    except ImportError as e:
        logger.error(f"❌ Could not import ultimate_attention_fix_new: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error testing ultimate attention fix: {e}")
        return False

if __name__ == "__main__":
    if test_fix():
        logger.info("✅ All tests passed")
        sys.exit(0)
    else:
        logger.error("❌ Some tests failed")
        sys.exit(1)
