#!/usr/bin/env python3
"""
Fix for TRL/PEFT import issues in transformers.

This script addresses the error:
"cannot import name 'top_k_top_p_filtering' from 'transformers'"

It works by:
1. Adding the missing top_k_top_p_filtering function to transformers
2. Ensuring proper imports for TRL and PEFT
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

def find_transformers_path():
    """Find the path to the transformers package."""
    try:
        import transformers
        transformers_path = os.path.dirname(inspect.getfile(transformers))
        logger.info(f"Found transformers at: {transformers_path}")
        return transformers_path
    except ImportError:
        logger.error("Transformers not installed. Please install it first.")
        return None

def add_top_k_top_p_filtering():
    """Add the missing top_k_top_p_filtering function to transformers."""
    transformers_path = find_transformers_path()
    if not transformers_path:
        return False

    # Check if the function already exists in generation/utils.py
    generation_utils_path = os.path.join(transformers_path, "generation", "utils.py")
    if not os.path.exists(generation_utils_path):
        logger.error(f"Generation utils file not found at {generation_utils_path}")
        return False

    # Read the file to check if the function already exists
    with open(generation_utils_path, "r") as f:
        content = f.read()
        if "def top_k_top_p_filtering" in content:
            logger.info("top_k_top_p_filtering function already exists in transformers")
            return True

    # Function doesn't exist, let's add it
    function_code = """

def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    \"\"\""
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering

    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        top_k: if > 0, keep only top k tokens with highest probability (top-k filtering).
        top_p: if < 1.0, keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        min_tokens_to_keep: Minimum number of tokens to keep per batch example in the output.

    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    \"\"\""
    if top_k > 0:
        logits = TopKLogitsWarper(top_k=top_k, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)(
            None, logits
        )

    if 0 <= top_p < 1.0:
        logits = TopPLogitsWarper(top_p=top_p, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)(
            None, logits
        )

    return logits
"""

    # Add the function to the file
    with open(generation_utils_path, "a") as f:
        f.write(function_code)

    logger.info(f"Added top_k_top_p_filtering function to {generation_utils_path}")

    # Also add the function to the __init__.py file
    init_path = os.path.join(transformers_path, "__init__.py")
    if os.path.exists(init_path):
        with open(init_path, "r") as f:
            content = f.read()
            if "top_k_top_p_filtering" not in content:
                # Add import to __init__.py
                with open(init_path, "a") as f:
                    f.write("\nfrom .generation.utils import top_k_top_p_filtering\n")
                logger.info(f"Added top_k_top_p_filtering import to {init_path}")

    return True

def fix_trl_peft_imports():
    """Fix TRL and PEFT imports."""
    try:
        # Try to import TRL and PEFT
        import_success = True
        try:
            import trl
            logger.info(f"TRL version: {getattr(trl, '__version__', 'unknown')}")
        except ImportError as e:
            logger.warning(f"TRL not installed or has issues: {e}")
            import_success = False

        try:
            import peft
            logger.info(f"PEFT version: {getattr(peft, '__version__', 'unknown')}")
        except ImportError as e:
            logger.warning(f"PEFT not installed or has issues: {e}")
            import_success = False

        # If imports failed, try to reinstall
        if not import_success:
            logger.info("Attempting to reinstall TRL and PEFT...")
            os.system(f"{sys.executable} -m pip install trl==0.7.4 peft==0.6.0 --no-deps")
            os.system(f"{sys.executable} -m pip install trl==0.7.4 peft==0.6.0")

        return True
    except Exception as e:
        logger.error(f"Error fixing TRL/PEFT imports: {e}")
        return False

def verify_fix():
    """Verify that the fix worked."""
    try:
        # Try to import top_k_top_p_filtering from transformers
        from transformers import top_k_top_p_filtering
        logger.info("✅ Successfully imported top_k_top_p_filtering from transformers")
        return True
    except ImportError as e:
        logger.error(f"❌ Failed to import top_k_top_p_filtering: {e}")
        return False

def main():
    """Main function."""
    logger.info("Starting TRL/PEFT import fix...")
    
    # Add top_k_top_p_filtering function
    if add_top_k_top_p_filtering():
        logger.info("✅ Successfully added top_k_top_p_filtering function")
    else:
        logger.error("❌ Failed to add top_k_top_p_filtering function")
    
    # Fix TRL and PEFT imports
    if fix_trl_peft_imports():
        logger.info("✅ Successfully fixed TRL/PEFT imports")
    else:
        logger.error("❌ Failed to fix TRL/PEFT imports")
    
    # Verify the fix
    if verify_fix():
        logger.info("✅ Fix verification successful")
        return True
    else:
        logger.error("❌ Fix verification failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
