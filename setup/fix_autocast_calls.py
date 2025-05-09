#!/usr/bin/env python3
"""
Fix autocast calls in the unified_deepseek_training.py file.
This script patches all torch.cuda.amp.autocast calls to use our safe_autocast function.
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

def fix_autocast_calls():
    """Fix autocast calls in the unified_deepseek_training.py file"""
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
        backup_path = file_path.with_suffix(".py.bak")
        with open(backup_path, "w") as f:
            f.write(content)
        logger.info(f"Created backup of original file: {backup_path}")

        # Add the safe_autocast function if it doesn't exist
        if "def safe_autocast" not in content:
            # Add the safe_autocast function after the imports
            safe_autocast_func = """
# Define a helper function to handle autocast compatibility
def safe_autocast(dtype=None):
    \"\"\"Create a safe autocast context that works with different PyTorch versions\"\"\"
    from contextlib import contextmanager
    import torch

    @contextmanager
    def dummy_context_manager():
        yield

    # Check if torch.cuda.amp.autocast exists
    if not hasattr(torch.cuda, 'amp') or not hasattr(torch.cuda.amp, 'autocast'):
        logger.warning("torch.cuda.amp.autocast not available. Using no-op context manager.")
        yield dummy_context_manager()
        return

    # Check if autocast supports dtype parameter (newer PyTorch versions)
    try:
        if dtype is not None:
            with torch.cuda.amp.autocast(dtype=dtype) as ctx:
                yield ctx
        else:
            with torch.cuda.amp.autocast() as ctx:
                yield ctx
    except TypeError:
        # Older PyTorch versions don't support dtype parameter
        logger.warning("PyTorch version doesn't support dtype in autocast. Using default dtype.")
        with torch.cuda.amp.autocast() as ctx:
            yield ctx
"""
            # Find the position to insert the function
            import_section_end = content.find("def main(")
            if import_section_end == -1:
                import_section_end = content.find("def train_with_unsloth")

            if import_section_end != -1:
                content = content[:import_section_end] + safe_autocast_func + content[import_section_end:]
                logger.info("Added safe_autocast function to the file")
            else:
                logger.warning("Could not find a suitable position to add the safe_autocast function")

        # Replace all torch.cuda.amp.autocast calls with safe_autocast
        # First, replace calls with dtype parameter
        pattern1 = r"with\s+torch\.cuda\.amp\.autocast\(dtype=([^)]+)\)"
        replacement1 = r"with safe_autocast(dtype=\1)"
        content = re.sub(pattern1, replacement1, content)

        # Then, replace calls without parameters
        pattern2 = r"with\s+torch\.cuda\.amp\.autocast\(\)"
        replacement2 = r"with safe_autocast()"
        content = re.sub(pattern2, replacement2, content)

        # Finally, replace any remaining calls with other parameters
        pattern3 = r"with\s+torch\.cuda\.amp\.autocast\(([^)]+)\)"
        replacement3 = r"with safe_autocast(\1)"
        content = re.sub(pattern3, replacement3, content)

        logger.info("Replaced torch.cuda.amp.autocast calls with safe_autocast")

        # Write the modified content back to the file
        with open(file_path, "w") as f:
            f.write(content)
        logger.info(f"Successfully updated file: {file_path}")

        return True
    except Exception as e:
        logger.error(f"Error fixing autocast calls: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    # Fix autocast calls
    success = fix_autocast_calls()

    # Exit with appropriate code
    sys.exit(0 if success else 1)
