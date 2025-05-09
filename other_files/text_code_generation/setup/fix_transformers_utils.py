#!/usr/bin/env python3
"""
Fix for transformers.utils import issue.

This script ensures that transformers.utils is available by:
1. Checking if transformers is installed
2. Creating a utils.py file in the transformers package if it doesn't exist'
3. Adding necessary imports and functions to the utils.py file
"""

import os
import sys
import importlib
import logging
import shutil
import subprocess
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

def find_transformers_package():
    """Find the transformers package directory."""
    try:
        import transformers
        return os.path.dirname(transformers.__file__)
    except ImportError:
        logger.error("Transformers package not found. Please install it first.")
        return None

def create_utils_module(transformers_dir):
    """Create a utils.py file in the transformers package."""
    utils_path = os.path.join(transformers_dir, "utils.py")
    
    # Check if utils.py already exists
    if os.path.exists(utils_path):
        logger.info(f"utils.py already exists at {utils_path}")
        return True
    
    # Create utils.py with necessary imports and functions
    logger.info(f"Creating utils.py at {utils_path}")
    with open(utils_path, "w") as f:
        f.write("""
# Auto-generated utils.py for transformers package
import os
import sys
import importlib
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

def is_torch_available():
    \"\"\"Check if PyTorch is available.\"\"\"
    try:
        import torch
        return True
    except ImportError:
        return False

def is_tf_available():
    \"\"\"Check if TensorFlow is available.\"\"\"
    try:
        import tensorflow as tf
        return True
    except ImportError:
        return False

def is_flax_available():
    \"\"\"Check if Flax is available.\"\"\"
    try:
        import jax
        import flax
        return True
    except ImportError:
        return False

def is_safetensors_available():
    \"\"\"Check if safetensors is available.\"\"\"
    try:
        import safetensors
        return True
    except ImportError:
        return False

def is_torch_cuda_available():
    \"\"\"Check if CUDA is available for PyTorch.\"\"\"
    if is_torch_available():
        import torch
        return torch.cuda.is_available()
    return False

def is_torch_bf16_available():
    \"\"\"Check if BF16 is available for PyTorch.\"\"\"
    if is_torch_available():
        import torch
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return False

def is_torch_fp16_available():
    \"\"\"Check if FP16 is available for PyTorch.\"\"\"
    if is_torch_available():
        import torch
        return torch.cuda.is_available()
    return False

def is_accelerate_available():
    \"\"\"Check if accelerate is available.\"\"\"
    try:
        import accelerate
        return True
    except ImportError:
        return False

def is_bitsandbytes_available():
    \"\"\"Check if bitsandbytes is available.\"\"\"
    try:
        import bitsandbytes
        return True
    except ImportError:
        return False

def is_peft_available():
    \"\"\"Check if peft is available.\"\"\"
    try:
        import peft
        return True
    except ImportError:
        return False

def is_datasets_available():
    \"\"\"Check if datasets is available.\"\"\"
    try:
        import datasets
        return True
    except ImportError:
        return False

def is_sentencepiece_available():
    \"\"\"Check if sentencepiece is available.\"\"\"
    try:
        import sentencepiece
        return True
    except ImportError:
        return False

def is_tokenizers_available():
    \"\"\"Check if tokenizers is available.\"\"\"
    try:
        import tokenizers
        return True
    except ImportError:
        return False

def is_tqdm_available():
    \"\"\"Check if tqdm is available.\"\"\"
    try:
        import tqdm
        return True
    except ImportError:
        return False

def is_flash_attn_available():
    \"\"\"Check if flash-attn is available.\"\"\"
    try:
        import flash_attn
        return True
    except ImportError:
        return False

def is_xformers_available():
    \"\"\"Check if xformers is available.\"\"\"
    try:
        import xformers
        return True
    except ImportError:
        return False
""")""
    
    return os.path.exists(utils_path)

def fix_transformers_utils():
    """Fix the transformers.utils import issue."""
    # Find transformers package
    transformers_dir = find_transformers_package()
    if not transformers_dir:
        logger.error("Could not find transformers package directory.")
        return False
    
    logger.info(f"Found transformers package at {transformers_dir}")
    
    # Create utils.py if it doesn't exist
    if create_utils_module(transformers_dir):
        logger.info("Successfully created or verified utils.py")
        
        # Try to import transformers.utils to verify the fix
        try:
            import transformers.utils
            logger.info("Successfully imported transformers.utils")
            return True
        except ImportError as e:
            logger.error(f"Failed to import transformers.utils after fix: {e}")
            return False
    else:
        logger.error("Failed to create utils.py")
        return False

def main():
    """Main function."""
    logger.info("Starting transformers.utils fix...")
    success = fix_transformers_utils()
    
    if success:
        logger.info("✅ transformers.utils fix applied successfully!")
    else:
        logger.error("❌ Failed to fix transformers.utils")
        
        # Try reinstalling transformers as a last resort
        logger.info("Attempting to reinstall transformers...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "transformers"])
        subprocess.run([sys.executable, "-m", "pip", "install", "transformers==4.36.2"])
        
        # Try the fix again
        logger.info("Trying the fix again after reinstall...")
        success = fix_transformers_utils()
        
        if success:
            logger.info("✅ transformers.utils fix applied successfully after reinstall!")
        else:
            logger.error("❌ Failed to fix transformers.utils even after reinstall")
    
    return success

if __name__ == "__main__":
    main()
