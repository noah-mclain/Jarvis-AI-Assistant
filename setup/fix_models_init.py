#!/usr/bin/env python3
"""
Fix the models/__init__.py file in transformers.

This script fixes the syntax error in the models/__init__.py file.
"""

import os
import sys
import logging
import shutil
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

def fix_models_init(transformers_dir):
    """Fix the models/__init__.py file."""
    models_dir = os.path.join(transformers_dir, "models")
    init_path = os.path.join(models_dir, "__init__.py")
    
    if not os.path.exists(init_path):
        logger.error(f"models/__init__.py not found at {init_path}")
        return False
    
    logger.info(f"Fixing {init_path}...")
    
    # Create a backup
    backup_path = init_path + ".bak"
    shutil.copy2(init_path, backup_path)
    logger.info(f"Created backup at {backup_path}")
    
    # Read the file
    with open(init_path, "r") as f:
        lines = f.readlines()
    
    # Create a new file without the deepseek import
    new_lines = []
    for line in lines:
        if "deepseek" not in line:
            new_lines.append(line)
    
    # Write the new file
    with open(init_path, "w") as f:
        f.writelines(new_lines)
    
    logger.info(f"Removed deepseek import from {init_path}")
    
    # Try to import transformers.models
    try:
        import importlib
        import transformers.models
        importlib.reload(transformers.models)
        logger.info("Successfully imported transformers.models")
        return True
    except Exception as e:
        logger.error(f"Failed to import transformers.models: {e}")
        
        # Restore the backup
        shutil.copy2(backup_path, init_path)
        logger.info(f"Restored backup from {backup_path}")
        return False

def create_empty_init(transformers_dir):
    """Create an empty models/__init__.py file."""
    models_dir = os.path.join(transformers_dir, "models")
    init_path = os.path.join(models_dir, "__init__.py")
    
    if not os.path.exists(init_path):
        logger.error(f"models/__init__.py not found at {init_path}")
        return False
    
    logger.info(f"Creating empty {init_path}...")
    
    # Create a backup
    backup_path = init_path + ".bak2"
    shutil.copy2(init_path, backup_path)
    logger.info(f"Created backup at {backup_path}")
    
    # Create a minimal __init__.py file
    with open(init_path, "w") as f:
        f.write("""
# This file was created by fix_models_init.py
# It replaces the original __init__.py file to fix syntax errors

# Import all models
from . import albert
from . import auto
from . import bart
from . import bert
from . import clip
from . import deberta
from . import gpt2
from . import llama
from . import mistral
from . import t5
""")""
    
    logger.info(f"Created minimal {init_path}")
    
    # Try to import transformers.models
    try:
        import importlib
        import transformers.models
        importlib.reload(transformers.models)
        logger.info("Successfully imported transformers.models")
        return True
    except Exception as e:
        logger.error(f"Failed to import transformers.models: {e}")
        
        # Restore the backup
        shutil.copy2(backup_path, init_path)
        logger.info(f"Restored backup from {backup_path}")
        return False

def main():
    """Main function."""
    logger.info("Starting models/__init__.py fix...")
    
    # Find transformers package
    transformers_dir = find_transformers_package()
    if not transformers_dir:
        logger.error("Could not find transformers package directory.")
        return False
    
    logger.info(f"Found transformers package at {transformers_dir}")
    
    # Try to fix models/__init__.py
    if fix_models_init(transformers_dir):
        logger.info("✅ Successfully fixed models/__init__.py")
        return True
    
    # If that fails, try creating an empty __init__.py
    if create_empty_init(transformers_dir):
        logger.info("✅ Successfully created minimal models/__init__.py")
        return True
    
    logger.error("❌ Failed to fix models/__init__.py")
    return False

if __name__ == "__main__":
    success = main()
    if success:
        print("✅ models/__init__.py fix applied successfully!")
    else:
        print("❌ Failed to fix models/__init__.py")
        sys.exit(1)
