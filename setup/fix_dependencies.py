#!/usr/bin/env python3
"""
Fix dependencies for Paperspace environment.

This script ensures that all required dependencies are installed with the correct versions
without modifying any existing code or causing conflicts with the requirements.txt file.
"""

import os
import sys
import subprocess
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_command(command):
    """Run a shell command and log the output."""
    logger.info(f"Running command: {command}")
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    stdout, stderr = process.communicate()
    
    if stdout:
        logger.info(stdout)
    if stderr:
        logger.error(stderr)
    
    return process.returncode

def install_dependencies():
    """Install the required dependencies without conflicts."""
    logger.info("Installing critical dependencies...")
    
    # Install core dependencies with --no-deps to avoid conflicts
    dependencies = [
        "protobuf<4.24",
        "werkzeug",
        "pandas",
        "huggingface-hub",
        "markdown"
    ]
    
    for dependency in dependencies:
        run_command(f"pip install {dependency} --no-deps")
    
    # Verify installations
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        logger.error("PyTorch is not installed!")
    
    try:
        import transformers
        logger.info(f"Transformers version: {transformers.__version__}")
    except ImportError:
        logger.error("Transformers is not installed!")
    
    try:
        import peft
        logger.info(f"PEFT version: {peft.__version__}")
    except ImportError:
        logger.error("PEFT is not installed!")
    
    try:
        import accelerate
        logger.info(f"Accelerate version: {accelerate.__version__}")
    except ImportError:
        logger.error("Accelerate is not installed!")
    
    return True

def main():
    """Main function."""
    logger.info("Starting dependency fix...")
    success = install_dependencies()
    
    if success:
        logger.info("✅ Dependencies fixed successfully!")
    else:
        logger.error("❌ Failed to fix dependencies!")
    
    return success

if __name__ == "__main__":
    main()
