"""
Environment setup for Paperspace Gradient.

This module handles setting up the environment for Paperspace Gradient,
including creating necessary directories and configuring paths.
"""

import os
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories for Jarvis AI Assistant"""
    base_dir = "notebooks/Jarvis_AI_Assistant"
    
    directories = [
        "models",
        "datasets",
        "checkpoints",
        "logs",
        "visualizations",
        "evaluation_metrics"
    ]
    
    for directory in directories:
        dir_path = os.path.join(base_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

def is_paperspace_environment():
    """Checks if code is running in Paperspace Gradient environment."""
    return os.path.exists('/notebooks') and (
        os.environ.get('PAPERSPACE') == 'true' or 
        os.path.exists('/etc/paperspace') or
        os.path.exists('notebooks/Jarvis_AI_Assistant')
    )

def setup_paperspace_env():
    """Set up the Paperspace environment"""
    if is_paperspace_environment():
        logger.info("Setting up Paperspace environment...")
        
        # Create necessary directories
        create_directories()
        
        # Set environment variables for Paperspace
        os.environ['PAPERSPACE'] = 'true'
        os.environ['JARVIS_STORAGE_BASE'] = 'notebooks/Jarvis_AI_Assistant'
        
        logger.info("Paperspace environment setup complete")
    else:
        logger.info("Not running in Paperspace environment, skipping setup")

def get_storage_base_path():
    """Get the base path for storage"""
    if is_paperspace_environment():
        return "notebooks/Jarvis_AI_Assistant"
    else:
        # For local development, use the project root
        current_file = Path(__file__).resolve()
        project_root = str(current_file.parent.parent.parent)
        return project_root

def main():
    """Main function to set up the Paperspace environment"""
    setup_paperspace_env()
    
    # Print storage paths for verification
    base_path = get_storage_base_path()
    logger.info(f"Storage base path: {base_path}")
    
    for directory in ["models", "datasets", "checkpoints", "logs", "visualizations", "evaluation_metrics"]:
        path = os.path.join(base_path, directory)
        logger.info(f"{directory} path: {path}")

if __name__ == "__main__":
    main()
