#!/usr/bin/env python3
"""
Set up the Paperspace environment for Jarvis AI Assistant

This script:
1. Sets environment variables to indicate we're in a Paperspace environment
2. Creates necessary directories for Jarvis AI Assistant
3. Checks if the required packages are installed
4. Ensures spaCy is properly installed with en_core_web_sm model
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("paperspace_setup")

def set_paperspace_env_vars():
    """Set environment variables to indicate we're in a Paperspace environment"""
    os.environ['PAPERSPACE'] = 'true'
    os.environ['PAPERSPACE_ENVIRONMENT'] = 'true'
    
    # Also add to .bashrc for persistence
    with open(os.path.expanduser('~/.bashrc'), 'a') as f:
        f.write('\n# Added by Jarvis AI Assistant\n')
        f.write('export PAPERSPACE=true\n')
        f.write('export PAPERSPACE_ENVIRONMENT=true\n')
    
    logger.info("Set Paperspace environment variables")
    
    # Source .bashrc in the current shell
    try:
        subprocess.run(['source', os.path.expanduser('~/.bashrc')], shell=True)
    except:
        logger.warning("Could not source ~/.bashrc automatically. Please run 'source ~/.bashrc' manually.")

def create_directories():
    """Create necessary directories for Jarvis AI Assistant"""
    base_dir = "/notebooks/Jarvis_AI_Assistant"
    
    directories = [
        "models",
        "datasets",
        "metrics",
        "logs",
        "checkpoints",
        "evaluation_metrics",
        "visualizations"
    ]
    
    for directory in directories:
        dir_path = os.path.join(base_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'torch',
        'transformers',
        'datasets',
        'accelerate',
        'unsloth',
        'spacy'  # Add spaCy to the required packages
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Installing missing packages...")
        
        install_cmd = [sys.executable, '-m', 'pip', 'install'] + missing_packages
        subprocess.run(install_cmd)
        
        logger.info("Packages installed successfully")
    else:
        logger.info("All required packages are installed")

def check_and_install_spacy():
    """Check if spaCy and en_core_web_sm model are installed and install if needed"""
    logger.info("Checking spaCy installation...")
    
    # Check if spaCy is installed
    try:
        import spacy
        logger.info(f"spaCy version {spacy.__version__} is installed")
        
        # Check if the model is installed
        try:
            nlp = spacy.load("en_core_web_sm")
            logger.info("en_core_web_sm model is installed")
        except OSError:
            logger.warning("en_core_web_sm model is not installed")
            
            # Install the model
            logger.info("Installing en_core_web_sm model...")
            try:
                # Try direct download via spacy download command
                subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
                logger.info("en_core_web_sm model installed successfully")
            except subprocess.CalledProcessError:
                # If that fails, try installing from URL
                logger.warning("Standard installation failed, trying direct URL installation...")
                model_url = "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0.tar.gz"
                subprocess.run([sys.executable, "-m", "pip", "install", model_url], check=True)
                logger.info("en_core_web_sm model installed successfully from URL")
    
    except ImportError:
        logger.warning("spaCy is not installed")
        
        # Install spaCy
        logger.info("Installing spaCy...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "spacy==3.7.4"], check=True)
            
            # Now install the model
            logger.info("Installing en_core_web_sm model...")
            try:
                # Try direct download via spacy download command
                subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
            except subprocess.CalledProcessError:
                # If that fails, try installing from URL
                logger.warning("Standard installation failed, trying direct URL installation...")
                model_url = "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0.tar.gz"
                subprocess.run([sys.executable, "-m", "pip", "install", model_url], check=True)
            
            logger.info("spaCy and en_core_web_sm model installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install spaCy: {e}")
            logger.info("Please run setup/setup_spacy.py to install spaCy manually")

def main():
    """Main entry point"""
    logger.info("Setting up Paperspace environment for Jarvis AI Assistant")
    
    set_paperspace_env_vars()
    create_directories()
    check_requirements()
    check_and_install_spacy()
    
    logger.info("Paperspace environment setup complete")
    
    # Print instructions
    print("\n" + "="*80)
    print("Paperspace environment setup complete")
    print("You can now run the Jarvis AI Assistant commands from the guide")
    print("="*80)

if __name__ == "__main__":
    main() 