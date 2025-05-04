#!/usr/bin/env python3
"""
spaCy Integration Check for Jarvis AI Assistant

This script checks if spaCy and the en_core_web_sm model are properly installed and working.
If not, it offers to fix the installation.

Usage:
    python check_spacy.py [--fix] [--quiet]

Options:
    --fix      Automatically fix issues without prompting
    --quiet    Run in quiet mode (fewer log messages)
"""

import os
import sys
import logging
import argparse
import subprocess
from pathlib import Path

# Add the parent directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.generative_ai_module.utils import is_spacy_available, is_spacy_model_loaded, initialize_spacy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("check_spacy")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Check spaCy integration for Jarvis AI")
    parser.add_argument("--fix", action="store_true", help="Automatically fix issues without prompting")
    parser.add_argument("--quiet", action="store_true", help="Run in quiet mode")
    return parser.parse_args()

def run_command(cmd, check=True):
    """Run a shell command with proper error handling"""
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

def check_environment():
    """Check if we're in a Paperspace environment or local"""
    is_paperspace = os.environ.get('PAPERSPACE', '').lower() == 'true'
    if is_paperspace:
        logger.info("Running in Paperspace environment")
    else:
        logger.info("Running in local environment")
    return is_paperspace

def fix_spacy_installation():
    """Fix spaCy installation issues"""
    logger.info("Fixing spaCy installation...")
    
    # Try to run the setup_spacy.py script if it exists
    setup_script = os.path.join(project_root, "setup", "setup_spacy.py")
    if os.path.exists(setup_script):
        logger.info("Running setup_spacy.py script...")
        success, stdout, stderr = run_command([sys.executable, setup_script])
        if success:
            logger.info("spaCy setup script completed successfully")
            return True
        else:
            logger.error(f"Error running setup script: {stderr}")
    
    # If the script doesn't exist or fails, try direct installation
    logger.info("Attempting direct installation...")
    
    # First try to install spaCy
    logger.info("Installing spaCy...")
    success, stdout, stderr = run_command([
        sys.executable, "-m", "pip", "install", "spacy==3.7.4"
    ], check=False)
    
    if not success:
        logger.warning("Standard installation failed, trying no-deps installation...")
        success, stdout, stderr = run_command([
            sys.executable, "-m", "pip", "install", "spacy==3.7.4", "--no-deps"
        ])
        
        if not success:
            logger.error(f"Failed to install spaCy: {stderr}")
            return False
    
    # Now install the model
    logger.info("Installing en_core_web_sm model...")
    success, stdout, stderr = run_command([
        sys.executable, "-m", "pip", "install",
        "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0.tar.gz"
    ], check=False)
    
    if not success:
        logger.warning("Standard model installation failed, trying with Python -m spacy download...")
        success, stdout, stderr = run_command([
            sys.executable, "-m", "spacy", "download", "en_core_web_sm"
        ])
    
    return success

def check_and_fix_spacy(args):
    """Check spaCy installation and fix if needed"""
    # Check if spaCy is installed
    spacy_available, spacy_version = is_spacy_available()
    if spacy_available:
        logger.info(f"spaCy version {spacy_version} is installed")
    else:
        logger.warning("spaCy is not installed")
        
        if args.fix or input("Do you want to install spaCy? [y/N] ").lower() == 'y':
            if fix_spacy_installation():
                logger.info("spaCy has been installed successfully")
                spacy_available = True
            else:
                logger.error("Failed to install spaCy")
                return False
        else:
            logger.info("Skipping spaCy installation")
            return False
    
    # Check if the model is loaded
    model_available, model_message = is_spacy_model_loaded()
    if model_available:
        logger.info(model_message)
    else:
        logger.warning(f"spaCy model issue: {model_message}")
        
        if args.fix or input("Do you want to install the en_core_web_sm model? [y/N] ").lower() == 'y':
            # Try to download the model
            logger.info("Installing en_core_web_sm model...")
            success, stdout, stderr = run_command([
                sys.executable, "-m", "spacy", "download", "en_core_web_sm"
            ])
            
            if success:
                logger.info("en_core_web_sm model installed successfully")
                # Verify the installation
                model_available, model_message = is_spacy_model_loaded()
                if model_available:
                    logger.info("Model verification successful")
                else:
                    logger.error(f"Model verification failed: {model_message}")
                    return False
            else:
                logger.error(f"Failed to install model: {stderr}")
                return False
        else:
            logger.info("Skipping model installation")
            return False
    
    # Final verification - try to use spaCy
    nlp = initialize_spacy(fallback_to_basic=False, log_errors=not args.quiet)
    if nlp:
        # Test the pipeline with a simple sentence
        doc = nlp("Jarvis AI Assistant with spaCy is ready to process text effectively.")
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        tokens = [token.text for token in doc[:5]]
        
        logger.info(f"spaCy pipeline test successful. First 5 tokens: {tokens}")
        if entities:
            logger.info(f"Entities detected: {entities}")
        
        logger.info("✅ spaCy is fully operational!")
        return True
    else:
        logger.error("❌ Failed to initialize spaCy pipeline in final verification")
        return False

def main():
    """Main entry point"""
    args = parse_args()
    
    # Set logging level based on quiet flag
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    logger.info("Checking spaCy integration for Jarvis AI Assistant")
    
    # Check if we're in a Paperspace environment
    is_paperspace = check_environment()
    
    # Check and fix spaCy
    if check_and_fix_spacy(args):
        logger.info("spaCy integration check completed successfully")
        return 0
    else:
        if args.fix:
            logger.error("Failed to fix spaCy integration automatically")
        else:
            logger.warning("spaCy integration issues detected")
            logger.info("Run with --fix to automatically fix issues")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 