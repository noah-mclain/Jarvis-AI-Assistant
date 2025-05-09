#!/usr/bin/env python3
"""
SpaCy Setup Script for Jarvis AI Assistant

This script ensures spaCy is correctly installed with the en_core_web_sm model
without disrupting other dependencies in the environment.

Features:
- Safe installation of spaCy 3.7.4
- Installation of en_core_web_sm model
- Verification of installation
- Automatic dependency resolution for common issues
"""

import os
import sys
import subprocess
import logging
import pkg_resources
from importlib import util
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("spacy_setup")

# Constants
SPACY_VERSION = "3.7.4"
SPACY_MODEL = "en_core_web_sm"
SPACY_MODEL_VERSION = "3.7.0"
SPACY_MODEL_URL = f"https://github.com/explosion/spacy-models/releases/download/{SPACY_MODEL}-{SPACY_MODEL_VERSION}/{SPACY_MODEL}-{SPACY_MODEL_VERSION}.tar.gz"

def is_package_installed(package_name, version=None):
    """Check if a package is installed, optionally with specific version"""
    try:
        pkg = pkg_resources.get_distribution(package_name)
        if version is None:
            return True
        return pkg.version == version
    except pkg_resources.DistributionNotFound:
        return False

def run_command(cmd, check=True):
    """Run a shell command with proper error handling"""
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        return result.returncode == 0, result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        logger.error(e.stderr)
        return False, e.stderr

def test_spacy_import():
    """Try to import spaCy and the model to check if they work"""
    try:
        spec = util.find_spec("spacy")
        if spec is None:
            return False, "spaCy module not found"
        
        import spacy
        logger.info(f"spaCy version {spacy.__version__} installed")
        
        # Try loading the model
        try:
            nlp = spacy.load(SPACY_MODEL)
            # Test a simple sentence
            doc = nlp("Jarvis AI Assistant is working with spaCy.")
            pos_tags = [(token.text, token.pos_) for token in doc]
            logger.info(f"Model test successful. POS tags: {pos_tags[:3]}...")
            return True, f"spaCy {spacy.__version__} with {SPACY_MODEL} is working correctly"
        except Exception as e:
            return False, f"Error loading model: {str(e)}"
            
    except Exception as e:
        return False, f"Error importing spaCy: {str(e)}"

def install_spacy_safely():
    """Install spaCy without affecting other dependencies"""
    logger.info(f"Installing spaCy {SPACY_VERSION} safely...")
    
    # First try to install without --no-deps which is simpler
    success, output = run_command([
        sys.executable, "-m", "pip", "install", 
        f"spacy=={SPACY_VERSION}"
    ], check=False)
    
    if success:
        logger.info("Standard installation successful")
    else:
        logger.warning("Standard installation failed, trying no-deps installation...")
        
        # Install spaCy with no dependencies
        success, output = run_command([
            sys.executable, "-m", "pip", "install",
            f"spacy=={SPACY_VERSION}", "--no-deps"
        ])
        
        if not success:
            logger.error("Failed to install spaCy")
            return False
        
        # Install minimal required dependencies
        min_deps = [
            "wasabi==1.1.3",
            "srsly==2.4.8", 
            "catalogue==2.0.10",
            "typer==0.9.0",
            "weasel==0.3.4",
            "cloudpathlib==0.16.0",
            "pydantic==1.10.13"
        ]
        
        for dep in min_deps:
            run_command([
                sys.executable, "-m", "pip", "install",
                dep, "--no-deps"
            ], check=False)
    
    # Now install the language model
    logger.info(f"Installing {SPACY_MODEL} model...")
    success, output = run_command([
        sys.executable, "-m", "pip", "install", SPACY_MODEL_URL
    ], check=False)
    
    if not success:
        # Try with no-deps
        logger.warning("Model installation failed, trying with --no-deps...")
        success, output = run_command([
            sys.executable, "-m", "pip", "install", 
            SPACY_MODEL_URL, "--no-deps"
        ])
    
    return success

def fix_spacy_dependencies():
    """Fix common spaCy dependency issues"""
    logger.info("Fixing spaCy dependencies...")
    
    # Check if thinc is installed properly (required by spaCy)
    if not is_package_installed("thinc"):
        logger.info("Installing thinc (required by spaCy)...")
        run_command([
            sys.executable, "-m", "pip", "install",
            "thinc==8.1.10", "--no-deps"
        ])
    
    # Check other critical dependencies
    for dep, version in [
        ("cymem", "2.0.11"),
        ("preshed", "3.0.9"),
        ("murmurhash", "1.0.12"),
        ("blis", "0.7.11")
    ]:
        if not is_package_installed(dep):
            logger.info(f"Installing {dep} {version}...")
            run_command([
                sys.executable, "-m", "pip", "install",
                f"{dep}=={version}", "--no-deps"
            ])

def main():
    """Main entry point"""
    logger.info("Setting up spaCy for Jarvis AI Assistant")
    
    # Check if spaCy is already installed and working
    success, message = test_spacy_import()
    if success:
        logger.info("spaCy is already installed and working correctly!")
        logger.info(message)
        return 0
    
    # spaCy not working, need to install or fix
    logger.warning(f"spaCy check failed: {message}")
    
    # First try fixing dependencies if spaCy is installed but not working
    if is_package_installed("spacy"):
        logger.info("spaCy is installed but not working correctly. Trying to fix dependencies...")
        fix_spacy_dependencies()
        
        # Check if fixing worked
        success, message = test_spacy_import()
        if success:
            logger.info("Fixed spaCy installation!")
            return 0
    
    # Need to install spaCy
    logger.info("Installing spaCy...")
    if install_spacy_safely():
        logger.info("spaCy installation complete")
        
        # Final verification
        success, message = test_spacy_import()
        if success:
            logger.info("spaCy setup completed successfully!")
            return 0
        else:
            logger.error(f"Installation completed but verification failed: {message}")
            return 1
    else:
        logger.error("Failed to install spaCy")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 