#!/usr/bin/env python3
"""
Fix spaCy Installation for Paperspace Environment

This script safely fixes spaCy installation issues in Paperspace without triggering
segmentation faults that can occur with certain combinations of spaCy and thinc.

The script:
1. Uninstalls conflicting packages
2. Installs packages in the correct order
3. Verifies the installation with minimal imports
4. Provides a working spaCy installation with en_core_web_sm model
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging to file to avoid losing output if segfault occurs
log_path = "spacy_fix.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("spacy_fix")

def run_command(cmd, check=False):
    """
    Run a shell command safely, logging output and errors.
    Returns success status and output.
    """
    logger.info(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        if result.stdout:
            logger.info(f"Output: {result.stdout[:100]}...")
        if result.stderr:
            logger.warning(f"Stderr: {result.stderr[:100]}...")
        return result.returncode == 0, result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        if hasattr(e, 'stderr'):
            logger.error(e.stderr)
        return False, str(e)
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return False, str(e)

def uninstall_conflicting_packages():
    """Uninstall all potentially conflicting packages"""
    logger.info("Uninstalling conflicting packages...")
    packages = [
        "spacy", "thinc", "spacy-legacy", "spacy-loggers", 
        "catalogue", "wasabi", "srsly", "murmurhash", 
        "cymem", "preshed", "blis", "langcodes", 
        "pydantic", "pydantic-core", "typer"
    ]
    
    for package in packages:
        run_command([sys.executable, "-m", "pip", "uninstall", "-y", package])
    
    logger.info("Uninstallation complete")

def install_dependencies():
    """Install dependencies in the correct order"""
    logger.info("Installing foundation packages...")
    run_command([sys.executable, "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"])
    
    # Install numpy first as it's a critical dependency
    logger.info("Installing NumPy...")
    run_command([sys.executable, "-m", "pip", "install", "numpy==1.26.4"])
    
    # Install core dependencies with exact versions
    logger.info("Installing core dependencies...")
    dependencies = [
        ["cymem==2.0.11", "preshed==3.0.9", "murmurhash==1.0.12", "blis==0.7.11"],
        ["typer==0.9.0", "catalogue==2.0.10", "wasabi==1.1.3", "srsly==2.4.8"],
        ["pydantic==1.10.13"]
    ]
    
    for dep_group in dependencies:
        run_command([sys.executable, "-m", "pip", "install"] + dep_group)
    
    # Install Thinc separately with proper version
    logger.info("Installing Thinc...")
    run_command([sys.executable, "-m", "pip", "install", "thinc==8.1.10", "--no-deps"])
    run_command([sys.executable, "-m", "pip", "install", "thinc==8.1.10"])
    
    # Install spaCy with specific version
    logger.info("Installing spaCy...")
    run_command([sys.executable, "-m", "pip", "install", "spacy==3.7.4", "--no-deps"])
    run_command([sys.executable, "-m", "pip", "install", "spacy==3.7.4"])
    
    # Install model
    logger.info("Installing en_core_web_sm model...")
    model_url = "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0.tar.gz"
    run_command([sys.executable, "-m", "pip", "install", model_url, "--no-deps"])
    
    logger.info("All dependencies installed")

def verify_installation():
    """Verify that spaCy is installed and working correctly"""
    logger.info("Verifying installation...")
    
    # Use a separate process to isolate potential issues
    verification_script = """
import sys
try:
    import spacy
    print(f'SpaCy version: {spacy.__version__}')
    try:
        # Try loading without doing anything complex
        nlp = spacy.load('en_core_web_sm')
        print('English model loaded successfully')
        doc = nlp('Simple test sentence.')
        print('Basic test successful')
        sys.exit(0)
    except Exception as e:
        print(f'Error with model: {str(e)}')
        sys.exit(1)
except Exception as e:
    print(f'Error importing spaCy: {str(e)}')
    sys.exit(1)
"""
    
    # Write script to a temporary file
    temp_script_path = "verify_spacy.py"
    with open(temp_script_path, "w") as f:
        f.write(verification_script)
    
    # Run the verification script
    success, output = run_command([sys.executable, temp_script_path])
    
    # Clean up
    try:
        os.remove(temp_script_path)
    except:
        pass
    
    if success:
        logger.info("Verification successful!")
        return True
    else:
        logger.error("Verification failed")
        return False

def create_working_test_script():
    """Create a minimal working test script for spaCy"""
    test_script = """#!/usr/bin/env python3
'''
Simple spaCy test script that avoids segmentation faults
'''

import sys

def test_spacy():
    try:
        import spacy
        print(f"spaCy version: {spacy.__version__}")
        
        # Load model with minimal functionality
        nlp = spacy.load("en_core_web_sm")
        print("Model loaded successfully")
        
        # Simple text processing (minimal)
        text = "Jarvis AI is working with spaCy."
        doc = nlp(text)
        
        # Print tokens only - avoid complex processing
        print("\\nTokens:")
        for token in doc:
            print(f"  {token.text}")
            
        print("\\nTest completed successfully!")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = test_spacy()
    sys.exit(0 if success else 1)
"""
    
    # Save the test script
    test_script_path = "test_spacy_minimal.py"
    with open(test_script_path, "w") as f:
        f.write(test_script)
    
    os.chmod(test_script_path, 0o755)  # Make executable
    logger.info(f"Created minimal test script at {test_script_path}")
    return test_script_path

def main():
    """Main entry point"""
    logger.info("Starting spaCy fix for Paperspace environment")
    
    # Step 1: Uninstall conflicting packages
    uninstall_conflicting_packages()
    
    # Step 2: Install dependencies in correct order
    install_dependencies()
    
    # Step 3: Verify installation
    if verify_installation():
        # Step 4: Create a working test script
        test_script_path = create_working_test_script()
        
        logger.info("="*60)
        logger.info("✅ spaCy has been successfully installed!")
        logger.info(f"You can verify it works with: python {test_script_path}")
        logger.info("="*60)
        return 0
    else:
        logger.error("="*60)
        logger.error("❌ Failed to install spaCy properly")
        logger.error("Please check the log file for details: " + log_path)
        logger.error("="*60)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 