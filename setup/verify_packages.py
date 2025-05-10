#!/usr/bin/env python3
"""
Verify Packages for Jarvis AI Assistant

This script verifies that all required packages are installed and have the correct versions.
If a package is missing or has an incorrect version, it will be installed or upgraded.
"""

import os
import sys
import logging
import importlib
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

# Required packages and their versions
REQUIRED_PACKAGES = {
    "torch": "2.1.2",
    "transformers": "4.36.2",
    "peft": "0.6.0",
    "accelerate": "0.25.0",
    "bitsandbytes": "0.43.0",
    "datasets": "2.14.5",
    "numpy": "1.26.4",
    "scipy": "1.12.0",
    "pandas": "2.0.3",
    "matplotlib": "3.8.3",
    "scikit-learn": "1.4.2",
    "joblib": "1.3.2",
    "tqdm": "4.66.1",
    "huggingface_hub": "0.19.4",
    "tokenizers": "0.14.0",
    "einops": "0.7.0",
    "sentencepiece": "0.1.99",
    "nltk": "3.8.1",
    "tensorboard": "2.15.2",
    "protobuf": "3.20.3",
    "werkzeug": "2.3.7",
    "markdown": "3.5.2",
    "xformers": "0.0.23.post1"
}

def verify_packages():
    """Verify that all required packages are installed and have the correct versions"""
    logger.info("Verifying packages")
    
    missing_packages = []
    incorrect_versions = []
    
    for package, required_version in REQUIRED_PACKAGES.items():
        try:
            module = importlib.import_module(package)
            
            # Get the package version
            if hasattr(module, "__version__"):
                version = module.__version__
            elif hasattr(module, "version"):
                version = module.version
            else:
                version = "unknown"
            
            logger.info(f"{package}: {version} (required: {required_version})")
            
            # Check if the version is correct
            if version != "unknown" and version != required_version:
                incorrect_versions.append((package, version, required_version))
        except ImportError:
            logger.warning(f"{package}: Not installed (required: {required_version})")
            missing_packages.append((package, required_version))
    
    # Install missing packages
    if missing_packages:
        logger.info("Installing missing packages")
        for package, version in missing_packages:
            install_package(package, version)
    
    # Upgrade packages with incorrect versions
    if incorrect_versions:
        logger.info("Upgrading packages with incorrect versions")
        for package, current_version, required_version in incorrect_versions:
            upgrade_package(package, current_version, required_version)
    
    # Verify transformers.utils
    verify_transformers_utils()
    
    logger.info("Package verification complete")

def install_package(package, version):
    """Install a package with the specified version"""
    logger.info(f"Installing {package} {version}")
    
    try:
        # Install the package with the specified version
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}=={version}", "--no-deps"])
        logger.info(f"Successfully installed {package} {version}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install {package} {version}: {e}")
        
        # Try to install without version constraint
        try:
            logger.info(f"Trying to install {package} without version constraint")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--no-deps"])
            logger.info(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {package}: {e}")

def upgrade_package(package, current_version, required_version):
    """Upgrade a package to the specified version"""
    logger.info(f"Upgrading {package} from {current_version} to {required_version}")
    
    try:
        # Upgrade the package to the specified version
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}=={required_version}", "--no-deps", "--force-reinstall"])
        logger.info(f"Successfully upgraded {package} to {required_version}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to upgrade {package} to {required_version}: {e}")

def verify_transformers_utils():
    """Verify that transformers.utils is available"""
    logger.info("Verifying transformers.utils")
    
    try:
        import transformers.utils
        logger.info("transformers.utils is available")
    except ImportError:
        logger.warning("transformers.utils is not available")
        
        # Try to fix transformers.utils
        try:
            logger.info("Trying to fix transformers.utils")
            
            # Check if fix_transformers_utils.py exists
            if os.path.exists("setup/fix_transformers_utils.py"):
                logger.info("Running fix_transformers_utils.py")
                subprocess.check_call([sys.executable, "setup/fix_transformers_utils.py"])
                logger.info("Successfully ran fix_transformers_utils.py")
            else:
                logger.warning("fix_transformers_utils.py not found")
                
                # Try to create transformers.utils module
                try:
                    import transformers
                    
                    # Get the transformers package directory
                    transformers_dir = os.path.dirname(transformers.__file__)
                    
                    # Create the utils.py file
                    utils_path = os.path.join(transformers_dir, "utils.py")
                    
                    # Check if the file already exists
                    if not os.path.exists(utils_path):
                        logger.info(f"Creating {utils_path}")
                        
                        with open(utils_path, "w") as f:
                            f.write("""
# Minimal transformers.utils module
import torch

def get_attention_mask_dtype(dtype):
    if dtype == torch.float16:
        return torch.float32
    elif dtype == torch.bfloat16:
        return torch.float32
    else:
        return dtype

def convert_attention_mask(attention_mask, dtype):
    if attention_mask is None:
        return None
    
    # Convert to the appropriate dtype
    if attention_mask.dtype != dtype:
        attention_mask = attention_mask.to(dtype=dtype)
    
    return attention_mask
""")
                        
                        logger.info(f"Successfully created {utils_path}")
                    else:
                        logger.info(f"{utils_path} already exists")
                except Exception as e:
                    logger.error(f"Failed to create transformers.utils module: {e}")
        except Exception as e:
            logger.error(f"Failed to fix transformers.utils: {e}")

if __name__ == "__main__":
    verify_packages()
