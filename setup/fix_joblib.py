#!/usr/bin/env python3
"""
Fix joblib dependency issues.

This script ensures that joblib is properly installed and available for scikit-learn.
"""

import os
import sys
import subprocess
import logging
import importlib.util
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def install_package(package, version=None, no_deps=False):
    """Install a package using pip."""
    cmd = [sys.executable, "-m", "pip", "install"]
    
    if version:
        cmd.append(f"{package}=={version}")
    else:
        cmd.append(package)
    
    if no_deps:
        cmd.append("--no-deps")
    
    logger.info(f"Installing {package}{f'=={version}' if version else ''}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Failed to install {package}: {result.stderr}")
        return False
    
    logger.info(f"Successfully installed {package}")
    return True

def check_package(package):
    """Check if a package is installed and importable."""
    try:
        spec = importlib.util.find_spec(package)
        if spec is None:
            logger.warning(f"{package} is not installed")
            return False
        
        module = importlib.import_module(package)
        version = getattr(module, "__version__", "unknown")
        logger.info(f"{package} version: {version}")
        return True
    except ImportError as e:
        logger.warning(f"Failed to import {package}: {e}")
        return False

def fix_joblib():
    """Fix joblib dependency issues."""
    # Check if joblib is installed
    if not check_package("joblib"):
        # Try to install joblib
        if not install_package("joblib", "1.3.2"):
            logger.error("Failed to install joblib")
            return False
    
    # Check if scikit-learn is installed
    if not check_package("sklearn"):
        # Install dependencies for scikit-learn
        install_package("threadpoolctl", "3.2.0")
        install_package("scipy", "1.12.0")
        install_package("numpy", "1.26.4")
        
        # Try to install scikit-learn
        if not install_package("scikit-learn", "1.4.2"):
            logger.error("Failed to install scikit-learn")
            return False
    
    # Check if sklearn.utils._joblib is available
    try:
        from sklearn.utils import _joblib
        logger.info("sklearn.utils._joblib is available")
    except ImportError as e:
        logger.warning(f"Failed to import sklearn.utils._joblib: {e}")
        
        # Try to fix by creating a symlink
        try:
            import sklearn
            import joblib
            
            sklearn_dir = os.path.dirname(sklearn.__file__)
            utils_dir = os.path.join(sklearn_dir, "utils")
            
            # Check if _joblib.py exists
            joblib_py = os.path.join(utils_dir, "_joblib.py")
            if not os.path.exists(joblib_py):
                # Create _joblib.py
                with open(joblib_py, "w") as f:
                    f.write("""
# This file was created by fix_joblib.py to fix joblib dependency issues
import joblib

from joblib import *
""")""
                logger.info(f"Created {joblib_py}")
        except Exception as e2:
            logger.error(f"Failed to create symlink: {e2}")
            return False
    
    # Final check
    try:
        import joblib
        import sklearn
        from sklearn.utils import _joblib
        
        logger.info(f"joblib version: {joblib.__version__}")
        logger.info(f"scikit-learn version: {sklearn.__version__}")
        logger.info("sklearn.utils._joblib is available")
        
        return True
    except Exception as e:
        logger.error(f"Final check failed: {e}")
        return False

def main():
    """Main function."""
    logger.info("Starting joblib fix...")
    
    # Fix joblib
    if fix_joblib():
        logger.info("✅ joblib fix applied successfully!")
        return True
    else:
        logger.error("❌ Failed to fix joblib")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
