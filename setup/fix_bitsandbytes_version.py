#!/usr/bin/env python3
"""
Fix for bitsandbytes version issues with 4-bit quantization.

This script checks the installed bitsandbytes version and upgrades it if needed
to support 4-bit quantization with the `to()` method.

Usage:
    python fix_bitsandbytes_version.py

The script will:
1. Check the current bitsandbytes version
2. Upgrade to a compatible version if needed
3. Apply any necessary patches for compatibility
"""

import os
import sys
import subprocess
import importlib
import logging
import pkg_resources

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Minimum required version for 4-bit quantization with to() method
MIN_VERSION = "0.42.0"

def get_installed_version():
    """Get the installed bitsandbytes version."""
    try:
        import bitsandbytes
        if hasattr(bitsandbytes, "__version__"):
            return bitsandbytes.__version__
        else:
            # Try to get version from pkg_resources
            try:
                return pkg_resources.get_distribution("bitsandbytes").version
            except pkg_resources.DistributionNotFound:
                logger.warning("Could not determine bitsandbytes version from pkg_resources")
                return None
    except ImportError:
        logger.warning("bitsandbytes is not installed")
        return None

def parse_version(version_str):
    """Parse version string into a tuple of integers."""
    if not version_str:
        return (0, 0, 0)
    
    try:
        # Handle versions like "0.41.1.post2"
        parts = version_str.split('.')
        if len(parts) >= 3:
            # Handle post-release versions
            patch = parts[2].split('post')[0] if 'post' in parts[2] else parts[2]
            return (int(parts[0]), int(parts[1]), int(patch))
        elif len(parts) == 2:
            return (int(parts[0]), int(parts[1]), 0)
        else:
            return (int(parts[0]), 0, 0)
    except (ValueError, IndexError) as e:
        logger.warning(f"Could not parse version string '{version_str}': {e}")
        return (0, 0, 0)

def version_is_compatible(version_str):
    """Check if the version is compatible with 4-bit quantization."""
    if not version_str:
        return False
    
    current = parse_version(version_str)
    minimum = parse_version(MIN_VERSION)
    
    # Compare versions
    return current >= minimum

def install_compatible_version():
    """Install a compatible version of bitsandbytes."""
    logger.info(f"Installing bitsandbytes >= {MIN_VERSION}")
    
    # First uninstall current version
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "bitsandbytes"])
        logger.info("Successfully uninstalled existing bitsandbytes")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to uninstall existing bitsandbytes: {e}")
    
    # Install new version
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"bitsandbytes>={MIN_VERSION}"])
        logger.info(f"Successfully installed bitsandbytes >= {MIN_VERSION}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install bitsandbytes >= {MIN_VERSION}: {e}")
        
        # Try installing the latest available version
        try:
            logger.info("Trying to install the latest available version")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "bitsandbytes==0.42.0"])
            logger.info("Successfully installed bitsandbytes 0.42.0")
            return True
        except subprocess.CalledProcessError as e2:
            logger.error(f"Failed to install bitsandbytes 0.42.0: {e2}")
            return False

def add_version_attribute():
    """Add __version__ attribute to bitsandbytes if it doesn't exist."""
    try:
        import bitsandbytes
        if not hasattr(bitsandbytes, "__version__"):
            logger.info("Adding __version__ attribute to bitsandbytes")
            
            # Find the bitsandbytes package location
            spec = importlib.util.find_spec('bitsandbytes')
            if spec and spec.origin:
                init_path = os.path.join(os.path.dirname(spec.origin), '__init__.py')
                
                # Read the current content
                with open(init_path, 'r') as f:
                    content = f.read()
                
                # Add version if not already there
                if '__version__' not in content:
                    # Get version from pkg_resources
                    try:
                        version = pkg_resources.get_distribution("bitsandbytes").version
                    except pkg_resources.DistributionNotFound:
                        version = MIN_VERSION  # Default to minimum version
                    
                    with open(init_path, 'a') as f:
                        f.write(f'\n\n# Added by fix_bitsandbytes_version.py\n__version__ = "{version}"\n')
                    logger.info(f"Added __version__ = {version} to bitsandbytes")
                    
                    # Reload the module to apply changes
                    importlib.reload(bitsandbytes)
                    logger.info(f"Reloaded bitsandbytes, version: {bitsandbytes.__version__}")
                    return True
                else:
                    logger.info("__version__ attribute already exists in bitsandbytes")
                    return True
            else:
                logger.warning("Could not find bitsandbytes package location")
                return False
        else:
            logger.info(f"bitsandbytes already has __version__ attribute: {bitsandbytes.__version__}")
            return True
    except Exception as e:
        logger.error(f"Error adding __version__ attribute to bitsandbytes: {e}")
        return False

def main():
    """Main function to fix bitsandbytes version."""
    logger.info("Checking bitsandbytes version for 4-bit quantization compatibility")
    
    # Get current version
    current_version = get_installed_version()
    logger.info(f"Current bitsandbytes version: {current_version}")
    
    # Check if version is compatible
    if version_is_compatible(current_version):
        logger.info(f"bitsandbytes version {current_version} is compatible with 4-bit quantization")
        
        # Ensure __version__ attribute exists
        add_version_attribute()
        
        return True
    else:
        logger.warning(f"bitsandbytes version {current_version} is not compatible with 4-bit quantization")
        logger.info(f"Minimum required version: {MIN_VERSION}")
        
        # Install compatible version
        success = install_compatible_version()
        if success:
            # Verify installation
            new_version = get_installed_version()
            logger.info(f"Installed bitsandbytes version: {new_version}")
            
            if version_is_compatible(new_version):
                logger.info(f"Successfully upgraded bitsandbytes to compatible version {new_version}")
                
                # Ensure __version__ attribute exists
                add_version_attribute()
                
                return True
            else:
                logger.error(f"Upgraded to version {new_version}, but it's still not compatible")
                return False
        else:
            logger.error("Failed to install compatible bitsandbytes version")
            return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
