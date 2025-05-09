#!/usr/bin/env python3
"""
Check and fix bitsandbytes version for 4-bit quantization compatibility.
"""
import sys
import logging
import importlib
import subprocess
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_bitsandbytes_version():
    """Check if bitsandbytes version is compatible with 4-bit quantization"""
    try:
        import bitsandbytes
        if hasattr(bitsandbytes, '__version__'):
            version = bitsandbytes.__version__
            logger.info(f"bitsandbytes version: {version}")

            # Parse version
            try:
                major, minor, patch = map(int, version.split('.'))
                # Check if version is >= 0.42.0 for 4-bit quantization
                if (major > 0) or (major == 0 and minor >= 42):
                    logger.info("✅ bitsandbytes version is compatible with 4-bit quantization")
                    return True
                else:
                    logger.warning("⚠️ bitsandbytes version is too old for 4-bit quantization")
                    logger.warning("Minimum required: 0.42.0 for 4-bit quantization")
                    return False
            except ValueError:
                logger.warning(f"Could not parse bitsandbytes version: {version}")
                return False
        else:
            logger.warning("bitsandbytes version attribute not found")
            return False
    except ImportError:
        logger.error("bitsandbytes is not installed")
        return False

def fix_bitsandbytes_version():
    """Fix bitsandbytes version for 4-bit quantization compatibility"""
    if check_bitsandbytes_version():
        logger.info("bitsandbytes version is already compatible with 4-bit quantization")
        return True

    logger.info("Attempting to fix bitsandbytes version...")

    # Try to upgrade bitsandbytes
    try:
        logger.info("Upgrading bitsandbytes to version 0.43.0...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "bitsandbytes>=0.43.0", "--no-deps"])
        logger.info("bitsandbytes upgraded successfully")

        # Verify the upgrade
        if check_bitsandbytes_version():
            logger.info("✅ bitsandbytes version is now compatible with 4-bit quantization")
            return True
        else:
            logger.warning("⚠️ bitsandbytes version is still not compatible with 4-bit quantization")
            return False
    except Exception as e:
        logger.error(f"Error upgrading bitsandbytes: {e}")
        return False

if __name__ == "__main__":
    # Fix bitsandbytes version
    success = fix_bitsandbytes_version()

    # Exit with appropriate code
    sys.exit(0 if success else 1)
