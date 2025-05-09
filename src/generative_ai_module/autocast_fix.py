#!/usr/bin/env python3
"""
Fix for autocast compatibility issues across different PyTorch versions.
This module provides a safe_autocast context manager that works with different PyTorch versions,
and a function to patch all torch.cuda.amp.autocast calls in the codebase.
"""

import logging
import re
import sys
import os
import importlib
from contextlib import contextmanager
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@contextmanager
def safe_autocast(dtype=None):
    """
    Create a safe autocast context that works with different PyTorch versions.

    Args:
        dtype: The dtype to use for autocast (only supported in newer PyTorch versions)

    Yields:
        An autocast context manager that works with the current PyTorch version
    """
    # Check if torch.cuda.amp.autocast exists
    if not hasattr(torch.cuda, 'amp') or not hasattr(torch.cuda.amp, 'autocast'):
        logger.warning("torch.cuda.amp.autocast not available. Using no-op context manager.")
        # Create a no-op context manager
        @contextmanager
        def dummy_context_manager():
            yield
        yield dummy_context_manager()
        return

    # Check if autocast supports dtype parameter (newer PyTorch versions)
    try:
        if dtype is not None:
            # Try to create an autocast with dtype parameter
            with torch.cuda.amp.autocast(dtype=dtype) as ctx:
                yield ctx
        else:
            # Use default dtype
            with torch.cuda.amp.autocast() as ctx:
                yield ctx
    except TypeError:
        # Older PyTorch versions don't support dtype parameter
        logger.warning("PyTorch version doesn't support dtype in autocast. Using default dtype.")
        with torch.cuda.amp.autocast() as ctx:
            yield ctx

def patch_autocast_calls():
    """
    Patch all autocast calls in the codebase.
    This function monkey-patches torch.cuda.amp.autocast to use our safe_autocast function.
    """
    try:
        # Check if torch.cuda.amp.autocast exists
        if not hasattr(torch.cuda, 'amp') or not hasattr(torch.cuda.amp, 'autocast'):
            logger.warning("torch.cuda.amp.autocast not available. Nothing to patch.")
            return False

        # Store the original autocast function
        original_autocast = torch.cuda.amp.autocast

        # Define a patched autocast function
        class patched_autocast(torch.cuda.amp.autocast):
            def __init__(self, *args, **kwargs):
                # Extract dtype if present
                dtype = kwargs.get('dtype', None)

                # Remove dtype if it's not supported in this PyTorch version
                try:
                    # Try to create an autocast with the provided arguments
                    super().__init__(*args, **kwargs)
                except TypeError:
                    # If dtype is not supported, remove it and try again
                    if 'dtype' in kwargs:
                        logger.warning("PyTorch version doesn't support dtype in autocast. Removing dtype parameter.")
                        kwargs.pop('dtype')
                    super().__init__(*args, **kwargs)

        # Apply the patch
        torch.cuda.amp.autocast = patched_autocast
        logger.info("Successfully patched torch.cuda.amp.autocast")

        # Also patch the module-level import
        if 'torch.cuda.amp' in sys.modules:
            sys.modules['torch.cuda.amp'].autocast = patched_autocast
            logger.info("Successfully patched torch.cuda.amp module")

        return True
    except Exception as e:
        logger.error(f"Error in patch_autocast_calls: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def apply_autocast_fix():
    """Apply the autocast fix and return True if successful"""
    try:
        success = patch_autocast_calls()
        if success:
            logger.info("Successfully applied autocast fix")
        else:
            logger.warning("Failed to apply autocast fix")
        return success
    except Exception as e:
        logger.error(f"Failed to apply autocast fix: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    # Apply the autocast fix
    success = apply_autocast_fix()

    # Exit with appropriate code
    sys.exit(0 if success else 1)
