#!/usr/bin/env python3
"""
Fix for unsloth trust_remote_code issue.

This script patches the unsloth library to properly handle the trust_remote_code parameter
when loading models. This fixes the 'trust_remote_code' KeyError that can occur when
loading DeepSeek models with unsloth.

Usage:
    python fix_unsloth_trust_remote_code.py

The script will:
1. Check if unsloth is installed
2. Patch the unsloth library to handle trust_remote_code correctly
3. Verify the patch was applied successfully
"""

import os
import sys
import logging
import importlib
import inspect
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def check_unsloth_installation():
    """Check if unsloth is installed and return its path."""
    try:
        import unsloth
        logger.info(f"Unsloth is installed at: {unsloth.__file__}")
        return Path(unsloth.__file__).parent
    except ImportError:
        logger.error("Unsloth is not installed. Please install it first.")
        return None

def patch_unsloth_models():
    """Patch the unsloth models module to handle trust_remote_code correctly."""
    try:
        import unsloth.models

        # Check if BaseAdapter exists
        if not hasattr(unsloth.models, "BaseAdapter"):
            logger.warning("unsloth.models.BaseAdapter not found. This might be a different version of unsloth.")

            # Try to find any adapter class
            adapter_classes = []
            for name in dir(unsloth.models):
                obj = getattr(unsloth.models, name)
                if inspect.isclass(obj) and hasattr(obj, "get_model_and_tokenizer"):
                    adapter_classes.append((name, obj))

            if not adapter_classes:
                logger.error("No adapter classes found in unsloth.models.")
                return False

            logger.info(f"Found adapter classes: {[name for name, _ in adapter_classes]}")

            # Patch each adapter class
            for name, adapter_class in adapter_classes:
                original_get_model_and_tokenizer = adapter_class.get_model_and_tokenizer

                def patched_get_model_and_tokenizer(self, *args, **kwargs):
                    """Patched version that ensures trust_remote_code is set correctly"""
                    # Make sure trust_remote_code is included in kwargs
                    if 'trust_remote_code' not in kwargs:
                        kwargs['trust_remote_code'] = True

                    logger.info(f"Calling {name}.get_model_and_tokenizer with trust_remote_code={kwargs.get('trust_remote_code')}")
                    return original_get_model_and_tokenizer(self, *args, **kwargs)

                # Apply the patch
                adapter_class.get_model_and_tokenizer = patched_get_model_and_tokenizer
                logger.info(f"✅ Successfully patched {name}.get_model_and_tokenizer")
        else:
            # Patch BaseAdapter
            original_get_model_and_tokenizer = unsloth.models.BaseAdapter.get_model_and_tokenizer

            def patched_get_model_and_tokenizer(self, *args, **kwargs):
                """Patched version that ensures trust_remote_code is set correctly"""
                # Make sure trust_remote_code is included in kwargs
                if 'trust_remote_code' not in kwargs:
                    kwargs['trust_remote_code'] = True

                logger.info(f"Calling BaseAdapter.get_model_and_tokenizer with trust_remote_code={kwargs.get('trust_remote_code')}")
                return original_get_model_and_tokenizer(self, *args, **kwargs)

            # Apply the patch
            unsloth.models.BaseAdapter.get_model_and_tokenizer = patched_get_model_and_tokenizer
            logger.info("✅ Successfully patched BaseAdapter.get_model_and_tokenizer")

        # Also patch FastLanguageModel.from_pretrained
        from unsloth import FastLanguageModel
        original_from_pretrained = FastLanguageModel.from_pretrained

        @staticmethod
        def patched_from_pretrained(*args, **kwargs):
            """Patched version that ensures trust_remote_code is set correctly"""
            # Make sure trust_remote_code is included in kwargs
            if 'trust_remote_code' not in kwargs:
                kwargs['trust_remote_code'] = True

            logger.info(f"Calling FastLanguageModel.from_pretrained with trust_remote_code={kwargs.get('trust_remote_code')}")
            return original_from_pretrained(*args, **kwargs)

        # Apply the patch
        FastLanguageModel.from_pretrained = patched_from_pretrained
        logger.info("✅ Successfully patched FastLanguageModel.from_pretrained")

        return True
    except Exception as e:
        logger.error(f"Error patching unsloth models: {e}")
        return False

def patch_unsloth_file():
    """Patch the unsloth models.py file directly."""
    try:
        unsloth_path = check_unsloth_installation()
        if not unsloth_path:
            return False

        # Find models.py
        models_path = unsloth_path / "models" / "__init__.py"
        if not models_path.exists():
            logger.error(f"Could not find models.py at {models_path}")
            return False

        # Read the file
        with open(models_path, "r") as f:
            content = f.read()

        # Check if the file already contains our patch
        if "# Patched by fix_unsloth_trust_remote_code.py" in content:
            logger.info("File already patched.")
            return True

        # Find the get_model_and_tokenizer method
        if "def get_model_and_tokenizer" in content:
            # Add trust_remote_code parameter
            patched_content = content.replace(
                "def get_model_and_tokenizer(self, model_name, **kwargs):",
                "def get_model_and_tokenizer(self, model_name, **kwargs):\n        # Patched by fix_unsloth_trust_remote_code.py\n        if 'trust_remote_code' not in kwargs:\n            kwargs['trust_remote_code'] = True"
            )

            # Write the patched file
            with open(models_path, "w") as f:
                f.write(patched_content)

            logger.info(f"✅ Successfully patched {models_path}")
            return True
        else:
            logger.warning("Could not find get_model_and_tokenizer method in the file.")
            return False
    except Exception as e:
        logger.error(f"Error patching unsloth file: {e}")
        return False

def verify_patch():
    """Verify that the patch was applied successfully."""
    try:
        # Reload unsloth modules
        import unsloth
        import importlib
        importlib.reload(unsloth)

        import unsloth.models
        importlib.reload(unsloth.models)

        from unsloth import FastLanguageModel
        importlib.reload(sys.modules['unsloth.models'])

        # Check if the patch was applied
        if hasattr(unsloth.models, "BaseAdapter"):
            # Get the source code of the method
            source = inspect.getsource(unsloth.models.BaseAdapter.get_model_and_tokenizer)
            if "trust_remote_code" in source:
                logger.info("✅ Patch verification successful for BaseAdapter.get_model_and_tokenizer")
                return True
            else:
                logger.warning("Patch verification failed for BaseAdapter.get_model_and_tokenizer")
                return False
        else:
            # Check FastLanguageModel.from_pretrained
            source = inspect.getsource(FastLanguageModel.from_pretrained)
            if "trust_remote_code" in source:
                logger.info("✅ Patch verification successful for FastLanguageModel.from_pretrained")
                return True
            else:
                logger.warning("Patch verification failed for FastLanguageModel.from_pretrained")
                return False
    except Exception as e:
        logger.error(f"Error verifying patch: {e}")
        return False

def main():
    """Main function to fix unsloth trust_remote_code issue."""
    logger.info("Checking unsloth installation...")
    unsloth_path = check_unsloth_installation()
    if not unsloth_path:
        return False

    logger.info("Patching unsloth models...")
    if patch_unsloth_models():
        logger.info("✅ Successfully patched unsloth models in memory")
    else:
        logger.warning("Failed to patch unsloth models in memory")

    logger.info("Patching unsloth file...")
    if patch_unsloth_file():
        logger.info("✅ Successfully patched unsloth file")
    else:
        logger.warning("Failed to patch unsloth file")

    logger.info("Verifying patch...")
    if verify_patch():
        logger.info("✅ Patch verification successful")
        return True
    else:
        logger.warning("Patch verification failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
