#!/usr/bin/env python3
"""
Verify required Python packages for Jarvis AI Assistant.
"""

import sys
import os
import subprocess
import importlib

def check_required_packages():
    """
    Check for required Python packages and install missing ones.
    """
    required_packages = ["torch", "transformers", "datasets", "bitsandbytes", "numpy", "pandas", "huggingface-hub"]
    missing_packages = []
    version_conflicts = []
    has_transformers_utils = True
    bitsandbytes_version_ok = False

    for package in required_packages:
        try:
            if package == "huggingface-hub":
                # Special handling for huggingface-hub (import name is huggingface_hub)
                try:
                    import huggingface_hub
                    print(f"✓ {package} is installed")
                    if hasattr(huggingface_hub, "__version__"):
                        version = huggingface_hub.__version__
                        print(f"  huggingface-hub version: {version}")
                except ImportError:
                    print(f"✗ {package} is NOT installed")
                    missing_packages.append(package)
                    # Install huggingface-hub directly
                    print("Installing huggingface-hub...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface-hub==0.19.4", "--no-deps"])
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface-hub==0.19.4"])
                    try:
                        import huggingface_hub
                        print(f"✓ huggingface-hub installed successfully, version: {huggingface_hub.__version__}")
                    except ImportError as e:
                        print(f"✗ Failed to install huggingface-hub: {e}")
                continue

            # For other packages, use importlib to handle hyphens in package names
            try:
                if "-" in package:
                    # Convert package name with hyphens to import name with underscores
                    import_name = package.replace("-", "_")
                    module = importlib.import_module(import_name)
                else:
                    module = importlib.import_module(package)
                print(f"✓ {package} is installed")
            except ImportError:
                # Fall back to __import__ for backward compatibility
                module = __import__(package)
                print(f"✓ {package} is installed")

            # Special check for transformers.utils
            if package == "transformers":
                try:
                    import transformers.utils
                    print("✓ transformers.utils is available")
                except ImportError:
                    print("✗ transformers.utils is NOT available")
                    has_transformers_utils = False

            # Special check for bitsandbytes version for 4-bit quantization
            if package == "bitsandbytes":
                try:
                    import bitsandbytes
                    if hasattr(bitsandbytes, "__version__"):
                        version = bitsandbytes.__version__
                        print(f"  bitsandbytes version: {version}")
                        # Parse version
                        try:
                            major, minor, patch = map(int, version.split("."))
                            # Check if version is >= 0.42.0 for 4-bit quantization
                            if (major > 0) or (major == 0 and minor >= 42):
                                print("✓ bitsandbytes version is compatible with 4-bit quantization")
                                bitsandbytes_version_ok = True
                            else:
                                print("✗ bitsandbytes version is too old for 4-bit quantization")
                                print("  Minimum required: 0.42.0 for 4-bit quantization")
                                version_conflicts.append(("bitsandbytes", ">=0.42.0"))
                        except ValueError:
                            print(f"  Could not parse bitsandbytes version: {version}")
                    else:
                        print("  bitsandbytes version attribute not found")
                except Exception as e:
                    print(f"  Error checking bitsandbytes version: {e}")

            # Check for known version conflicts (just report, do not fix)
            if package == "numpy":
                if not hasattr(module, "__version__") or module.__version__ != "1.26.4":
                    version_conflicts.append(("numpy", "1.26.4"))
            elif package == "typing-extensions":
                if not hasattr(module, "__version__") or module.__version__ != "4.13.2":
                    version_conflicts.append(("typing-extensions", "4.13.2"))
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is NOT installed")

    # Report missing packages and version conflicts
    if missing_packages or version_conflicts or not has_transformers_utils:
        print("\n⚠️ Some required packages are missing or have version conflicts.")
        print("Please run the consolidated setup script ONCE to fix these issues:")
        print("./setup/consolidated_unified_setup.sh")
        print("\nContinuing with training anyway, but some features may not work correctly.")

    # Special warning for bitsandbytes version
    if not bitsandbytes_version_ok:
        print("\n⚠️ WARNING: Your bitsandbytes version may not support 4-bit quantization.")
        print("This can cause errors like: \"Calling `to()` is not supported for `4bit` quantized models\"")
        print("Consider upgrading bitsandbytes to version 0.42.0 or higher:")
        print("pip install bitsandbytes>=0.42.0")
        print("\nContinuing with training, but 4-bit quantization may fail.")

    print("\nContinuing with training...")

if __name__ == "__main__":
    check_required_packages()
