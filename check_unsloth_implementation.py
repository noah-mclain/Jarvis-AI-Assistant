#!/usr/bin/env python3
"""
Check Unsloth implementation details to help debug the model loading issue.
This script inspects the Unsloth implementation and prints relevant information.
"""

import os
import sys
import inspect
from pathlib import Path

def check_unsloth_paths():
    """Check if Unsloth is installed and where it's located"""
    print("=" * 50)
    print("CHECKING UNSLOTH IMPLEMENTATION")
    print("=" * 50)
    
    # Check if custom_unsloth directory exists
    custom_unsloth_path = Path("/notebooks/custom_unsloth")
    if custom_unsloth_path.exists():
        print(f"✓ Custom Unsloth implementation found at: {custom_unsloth_path}")
        
        # Check if it's in the Python path
        if str(custom_unsloth_path) in sys.path:
            print("✓ Custom Unsloth is in the Python path")
        else:
            print("✗ Custom Unsloth is NOT in the Python path")
            
        # List the files in the custom_unsloth directory
        print("\nFiles in custom_unsloth directory:")
        for file in custom_unsloth_path.glob("**/*.py"):
            print(f"  - {file.relative_to(custom_unsloth_path)}")
    else:
        print("✗ Custom Unsloth implementation NOT found at: {custom_unsloth_path}")
    
    # Check if standard unsloth is installed
    try:
        import unsloth
        print(f"\n✓ Standard Unsloth is installed at: {unsloth.__file__}")
        print(f"  Unsloth version: {getattr(unsloth, '__version__', 'Unknown')}")
        
        # Check FastLanguageModel implementation
        try:
            from unsloth import FastLanguageModel
            print("\nFastLanguageModel.from_pretrained implementation:")
            print(inspect.getsource(FastLanguageModel.from_pretrained))
        except (ImportError, AttributeError) as e:
            print(f"✗ Could not inspect FastLanguageModel: {e}")
    except ImportError:
        print("\n✗ Standard Unsloth is NOT installed")
    
    # Check if there's a custom implementation being used
    try:
        sys.path.insert(0, str(custom_unsloth_path))
        from unsloth.models import get_model_and_tokenizer
        print("\nCustom get_model_and_tokenizer implementation:")
        print(inspect.getsource(get_model_and_tokenizer))
    except (ImportError, AttributeError) as e:
        print(f"\n✗ Could not inspect custom get_model_and_tokenizer: {e}")
    finally:
        if str(custom_unsloth_path) in sys.path:
            sys.path.remove(str(custom_unsloth_path))
    
    print("\nPython path:")
    for path in sys.path:
        print(f"  - {path}")
    
    print("=" * 50)

def main():
    """Main function"""
    check_unsloth_paths()
    
    print("\nRecommendation:")
    print("Based on the error 'got multiple values for keyword argument 'device_map'',")
    print("the Unsloth implementation is likely passing device_map internally.")
    print("Modify the FastLanguageModel.from_pretrained call to remove device_map parameter.")
    print("Also check if there's a custom implementation in /notebooks/custom_unsloth that needs to be modified.")

if __name__ == "__main__":
    main()
