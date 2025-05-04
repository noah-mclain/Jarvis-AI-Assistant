#!/usr/bin/env python3

"""
Script to fix import errors in generative_ai_module/__init__.py
"""

import os
import sys
import shutil

def fix_init_imports():
    """
    Fix the import error in __init__.py by updating the import statement
    from 'from .jarvis_unified import UnifiedModel' or 'from .jarvis_unified import JarvisUnified as UnifiedModel'
    to 'from .jarvis_unified import JarvisAI as UnifiedModel'
    """
    # Find the __init__.py file
    base_paths = [
        "/notebooks/src/generative_ai_module/__init__.py",  # Paperspace path
        "src/generative_ai_module/__init__.py",             # Relative path
    ]
    
    init_file = None
    for path in base_paths:
        if os.path.exists(path):
            init_file = path
            break
    
    if not init_file:
        print("ERROR: __init__.py not found in expected locations.")
        if len(sys.argv) > 1:
            init_file = sys.argv[1]
            print(f"Using provided path: {init_file}")
        else:
            init_file = input("Please enter the full path to generative_ai_module/__init__.py: ")
            if not os.path.exists(init_file):
                print(f"ERROR: File not found at {init_file}")
                return False
    
    # Create backup
    backup_file = f"{init_file}.bak"
    if not os.path.exists(backup_file):
        shutil.copy2(init_file, backup_file)
        print(f"Created backup: {backup_file}")
    
    # Read the file
    with open(init_file, 'r') as f:
        content = f.read()
    
    # Fix the import statement
    if "from .jarvis_unified import UnifiedModel" in content:
        fixed_content = content.replace(
            "from .jarvis_unified import UnifiedModel",
            "from .jarvis_unified import JarvisAI as UnifiedModel"
        )
        fixed = True
    elif "from .jarvis_unified import JarvisUnified as UnifiedModel" in content:
        fixed_content = content.replace(
            "from .jarvis_unified import JarvisUnified as UnifiedModel",
            "from .jarvis_unified import JarvisAI as UnifiedModel"
        )
        fixed = True
    else:
        print("Import statement not found or already fixed.")
        return False
    
    # Write the fixed content
    with open(init_file, 'w') as f:
        f.write(fixed_content)
    
    print(f"✅ Fixed import in {init_file}")
    print(f"💡 Changed import to use 'from .jarvis_unified import JarvisAI as UnifiedModel'")
    return True

if __name__ == "__main__":
    print("🔄 Fixing import errors in generative_ai_module...")
    if fix_init_imports():
        print("✅ Fixed successfully! You should no longer see the import error.")
        print("   You can now run your commands without issues.")
    else:
        print("❌ No changes were made. Please check the files manually.") 