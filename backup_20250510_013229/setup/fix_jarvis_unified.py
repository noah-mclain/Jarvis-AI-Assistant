#!/usr/bin/env python3

"""
Script to fix indentation error in jarvis_unified.py
"""

import re
import os
import shutil

def fix_file(file_path):
    """Fix the indentation error in jarvis_unified.py"""
    # Create backup
    backup_path = f"{file_path}.bak"
    if not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)
        print(f"Created backup: {backup_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Look for the try statement and the unsloth import
    try_match = re.search(r'# Try to import Unsloth for optimized training\s*\ntry:\s*\n', content)
    if not try_match:
        print("Try statement not found in the file")
        return False
    
    try_end = try_match.end()
    
    # Check if there's a proper indentation after try:
    next_line = content[try_end:try_end+100].lstrip()
    if not next_line.startswith('from unsloth import') and not next_line.startswith('    from unsloth import'):
        # Fix by adding proper indentation
        fixed_content = content[:try_end] + '    ' + content[try_end:]
        
        # Write the fixed content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print(f"Fixed indentation in {file_path}")
        return True
    else:
        print("File doesn't appear to have an indentation error")'
        return False

if __name__ == "__main__":
    jarvis_file = "/notebooks/src/generative_ai_module/jarvis_unified.py"
    
    # Check if file exists
    if not os.path.exists(jarvis_file):
        # Try relative path
        jarvis_file = "src/generative_ai_module/jarvis_unified.py"
        if not os.path.exists(jarvis_file):
            print(f"jarvis_unified.py not found")
            import sys
            if len(sys.argv) > 1:
                jarvis_file = sys.argv[1]
                print(f"Using provided path: {jarvis_file}")
            else:
                jarvis_file = input("Please enter the full path to jarvis_unified.py: ")
    
    if not os.path.exists(jarvis_file):
        print(f"ERROR: File not found at {jarvis_file}")
        exit(1)
    
    print(f"Fixing indentation in {jarvis_file}")
    fix_file(jarvis_file) 