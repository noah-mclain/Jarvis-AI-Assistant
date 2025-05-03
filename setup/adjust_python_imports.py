#!/usr/bin/env python3

"""
Script to adjust Python files to use the minimal unsloth implementation.
This script will scan Python files in a given directory and modify import statements
for unsloth to use the custom minimal implementation.
"""

import os
import sys
import re
import argparse
from pathlib import Path
import shutil

def create_backup(file_path):
    """Create a backup of the file with .bak extension"""
    backup_path = f"{file_path}.bak"
    if not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)
        print(f"Created backup: {backup_path}")
    return backup_path

def adjust_file(file_path, custom_unsloth_path):
    """Adjust import statements in a Python file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if file imports unsloth
    if not re.search(r'import\s+unsloth|from\s+unsloth\s+import', content):
        print(f"No unsloth imports found in {file_path}")
        return False

    # Create backup
    create_backup(file_path)
    
    # Add sys.path modification at the top of the file
    unsloth_path_insertion = f"""import sys
import os
# Use custom minimal unsloth implementation
if "{custom_unsloth_path}" not in sys.path:
    sys.path.insert(0, "{custom_unsloth_path}")
"""
    
    # Check if there are existing imports
    if re.search(r'import\s+sys', content):
        # If sys is already imported, we need to add just the path insertion
        path_insertion = f"""
# Use custom minimal unsloth implementation
if "{custom_unsloth_path}" not in sys.path:
    sys.path.insert(0, "{custom_unsloth_path}")
"""
        # Insert after existing sys import
        content = re.sub(r'(import\s+sys.*?\n)', r'\1' + path_insertion, content, count=1, flags=re.DOTALL)
    else:
        # Add at top of file (after docstring if exists)
        if content.startswith('"""') or content.startswith("'''"):
            # Find end of docstring
            match = re.search(r'^(""".*?"""|\'\'\'.*?\'\'\')', content, re.DOTALL)
            if match:
                docstring_end = match.end()
                content = content[:docstring_end] + "\n" + unsloth_path_insertion + content[docstring_end:]
            else:
                content = unsloth_path_insertion + content
        else:
            content = unsloth_path_insertion + content
    
    # Add comment about using minimal unsloth
    content = re.sub(
        r'(from\s+unsloth\s+import|import\s+unsloth)',
        r'# Using minimal unsloth implementation\n\1',
        content
    )
    
    # Write the modified content back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Modified {file_path} to use minimal unsloth")
    return True

def scan_directory(directory, custom_unsloth_path):
    """Scan directory for Python files and adjust them"""
    modified_files = 0
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if adjust_file(file_path, custom_unsloth_path):
                    modified_files += 1
    
    return modified_files

def main():
    parser = argparse.ArgumentParser(description='Adjust Python files to use minimal unsloth')
    parser.add_argument('directory', help='Directory to scan for Python files')
    parser.add_argument('--custom-path', default='/notebooks/custom_unsloth',
                        help='Path to the custom unsloth implementation')
    
    args = parser.parse_args()
    
    print(f"Scanning directory: {args.directory}")
    print(f"Using custom unsloth at: {args.custom_path}")
    
    modified_files = scan_directory(args.directory, args.custom_path)
    
    print(f"\nModified {modified_files} Python files to use minimal unsloth implementation")
    print("Backups were created with .bak extension")

if __name__ == '__main__':
    main() 