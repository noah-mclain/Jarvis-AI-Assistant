#!/usr/bin/env python3
"""
Fix docstring quotes in Python files.
This script fixes incorrect docstring quotes in Python files in the setup directory.
"""

import os
import sys
from pathlib import Path

def fix_docstring_quotes(file_path):
    """Fix docstring quotes in a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for incorrect docstring quotes (""" instead of """)
        if '"""' in content:
            content = content.replace('"""', '"""')
            print(f"Fixed incorrect docstring quotes in {file_path}")

            # Write the fixed content back to the file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error fixing docstring quotes in {file_path}: {e}")
        return False

def main():
    """Main function to fix docstring quotes in Python files."""
    # Find all Python files in the setup directory
    setup_dir = Path("setup")
    if not setup_dir.exists():
        print(f"Directory not found: {setup_dir}")
        return False

    python_files = list(setup_dir.glob("**/*.py"))
    print(f"Found {len(python_files)} Python files in {setup_dir}")

    fixed_files = []

    for file_path in python_files:
        if fix_docstring_quotes(file_path):
            fixed_files.append(file_path)

    if fixed_files:
        print(f"Fixed docstring quotes in {len(fixed_files)} files:")
        for file_path in fixed_files:
            print(f"  - {file_path}")
        return True
    else:
        print("No incorrect docstring quotes found in Python files.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
"""
"""