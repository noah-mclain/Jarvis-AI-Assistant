#!/usr/bin/env python3
"""
Fix syntax errors in Python files, particularly unterminated string literals.
This script fixes common syntax errors in Python files in the setup directory.
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
        if '"""' in content:""
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

def fix_unterminated_strings(file_path):
    """Fix unterminated string literals in a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        fixed = False
        for i, line in enumerate(lines):
            # Skip comments
            if line.strip().startswith('#'):
                continue

            # Check for unterminated strings with single quotes
            if line.count("'") % 2 != 0:'
                # Don't fix lines that end with a backslash (line continuation)
                if not line.rstrip().endswith('\\'):
                    lines[i] = line.rstrip() + "'\n"'
                    print(f"Fixed unterminated single quote in {file_path}, line {i+1}")
                    fixed = True

            # Check for unterminated strings with double quotes
            if line.count('"') % 2 != 0:"
                # Don't fix lines that end with a backslash (line continuation)
                if not line.rstrip().endswith('\\'):
                    lines[i] = line.rstrip() + '"\n'"
                    print(f"Fixed unterminated double quote in {file_path}, line {i+1}")
                    fixed = True

        if fixed:
            # Write the fixed content back to the file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            return True
        return False
    except Exception as e:
        print(f"Error fixing unterminated strings in {file_path}: {e}")
        return False

def main():
    """Main function to fix syntax errors in Python files."""
    # Find all Python files in the setup directory
    setup_dir = Path("setup")
    if not setup_dir.exists():
        print(f"Directory not found: {setup_dir}")
        return False

    python_files = list(setup_dir.glob("**/*.py"))
    print(f"Found {len(python_files)} Python files in {setup_dir}")

    fixed_files = []

    for file_path in python_files:
        fixed_docstrings = fix_docstring_quotes(file_path)
        fixed_strings = fix_unterminated_strings(file_path)

        if fixed_docstrings or fixed_strings:
            fixed_files.append(file_path)

    if fixed_files:
        print(f"Fixed syntax errors in {len(fixed_files)} files:")
        for file_path in fixed_files:
            print(f"  - {file_path}")
        return True
    else:
        print("No syntax errors found in Python files.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
"""