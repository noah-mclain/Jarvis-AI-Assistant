#!/usr/bin/env python3
"""
Fix unterminated string literals in Python files.
This script scans Python files for unterminated string literals and fixes them.
"""

import os
import sys
import re
import tokenize
from io import BytesIO
from pathlib import Path

def find_unterminated_strings(file_path):
    """Find unterminated string literals in a Python file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check for unterminated strings using regex
    lines = content.split('\n')
    issues = []

    for i, line in enumerate(lines):
        # Check for single quotes
        single_quotes = line.count("'")
        if single_quotes % 2 != 0 and not line.strip().startswith('#'):
            issues.append((i+1, line))

        # Check for double quotes
        double_quotes = line.count('"')
        if double_quotes % 2 != 0 and not line.strip().startswith('#'):
            issues.append((i+1, line))

    return issues

def fix_unterminated_strings(file_path):
    """Fix unterminated string literals in a Python file."""
    issues = find_unterminated_strings(file_path)

    if not issues:
        return False

    print(f"Found {len(issues)} potential unterminated strings in {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    fixed = False

    for line_num, line_content in issues:
        print(f"Line {line_num}: {line_content}")

        # Try to fix the line
        fixed_line = line_content

        # Check for print statements with unterminated strings
        if "print('" in line_content and not line_content.endswith("')"):
            fixed_line = line_content + "')"
            print(f"Fixed to: {fixed_line}")
            lines[line_num-1] = fixed_line + '\n'
            fixed = True

        # Check for other unterminated strings
        elif "'" in line_content:
            # Count single quotes
            if line_content.count("'") % 2 != 0:
                fixed_line = line_content + "'"
                print(f"Fixed to: {fixed_line}")
                lines[line_num-1] = fixed_line + '\n'
                fixed = True

        # Check for double quotes
        elif '"' in line_content:
            # Count double quotes
            if line_content.count('"') % 2 != 0:
                fixed_line = line_content + '"'
                print(f"Fixed to: {fixed_line}")
                lines[line_num-1] = fixed_line + '\n'
                fixed = True

    if fixed:
        # Write the fixed content back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        print(f"Fixed unterminated strings in {file_path}")
        return True

    return False

def main():
    """Main function to fix unterminated string literals in Python files."""
    # Find all Python files in the setup directory
    setup_dir = Path("setup")
    if not setup_dir.exists():
        print(f"Directory not found: {setup_dir}")
        return False

    python_files = list(setup_dir.glob("**/*.py"))
    print(f"Found {len(python_files)} Python files in {setup_dir}")

    fixed_files = []

    for file_path in python_files:
        if fix_unterminated_strings(file_path):
            fixed_files.append(file_path)

    if fixed_files:
        print(f"Fixed unterminated strings in {len(fixed_files)} files:")
        for file_path in fixed_files:
            print(f"  - {file_path}")
        return True
    else:
        print("No unterminated strings found in Python files.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
