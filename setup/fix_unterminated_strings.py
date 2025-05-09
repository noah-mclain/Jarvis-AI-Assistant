#!/usr/bin/env python3
"""
Fix unterminated string literals in Python files.

This script scans Python files for unterminated string literals and fixes them.
It specifically targets common issues like:
1. Missing closing quotes
2. Unescaped quotes within strings
3. Trailing single quotes at the end of strings

Usage:
    python fix_unterminated_strings.py [file_or_directory]
"""

import os
import sys
import re
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def find_unterminated_strings(content):
    """
    Find unterminated string literals in the content.

    Args:
        content: The content to scan for unterminated strings

    Returns:
        A list of tuples (line_number, line_content, fixed_line)
    """
    issues = []
    lines = content.split('\n')

    for i, line in enumerate(lines):
        line_number = i + 1

        # Skip comment lines
        if line.strip().startswith('#'):
            continue

        # Check for unbalanced quotes
        single_quotes = line.count("'")'
        double_quotes = line.count('"')"

        # Check for triple quotes
        triple_single = line.count("'''")
        triple_double = line.count('"""')

        # Adjust counts for triple quotes
        single_quotes -= triple_single * 3
        double_quotes -= triple_double * 3

        # Check if we have an odd number of quotes (indicating unterminated string)
        if single_quotes % 2 == 1:
            # Try to fix by adding a closing quote
            fixed_line = line.rstrip() + "'"'
            issues.append((line_number, line, fixed_line))
        elif double_quotes % 2 == 1:
            # Try to fix by adding a closing quote
            fixed_line = line.rstrip() + '"'"
            issues.append((line_number, line, fixed_line))

    return issues

def fix_file(file_path):
    """
    Fix unterminated strings in a file.

    Args:
        file_path: Path to the file to fix

    Returns:
        True if the file was fixed, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find unterminated strings
        issues = find_unterminated_strings(content)

        if not issues:
            logger.info(f"No unterminated strings found in {file_path}")
            return False

        # Fix the issues
        lines = content.split('\n')
        for line_number, line, fixed_line in issues:
            logger.info(f"Fixing line {line_number} in {file_path}")
            logger.info(f"  Original: {line}")
            logger.info(f"  Fixed:    {fixed_line}")
            lines[line_number - 1] = fixed_line

        # Write the fixed content back to the file
        fixed_content = '\n'.join(lines)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)

        logger.info(f"Fixed {len(issues)} issues in {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error fixing {file_path}: {e}")
        return False

def scan_directory(directory):
    """
    Scan a directory for Python files and fix unterminated strings.

    Args:
        directory: Directory to scan

    Returns:
        Number of files fixed
    """
    fixed_count = 0
    directory_path = Path(directory)

    # Find all Python files in the directory
    python_files = list(directory_path.glob("**/*.py"))
    logger.info(f"Found {len(python_files)} Python files in {directory}")

    for file_path in python_files:
        if fix_file(file_path):
            fixed_count += 1

    return fixed_count

def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        # Default to the setup directory
        path = 'setup'

    if os.path.isfile(path):
        # Fix a single file
        if fix_file(path):
            logger.info(f"Fixed unterminated strings in {path}")
        else:
            logger.info(f"No unterminated strings found in {path}")
    elif os.path.isdir(path):
        # Scan a directory
        fixed_count = scan_directory(path)
        logger.info(f"Fixed unterminated strings in {fixed_count} files")
    else:
        logger.error(f"Path not found: {path}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
