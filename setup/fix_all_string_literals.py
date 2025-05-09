#!/usr/bin/env python3
"""
Comprehensive fix for all string literal issues in Python files.

This script scans all Python files in the setup directory and fixes:
1. Unterminated string literals
2. Incorrect triple-quoted docstrings
3. Other common string syntax errors

Run this script before training to ensure all syntax errors are fixed.
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

def fix_file(file_path):
    """
    Fix all string literal issues in a single file.

    Args:
        file_path: Path to the file to fix

    Returns:
        bool: True if any fixes were applied, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Fix 1: Replace """ with """ in docstrings
        content = re.sub(r'"""', '"""', content)

        # Fix 2: Fix unterminated triple-quoted docstrings
        # Find all triple-quoted strings that don't have a closing triple quote
        triple_quote_pattern = re.compile(r'"""(.*?)(?:"""|\Z)', re.DOTALL)
        for match in triple_quote_pattern.finditer(content):
            if '"""' not in match.group(1):
                # This is an unterminated triple-quoted string
                # Add the closing triple quotes
                start_pos = match.start()
                end_pos = match.end()
                if end_pos == len(content):
                    # If the match goes to the end of the file, add closing quotes
                    content = content[:end_pos] + '"""' + content[end_pos:]

        # Fix 3: Fix unterminated single-quoted strings
        lines = content.split('\n')
        fixed_lines = []
        in_multiline_string = False
        multiline_quote_char = None

        for line in lines:
            # Skip if we're in a multiline string
            if in_multiline_string:
                fixed_lines.append(line)
                # Check if this line ends the multiline string
                if multiline_quote_char in line:
                    in_multiline_string = False
                    multiline_quote_char = None
                continue

            # Count quotes in the line
            single_quotes = line.count("'")'
            double_quotes = line.count('"')"

            # Check for triple quotes which shouldn't be counted as unterminated
            triple_single = line.count("'''")
            triple_double = line.count('"""')

            # Adjust counts for triple quotes (each triple quote counts as 3 in the original count)
            single_quotes -= triple_single * 3
            double_quotes -= triple_double * 3

            # Check if we're starting a multiline string
            if triple_single % 2 != 0:
                in_multiline_string = True
                multiline_quote_char = "'''"
            elif triple_double % 2 != 0:
                in_multiline_string = True
                multiline_quote_char = '"""'

            # Fix unterminated single quotes
            if single_quotes % 2 != 0 and not line.strip().startswith('#'):
                # This line has an unterminated single quote
                fixed_line = line + "'"'
                logger.info(f"Fixed unterminated single quote in {file_path}, line: {line}")
                fixed_lines.append(fixed_line)
            # Fix unterminated double quotes
            elif double_quotes % 2 != 0 and not line.strip().startswith('#'):
                # This line has an unterminated double quote
                fixed_line = line + '"'"
                logger.info(f"Fixed unterminated double quote in {file_path}, line: {line}")
                fixed_lines.append(fixed_line)
            else:
                fixed_lines.append(line)

        # Reconstruct the content
        content = '\n'.join(fixed_lines)

        # Fix 4: Fix specific issues with f-strings
        # Replace problematic patterns like f'string{var}" with f'string{var}'
        content = re.sub(r"f'([^']*)'\"", r"f'\1', content)'
        content = re.sub(r"f\"([^\"]*)\"'", r"f\"\1\"", content)'

        # Write the fixed content back to the file if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Fixed string literal issues in {file_path}")
            return True
        else:
            logger.debug(f"No string literal issues found in {file_path}")
            return False
    except Exception as e:
        logger.error(f"Error fixing string literals in {file_path}: {e}")
        return False

def fix_all_files(directory=None):
    """
    Fix all string literal issues in all Python files in the given directory.

    Args:
        directory: Directory to scan for Python files (default: setup)

    Returns:
        tuple: (fixed_files, total_files)
    """
    if directory is None:
        # Default to the setup directory
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".")

    logger.info(f"Scanning directory: {directory}")

    # Find all Python files in the directory
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))

    logger.info(f"Found {len(python_files)} Python files")

    # Fix each file
    fixed_files = 0
    for file_path in python_files:
        if fix_file(file_path):
            fixed_files += 1

    return fixed_files, len(python_files)

def verify_syntax(file_path):
    """
    Verify that a Python file has valid syntax.

    Args:
        file_path: Path to the file to verify

    Returns:
        bool: True if the file has valid syntax, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Try to compile the file to check for syntax errors
        compile(content, file_path, 'exec')
        return True
    except SyntaxError as e:
        logger.error(f"Syntax error in {file_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error verifying syntax in {file_path}: {e}")
        return False

def verify_all_files(directory=None):
    """
    Verify that all Python files in the given directory have valid syntax.

    Args:
        directory: Directory to scan for Python files (default: setup)

    Returns:
        tuple: (valid_files, total_files)
    """
    if directory is None:
        # Default to the setup directory
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".")

    logger.info(f"Verifying syntax in directory: {directory}")

    # Find all Python files in the directory
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))

    logger.info(f"Found {len(python_files)} Python files")

    # Verify each file
    valid_files = 0
    for file_path in python_files:
        if verify_syntax(file_path):
            valid_files += 1

    return valid_files, len(python_files)

def main():
    """Main function to fix all string literal issues."""
    logger.info("Starting comprehensive string literal fix")

    # Fix all files in the setup directory
    fixed_files, total_files = fix_all_files()
    logger.info(f"Fixed string literal issues in {fixed_files} out of {total_files} files")

    # Verify syntax of all files
    valid_files, total_files = verify_all_files()
    logger.info(f"Verified syntax in {valid_files} out of {total_files} files")

    if valid_files == total_files:
        logger.info("✅ All files have valid syntax")
        return True
    else:
        logger.warning(f"⚠️ {total_files - valid_files} files still have syntax errors")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
