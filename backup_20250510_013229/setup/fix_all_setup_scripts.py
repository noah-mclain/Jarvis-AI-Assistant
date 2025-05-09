#!/usr/bin/env python3
"""
Comprehensive fix for all setup scripts.

This script fixes syntax errors in all Python files in the setup directory,
focusing on string literal issues and other common syntax errors.
"""

import os
import sys
import re
import logging
import traceback
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
    Fix syntax errors in a single file.
    
    Args:
        file_path: Path to the file to fix
        
    Returns:
        bool: True if the file was fixed successfully, False otherwise
    """
    logger.info(f"Fixing file: {file_path}")
    
    try:
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if the file has syntax errors
        try:
            compile(content, file_path, 'exec')
            logger.debug(f"File has no syntax errors: {file_path}")
            return True
        except SyntaxError as e:
            logger.info(f"Found syntax error in {file_path}: {e}")
        
        # Create a backup of the original file
        backup_path = file_path + ".bak"
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Fix 1: Replace """" with """
        content = content.replace('""""', '"""')"
        
        # Fix 2: Fix unterminated triple-quoted docstrings
        lines = content.split('\n')
        fixed_lines = []
        in_docstring = False
        docstring_quote = None
        
        for i, line in enumerate(lines):
            # Check for docstring start
            if not in_docstring:
                if '"""' in line and line.count('"""') % 2 != 0:
                    in_docstring = True
                    docstring_quote = '"""'
                elif "'''" in line and line.count("'''") % 2 != 0:
                    in_docstring = True
                    docstring_quote = "'''"
            
            # Check for docstring end
            elif in_docstring and docstring_quote in line:
                in_docstring = False
                docstring_quote = None
            
            fixed_lines.append(line)
        
        # If we're still in a docstring at the end of the file, add closing quotes
        if in_docstring and docstring_quote:
            logger.info(f"Found unterminated docstring in {file_path}")
            fixed_lines.append(docstring_quote)
            logger.info(f"Added closing quotes: {docstring_quote}")
        
        # Fix 3: Fix other string literal issues
        for i in range(len(fixed_lines)):
            line = fixed_lines[i]
            
            # Skip comments
            if line.strip().startswith('#'):
                continue
            
            # Count quotes in the line
            single_quotes = line.count("'")'
            double_quotes = line.count('"')"
            
            # Adjust for triple quotes
            triple_single = line.count("'''")
            triple_double = line.count('"""')
            
            # Adjust counts
            single_quotes -= triple_single * 3
            double_quotes -= triple_double * 3
            
            # Fix unterminated single quotes
            if single_quotes % 2 != 0:
                logger.info(f"Fixed unterminated single quote in {file_path}, line {i+1}: {line}")
                fixed_lines[i] = line + "'"'
            
            # Fix unterminated double quotes
            elif double_quotes % 2 != 0:
                logger.info(f"Fixed unterminated double quote in {file_path}, line {i+1}: {line}")
                fixed_lines[i] = line + '"'"
        
        # Fix 4: Fix indentation issues
        # This is a simple fix that just ensures consistent indentation
        for i in range(1, len(fixed_lines)):
            line = fixed_lines[i]
            prev_line = fixed_lines[i-1]
            
            # Skip empty lines
            if not line.strip():
                continue
            
            # Check for inconsistent indentation
            if line.startswith(' ') and prev_line.startswith(' '):
                curr_indent = len(line) - len(line.lstrip(' '))
                prev_indent = len(prev_line) - len(prev_line.lstrip(' '))
                
                # If indentation is off by 1-3 spaces, fix it
                if 0 < abs(curr_indent - prev_indent) < 4:
                    if curr_indent > prev_indent:
                        # Current line has more indentation
                        fixed_lines[i] = ' ' * prev_indent + line.lstrip(' ')
                        logger.info(f"Fixed indentation in {file_path}, line {i+1}")
                    else:
                        # Previous line has more indentation
                        # Only fix if current line should be at same level
                        if not prev_line.strip().endswith(':'):
                            fixed_lines[i] = ' ' * prev_indent + line.lstrip(' ')
                            logger.info(f"Fixed indentation in {file_path}, line {i+1}")
        
        # Write the fixed content back to the file
        fixed_content = '\n'.join(fixed_lines)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        # Verify the fix worked
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                new_content = f.read()
            compile(new_content, file_path, 'exec')
            logger.info(f"✅ Successfully fixed {file_path}")
            return True
        except SyntaxError as e:
            logger.error(f"❌ Failed to fix {file_path}: {e}")
            
            # Try a more aggressive fix for this specific file
            try:
                # Read the file line by line and fix each line
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                fixed_lines = []
                for line in lines:
                    # Remove any trailing whitespace
                    line = line.rstrip()
                    
                    # Ensure proper string termination
                    if "'" in line and line.count("'") % 2 != 0:
                        line += "'"'
                    if '"' in line and line.count('"') % 2 != 0:
                        line += '"'"
                    
                    fixed_lines.append(line)
                
                # Write the fixed content back to the file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(fixed_lines))
                
                # Verify the fix worked
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        new_content = f.read()
                    compile(new_content, file_path, 'exec')
                    logger.info(f"✅ Successfully fixed {file_path} with aggressive fix")
                    return True
                except SyntaxError as e2:
                    logger.error(f"❌ Failed to fix {file_path} with aggressive fix: {e2}")
                    # Restore from backup
                    with open(backup_path, 'r', encoding='utf-8') as f:
                        original_content = f.read()
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(original_content)
                    logger.info(f"Restored original file from backup: {file_path}")
                    return False
            except Exception as e2:
                logger.error(f"Error during aggressive fix of {file_path}: {e2}")
                # Restore from backup
                with open(backup_path, 'r', encoding='utf-8') as f:
                    original_content = f.read()
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                logger.info(f"Restored original file from backup: {file_path}")
                return False
    except Exception as e:
        logger.error(f"Error fixing {file_path}: {e}")
        logger.error(traceback.format_exc())
        return False

def fix_all_files(directory=None):
    """
    Fix all Python files in the given directory.
    
    Args:
        directory: Directory to scan for Python files (default: setup)
        
    Returns:
        tuple: (fixed_files, total_files)
    """
    if directory is None:
        # Default to the setup directory
        directory = os.path.dirname(os.path.abspath(__file__))
    
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

def main():
    """Main function."""
    logger.info("Starting comprehensive fix for all setup scripts")
    fixed_files, total_files = fix_all_files()
    logger.info(f"Fixed {fixed_files} out of {total_files} files")
    return fixed_files == total_files

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
