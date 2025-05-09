#!/usr/bin/env python3
"""
Fix for unterminated string literals in ultimate_attention_fix.py.

This script specifically fixes the syntax errors in ultimate_attention_fix.py
to ensure it can be imported without errors.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def fix_ultimate_attention_fix():
    """Fix the ultimate_attention_fix.py file."""
    # Get the path to the ultimate_attention_fix.py file
    setup_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(setup_dir, "ultimate_attention_fix.py")
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False
    
    logger.info(f"Fixing file: {file_path}")
    
    try:
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if the file has syntax errors
        try:
            compile(content, file_path, 'exec')
            logger.info("File has no syntax errors, no fixes needed")
            return True
        except SyntaxError as e:
            logger.info(f"Found syntax error: {e}")
        
        # Create a backup of the original file
        backup_path = file_path + ".bak"
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Created backup at: {backup_path}")
        
        # Fix common issues with string literals
        
        # 1. Replace """" with """
        content = content.replace('""""', '"""')"
        
        # 2. Fix unterminated triple-quoted docstrings
        lines = content.split('\n')
        fixed_lines = []
        in_docstring = False
        docstring_start_line = -1
        
        for i, line in enumerate(lines):
            # Check for docstring start
            if not in_docstring and ('"""' in line) and line.count('"""') % 2 != 0:
                in_docstring = True
                docstring_start_line = i
            
            # Check for docstring end
            elif in_docstring and '"""' in line:
                in_docstring = False
                docstring_start_line = -1
            
            fixed_lines.append(line)
        
        # If we're still in a docstring at the end of the file, add closing quotes
        if in_docstring and docstring_start_line != -1:
            logger.info(f"Found unterminated docstring starting at line {docstring_start_line+1}")
            fixed_lines.append('"""')
            logger.info("Added closing triple quotes")
        
        # 3. Fix other string literal issues
        for i, line in enumerate(fixed_lines):
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
            if single_quotes % 2 != 0 and not line.strip().startswith('#'):
                logger.info(f"Fixed unterminated single quote at line {i+1}: {line}")
                fixed_lines[i] = line + "'"'
            
            # Fix unterminated double quotes
            elif double_quotes % 2 != 0 and not line.strip().startswith('#'):
                logger.info(f"Fixed unterminated double quote at line {i+1}: {line}")
                fixed_lines[i] = line + '"'"
        
        # Write the fixed content back to the file
        fixed_content = '\n'.join(fixed_lines)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        # Verify the fix worked
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                new_content = f.read()
            compile(new_content, file_path, 'exec')
            logger.info("✅ Successfully fixed syntax errors")
            return True
        except SyntaxError as e:
            logger.error(f"❌ Failed to fix syntax errors: {e}")
            # Restore from backup
            with open(backup_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(original_content)
            logger.info("Restored original file from backup")
            return False
    except Exception as e:
        logger.error(f"Error fixing file: {e}")
        return False

def main():
    """Main function."""
    logger.info("Starting fix for ultimate_attention_fix.py")
    success = fix_ultimate_attention_fix()
    if success:
        logger.info("✅ Successfully fixed ultimate_attention_fix.py")
    else:
        logger.error("❌ Failed to fix ultimate_attention_fix.py")
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
