#!/usr/bin/env python3
"""
Fix for consolidated_utils.py

This script fixes the syntax error in consolidated_utils.py.
"""

import os
import sys
import re

def fix_consolidated_utils():
    """
    Fix the syntax error in consolidated_utils.py.
    """
    file_path = os.path.join(os.path.dirname(__file__), "consolidated_utils.py")
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} does not exist.")
        return False
    
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for unterminated string literals
        fixed_content = content
        string_regex = r'(["\'])((?:\\.|[^\\])*?)(?:\1|$)'
        
        for match in re.finditer(string_regex, content):
            full_match = match.group(0)
            quote = match.group(1)
            
            # Check if the string is unterminated
            if not full_match.endswith(quote):
                # Fix the unterminated string by adding the closing quote
                fixed_content = fixed_content.replace(full_match, full_match + quote)
                print(f"Fixed unterminated string: {full_match[:20]}...")
        
        # Fix specific line 625 issue
        line_625_regex = r"optimize_memory_usage\(\)'$"
        if re.search(line_625_regex, fixed_content, re.MULTILINE):
            fixed_content = re.sub(line_625_regex, "optimize_memory_usage()", fixed_content)
            print("Fixed line 625 issue.")
        
        # Write the fixed content back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print(f"Successfully fixed {file_path}")
        return True
    
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

if __name__ == "__main__":
    success = fix_consolidated_utils()
    sys.exit(0 if success else 1)
