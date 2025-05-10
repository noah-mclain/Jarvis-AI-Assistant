#!/usr/bin/env python3
"""
Fix all setup scripts

This script fixes syntax errors in all setup scripts.
"""

import os
import sys
import re
import glob

def fix_unterminated_strings(file_path):
    """
    Fix unterminated string literals in a Python file.
    
    Args:
        file_path (str): Path to the Python file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find unterminated string literals
        fixed_content = content
        string_regex = r'(["\'])((?:\\.|[^\\])*?)(?:\1|$)'
        
        fixed = False
        for match in re.finditer(string_regex, content):
            full_match = match.group(0)
            quote = match.group(1)
            
            # Check if the string is unterminated
            if not full_match.endswith(quote):
                # Fix the unterminated string by adding the closing quote
                fixed_content = fixed_content.replace(full_match, full_match + quote)
                print(f"Fixed unterminated string in {file_path}: {full_match[:20]}...")
                fixed = True
        
        # Fix specific issues
        if "optimize_memory_usage()'" in fixed_content:
            fixed_content = fixed_content.replace("optimize_memory_usage()'", "optimize_memory_usage()")
            print(f"Fixed optimize_memory_usage issue in {file_path}")
            fixed = True
        
        # Write the fixed content back to the file
        if fixed:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            
            print(f"Fixed unterminated strings in {file_path}")
            return True
        else:
            print(f"No unterminated strings found in {file_path}")
            return True
    
    except Exception as e:
        print(f"Error fixing unterminated strings in {file_path}: {e}")
        return False

def fix_syntax_errors(file_path):
    """
    Fix common syntax errors in a Python file.
    
    Args:
        file_path (str): Path to the Python file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix missing colons after if/for/while/def/class statements
        fixed_content = re.sub(r'(if\s+.*?)\s*\n', r'\1:\n', content)
        fixed_content = re.sub(r'(for\s+.*?)\s*\n', r'\1:\n', fixed_content)
        fixed_content = re.sub(r'(while\s+.*?)\s*\n', r'\1:\n', fixed_content)
        fixed_content = re.sub(r'(def\s+.*?\))\s*\n', r'\1:\n', fixed_content)
        fixed_content = re.sub(r'(class\s+.*?(?:\(.*?\))?)\s*\n', r'\1:\n', fixed_content)
        
        # Fix indentation (convert tabs to spaces)
        lines = fixed_content.split('\n')
        fixed_lines = []
        for line in lines:
            if line.startswith('\t'):
                fixed_line = line.replace('\t', '    ')
                fixed_lines.append(fixed_line)
            else:
                fixed_lines.append(line)
        
        fixed_content = '\n'.join(fixed_lines)
        
        # Write the fixed content back to the file
        if fixed_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            
            print(f"Fixed syntax errors in {file_path}")
            return True
        else:
            print(f"No syntax errors found in {file_path}")
            return True
    
    except Exception as e:
        print(f"Error fixing syntax errors in {file_path}: {e}")
        return False

def fix_all_setup_scripts():
    """
    Fix all setup scripts.
    
    Returns:
        int: Number of files fixed
    """
    setup_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Fixing setup scripts in {setup_dir}...")
    
    fixed_count = 0
    
    # Get all Python files in the setup directory
    python_files = glob.glob(os.path.join(setup_dir, "*.py"))
    
    for file_path in python_files:
        print(f"Checking {file_path}...")
        if fix_unterminated_strings(file_path):
            fixed_count += 1
        if fix_syntax_errors(file_path):
            fixed_count += 1
    
    print(f"Fixed {fixed_count} files.")
    return fixed_count

if __name__ == "__main__":
    fix_all_setup_scripts()
    print("All setup scripts have been fixed.")
    sys.exit(0)
