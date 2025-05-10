#!/usr/bin/env python3
"""
Wrapper script for fix_deepseek_model function in consolidated_deepseek_fixes.py
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import from the root
parent_dir = str(Path(__file__).resolve().parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the function from consolidated_deepseek_fixes.py
try:
    from consolidated_deepseek_fixes import fix_deepseek_model
    
    # Call the function
    success = fix_deepseek_model()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
except ImportError:
    print("Error: Could not import fix_deepseek_model from consolidated_deepseek_fixes.py")
    print("Make sure consolidated_deepseek_fixes.py is in the same directory as this script.")
    sys.exit(1)
