#!/usr/bin/env python3
"""
Wrapper script for fix_transformers_utils function in consolidated_utils.py
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import from the root
parent_dir = str(Path(__file__).resolve().parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the function from consolidated_utils.py
try:
    from consolidated_utils import fix_transformers_utils
    
    # Call the function
    success = fix_transformers_utils()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
except ImportError:
    print("Error: Could not import fix_transformers_utils from consolidated_utils.py")
    print("Make sure consolidated_utils.py is in the same directory as this script.")
    sys.exit(1)
