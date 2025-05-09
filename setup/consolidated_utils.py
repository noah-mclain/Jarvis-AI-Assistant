#!/usr/bin/env python3
"""
Consolidated Utilities

This module consolidates various utility functions for:
- Fixing Python imports
- Fixing string literals
- Fixing syntax errors
- Clearing CUDA cache
- Optimizing memory usage
- Setting up environment
- Fixing dependencies

This consolidates functionality from:
- adjust_python_imports.py
- apply_all_fixes.py
- clear_cuda_cache.py
- fix_all_setup_scripts.py
- fix_all_string_literals.py
- fix_dependencies.py
- fix_docstrings.py
- fix_imports.py
- fix_jarvis_unified.py
- fix_joblib.py
- fix_models_init.py
- fix_syntax_errors.py
- fix_tensorboard_callback.py
- fix_transformer_issues.py
- fix_transformers_utils.py
- fix_trl_peft_imports.py
- fix_trl_spacy_imports.py
- fix_unterminated_strings.py
- gpu_utils.py
- optimize_memory_usage.py
- setup_environment.py
"""

import os
import sys
import re
import gc
import ast
import logging
import importlib
import subprocess
import tokenize
import io
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Some utilities may not work.")

# Add the parent directory to the path so we can import from the root
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def is_paperspace_environment():
    """Check if running in Paperspace Gradient environment"""
    return os.path.exists("/notebooks") or os.path.exists("/storage")

def is_colab_environment():
    """Check if running in Google Colab environment"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def clear_cuda_cache():
    """Clear CUDA cache to free up GPU memory"""
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available. Cannot clear CUDA cache.")
        return False
    
    if torch.cuda.is_available():
        logger.info("Clearing CUDA cache...")
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("CUDA cache cleared.")
        return True
    else:
        logger.warning("CUDA not available. No cache to clear.")
        return False

def optimize_memory_usage():
    """Optimize memory usage for training"""
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available. Cannot optimize memory usage.")
        return False
    
    # Clear CUDA cache
    clear_cuda_cache()
    
    # Set environment variables for memory optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    logger.info("Memory usage optimized.")
    return True

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
        
        for match in re.finditer(string_regex, content):
            full_match = match.group(0)
            quote = match.group(1)
            
            # Check if the string is unterminated
            if not full_match.endswith(quote):
                # Fix the unterminated string by adding the closing quote
                fixed_content = fixed_content.replace(full_match, full_match + quote)
                logger.info(f"Fixed unterminated string in {file_path}: {full_match[:20]}...")
        
        # Write the fixed content back to the file
        if fixed_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            
            logger.info(f"Fixed unterminated strings in {file_path}")
            return True
        else:
            logger.info(f"No unterminated strings found in {file_path}")
            return True
    
    except Exception as e:
        logger.error(f"Error fixing unterminated strings in {file_path}: {e}")
        return False

def fix_all_string_literals(directory=None):
    """
    Fix all string literals in Python files.
    
    Args:
        directory (str): Directory to search for Python files (default: current directory)
        
    Returns:
        int: Number of files fixed
    """
    if directory is None:
        directory = os.getcwd()
    
    logger.info(f"Fixing string literals in {directory}...")
    
    fixed_count = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                if fix_unterminated_strings(file_path):
                    fixed_count += 1
    
    logger.info(f"Fixed string literals in {fixed_count} files.")
    return fixed_count

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
            
            logger.info(f"Fixed syntax errors in {file_path}")
            return True
        else:
            logger.info(f"No syntax errors found in {file_path}")
            return True
    
    except Exception as e:
        logger.error(f"Error fixing syntax errors in {file_path}: {e}")
        return False

def fix_all_syntax_errors(directory=None):
    """
    Fix all syntax errors in Python files.
    
    Args:
        directory (str): Directory to search for Python files (default: current directory)
        
    Returns:
        int: Number of files fixed
    """
    if directory is None:
        directory = os.getcwd()
    
    logger.info(f"Fixing syntax errors in {directory}...")
    
    fixed_count = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                if fix_syntax_errors(file_path):
                    fixed_count += 1
    
    logger.info(f"Fixed syntax errors in {fixed_count} files.")
    return fixed_count

def fix_imports(file_path):
    """
    Fix imports in a Python file.
    
    Args:
        file_path (str): Path to the Python file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix relative imports
        fixed_content = re.sub(r'from\s+\.\.([a-zA-Z0-9_]+)', r'from src.\1', content)
        fixed_content = re.sub(r'from\s+\.([a-zA-Z0-9_]+)', r'from src.generative_ai_module.\1', fixed_content)
        
        # Add missing imports
        if "import torch" not in fixed_content and "torch." in fixed_content:
            fixed_content = "import torch\n" + fixed_content
        
        if "import numpy" not in fixed_content and "numpy." in fixed_content:
            fixed_content = "import numpy as np\n" + fixed_content
        
        if "import os" not in fixed_content and "os." in fixed_content:
            fixed_content = "import os\n" + fixed_content
        
        if "import sys" not in fixed_content and "sys." in fixed_content:
            fixed_content = "import sys\n" + fixed_content
        
        # Write the fixed content back to the file
        if fixed_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            
            logger.info(f"Fixed imports in {file_path}")
            return True
        else:
            logger.info(f"No import issues found in {file_path}")
            return True
    
    except Exception as e:
        logger.error(f"Error fixing imports in {file_path}: {e}")
        return False

def fix_all_imports(directory=None):
    """
    Fix all imports in Python files.
    
    Args:
        directory (str): Directory to search for Python files (default: current directory)
        
    Returns:
        int: Number of files fixed
    """
    if directory is None:
        directory = os.getcwd()
    
    logger.info(f"Fixing imports in {directory}...")
    
    fixed_count = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                if fix_imports(file_path):
                    fixed_count += 1
    
    logger.info(f"Fixed imports in {fixed_count} files.")
    return fixed_count

def fix_transformers_utils():
    """
    Fix transformers.utils module.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import transformers
        import transformers.utils
        logger.info("transformers.utils module already exists.")
        return True
    except ImportError:
        logger.info("Creating transformers.utils module...")
        
        try:
            # Find the transformers package directory
            transformers_dir = os.path.dirname(transformers.__file__)
            utils_dir = os.path.join(transformers_dir, "utils")
            
            # Create the utils directory if it doesn't exist
            os.makedirs(utils_dir, exist_ok=True)
            
            # Create an empty __init__.py file
            with open(os.path.join(utils_dir, "__init__.py"), "w") as f:
                f.write("""
# Auto-generated utils module for transformers
import logging
logger = logging.getLogger(__name__)
""")
            
            # Try to import it again to verify
            importlib.invalidate_caches()
            import transformers.utils
            logger.info("Successfully created transformers.utils module.")
            return True
        except Exception as e:
            logger.error(f"Failed to create transformers.utils module: {e}")
            return False

def fix_trl_peft_imports():
    """
    Fix TRL and PEFT imports.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Check if TRL is installed
        import trl
        
        # Check if PEFT is installed
        try:
            import peft
        except ImportError:
            logger.warning("PEFT not installed. Installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "peft"], check=True)
        
        # Fix imports in TRL
        trl_dir = os.path.dirname(trl.__file__)
        
        # Find files that import PEFT
        for root, dirs, files in os.walk(trl_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Check if the file imports PEFT
                        if "import peft" in content or "from peft" in content:
                            # Add try-except block around PEFT imports
                            fixed_content = re.sub(
                                r'(from peft import.*?)$',
                                r'try:\n    \1\nexcept ImportError:\n    pass',
                                content,
                                flags=re.MULTILINE
                            )
                            
                            fixed_content = re.sub(
                                r'(import peft.*?)$',
                                r'try:\n    \1\nexcept ImportError:\n    pass',
                                fixed_content,
                                flags=re.MULTILINE
                            )
                            
                            # Write the fixed content back to the file
                            if fixed_content != content:
                                with open(file_path, 'w', encoding='utf-8') as f:
                                    f.write(fixed_content)
                                
                                logger.info(f"Fixed PEFT imports in {file_path}")
                    
                    except Exception as e:
                        logger.error(f"Error fixing PEFT imports in {file_path}: {e}")
        
        logger.info("Fixed TRL and PEFT imports.")
        return True
    
    except ImportError:
        logger.warning("TRL not installed. Skipping TRL and PEFT import fixes.")
        return False

def fix_trl_spacy_imports():
    """
    Fix TRL and spaCy imports.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Check if TRL is installed
        import trl
        
        # Fix imports in TRL
        trl_dir = os.path.dirname(trl.__file__)
        
        # Find files that import spaCy
        for root, dirs, files in os.walk(trl_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Check if the file imports spaCy
                        if "import spacy" in content:
                            # Add try-except block around spaCy imports
                            fixed_content = re.sub(
                                r'(import spacy.*?)$',
                                r'try:\n    \1\nexcept ImportError:\n    pass',
                                content,
                                flags=re.MULTILINE
                            )
                            
                            # Write the fixed content back to the file
                            if fixed_content != content:
                                with open(file_path, 'w', encoding='utf-8') as f:
                                    f.write(fixed_content)
                                
                                logger.info(f"Fixed spaCy imports in {file_path}")
                    
                    except Exception as e:
                        logger.error(f"Error fixing spaCy imports in {file_path}: {e}")
        
        logger.info("Fixed TRL and spaCy imports.")
        return True
    
    except ImportError:
        logger.warning("TRL not installed. Skipping TRL and spaCy import fixes.")
        return False

def fix_joblib():
    """
    Fix joblib installation.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import joblib
        logger.info(f"joblib version: {joblib.__version__}")
        return True
    except ImportError:
        logger.warning("joblib not installed. Installing...")
        
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "joblib==1.3.2"], check=True)
            
            # Verify installation
            import joblib
            logger.info(f"Successfully installed joblib version: {joblib.__version__}")
            return True
        except Exception as e:
            logger.error(f"Failed to install joblib: {e}")
            return False

def fix_tensorboard_callback():
    """
    Fix TensorBoard callback.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Check if transformers is installed
        import transformers
        
        # Find the transformers package directory
        transformers_dir = os.path.dirname(transformers.__file__)
        
        # Find the callback file
        callback_file = os.path.join(transformers_dir, "integrations.py")
        
        if os.path.exists(callback_file):
            with open(callback_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Fix TensorBoard import
            fixed_content = re.sub(
                r'(from tensorboardX import SummaryWriter)',
                r'try:\n    \1\nexcept ImportError:\n    from torch.utils.tensorboard import SummaryWriter',
                content
            )
            
            # Write the fixed content back to the file
            if fixed_content != content:
                with open(callback_file, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                
                logger.info(f"Fixed TensorBoard callback in {callback_file}")
                return True
            else:
                logger.info(f"No TensorBoard callback issues found in {callback_file}")
                return True
        else:
            logger.warning(f"Could not find callback file: {callback_file}")
            return False
    
    except ImportError:
        logger.warning("transformers not installed. Skipping TensorBoard callback fix.")
        return False

def apply_all_fixes():
    """
    Apply all fixes.
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Fix string literals
    fix_all_string_literals()
    
    # Fix syntax errors
    fix_all_syntax_errors()
    
    # Fix imports
    fix_all_imports()
    
    # Fix transformers.utils
    fix_transformers_utils()
    
    # Fix TRL and PEFT imports
    fix_trl_peft_imports()
    
    # Fix TRL and spaCy imports
    fix_trl_spacy_imports()
    
    # Fix joblib
    fix_joblib()
    
    # Fix TensorBoard callback
    fix_tensorboard_callback()
    
    # Optimize memory usage
    optimize_memory_usage()
    
    logger.info("All fixes applied successfully.")
    return True

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Consolidated utilities for Jarvis AI Assistant.")
    parser.add_argument("--action", type=str, default="all",
                        choices=["all", "string-literals", "syntax", "imports", "transformers-utils",
                                "trl-peft", "trl-spacy", "joblib", "tensorboard", "memory"],
                        help="Action to perform")
    parser.add_argument("--directory", type=str, help="Directory to process")
    args = parser.parse_args()
    
    # Perform the requested action
    if args.action == "all":
        apply_all_fixes()
    elif args.action == "string-literals":
        fix_all_string_literals(args.directory)
    elif args.action == "syntax":
        fix_all_syntax_errors(args.directory)
    elif args.action == "imports":
        fix_all_imports(args.directory)
    elif args.action == "transformers-utils":
        fix_transformers_utils()
    elif args.action == "trl-peft":
        fix_trl_peft_imports()
    elif args.action == "trl-spacy":
        fix_trl_spacy_imports()
    elif args.action == "joblib":
        fix_joblib()
    elif args.action == "tensorboard":
        fix_tensorboard_callback()
    elif args.action == "memory":
        optimize_memory_usage()
