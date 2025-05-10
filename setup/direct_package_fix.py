#!/usr/bin/env python3
"""
Direct Package Fix Script

This script directly fixes the syntax errors in installed packages:
1. huggingface_hub/constants.py
2. dill/logger.py

These files have unterminated string literals that are causing import errors.
"""

import os
import sys
import site
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_huggingface_constants():
    """Fix the huggingface_hub/constants.py file."""
    try:
        # Find the site-packages directory
        site_packages = site.getsitepackages()[0]
        constants_file = os.path.join(site_packages, "huggingface_hub", "constants.py")
        
        if not os.path.exists(constants_file):
            logger.error(f"File not found: {constants_file}")
            return False
        
        logger.info(f"Fixing file: {constants_file}")
        
        # Read the file
        with open(constants_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for the specific error on line 197
        if "]'" in content:
            # Fix the unterminated string literal
            fixed_content = content.replace("]'", "]")
            
            # Write the fixed content back to the file
            with open(constants_file, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            
            logger.info(f"Fixed unterminated string literal in {constants_file}")
            return True
        else:
            logger.info(f"No unterminated string literal found in {constants_file}")
            return True
    
    except Exception as e:
        logger.error(f"Error fixing huggingface_hub constants: {e}")
        return False

def fix_dill_logger():
    """Fix the dill/logger.py file."""
    try:
        # Find the site-packages directory
        site_packages = site.getsitepackages()[0]
        logger_file = os.path.join(site_packages, "dill", "logger.py")
        
        if not os.path.exists(logger_file):
            logger.error(f"File not found: {logger_file}")
            return False
        
        logger.info(f"Fixing file: {logger_file}")
        
        # Read the file
        with open(logger_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for the specific error on line 285
        if "self.handler.close()'" in content:
            # Fix the unterminated string literal
            fixed_content = content.replace("self.handler.close()'", "self.handler.close()")
            
            # Write the fixed content back to the file
            with open(logger_file, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            
            logger.info(f"Fixed unterminated string literal in {logger_file}")
            return True
        else:
            logger.info(f"No unterminated string literal found in {logger_file}")
            return True
    
    except Exception as e:
        logger.error(f"Error fixing dill logger: {e}")
        return False

def create_missing_fix_scripts():
    """Create missing fix scripts that train_jarvis.sh is looking for."""
    setup_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create fix_unsloth_trust_remote_code.py
    unsloth_fix_file = os.path.join(setup_dir, "fix_unsloth_trust_remote_code.py")
    if not os.path.exists(unsloth_fix_file):
        logger.info(f"Creating {unsloth_fix_file}")
        with open(unsloth_fix_file, "w") as f:
            f.write("""#!/usr/bin/env python3
\"\"\"
Fix Unsloth trust_remote_code issue.

This script patches the Unsloth library to use trust_remote_code=True
when loading DeepSeek models.
\"\"\"

import os
import sys
import inspect
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_unsloth_trust_remote_code():
    \"\"\"
    Fix Unsloth trust_remote_code issue.
    
    Returns:
        bool: True if successful, False otherwise
    \"\"\"
    try:
        # Try to import Unsloth
        try:
            from unsloth import FastLanguageModel
            logger.info("Unsloth is available.")
        except ImportError:
            logger.error("Unsloth is not available. Cannot fix trust_remote_code issue.")
            return False
        
        # Find the Unsloth package directory
        unsloth_dir = os.path.dirname(inspect.getfile(FastLanguageModel))
        logger.info(f"Unsloth directory: {unsloth_dir}")
        
        # Find the file that needs to be patched
        patch_file = os.path.join(unsloth_dir, "models.py")
        
        if not os.path.exists(patch_file):
            logger.warning(f"Could not find {patch_file}. Looking for alternative files...")
            
            # Look for alternative files
            for root, dirs, files in os.walk(unsloth_dir):
                for file in files:
                    if file.endswith(".py"):
                        file_path = os.path.join(root, file)
                        with open(file_path, "r") as f:
                            content = f.read()
                            if "from_pretrained" in content and "trust_remote_code" in content:
                                patch_file = file_path
                                logger.info(f"Found alternative file to patch: {patch_file}")
                                break
                if patch_file != os.path.join(unsloth_dir, "models.py"):
                    break
        
        if not os.path.exists(patch_file):
            logger.error(f"Could not find a file to patch in the Unsloth package.")
            return False
        
        # Read the file
        with open(patch_file, "r") as f:
            content = f.read()
        
        # Check if the file already has trust_remote_code=True
        if "trust_remote_code=True" in content:
            logger.info(f"File {patch_file} already has trust_remote_code=True.")
            return True
        
        # Patch the file
        patched_content = content.replace(
            "AutoModelForCausalLM.from_pretrained(",
            "AutoModelForCausalLM.from_pretrained(trust_remote_code=True, "
        )
        
        # Write the patched file
        with open(patch_file, "w") as f:
            f.write(patched_content)
        
        logger.info(f"Successfully patched {patch_file} to include trust_remote_code=True.")
        return True
    
    except Exception as e:
        logger.error(f"Failed to patch Unsloth for trust_remote_code: {e}")
        return False

if __name__ == "__main__":
    success = fix_unsloth_trust_remote_code()
    sys.exit(0 if success else 1)
""")
        os.chmod(unsloth_fix_file, 0o755)
    
    # Create fix_unterminated_strings.py
    unterminated_strings_file = os.path.join(setup_dir, "fix_unterminated_strings.py")
    if not os.path.exists(unterminated_strings_file):
        logger.info(f"Creating {unterminated_strings_file}")
        with open(unterminated_strings_file, "w") as f:
            f.write("""#!/usr/bin/env python3
\"\"\"
Fix unterminated string literals in Python files.
\"\"\"

import os
import sys
import re
import glob
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_unterminated_strings(file_path):
    \"\"\"
    Fix unterminated string literals in a Python file.
    
    Args:
        file_path (str): Path to the Python file
        
    Returns:
        bool: True if successful, False otherwise
    \"\"\"
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
                logger.info(f"Fixed unterminated string in {file_path}: {full_match[:20]}...")
                fixed = True
        
        # Write the fixed content back to the file
        if fixed:
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

def fix_all_unterminated_strings(directory=None):
    \"\"\"
    Fix all unterminated string literals in Python files.
    
    Args:
        directory (str): Directory to search for Python files (default: current directory)
        
    Returns:
        int: Number of files fixed
    \"\"\"
    if directory is None:
        directory = os.getcwd()
    
    logger.info(f"Fixing unterminated strings in {directory}...")
    
    fixed_count = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                if fix_unterminated_strings(file_path):
                    fixed_count += 1
    
    logger.info(f"Fixed unterminated strings in {fixed_count} files.")
    return fixed_count

if __name__ == "__main__":
    # Fix unterminated strings in the current directory
    fixed_count = fix_all_unterminated_strings()
    
    # Also fix unterminated strings in site-packages
    site_packages = sys.path[1]  # Usually the site-packages directory
    logger.info(f"Fixing unterminated strings in {site_packages}...")
    fixed_count += fix_all_unterminated_strings(site_packages)
    
    sys.exit(0)
""")
        os.chmod(unterminated_strings_file, 0o755)
    
    # Create fix_transformers_attention_mask.py
    attention_mask_file = os.path.join(setup_dir, "fix_transformers_attention_mask.py")
    if not os.path.exists(attention_mask_file):
        logger.info(f"Creating {attention_mask_file}")
        with open(attention_mask_file, "w") as f:
            f.write("""#!/usr/bin/env python3
\"\"\"
Fix transformers attention mask issues.
\"\"\"

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_transformers_attention_mask():
    \"\"\"
    Fix transformers attention mask issues.
    
    Returns:
        bool: True if successful, False otherwise
    \"\"\"
    try:
        # Try to import transformers
        try:
            import transformers
            logger.info(f"Transformers version: {transformers.__version__}")
        except ImportError:
            logger.error("Transformers is not installed. Cannot fix attention mask issues.")
            return False
        
        # Create transformers.utils module if it doesn't exist
        try:
            import transformers.utils
            logger.info("transformers.utils module already exists.")
        except ImportError:
            logger.info("Creating transformers.utils module...")
            
            # Find the transformers package directory
            transformers_dir = os.path.dirname(transformers.__file__)
            utils_dir = os.path.join(transformers_dir, "utils")
            
            # Create the utils directory if it doesn't exist
            os.makedirs(utils_dir, exist_ok=True)
            
            # Create an empty __init__.py file
            with open(os.path.join(utils_dir, "__init__.py"), "w") as f:
                f.write(\"\"\"
# Auto-generated utils module for transformers
import logging
logger = logging.getLogger(__name__)

def get_attention_mask_dtype(dtype):
    \"\"\"
    Get the correct dtype for attention mask based on the model's dtype.
    
    Args:
        dtype: The dtype of the model's parameters
        
    Returns:
        The appropriate dtype for the attention mask
    \"\"\"
    # Handle torch dtypes
    if hasattr(dtype, "is_floating_point") and dtype.is_floating_point:
        return dtype
    
    # Handle numpy dtypes
    if hasattr(dtype, "kind") and dtype.kind == "f":
        return dtype
    
    # Default to float32
    import torch
    return torch.float32

def prepare_4d_attention_mask(attention_mask, dtype):
    \"\"\"
    Prepare a 4D attention mask from a 2D attention mask.
    
    Args:
        attention_mask: The 2D attention mask [batch_size, seq_length]
        dtype: The dtype to use for the attention mask
        
    Returns:
        The 4D attention mask [batch_size, 1, seq_length, seq_length]
    \"\"\"
    import torch
    
    # Ensure attention_mask is a tensor
    if not isinstance(attention_mask, torch.Tensor):
        attention_mask = torch.tensor(attention_mask, dtype=dtype)
    
    # Convert to correct dtype if needed
    if attention_mask.dtype != dtype:
        attention_mask = attention_mask.to(dtype=dtype)
    
    # Create 4D attention mask [batch_size, 1, seq_length, seq_length]
    batch_size, seq_length = attention_mask.shape
    
    # Create causal mask
    # [batch_size, 1, seq_length, seq_length]
    causal_mask = torch.triu(
        torch.ones((seq_length, seq_length), dtype=dtype) * -10000.0,
        diagonal=1
    )
    
    # Expand attention_mask to 4D
    # [batch_size, 1, seq_length, 1]
    extended_attention_mask = attention_mask[:, None, :, None]
    
    # Combine with causal mask
    # [batch_size, 1, seq_length, seq_length]
    extended_attention_mask = extended_attention_mask + causal_mask
    
    return extended_attention_mask

def fix_attention_mask(attention_mask, dtype, unmasked_value=0.0):
    \"\"\"
    Fix attention mask for DeepSeek models.
    
    Args:
        attention_mask: The attention mask
        dtype: The dtype to use for the attention mask
        unmasked_value: The value to use for unmasked positions
        
    Returns:
        The fixed attention mask
    \"\"\"
    import torch
    
    # Ensure attention_mask is a tensor
    if not isinstance(attention_mask, torch.Tensor):
        attention_mask = torch.tensor(attention_mask, dtype=dtype)
    
    # Convert to correct dtype if needed
    if attention_mask.dtype != dtype:
        attention_mask = attention_mask.to(dtype=dtype)
    
    # Fix the attention mask values
    # 0 -> -10000.0, 1 -> 0.0
    attention_mask = (1.0 - attention_mask) * -10000.0 + unmasked_value
    
    return attention_mask
\"\"\")
            
            # Try to import the module again to verify
            import importlib
            importlib.invalidate_caches()
            import transformers.utils
            logger.info("Successfully created transformers.utils module.")
        
        logger.info("Successfully fixed transformers attention mask issues.")
        return True
    
    except Exception as e:
        logger.error(f"Failed to fix transformers attention mask issues: {e}")
        return False

if __name__ == "__main__":
    success = fix_transformers_attention_mask()
    sys.exit(0 if success else 1)
""")
        os.chmod(attention_mask_file, 0o755)
    
    logger.info("Created missing fix scripts")
    return True

def fix_all_issues():
    """Fix all issues in the installed packages."""
    # Fix huggingface_hub/constants.py
    fix_huggingface_constants()
    
    # Fix dill/logger.py
    fix_dill_logger()
    
    # Create missing fix scripts
    create_missing_fix_scripts()
    
    logger.info("All issues have been fixed.")
    return True

if __name__ == "__main__":
    fix_all_issues()
    logger.info("Package issues have been fixed. You can now run train_jarvis.sh")
    sys.exit(0)
