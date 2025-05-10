#!/usr/bin/env python3
"""
Comprehensive Fix Script for Paperspace Environment

This script fixes various issues in the Paperspace environment:
1. Unterminated string literals in installed packages (huggingface_hub, dill)
2. Creates missing setup files
3. Fixes syntax errors in existing setup files

Run this script before running consolidated_unified_setup.sh or train_jarvis.sh
"""

import os
import sys
import re
import glob
import importlib
import site
import subprocess
from pathlib import Path

# Configure basic logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_unterminated_strings_in_file(file_path):
    """Fix unterminated string literals in a file."""
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
        
        # Fix specific issues
        if "optimize_memory_usage()'" in fixed_content:
            fixed_content = fixed_content.replace("optimize_memory_usage()'", "optimize_memory_usage()")
            logger.info(f"Fixed optimize_memory_usage issue in {file_path}")
            fixed = True
        
        # Fix double colon issue
        if "sys.path::" in fixed_content:
            fixed_content = fixed_content.replace("sys.path::", "sys.path:")
            logger.info(f"Fixed double colon issue in {file_path}")
            fixed = True
        
        # Fix colon after docstring
        if '"""Fix' in fixed_content and '""":' in fixed_content:
            fixed_content = fixed_content.replace('""":',  '"""')
            logger.info(f"Fixed colon after docstring in {file_path}")
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

def fix_huggingface_hub_constants():
    """Fix unterminated string literals in huggingface_hub/constants.py."""
    try:
        # Find the huggingface_hub package
        site_packages = site.getsitepackages()[0]
        constants_file = os.path.join(site_packages, "huggingface_hub", "constants.py")
        
        if os.path.exists(constants_file):
            logger.info(f"Fixing unterminated strings in {constants_file}")
            fix_unterminated_strings_in_file(constants_file)
            return True
        else:
            logger.warning(f"Could not find {constants_file}")
            return False
    
    except Exception as e:
        logger.error(f"Error fixing huggingface_hub constants: {e}")
        return False

def fix_dill_logger():
    """Fix unterminated string literals in dill/logger.py."""
    try:
        # Find the dill package
        site_packages = site.getsitepackages()[0]
        logger_file = os.path.join(site_packages, "dill", "logger.py")
        
        if os.path.exists(logger_file):
            logger.info(f"Fixing unterminated strings in {logger_file}")
            fix_unterminated_strings_in_file(logger_file)
            return True
        else:
            logger.warning(f"Could not find {logger_file}")
            return False
    
    except Exception as e:
        logger.error(f"Error fixing dill logger: {e}")
        return False

def create_missing_setup_files():
    """Create missing setup files."""
    setup_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create setup_environment.py
    setup_environment_file = os.path.join(setup_dir, "setup_environment.py")
    if not os.path.exists(setup_environment_file):
        logger.info(f"Creating {setup_environment_file}")
        with open(setup_environment_file, "w") as f:
            f.write("""#!/usr/bin/env python3
\"\"\"
Setup environment and create directories for Jarvis AI Assistant.
\"\"\"

import os
import sys

def setup_paperspace_env():
    \"\"\"
    Set up Paperspace environment.
    \"\"\"
    # Force Paperspace environment detection
    os.environ["PAPERSPACE"] = "true"
    print("✅ Forced Paperspace environment detection")

    # Set up environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,garbage_collection_threshold:0.8"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    # Force certain operations on CPU to save GPU memory
    os.environ["FORCE_CPU_ONLY_FOR_INITIAL_LOAD"] = "1"
    os.environ["FORCE_CPU_ONLY_FOR_TOKENIZATION"] = "1"
    os.environ["FORCE_CPU_ONLY_FOR_DATASET_PROCESSING"] = "1"
    os.environ["TOKENIZERS_FORCE_CPU"] = "1"
    os.environ["HF_DATASETS_CPU_ONLY"] = "1"
    os.environ["JARVIS_FORCE_CPU_TOKENIZER"] = "1"

    # Set PyTorch to use deterministic algorithms for reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = "42"

    print("✅ Environment variables set for Paperspace")

def create_directories():
    \"\"\"
    Create necessary directories.
    \"\"\"
    directories = [
        "models",
        "datasets",
        "checkpoints",
        "logs",
        "visualizations",
        "preprocessed_data",
        "metrics"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

if __name__ == "__main__":
    # Set up environment
    setup_paperspace_env()
    
    # Create directories
    create_directories()
    
    print("✅ Environment setup complete")
""")
        os.chmod(setup_environment_file, 0o755)
    
    # Create verify_packages.py
    verify_packages_file = os.path.join(setup_dir, "verify_packages.py")
    if not os.path.exists(verify_packages_file):
        logger.info(f"Creating {verify_packages_file}")
        with open(verify_packages_file, "w") as f:
            f.write("""#!/usr/bin/env python3
\"\"\"
Verify required Python packages for Jarvis AI Assistant.
\"\"\"

import os
import sys
import subprocess
import importlib

def check_package(package_name, min_version=None):
    \"\"\"
    Check if a package is installed and meets the minimum version requirement.
    
    Args:
        package_name (str): Name of the package to check
        min_version (str, optional): Minimum version required
        
    Returns:
        bool: True if the package is installed and meets the version requirement
    \"\"\"
    try:
        module = importlib.import_module(package_name)
        if hasattr(module, "__version__"):
            version = module.__version__
            print(f"✅ {package_name} {version} is installed")
            
            if min_version and version < min_version:
                print(f"⚠️ {package_name} {version} is older than required {min_version}")
                return False
            
            return True
        else:
            print(f"✅ {package_name} is installed (version unknown)")
            return True
    except ImportError:
        print(f"❌ {package_name} is not installed")
        return False

def install_package(package_name, version=None):
    \"\"\"
    Install a package using pip.
    
    Args:
        package_name (str): Name of the package to install
        version (str, optional): Version to install
        
    Returns:
        bool: True if the installation was successful
    \"\"\"
    try:
        if version:
            package_spec = f"{package_name}=={version}"
        else:
            package_spec = package_name
        
        print(f"Installing {package_spec}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_spec])
        
        return check_package(package_name)
    except Exception as e:
        print(f"❌ Failed to install {package_name}: {e}")
        return False

def verify_required_packages():
    \"\"\"
    Verify all required packages and install missing ones.
    
    Returns:
        bool: True if all required packages are installed
    \"\"\"
    required_packages = {
        "torch": "2.0.0",
        "transformers": "4.30.0",
        "datasets": "2.12.0",
        "accelerate": "0.20.0",
        "peft": "0.4.0",
        "bitsandbytes": "0.40.0",
        "trl": "0.4.7",
        "huggingface_hub": "0.15.0",
        "numpy": "1.24.0",
        "scipy": "1.10.0",
        "matplotlib": "3.7.0",
        "pandas": "2.0.0",
        "scikit-learn": "1.2.0",
        "nltk": "3.8.0",
        "tqdm": "4.65.0",
        "tensorboard": "2.12.0",
        "einops": "0.6.0",
        "sentencepiece": "0.1.99"
    }
    
    all_installed = True
    
    for package, min_version in required_packages.items():
        if not check_package(package, min_version):
            if not install_package(package):
                all_installed = False
    
    return all_installed

if __name__ == "__main__":
    if verify_required_packages():
        print("✅ All required packages are installed")
        sys.exit(0)
    else:
        print("❌ Some required packages are missing or have incorrect versions")
        sys.exit(1)
""")
        os.chmod(verify_packages_file, 0o755)
    
    # Create clear_cuda_cache.py
    clear_cuda_cache_file = os.path.join(setup_dir, "clear_cuda_cache.py")
    if not os.path.exists(clear_cuda_cache_file):
        logger.info(f"Creating {clear_cuda_cache_file}")
        with open(clear_cuda_cache_file, "w") as f:
            f.write("""#!/usr/bin/env python3
\"\"\"
Clear CUDA cache to free up GPU memory.
\"\"\"

import gc
import sys

def clear_cuda_cache():
    \"\"\"
    Clear CUDA cache to free up GPU memory.
    
    Returns:
        bool: True if successful, False otherwise
    \"\"\"
    try:
        import torch
        
        if torch.cuda.is_available():
            print("Clearing CUDA cache...")
            torch.cuda.empty_cache()
            gc.collect()
            
            # Get GPU memory usage
            memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
            memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # Convert to GB
            
            print(f"GPU memory allocated: {memory_allocated:.2f} GB")
            print(f"GPU memory reserved: {memory_reserved:.2f} GB")
            
            print("✅ CUDA cache cleared successfully")
            return True
        else:
            print("⚠️ CUDA is not available. No cache to clear.")
            return True
    except ImportError:
        print("❌ PyTorch is not installed. Cannot clear CUDA cache.")
        return False
    except Exception as e:
        print(f"❌ Error clearing CUDA cache: {e}")
        return False

if __name__ == "__main__":
    success = clear_cuda_cache()
    sys.exit(0 if success else 1)
""")
        os.chmod(clear_cuda_cache_file, 0o755)
    
    # Create verify_gpu_code.py
    verify_gpu_code_file = os.path.join(setup_dir, "verify_gpu_code.py")
    if not os.path.exists(verify_gpu_code_file):
        logger.info(f"Creating {verify_gpu_code_file}")
        with open(verify_gpu_code_file, "w") as f:
            f.write("""#!/usr/bin/env python3
\"\"\"
Verify GPU availability for code model training.
\"\"\"

import sys

def verify_gpu():
    \"\"\"
    Verify GPU availability for code model training.
    
    Returns:
        bool: True if GPU is available and has sufficient memory
    \"\"\"
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("❌ CUDA is not available. GPU training is not possible.")
            return False
        
        # Get GPU information
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        
        print(f"Found {device_count} GPU(s)")
        print(f"Using GPU: {device_name}")
        
        # Check memory
        memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
        memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # Convert to GB
        
        print(f"GPU memory allocated: {memory_allocated:.2f} GB")
        print(f"GPU memory reserved: {memory_reserved:.2f} GB")
        
        # Try a simple tensor operation on GPU
        try:
            x = torch.rand(1000, 1000).cuda()
            y = torch.rand(1000, 1000).cuda()
            z = torch.matmul(x, y)
            del x, y, z
            torch.cuda.empty_cache()
            print("✅ GPU tensor operations successful")
        except Exception as e:
            print(f"❌ GPU tensor operations failed: {e}")
            return False
        
        print("✅ GPU verification successful")
        return True
    
    except ImportError:
        print("❌ PyTorch is not installed. Cannot verify GPU.")
        return False
    except Exception as e:
        print(f"❌ Error verifying GPU: {e}")
        return False

if __name__ == "__main__":
    success = verify_gpu()
    sys.exit(0 if success else 1)
""")
        os.chmod(verify_gpu_code_file, 0o755)
    
    logger.info("Created missing setup files")
    return True

def fix_all_setup_scripts():
    """Fix all setup scripts."""
    setup_dir = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Fixing setup scripts in {setup_dir}...")
    
    # Fix syntax errors in all Python files
    python_files = glob.glob(os.path.join(setup_dir, "*.py"))
    for file_path in python_files:
        fix_unterminated_strings_in_file(file_path)
    
    logger.info("All setup scripts have been fixed.")
    return True

def fix_all_issues():
    """Fix all issues in the Paperspace environment."""
    # Fix unterminated strings in huggingface_hub/constants.py
    fix_huggingface_hub_constants()
    
    # Fix unterminated strings in dill/logger.py
    fix_dill_logger()
    
    # Create missing setup files
    create_missing_setup_files()
    
    # Fix all setup scripts
    fix_all_setup_scripts()
    
    logger.info("All issues have been fixed.")
    return True

if __name__ == "__main__":
    fix_all_issues()
    logger.info("Paperspace environment has been fixed. You can now run train_jarvis.sh")
    sys.exit(0)
