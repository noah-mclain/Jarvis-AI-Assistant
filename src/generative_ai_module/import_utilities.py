"""
Import Utilities Module

This module provides utility functions for handling imports and
fixing common import issues in the Jarvis AI Assistant.
"""

import sys
import os
import importlib
import logging
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_imports(module_names=None):
    """
    Check if specified modules can be imported correctly
    
    Args:
        module_names: List of module names to check (None for default set)
        
    Returns:
        Dict mapping module names to (success, error_message)
    """
    # Default modules to check
    if module_names is None:
        module_names = [
            "src.generative_ai_module.evaluation_metrics",
            "src.generative_ai_module.train_models",
            "src.generative_ai_module.text_generator",
            "src.generative_ai_module.code_generator",
            "src.generative_ai_module.utils"
        ]
    
    results = {}
    
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
            results[module_name] = (True, module)
        except Exception as e:
            results[module_name] = (False, str(e))
    
    return results

def monkey_patch_modules():
    """Patch modules with missing functions"""
    from .train_models import calculate_metrics
    from .evaluation_metrics import save_metrics
    from .evaluation_metrics import EvaluationMetrics
    
    try:
        import src.generative_ai_module.evaluation_metrics
        sys.modules['src.generative_ai_module.evaluation_metrics'].calculate_metrics = calculate_metrics
        sys.modules['src.generative_ai_module.evaluation_metrics'].save_metrics = save_metrics
        sys.modules['src.generative_ai_module.evaluation_metrics'].EvaluationMetrics = EvaluationMetrics
        logger.info("✅ Monkey patched evaluation_metrics module")
    except ImportError:
        # Module not imported yet, that's fine
        pass
    
    # Add ourselves to sys.modules
    if __name__ != "__main__":
        sys.modules['src.generative_ai_module.calculate_metrics'] = sys.modules[__name__]
        sys.modules['src.generative_ai_module.evaluate_generation'] = sys.modules[__name__]
        logger.info("✅ Registered import_utilities as backup for missing modules")

def ensure_import_paths():
    """
    Ensure that the necessary import paths are in sys.path.
    This is useful when running scripts from different directories.
    """
    # Get the absolute path to the project root directory
    current_file = os.path.abspath(__file__)
    generative_ai_module_dir = os.path.dirname(current_file)
    src_dir = os.path.dirname(generative_ai_module_dir)
    project_root = os.path.dirname(src_dir)
    
    # Add the project root and src directory to sys.path if they're not already there
    for path in [project_root, src_dir]:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    return project_root, src_dir, generative_ai_module_dir

def fix_imports():
    """
    Fix common import issues by:
    1. Ensuring the correct paths are in sys.path
    2. Monkey-patching modules with missing functions
    3. Providing fallbacks for missing modules
    """
    # Ensure import paths
    project_root, src_dir, generative_ai_module_dir = ensure_import_paths()
    
    # Monkey-patch modules
    monkey_patch_modules()
    
    # Check imports and report results
    results = check_imports()
    success_count = sum(1 for success, _ in results.values() if success)
    logger.info(f"Import check: {success_count}/{len(results)} modules imported successfully")
    
    return success_count == len(results)

# Apply import fixes immediately when this module is imported
fix_imports()

# For backward compatibility, expose some functions from train_models and evaluation_metrics
try:
    from .train_models import calculate_metrics
except ImportError:
    def calculate_metrics(reference_texts, generated_texts):
        """Fallback implementation of calculate_metrics"""
        logger.warning("Using fallback implementation of calculate_metrics")
        return {
            "bleu": 0.0,
            "rouge1_f": 0.0,
            "rouge2_f": 0.0,
            "rougeL_f": 0.0
        }

try:
    from .evaluation_metrics import save_metrics
except ImportError:
    def save_metrics(metrics, output_file):
        """Fallback implementation of save_metrics"""
        logger.warning("Using fallback implementation of save_metrics")
        import json
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        return True