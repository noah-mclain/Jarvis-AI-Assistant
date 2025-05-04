"""
Import Utilities for Jarvis AI Assistant

This module provides comprehensive utilities for fixing import problems in the 
Jarvis AI Assistant codebase. It includes:
1. Path fixing to ensure modules can be found
2. Monkey patching for missing modules
3. Import verification tools
4. Functions to dynamically create import fix blocks

Usage:
    from src.generative_ai_module.import_utilities import fix_imports, check_imports
"""

import os
import sys
import inspect
import importlib
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Tuple, Union, Optional
import logging
import torch
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# Common import fix block that can be added to files
IMPORT_FIX_BLOCK = '''
# ===== BEGIN JARVIS IMPORT FIX =====
# This block was added by the fix_jarvis_imports.py script
import sys
import os

# Add the project root to sys.path
_jarvis_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _jarvis_project_root not in sys.path:
    sys.path.insert(0, _jarvis_project_root)

# Import the necessary functions directly
try:
    from src.generative_ai_module.import_utilities import calculate_metrics, save_metrics, EvaluationMetrics
except ImportError:
    # If that fails, define them locally
    import torch
    import numpy as np
    import math
    
    def calculate_metrics(model, data_batches, device, task_type="generation"):
        """Calculate metrics on a dataset (loss, perplexity, accuracy)"""
        model.eval()
        total_loss = 0.0
        total_batches = 0
        total_correct = 0
        total_samples = 0
        
        # Get the appropriate loss function
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for input_batch, target_batch in data_batches:
                try:
                    # Move data to the model's device
                    input_batch = input_batch.to(device)
                    target_batch = target_batch.to(device)
                    
                    # Get vocabulary size (for safety checks)
                    vocab_size = model.embedding.num_embeddings
                    
                    # Safety check: Ensure target indices are within valid range
                    if target_batch.max() >= vocab_size:
                        target_batch = torch.clamp(target_batch, 0, vocab_size - 1)
                    
                    # Forward pass
                    output, _ = model(input_batch)
                    
                    # Handle different target shapes
                    if target_batch.dim() == 1:
                        # For 1D targets (just indices)
                        loss = criterion(output, target_batch)
                        
                        # Calculate accuracy
                        pred = output.argmax(dim=1)
                        total_correct += (pred == target_batch).sum().item()
                        total_samples += target_batch.size(0)
                    else:
                        # For 2D targets
                        loss = criterion(output.view(-1, output.size(-1)), target_batch.view(-1))
                        
                        # Calculate accuracy
                        pred = output.view(-1, output.size(-1)).argmax(dim=1)
                        # Create mask to ignore padding tokens (value 0)
                        mask = target_batch.view(-1) != 0  
                        total_correct += ((pred == target_batch.view(-1)) & mask).sum().item()
                        total_samples += mask.sum().item()
                    
                    total_loss += loss.item()
                    total_batches += 1
                
                except Exception as e:
                    print(f"Error calculating metrics for batch: {str(e)}")
                    continue
        
        # Calculate average metrics
        avg_loss = total_loss / max(1, total_batches)
        perplexity = math.exp(min(avg_loss, 20))  # Cap perplexity to prevent overflow
        accuracy = total_correct / max(1, total_samples)
        
        return {
            "loss": avg_loss,
            "perplexity": perplexity,
            "accuracy": accuracy
        }
    
    class EvaluationMetrics:
        """Class for evaluating generative models"""
        def __init__(self, metrics_dir="evaluation_metrics", use_gpu=None):
            self.metrics_dir = metrics_dir
            os.makedirs(metrics_dir, exist_ok=True)
            self.use_gpu = torch.cuda.is_available() if use_gpu is None else use_gpu
        
        def evaluate_generation(self, prompt, generated_text, reference_text=None, 
                              dataset_name="unknown", save_results=True):
            import json
            from datetime import datetime
            
            results = {
                "prompt": prompt,
                "generated_text": generated_text,
                "dataset": dataset_name,
                "timestamp": datetime.now().isoformat()
            }
            
            if reference_text:
                results["reference_text"] = reference_text
            
            if save_results:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_path = os.path.join(self.metrics_dir, 
                                           f"evaluation_{dataset_name}_{timestamp}.json")
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2)
            
            return results
    
    def save_metrics(metrics, model_name, dataset_name, timestamp=None):
        """Save evaluation metrics to a JSON file"""
        import json
        from datetime import datetime
        
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        metrics_dir = "metrics"
        os.makedirs(metrics_dir, exist_ok=True)
        
        model_name_clean = model_name.replace('/', '_')
        dataset_name_clean = dataset_name.replace('/', '_')
        
        filename = f"{model_name_clean}_{dataset_name_clean}_{timestamp}.json"
        filepath = os.path.join(metrics_dir, filename)
        
        metrics_with_meta = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "timestamp": timestamp,
            "metrics": metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_with_meta, f, indent=2)
        
        print(f"Saved metrics to {filepath}")
        return filepath
# ===== END JARVIS IMPORT FIX =====
'''

def fix_path():
    """Add project root to sys.path to enable imports to work correctly"""
    # Determine the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    # Add to path if not already there
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    return project_root

def load_module_directly(module_name, file_path):
    """
    Load a module directly from its file path without relying on package structure
    
    Args:
        module_name: Name to assign to the module
        file_path: Path to the module file
        
    Returns:
        Tuple of (success, module/error_message)
    """
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        
        # Add it to sys.modules
        sys.modules[module_name] = module
        
        try:
            spec.loader.exec_module(module)
            return True, module
        except Exception as e:
            return False, str(e)
    except Exception as e:
        return False, str(e)

def backup_file(file_path):
    """
    Create a backup of a file before modifying it
    
    Args:
        file_path: Path to the file to back up
        
    Returns:
        Path to the backup file
    """
    import shutil
    backup_path = f"{file_path}.bak"
    shutil.copy2(file_path, backup_path)
    print(f"✅ Created backup at {backup_path}")
    return backup_path

def fix_imports(file_path, force=False, create_backup=True):
    """
    Add the import fix block to the top of a file
    
    Args:
        file_path: Path to the file to fix
        force: Whether to force adding the fix even if not needed
        create_backup: Whether to create a backup of the file
        
    Returns:
        bool: Whether changes were made to the file
    """
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return False

    if not file_path.endswith('.py'):
        print(f"Error: {file_path} is not a Python file.")
        return False

    print(f"Analyzing {file_path}...")

    # Create a backup if requested
    if create_backup:
        backup_file(file_path)

    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()

    # Check if the fix has already been applied
    if "# ===== BEGIN JARVIS IMPORT FIX =====" in content:
        print("Import fix already appears to be applied to this file.")
        return False

    # Find the insertion point (after any shebang, module docstrings, and initial imports)
    lines = content.split('\n')

    # Skip shebang if present
    start_index = 0
    if lines and lines[0].startswith('#!'):
        start_index = 1

    # Skip module docstring if present
    in_docstring = False
    for i in range(start_index, len(lines)):
        line = lines[i].strip()

        if line.startswith('"""') or line.startswith("'''"):
            if line.endswith('"""') or line.endswith("'''"):
                # Single line docstring
                start_index = i + 1
                break
            else:
                # Start of multi-line docstring
                in_docstring = True
                continue

        if in_docstring:
            if line.endswith('"""') or line.endswith("'''"):
                # End of multi-line docstring
                in_docstring = False
                start_index = i + 1
                break

    # Determine if the file needs the fix
    if not force:
        needs_fix = (
            "from src.generative_ai_module.evaluation_metrics import" in content or
            "from .evaluation_metrics import" in content or
            "from src.generative_ai_module.train_models import calculate_metrics" in content or
            "from .train_models import calculate_metrics" in content or
            "import src.generative_ai_module.evaluation_metrics" in content or
            "import src.generative_ai_module" in content
        )

        if not needs_fix:
            print("This file doesn't appear to need the import fix.")
            return False

    # Add the import fix block
    modified_content = '\n'.join(lines[:start_index]) + IMPORT_FIX_BLOCK + '\n'.join(lines[start_index:])

    # Write the modified content back to the file
    with open(file_path, 'w') as f:
        f.write(modified_content)

    print(f"Added import fix to {file_path}")
    return True

def extract_module_content(source_file, target_file, function_names=None):
    """
    Extract specified functions from a source file and save them to a target file
    
    Args:
        source_file: Path to the source file
        target_file: Path to save the extracted functions
        function_names: List of function names to extract (None for all)
        
    Returns:
        bool: Whether the extraction was successful
    """
    try:
        # First, load the source module
        module_name = os.path.basename(source_file).split('.')[0]
        success, module = load_module_directly(module_name, source_file)
        
        if not success:
            print(f"Error loading source module: {module}")
            return False
        
        # If no function names specified, extract all functions
        if function_names is None:
            function_names = [name for name, obj in inspect.getmembers(module) 
                           if inspect.isfunction(obj) and not name.startswith('_')]
        
        # Get the function code
        extracted_functions = []
        
        for func_name in function_names:
            if hasattr(module, func_name):
                func = getattr(module, func_name)
                if inspect.isfunction(func):
                    func_code = inspect.getsource(func)
                    extracted_functions.append(func_code)
                elif inspect.isclass(func):
                    class_code = inspect.getsource(func)
                    extracted_functions.append(class_code)
        
        if not extracted_functions:
            print(f"No functions found to extract from {source_file}")
            return False
        
        # Create output file
        with open(target_file, 'w') as f:
            f.write('"""\n')
            f.write(f'Standalone functions extracted from {os.path.basename(source_file)}\n')
            f.write('This file was automatically generated to resolve import issues.\n')
            f.write('"""\n\n')
            
            # Add necessary imports
            f.write('import os\n')
            f.write('import sys\n')
            f.write('import torch\n')
            f.write('import numpy as np\n')
            f.write('from datetime import datetime\n\n')
            
            # Write each function
            for func_code in extracted_functions:
                f.write(func_code)
                f.write('\n\n')
        
        print(f"Successfully extracted {len(extracted_functions)} functions to {target_file}")
        return True
    
    except Exception as e:
        print(f"Error extracting module content: {e}")
        return False

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

# The functions that were previously in import_fix.py
def calculate_metrics(model, data_batches, device):
    """Calculate metrics on a dataset (loss, perplexity, accuracy)"""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    total_correct = 0
    total_samples = 0
    
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for input_batch, target_batch in data_batches:
            # Move data to the model's device
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            
            # Forward pass
            output, _ = model(input_batch)
            
            # Calculate loss
            loss = criterion(output.view(-1, output.size(-1)), target_batch.view(-1))
            total_loss += loss.item()
            
            # Calculate accuracy
            predictions = output.argmax(dim=-1)
            correct = (predictions == target_batch).sum().item()
            total_correct += correct
            total_samples += target_batch.numel()
            
            total_batches += 1
    
    # Calculate metrics
    avg_loss = total_loss / max(1, total_batches)
    perplexity = np.exp(avg_loss)
    accuracy = total_correct / max(1, total_samples)
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'accuracy': accuracy
    }

def save_metrics(metrics, model_name, dataset_name, timestamp=None):
    """Save evaluation metrics to a JSON file."""
    import json
    from datetime import datetime

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create metrics directory
    metrics_dir = "metrics"
    os.makedirs(metrics_dir, exist_ok=True)

    # Create a clean filename
    model_name_clean = model_name.replace('/', '_')
    dataset_name_clean = dataset_name.replace('/', '_')

    filename = f"{model_name_clean}_{dataset_name_clean}_{timestamp}.json"
    filepath = os.path.join(metrics_dir, filename)

    # Add metadata
    metrics_with_meta = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "timestamp": timestamp,
        "metrics": metrics
    }

    # Save the metrics
    with open(filepath, 'w') as f:
        json.dump(metrics_with_meta, f, indent=2)

    print(f"Saved metrics to {filepath}")

    return filepath

class EvaluationMetrics:
    """Class for evaluating generative models"""

    def __init__(self, metrics_dir="evaluation_metrics", use_gpu=None):
        """Initialize the metrics"""
        self.metrics_dir = metrics_dir
        os.makedirs(metrics_dir, exist_ok=True)
        self.use_gpu = torch.cuda.is_available() if use_gpu is None else use_gpu

    def evaluate_generation(self, prompt, generated_text, reference_text=None,
                          dataset_name="unknown", save_results=True):
        """Evaluate generated text against reference"""
        import json
        from datetime import datetime
        
        results = {
            "prompt": prompt,
            "generated_text": generated_text,
            "dataset": dataset_name,
            "timestamp": datetime.now().isoformat()
        }

        if reference_text:
            results["reference_text"] = reference_text

        # Save results if requested
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = os.path.join(self.metrics_dir, f"evaluation_{dataset_name}_{timestamp}.json")

            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)

        return results

def fix_init_imports():
    """
    Fix the import error in __init__.py by updating the import statement
    from 'from .jarvis_unified import UnifiedModel' or 'from .jarvis_unified import JarvisUnified as UnifiedModel'
    to 'from .jarvis_unified import JarvisAI as UnifiedModel'
    """
    # Find the __init__.py file
    base_paths = [
        "/notebooks/src/generative_ai_module/__init__.py",  # Paperspace path
        "src/generative_ai_module/__init__.py",             # Relative path
    ]
    
    init_file = None
    for path in base_paths:
        if os.path.exists(path):
            init_file = path
            break
    
    if not init_file:
        print("ERROR: __init__.py not found in expected locations.")
        return False
    
    # Create backup
    backup_file = f"{init_file}.bak"
    if not os.path.exists(backup_file):
        import shutil
        shutil.copy2(init_file, backup_file)
        print(f"Created backup: {backup_file}")
    
    # Read the file
    with open(init_file, 'r') as f:
        content = f.read()
    
    # Check content for different patterns
    has_old_import_1 = "from .jarvis_unified import UnifiedModel" in content
    has_old_import_2 = "from .jarvis_unified import JarvisUnified as UnifiedModel" in content
    
    if not (has_old_import_1 or has_old_import_2):
        # Check if it already has the correct import
        if "from .jarvis_unified import JarvisAI as UnifiedModel" in content:
            print("__init__.py already has the correct import statement")
            return True
        print("Did not find expected import pattern in __init__.py")
        return False
    
    # Replace the import statement
    if has_old_import_1:
        new_content = content.replace(
            "from .jarvis_unified import UnifiedModel",
            "from .jarvis_unified import JarvisAI as UnifiedModel"
        )
    else:
        new_content = content.replace(
            "from .jarvis_unified import JarvisUnified as UnifiedModel",
            "from .jarvis_unified import JarvisAI as UnifiedModel"
        )
    
    # Write the updated content
    with open(init_file, 'w') as f:
        f.write(new_content)
    
    print("✅ Updated import statement in __init__.py")
    return True

def test_all_modules():
    """Test loading all modules directly from their file paths"""
    module_dir = os.path.dirname(os.path.abspath(__file__))
    module_files = {
        "evaluation_metrics": os.path.join(module_dir, "evaluation_metrics.py"),
        "train_models": os.path.join(module_dir, "train_models.py"),
        "text_generator": os.path.join(module_dir, "text_generator.py"),
        "code_generator": os.path.join(module_dir, "code_generator.py"),
        "utils": os.path.join(module_dir, "utils.py")
    }
    
    results = {}
    for module_name, file_path in module_files.items():
        if os.path.exists(file_path):
            success, result = load_module_directly(module_name, file_path)
            results[module_name] = (success, result)
        else:
            results[module_name] = (False, f"File not found: {file_path}")
    
    # Print summary
    print("\nModule Import Test Results:")
    for module_name, (success, result) in results.items():
        if success:
            print(f"✅ {module_name}: Successfully loaded")
        else:
            print(f"❌ {module_name}: Failed to load - {result}")
    
    return results

def fix_all_imports(target_dir=None, force=False):
    """
    Apply import fixes to all Python files in a directory
    
    Args:
        target_dir: Directory to search for Python files (default: current module directory)
        force: Whether to force adding the fix even if not needed
        
    Returns:
        Tuple of (num_files_fixed, num_files_skipped)
    """
    if target_dir is None:
        target_dir = os.path.dirname(os.path.abspath(__file__))
    
    fixed_files = []
    skipped_files = []
    
    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                
                # Skip this file (import_utilities.py) to avoid self-modification
                if os.path.basename(file_path) == 'import_utilities.py':
                    continue
                
                try:
                    if fix_imports(file_path, force=force):
                        fixed_files.append(file_path)
                    else:
                        skipped_files.append(file_path)
                except Exception as e:
                    print(f"Error fixing imports in {file_path}: {e}")
                    skipped_files.append(file_path)
    
    print(f"\nFixed imports in {len(fixed_files)} files")
    print(f"Skipped {len(skipped_files)} files")
    
    return len(fixed_files), len(skipped_files)

# Monkey patch the required modules
def monkey_patch_modules():
    """Patch modules with missing functions"""
    try:
        import src.generative_ai_module.evaluation_metrics
        sys.modules['src.generative_ai_module.evaluation_metrics'].calculate_metrics = calculate_metrics
        sys.modules['src.generative_ai_module.evaluation_metrics'].save_metrics = save_metrics
        sys.modules['src.generative_ai_module.evaluation_metrics'].EvaluationMetrics = EvaluationMetrics
        print("✅ Monkey patched evaluation_metrics module")
    except ImportError:
        # Module not imported yet, that's fine
        pass
    
    # Add ourselves to sys.modules
    if __name__ != "__main__":
        sys.modules['src.generative_ai_module.calculate_metrics'] = sys.modules[__name__]
        sys.modules['src.generative_ai_module.evaluate_generation'] = sys.modules[__name__]
        print("✅ Registered import_utilities as backup for missing modules")

# Run the path fixing when the module is imported
fix_path()

# Main function when run directly
def main():
    """Main entry point when run as a script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix import issues in Jarvis AI Assistant")
    parser.add_argument("--check", action="store_true", help="Check imports without modifying files")
    parser.add_argument("--fix", type=str, help="Path to Python file to fix imports in")
    parser.add_argument("--fix-all", action="store_true", help="Fix imports in all Python files")
    parser.add_argument("--force", action="store_true", help="Force adding import fixes even if not needed")
    parser.add_argument("--fix-init", action="store_true", help="Fix imports in __init__.py")
    parser.add_argument("--test", action="store_true", help="Test loading all modules directly")
    parser.add_argument("--monkey-patch", action="store_true", help="Monkey patch missing modules")
    
    args = parser.parse_args()
    
    print("\n🛠️ Jarvis AI Assistant Import Utilities")
    
    if args.check:
        # Just check imports
        results = check_imports()
        
        print("\nImport Check Results:")
        for module_name, (success, result) in results.items():
            if success:
                print(f"✅ {module_name}: Successfully imported")
            else:
                print(f"❌ {module_name}: Failed to import - {result}")
    
    elif args.fix:
        # Fix a specific file
        if os.path.exists(args.fix):
            fix_imports(args.fix, force=args.force)
        else:
            print(f"Error: File {args.fix} not found")
    
    elif args.fix_all:
        # Fix all Python files
        fix_all_imports(force=args.force)
    
    elif args.fix_init:
        # Fix __init__.py
        fix_init_imports()
    
    elif args.test:
        # Test loading all modules
        test_all_modules()
    
    elif args.monkey_patch:
        # Monkey patch missing modules
        monkey_patch_modules()
    
    else:
        # Default behavior - run all checks
        print("\nRunning all checks...")
        test_all_modules()
        
        print("\nChecking imports...")
        check_imports()
        
        print("\nChecking __init__.py...")
        fix_init_imports()
    
    print("\n✅ Done!")

if __name__ == "__main__":
    main() 