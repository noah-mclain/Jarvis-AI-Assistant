#!/usr/bin/env python3
"""
Direct import script that bypasses __init__.py to test module imports.

This script directly imports the necessary components to fix the import issues.
"""

import os
import sys
import importlib.util
from pprint import pprint

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

# Directory where this script is located
module_dir = os.path.dirname(os.path.abspath(__file__))

def load_module_directly(module_name, file_path):
    """Load a module directly from its file path without relying on package structure"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    
    # Add it to sys.modules
    sys.modules[module_name] = module
    
    try:
        spec.loader.exec_module(module)
        return True, module
    except Exception as e:
        return False, str(e)

def test_all_modules():
    """Test loading all modules directly"""
    print("🔍 Testing direct module imports...")
    
    # Dictionary to store our directly loaded modules
    modules = {}
    
    # 1. Load utils.py first (it has no dependencies)
    utils_path = os.path.join(module_dir, "utils.py")
    print(f"\nLoading utils.py from {utils_path}")
    success, utils = load_module_directly("utils_direct", utils_path)
    
    if success:
        print("✅ Successfully loaded utils.py")
        # Make it available for other modules
        sys.modules["utils"] = utils
        modules["utils"] = utils
        
        # Print utils functions (important ones)
        print("Available utils functions:")
        for name in dir(utils):
            if not name.startswith("_") and callable(getattr(utils, name)):
                print(f"  - {name}")
    else:
        print(f"❌ Failed to load utils.py: {utils}")
        return
    
    # 2. Load evaluation_metrics.py 
    metrics_path = os.path.join(module_dir, "evaluation_metrics.py")
    print(f"\nLoading evaluation_metrics.py from {metrics_path}")
    success, metrics = load_module_directly("evaluation_metrics_direct", metrics_path)
    
    if success:
        print("✅ Successfully loaded evaluation_metrics.py")
        modules["evaluation_metrics"] = metrics
        
        # Check if it has the EvaluationMetrics class
        if hasattr(metrics, "EvaluationMetrics"):
            print("  - Found EvaluationMetrics class")
        
        # Check if it has the save_metrics function
        if hasattr(metrics, "save_metrics"):
            print("  - Found save_metrics function")
    else:
        print(f"❌ Failed to load evaluation_metrics.py: {metrics}")
    
    # 3. Load train_models.py
    train_path = os.path.join(module_dir, "train_models.py")
    print(f"\nLoading train_models.py from {train_path}")
    success, train = load_module_directly("train_models_direct", train_path)
    
    if success:
        print("✅ Successfully loaded train_models.py")
        modules["train_models"] = train
        
        # Check if it has the calculate_metrics function
        if hasattr(train, "calculate_metrics"):
            print("  - Found calculate_metrics function")
    else:
        print(f"❌ Failed to load train_models.py: {train}")
    
    return modules

def extract_module_content(module_path, target_file):
    """Extract key functions/classes from a module to a standalone file"""
    print(f"\n📝 Extracting content from {module_path} to {target_file}...")
    
    with open(module_path, "r") as source_file:
        content = source_file.read()
    
    # Extract imports
    import_lines = []
    for line in content.split("\n"):
        if line.strip().startswith("import ") or line.strip().startswith("from "):
            if "from ." not in line:  # Skip relative imports
                import_lines.append(line)
    
    # Write the extracted content
    with open(target_file, "w") as target:
        target.write("#!/usr/bin/env python3\n")
        target.write('"""\nExtracted code from module - standalone version\n"""\n\n')
        
        # Write imports
        for imp in import_lines:
            target.write(imp + "\n")
        
        # Add separator
        target.write("\n# " + "="*50 + "\n\n")
        
        # For train_models.py, extract calculate_metrics function
        if "train_models" in module_path:
            in_function = False
            function_lines = []
            
            for line in content.split("\n"):
                if line.startswith("def calculate_metrics("):
                    in_function = True
                    function_lines.append(line)
                elif in_function:
                    function_lines.append(line)
                    if line.strip() == "":
                        # Empty line after function definition typically indicates end of function
                        if any("return" in l for l in function_lines[-10:]):
                            in_function = False
            
            # Write the function
            for line in function_lines:
                target.write(line + "\n")
                
        # For evaluation_metrics.py, extract EvaluationMetrics class and save_metrics function
        elif "evaluation_metrics" in module_path:
            # First extract save_metrics function
            in_function = False
            function_lines = []
            
            for line in content.split("\n"):
                if line.startswith("def save_metrics("):
                    in_function = True
                    function_lines.append(line)
                elif in_function:
                    function_lines.append(line)
                    if line.strip() == "":
                        # Empty line after function definition typically indicates end of function
                        if any("return" in l for l in function_lines[-10:]):
                            in_function = False
            
            # Write the function
            for line in function_lines:
                target.write(line + "\n")
        
        target.write("\n\nprint('Successfully loaded standalone version')\n")
    
    print(f"✅ Created standalone file: {target_file}")
    return target_file

def main():
    print("\n🛠️ Fixing import issues in the Generative AI Module...")
    
    # First, test direct module imports
    modules = test_all_modules()
    
    # If we need to extract content, create standalone files
    if "train_models" in modules:
        extract_module_content(
            os.path.join(module_dir, "train_models.py"),
            os.path.join(module_dir, "standalone_calculate_metrics.py")
        )
    
    if "evaluation_metrics" in modules:
        extract_module_content(
            os.path.join(module_dir, "evaluation_metrics.py"),
            os.path.join(module_dir, "standalone_save_metrics.py")
        )
    
    print("\n✅ Import analysis complete!")

if __name__ == "__main__":
    main() 