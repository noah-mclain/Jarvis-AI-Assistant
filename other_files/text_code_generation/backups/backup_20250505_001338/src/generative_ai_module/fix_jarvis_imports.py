#!/usr/bin/env python3
"""
Jarvis AI Assistant Import Fix Tool

This script fixes import issues in the Jarvis AI Assistant codebase by:
1. Creating a standalone copy of the functions that have import issues
2. Adding the necessary import statements to the top of any file that needs them

Usage:
python fix_jarvis_imports.py <file_to_fix>
"""

import os
import sys
import re
import argparse
import shutil
from pathlib import Path

# Special import block to add to files with import issues
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
    from src.generative_ai_module.import_fix import calculate_metrics, save_metrics, EvaluationMetrics
except ImportError:
    # If that fails, define them locally
    import torch
    import numpy as np
    
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
                input_batch = input_batch.to(device)
                target_batch = target_batch.to(device)
                output, _ = model(input_batch)
                loss = criterion(output.view(-1, output.size(-1)), target_batch.view(-1))
                total_loss += loss.item()
                predictions = output.argmax(dim=-1)
                correct = (predictions == target_batch).sum().item()
                total_correct += correct
                total_samples += target_batch.numel()
                total_batches += 1
        
        avg_loss = total_loss / max(1, total_batches)
        perplexity = np.exp(avg_loss)
        accuracy = total_correct / max(1, total_samples)
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'accuracy': accuracy
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

def backup_file(file_path):
    """Create a backup of the file before modifying it"""
    backup_path = f"{file_path}.bak"
    shutil.copy2(file_path, backup_path)
    print(f"✅ Created backup at {backup_path}")
    return backup_path

def fix_imports(file_path):
    """Add the import fix block to the top of the file"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if the fix has already been applied
    if "# ===== BEGIN JARVIS IMPORT FIX =====" in content:
        print("⚠️ Import fix already appears to be applied to this file.")
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
    
    # Determine if the file imports from evaluation_metrics or train_models
    needs_fix = (
        "from src.generative_ai_module.evaluation_metrics import" in content or
        "from .evaluation_metrics import" in content or
        "from src.generative_ai_module.train_models import calculate_metrics" in content or
        "from .train_models import calculate_metrics" in content or
        "import src.generative_ai_module.evaluation_metrics" in content or
        "import src.generative_ai_module" in content
    )
    
    if not needs_fix:
        print("ℹ️ This file doesn't appear to need the import fix.")
        return False
    
    # Add the import fix block
    modified_content = '\n'.join(lines[:start_index]) + IMPORT_FIX_BLOCK + '\n'.join(lines[start_index:])
    
    # Write the modified content back to the file
    with open(file_path, 'w') as f:
        f.write(modified_content)
    
    print(f"✅ Added import fix to {file_path}")
    return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Fix import issues in Jarvis AI Assistant Python files")
    parser.add_argument("file", help="Path to the Python file to fix")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating a backup of the file")
    parser.add_argument("--force", action="store_true", help="Apply the fix even if the file doesn't appear to need it")
    
    args = parser.parse_args()
    
    file_path = os.path.abspath(args.file)
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File {file_path} does not exist.")
        return 1
    
    if not file_path.endswith('.py'):
        print(f"❌ Error: {file_path} is not a Python file.")
        return 1
    
    print(f"🔍 Analyzing {file_path}...")
    
    # Create a backup if requested
    if not args.no_backup:
        backup_file(file_path)
    
    # Apply the fix
    if args.force:
        # Skip the check and apply the fix regardless
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check if the fix has already been applied
        if "# ===== BEGIN JARVIS IMPORT FIX =====" in content:
            print("⚠️ Import fix already appears to be applied to this file.")
            return 0
        
        # Add the import fix block at the start
        with open(file_path, 'r') as f:
            lines = f.read().split('\n')
        
        # Skip shebang if present
        start_index = 0
        if lines and lines[0].startswith('#!'):
            start_index = 1
        
        modified_content = '\n'.join(lines[:start_index]) + IMPORT_FIX_BLOCK + '\n'.join(lines[start_index:])
        
        with open(file_path, 'w') as f:
            f.write(modified_content)
        
        print(f"✅ Added import fix to {file_path} (force mode)")
    else:
        # Normal mode - only apply if needed
        if not fix_imports(file_path):
            print("ℹ️ No changes were made to the file.")
            return 0
    
    print("\n✅ Import fix applied successfully!")
    print("📝 You may now run the file and it should import correctly.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 