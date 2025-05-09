echo "===================================================================="
echo "Jarvis AI Assistant - Setup Script (No Dependency Conflicts)"
echo "===================================================================="

# Set up Python environment
sudo apt-get update
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev \
python3-dev sqlite3

curl https://pyenv.run | bash

# Add pyenv to your shell config
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init)"' >> ~/.bashrc
source ~/.bashrc

pyenv install 3.11.5
pyenv global 3.11.5
python --version  # Should output "Python 3.11.5"
python -m venv jarvis_env  # Uses pyenv's 3.11.5
# Activate Python environment
if [ -f "jarvis_env/bin/activate" ]; then
    echo "Activating Python environment from jarvis_env/bin/activate"
    source jarvis_env/bin/activate
elif [ -f "/notebooks/jarvis_env/bin/activate" ]; then
    echo "Activating Python environment from /notebooks/jarvis_env/bin/activate"
    source /notebooks/jarvis_env/bin/activate
else
    echo "Creating new Python environment"
    python -m venv jarvis_env
    source jarvis_env/bin/activate
fi

# Verify Python environment is activated
if [[ "$VIRTUAL_ENV" == *"jarvis_env"* ]]; then
    echo "✅ Python environment successfully activated: $VIRTUAL_ENV"
else
    echo "⚠️ Warning: Python environment may not be properly activated"
    echo "Current VIRTUAL_ENV: $VIRTUAL_ENV"
fi

pip cache purge

# Install dependencies
sudo apt-get update
sudo apt-get install -y curl unzip

# Download and run the rclone installer
curl https://rclone.org/install.sh | sudo bash
rclone version

rclone config

# Sync entire folder to Paperspace storage
rclone sync gdrive: \
  /notebooks/Jarvis_AI_Assistant \
  --drive-root-folder-id 1bGqUvfzVrZdxBqLz9rl0sN3_j5UjpSTE \
  --progress \
  --transfers 4 \
  --checkers 8 \
  --drive-acknowledge-abuse

  # Compare source and destination
rclone check gdrive: /notebooks/Jarvis_AI_Assistant \
  --drive-root-folder-id 1bGqUvfzVrZdxBqLz9rl0sN3_j5UjpSTE \
  --size-only

# Install Google Drive integration for data persistence
pip install gdown google-auth google-auth-oauthlib google-auth-httplib2

chmod +x setup/*
./setup/setup.sh
./setup/fix_unsloth_final.sh
./setup/create_minimal_unsloth.sh
./setup/apply_fixed_unsloth.sh
./setup/install_flash_attention.sh
./setup/install_additional_deps.sh
./setup/install_enhanced_attention.sh

# Activate minimal Unsloth
CUSTOM_UNSLOTH_DIR="/notebooks/custom_unsloth"
if [ -f "$CUSTOM_UNSLOTH_DIR/activate_minimal_unsloth.sh" ]; then
    echo "Activating minimal Unsloth from $CUSTOM_UNSLOTH_DIR/activate_minimal_unsloth.sh"
    source "$CUSTOM_UNSLOTH_DIR/activate_minimal_unsloth.sh"

    # Verify Unsloth activation by running the test script
    if [ -f "$CUSTOM_UNSLOTH_DIR/use_minimal_unsloth.py" ]; then
        echo "Testing minimal Unsloth installation..."
        python "$CUSTOM_UNSLOTH_DIR/use_minimal_unsloth.py"

        if [ $? -eq 0 ]; then
            echo "✅ Minimal Unsloth successfully activated and tested"
        else
            echo "⚠️ Warning: Minimal Unsloth test failed"
        fi
    else
        echo "⚠️ Warning: Minimal Unsloth test script not found at $CUSTOM_UNSLOTH_DIR/use_minimal_unsloth.py"
    fi
else
    echo "⚠️ Warning: Minimal Unsloth activation script not found at $CUSTOM_UNSLOTH_DIR/activate_minimal_unsloth.sh"
    echo "Creating minimal Unsloth installation..."

    # Run the create_minimal_unsloth.sh script if it exists
    if [ -f "setup/create_minimal_unsloth.sh" ]; then
        chmod +x setup/create_minimal_unsloth.sh
        ./setup/create_minimal_unsloth.sh

        # Now try to activate it
        if [ -f "$CUSTOM_UNSLOTH_DIR/activate_minimal_unsloth.sh" ]; then
            echo "Activating newly created minimal Unsloth..."
            source "$CUSTOM_UNSLOTH_DIR/activate_minimal_unsloth.sh"
            python "$CUSTOM_UNSLOTH_DIR/use_minimal_unsloth.py"
        fi
    else
        echo "❌ Error: create_minimal_unsloth.sh script not found"
    fi
fi

# Add minimal Unsloth to PYTHONPATH permanently
if [ -d "$CUSTOM_UNSLOTH_DIR" ]; then
    export PYTHONPATH="$CUSTOM_UNSLOTH_DIR:$PYTHONPATH"
    echo "export PYTHONPATH=\"$CUSTOM_UNSLOTH_DIR:\$PYTHONPATH\"" >> ~/.bashrc
    echo "✅ Added minimal Unsloth to PYTHONPATH permanently"
fi

## Fix Import Issues

# Create import_fix.py
cat > /notebooks/src/generative_ai_module/import_fix.py << 'EOL'
"""
Import Fix Module

This module provides all the functions and classes that were missing or had issues in imports.
Simply import this module first to ensure all dependencies are properly available.
"""

import os
import sys
import torch
import numpy as np
import json
from datetime import datetime

# Add the parent directory to sys.path for proper imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Calculate metrics function
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

# EvaluationMetrics class
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

# Update the module's __all__ to include these functions
__all__ = [
    'calculate_metrics',
    'save_metrics',
    'EvaluationMetrics'
]

# Monkey patch the required modules
try:
    import src.generative_ai_module.evaluation_metrics
    sys.modules['src.generative_ai_module.evaluation_metrics'].calculate_metrics = calculate_metrics
    sys.modules['src.generative_ai_module.evaluation_metrics'].save_metrics = save_metrics
    sys.modules['src.generative_ai_module.evaluation_metrics'].EvaluationMetrics = EvaluationMetrics
except ImportError:
    # Module not imported yet, that's fine
    pass

# Fix the module import if this is imported directly
if __name__ != "__main__":
    # Add ourselves to sys.modules
    sys.modules['src.generative_ai_module.calculate_metrics'] = sys.modules[__name__]
    sys.modules['src.generative_ai_module.evaluate_generation'] = sys.modules[__name__]
EOL

# Create fix_jarvis_imports.py
cat > /notebooks/src/generative_ai_module/fix_jarvis_imports.py << 'EOL'
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
EOL

# Make fix_jarvis_imports.py executable
chmod +x /notebooks/src/generative_ai_module/fix_jarvis_imports.py

./setup/install_spacy_isolated.sh
./setup/fix_spacy_for_paperspace.sh

# Apply attention mask fix for DeepSeek models
echo "Applying attention mask fix for DeepSeek models..."

# Make the scripts executable
chmod +x setup/fix_transformers_attention_mask.py
chmod +x setup/fix_attention_mask_params.py
chmod +x setup/fix_tensor_size_mismatch.py
chmod +x setup/fix_attention_dimension_mismatch.py
chmod +x setup/fix_tuple_unpacking_error.py
chmod +x setup/comprehensive_attention_mask_fix.py
chmod +x setup/fix_all_attention_issues.py
chmod +x setup/ultimate_attention_fix.py

# Run the ultimate fix script first (most comprehensive)
echo "Applying ultimate fix for all attention-related issues..."
python setup/ultimate_attention_fix.py

# Run the all-in-one fix script as first fallback
echo "Applying all-in-one attention mask and tuple unpacking fixes..."
python setup/fix_all_attention_issues.py

# Run individual fix scripts as additional fallbacks
echo "Applying individual fixes as additional fallbacks..."

# Run the general fix script
echo "Applying general attention mask fixes..."
python setup/fix_transformers_attention_mask.py

# Run the parameter-specific fix script
echo "Applying parameter-specific attention mask fixes..."
python setup/fix_attention_mask_params.py

# Run the tensor size mismatch fix script
echo "Applying tensor size mismatch fixes..."
python setup/fix_tensor_size_mismatch.py

# Run the attention dimension mismatch fix script
echo "Applying attention dimension mismatch fixes..."
python setup/fix_attention_dimension_mismatch.py

# Run the tuple unpacking error fix script
echo "Applying fix for 'too many values to unpack (expected 2)' error..."
python setup/fix_tuple_unpacking_error.py

# Run the comprehensive attention mask fix script
echo "Applying comprehensive attention mask fix..."
python setup/comprehensive_attention_mask_fix.py

if [ $? -ne 0 ]; then
    echo "⚠️ Warning: Attention mask fix scripts failed, but continuing anyway..."
else
    echo "✅ Attention mask fixes applied successfully"
fi

# Apply fixes to key files
cd /notebooks
python src/generative_ai_module/fix_jarvis_imports.py --force src/generative_ai_module/train_models.py
python src/generative_ai_module/fix_jarvis_imports.py --force src/generative_ai_module/finetune_deepseek.py
python src/generative_ai_module/fix_jarvis_imports.py --force src/generative_ai_module/evaluate_generation.py
python src/generative_ai_module/fix_jarvis_imports.py --force src/generative_ai_module/unified_generation_pipeline.py

# Create required directories
# mkdir -p /notebooks/Jarvis_AI_Assistant/{models,datasets,metrics,logs,checkpoints,evaluation_metrics,visualizations}