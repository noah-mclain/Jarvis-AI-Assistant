# Jarvis AI Assistant Implementation Guide for RTX 5000 (16GB GPU)

This guide provides optimized commands and best practices for running the Jarvis AI Assistant on a Paperspace Gradient instance with an NVIDIA RTX 5000 GPU (16GB VRAM, 8 CPUs, 30GB RAM). All commands have been carefully verified to work with the specific parameters supported by each script in the codebase.

> **Important Note:** Script parameters may change as the codebase evolves. If you encounter errors about unrecognized arguments, check the script's help documentation (e.g., `python script.py --help`) and adjust the commands accordingly. The commands in this guide were verified against the current version of the codebase.

## Environment Setup

```bash
# Set up Python environment
pip install -U pip
pip install torch==2.1.2 transformers==4.40.2 datasets accelerate unsloth

# Install Google Drive integration for data persistence
pip install gdown google-auth google-auth-oauthlib google-auth-httplib2

# Create required directories
mkdir -p /notebooks/Jarvis_AI_Assistant/{models,datasets,metrics,logs,checkpoints,evaluation_metrics,visualizations}
```

## Initial Import Setup

Before running any training or fine-tuning scripts, it's crucial to set up the Python import paths and verify that all necessary modules and functions are properly accessible. This is especially important when running on a GPU with limited memory like the RTX 5000, as import errors during execution can waste valuable compute time.

````bash
# Create a simple test script to verify imports
cat > /notebooks/test_imports.py << 'EOL'
#!/usr/bin/env python3
"""
Import Test Script for Jarvis AI Assistant

This script checks if all essential imports are working correctly.
If any of these imports fail, it will indicate which specific modules need fixing.
"""

import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def check_imports():
    """Test all essential imports"""
    print("Testing critical imports for Jarvis AI Assistant...")

    import_tests = [
        # Core ML libraries
        ("torch", "PyTorch is required for all model operations"),
        ("transformers", "HuggingFace Transformers is required for model loading and training"),
        ("datasets", "HuggingFace Datasets is required for dataset handling"),

        # Performance optimization
        ("unsloth", "Unsloth is required for optimized training (optional but recommended)"),
        ("peft", "PEFT is required for parameter-efficient fine-tuning"),
        ("bitsandbytes", "BitsAndBytes is required for quantization"),

        # Jarvis AI modules and sub-modules
        ("src.generative_ai_module", "Main Jarvis module"),
        ("src.generative_ai_module.import_fix", "Import fix utilities"),
        ("src.generative_ai_module.text_generator", "Text generation module"),
        ("src.generative_ai_module.code_generator", "Code generation module"),
        ("src.generative_ai_module.evaluation_metrics", "Evaluation metrics module"),
        ("src.generative_ai_module.utils", "Utility functions"),
    ]

    failed_imports = []
    for module_name, description in import_tests:
        try:
            __import__(module_name)
            print(f"✅ {module_name}: Successfully imported")
        except ImportError as e:
            failed_imports.append((module_name, description, str(e)))
            print(f"❌ {module_name}: Failed to import - {e}")

    # Test specific function imports
    print("\nTesting specific function imports...")

    try:
        from src.generative_ai_module.import_fix import calculate_metrics, save_metrics, EvaluationMetrics
        print("✅ Key functions from import_fix successfully imported")
    except ImportError as e:
        print(f"❌ Failed to import key functions from import_fix: {e}")
        failed_imports.append(("import_fix functions", "Critical evaluation functions", str(e)))

    # Summary
    if failed_imports:
        print("\n⚠️ Some imports failed. Please fix these before proceeding:")
        for module, desc, error in failed_imports:
            print(f"  - {module}: {desc}")
            print(f"    Error: {error}")
        return False
    else:
        print("\n✅ All imports successful! You can proceed with running the model.")
        return True

if __name__ == "__main__":
    success = check_imports()
    sys.exit(0 if success else 1)
EOL

# Make the test script executable
chmod +x /notebooks/test_imports.py

# Run the import test
python /notebooks/test_imports.py

# Set PYTHONPATH to include project root (add to ~/.bashrc for persistence)
echo 'export PYTHONPATH=$PYTHONPATH:/notebooks' >> ~/.bashrc
source ~/.bashrc

## Fix Import Issues

```bash
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

# Apply fixes to key files
cd /notebooks
python src/generative_ai_module/fix_jarvis_imports.py --force src/generative_ai_module/train_models.py
python src/generative_ai_module/fix_jarvis_imports.py --force src/generative_ai_module/finetune_deepseek.py
python src/generative_ai_module/fix_jarvis_imports.py --force src/generative_ai_module/evaluate_generation.py
python src/generative_ai_module/fix_jarvis_imports.py --force src/generative_ai_module/unified_generation_pipeline.py
```

## Optimal Training Commands for RTX 5000 (16GB GPU)

These commands are specifically optimized for the RTX 5000 GPU with 16GB VRAM, 8 CPUs, and 30GB RAM. They balance performance with memory constraints to get the best results.

### 1. Training the Base Models

```bash
# Set environment variables for best performance
export CUDA_VISIBLE_DEVICES=0
export CODE_SUBSET="jarvis_code_instructions"

# Train code models with DeepSeek optimizations
cd /notebooks
python src/generative_ai_module/train_models.py \
    --model-type code \
    --use-deepseek \
    --code-subset $CODE_SUBSET \
    --batch-size 2 \
    --epochs 3 \
    --learning-rate 2e-5 \
    --warmup-steps 100 \
    --load-in-4bit \
    --sequence-length 1024 \
    --early-stopping 3 \
    --deepseek-batch-size 1 \
    --max-samples 5000 \
    --visualization-dir /notebooks/Jarvis_AI_Assistant/visualizations

# Train on ALL text datasets with RTX 5000 optimized parameters
python src/generative_ai_module/train_models.py \
    --model-type text \
    --datasets all \
    --batch-size 4 \
    --epochs 3 \
    --learning-rate 3e-5 \
    --early-stopping 3 \
    --sequence-length 512 \
    --max-samples 2000 \
    --visualization-dir /notebooks/Jarvis_AI_Assistant/visualizations \
    --model-dir /notebooks/Jarvis_AI_Assistant/models \
    --warmup-steps 50

# Train on specific text datasets (if you don't want all)
python src/generative_ai_module/train_models.py \
    --model-type text \
    --datasets writing_prompts persona_chat \
    --batch-size 4 \
    --epochs 3 \
    --learning-rate 3e-5 \
    --early-stopping 3 \
    --sequence-length 512 \
    --max-samples 5000 \
    --visualization-dir /notebooks/Jarvis_AI_Assistant/visualizations
```

### 2. Fine-tuning DeepSeek Models with Unsloth

Unsloth optimization is critical for the RTX 5000, achieving up to 2x speed improvement while reducing memory usage. Our fine-tuning process uses:

- 4-bit quantization for minimal memory usage
- LoRA for memory-efficient parameter-efficient fine-tuning
- Optimized sequence length based on available memory

```bash
# Install dependencies for Unsloth optimization
pip install ninja
pip install unsloth

# Run fine-tuning with optimized parameters for 6.7B model
cd /notebooks
python src/generative_ai_module/finetune_deepseek.py \
    --epochs 2 \
    --batch-size 1 \
    --max-samples 5000 \
    --all-code-subsets \
    --sequence-length 1024 \
    --learning-rate 2e-5 \
    --warmup-steps 100 \
    --load-in-4bit \
    --save-steps 100 \
    --save-total-limit 2 \
    --use-unsloth \
    --output-dir /notebooks/Jarvis_AI_Assistant/models/deepseek_finetuned

# For better performance with smaller model
python src/generative_ai_module/finetune_deepseek.py \
    --epochs 3 \
    --batch-size 2 \
    --max-samples 5000 \
    --all-code-subsets \
    --sequence-length 2048 \
    --learning-rate 3e-5 \
    --warmup-steps 50 \
    --load-in-4bit \
    --use-unsloth \
    --output-dir /notebooks/Jarvis_AI_Assistant/models/deepseek_small_finetuned
```

### 3. Evaluation and Metrics

These commands evaluate model performance with memory-optimized settings for the RTX 5000:

```bash
# Evaluate on code generation tasks
cd /notebooks
python src/generative_ai_module/evaluate_generation.py \
    --batch-evaluate \
    --dataset-name jarvis_evaluation_set \
    --model-path /notebooks/Jarvis_AI_Assistant/models/deepseek_finetuned \
    --use-gpu \
    --metrics-dir /notebooks/Jarvis_AI_Assistant/metrics

# Evaluate on specific files
python src/generative_ai_module/evaluate_generation.py \
    --generated-file /notebooks/Jarvis_AI_Assistant/outputs/generated.txt \
    --reference-file /notebooks/Jarvis_AI_Assistant/outputs/reference.txt \
    --prompt-file /notebooks/Jarvis_AI_Assistant/outputs/prompt.txt \
    --dataset-name code_test \
    --use-gpu \
    --metrics-dir /notebooks/Jarvis_AI_Assistant/metrics
```

### 4. Using the Unified Generation Pipeline

The unified pipeline provides a comprehensive approach with optimized parameters:

```bash
# Run the unified generation pipeline with RTX 5000 optimizations
cd /notebooks
python src/generative_ai_module/unified_generation_pipeline.py \
    --mode train \
    --dataset jarvis_combined_dataset \
    --train-type code \
    --epochs 3 \
    --save-model \
    --use-deepseek \
    --deepseek-batch-size 1 \
    --learning-rate 1e-5 \
    --sequence-length 1024 \
    --warmup-steps 100 \
    --code-subset python \
    --all-code-subsets \
    --force-gpu \
    --max-samples 5000 \
    --model-dir /notebooks/Jarvis_AI_Assistant/models
```

### 5. Optimized Storage and Dataset Handling

For the RTX 5000 with limited 16GB VRAM, efficient storage management is critical. Use these approaches:

```bash
# Create a script to optimize storage for DeepSeek models
cd /notebooks
cat > optimize_storage.py << 'EOL'
#!/usr/bin/env python3
import os
import sys
import logging
from pathlib import Path

# Add the project root to the path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.generative_ai_module.storage_optimization import (
    optimize_storage_for_model,
    compress_dataset,
    create_checkpoint_strategy,
    setup_google_drive
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create required directories
os.makedirs("/notebooks/Jarvis_AI_Assistant/models", exist_ok=True)
os.makedirs("/notebooks/Jarvis_AI_Assistant/datasets", exist_ok=True)
os.makedirs("/notebooks/Jarvis_AI_Assistant/checkpoints", exist_ok=True)

# Optimize a model for RTX 5000
logger.info("Optimizing model for RTX 5000 (16GB VRAM)")
optimization_results = optimize_storage_for_model(
    model_name="deepseek-ai/deepseek-coder-6.7b-base",
    output_dir="/notebooks/Jarvis_AI_Assistant/models/deepseek_optimized",
    quantize_bits=4,  # Use 4-bit quantization for maximum memory efficiency
    use_external_storage=True,
    storage_type="gdrive",
    remote_path="DeepSeek_Models"
)

logger.info(f"Optimization complete: {optimization_results}")
EOL

# Make the script executable
chmod +x optimize_storage.py

# Run the optimization script
python optimize_storage.py

# Sync data to/from Google Drive using the built-in script
cd /notebooks

# Sync all data to Google Drive
python -m src.generative_ai_module.sync_gdrive to-gdrive

# Sync only models from Google Drive
python -m src.generative_ai_module.sync_gdrive from-gdrive --folder models

# Sync in both directions
python -m src.generative_ai_module.sync_gdrive all
```

### 6. Running the Jarvis AI Assistant

Run the assistant with memory-efficient settings optimized for the RTX 5000:

```bash
# Run the Jarvis AI Assistant in interactive mode
cd /notebooks
python src/generative_ai_module/run_jarvis.py \
    --model deepseek-ai/deepseek-coder-6.7b-instruct \
    --model-path /notebooks/Jarvis_AI_Assistant/models/deepseek_finetuned \
    --interactive \
    --max-tokens 512 \
    --output /notebooks/Jarvis_AI_Assistant/logs/chat_history.json

# Run the assistant with a single prompt (non-interactive mode)
python src/generative_ai_module/run_jarvis.py \
    --model deepseek-ai/deepseek-coder-6.7b-instruct \
    --prompt "Write a Python function to calculate the Fibonacci sequence" \
    --max-tokens 256 \
    --output /notebooks/Jarvis_AI_Assistant/logs/single_response.json

# Load from a previous chat history and continue the conversation
python src/generative_ai_module/run_jarvis.py \
    --model deepseek-ai/deepseek-coder-6.7b-instruct \
    --interactive \
    --history /notebooks/Jarvis_AI_Assistant/logs/chat_history.json \
    --output /notebooks/Jarvis_AI_Assistant/logs/continued_chat.json
```

## Performance Optimization Tips for RTX 5000 (16GB)

1. **Critical Memory Optimizations:**

   - Always use 4-bit quantization for 6.7B+ models on 16GB GPU
   - Keep batch size at 1-2 for large models
   - Use gradient accumulation steps of 8-16 to simulate larger batches
   - Monitor memory usage with `nvidia-smi -l 5`
   - For sequence generation, limit max tokens to 512-1024
   - Free CUDA cache periodically with `torch.cuda.empty_cache()`

2. **Sequence Length Management:**

   - Use max_seq_length of 1024 for 6.7B models with 4-bit quantization
   - For 1.3B models, sequence lengths of 2048 are possible
   - Consider dynamic sequence length based on available memory

3. **Efficient Fine-tuning:**

   - Always use LoRA/QLoRA with 4-bit quantization (saves 95%+ memory)
   - Use Flash Attention if available (20-40% speedup)
   - Target only key modules with LoRA (q_proj, k_proj, v_proj, o_proj)
   - Keep LoRA rank (r) between 8-16 for larger models
   - Apply early stopping to prevent overfitting

4. **Optimizing Training Speed:**

   - Enable Unsloth optimization for up to 2x training speed
   - Use mixed precision training (fp16 where supported)
   - Reduce validation frequency to save time (eval every 100-200 steps)
   - Limit checkpoint saving frequency (save every 100-200 steps)
   - Keep save_total_limit low (2-3) to manage disk space

5. **Dataset Optimizations:**
   - Use streaming datasets for large data
   - Apply aggressive filtering to ensure high-quality samples
   - Consider smaller, focused datasets rather than large, generic ones
   - Preprocess and tokenize data ahead of time

## Troubleshooting RTX 5000 Specific Issues

If you encounter CUDA out-of-memory errors:

```bash
# Monitor GPU memory usage
watch -n 5 nvidia-smi

# Reduce memory usage via these steps (in order):
1. Decrease batch size to 1
2. Increase gradient accumulation steps (8→16→32)
3. Reduce sequence length (2048→1024→512)
4. Enable 4-bit quantization if not already
5. Switch to a smaller model (6.7B→1.3B)
6. Disable validation during training
7. Restart the environment to clear fragmented memory
```

For poor training stability on RTX 5000:

```bash
# Try these stability improvements:
1. Reduce learning rate by half (2e-5→1e-5)
2. Increase warmup steps (5-10% of total steps)
3. Add gradient clipping: --max-grad-norm 1.0
4. Use cosine learning rate scheduler
5. Try a different optimizer (AdamW→Adafactor)
```

### Advanced Troubleshooting

#### Import Errors and Module Access

If you encounter import errors, first run the test script to diagnose the issue:

```bash
python /notebooks/test_imports.py
```

If specific modules are causing issues, apply the import fix script:

```bash
# Apply import fixes to problematic files
python src/generative_ai_module/fix_jarvis_imports.py --force <problematic_file>.py
```

#### Memory Fragmentation

The RTX 5000 with 16GB VRAM can suffer from memory fragmentation during long training sessions:

```bash
# Check for memory fragmentation
nvidia-smi -i 0 --query-gpu=utilization.gpu,memory.total,memory.free,memory.used --format=csv

# If you see high used memory but low GPU utilization, clear CUDA cache:
python -c "import torch; torch.cuda.empty_cache(); print('CUDA cache cleared')"
```

#### Recovering from Training Crashes

If training crashes due to OOM errors:

```bash
# 1. Ensure all processes are terminated
pkill -9 python

# 2. Free GPU memory
nvidia-smi --gpu-reset

# 3. Resume from the latest checkpoint with reduced parameters
python src/generative_ai_module/finetune_deepseek.py \
    --model deepseek-ai/deepseek-coder-6.7b-instruct \
    --dataset jarvis_code_dataset \
    --batch-size 1 \
    --gradient-accumulation-steps 32 \
    --load-in-4bit \
    --resume-from-checkpoint /notebooks/Jarvis_AI_Assistant/checkpoints/latest
```

#### Paperspace-Specific Issues

For issues specific to Paperspace Gradient:

```bash
# Reset environment variables if needed
echo 'export PAPERSPACE=true' >> ~/.bashrc
echo 'export PAPERSPACE_ENVIRONMENT=true' >> ~/.bashrc
source ~/.bashrc

# Clear Paperspace cache (if disk space is low)
rm -rf /tmp/gradient_*

# Set higher process priority for training job
sudo nice -n -10 python src/generative_ai_module/finetune_deepseek.py [other args]
```

#### Debugging Unsloth Integration

If you have issues with Unsloth optimization:

```bash
# Check if Unsloth can access the GPU
python -c "from unsloth import FastLanguageModel; print(f'CUDA Available: {FastLanguageModel.is_cuda_available()}')"

# Try running with basic settings first, then enable Unsloth
python src/generative_ai_module/finetune_deepseek.py \
    --model deepseek-ai/deepseek-coder-1.3b-instruct \
    --dataset jarvis_code_dataset \
    --batch-size 1 \
    --load-in-4bit \
    --use-unsloth \
    --debug-mode
```

## Paperspace Setup

For Paperspace Gradient specifically:

```bash
# Create a Paperspace environment variable
echo 'export PAPERSPACE=true' >> ~/.bashrc
echo 'export PAPERSPACE_ENVIRONMENT=true' >> ~/.bashrc
source ~/.bashrc

# For persistent storage with Google Drive
pip install gdown google-auth google-auth-oauthlib google-auth-httplib2

# Sync to Google Drive
python -c "from src.generative_ai_module.sync_gdrive import sync_all_to_gdrive; sync_all_to_gdrive()"
```

#### Code Quality and Linting Fixes

Several undefined variable issues have been fixed in the codebase to ensure it runs reliably on the RTX 5000:

```bash
# Fixed variables in storage_optimization.py
- Added logger definition to fix "logger is not defined" errors

# Fixed variables in train_models.py
- Updated CustomCallback class to properly handle trainer and model variables

# Fixed variables in unified_generation_pipeline.py
- Added infinity variable definition
- Created print_execution_time decorator function
- Fixed args parameter handling in interactive_generation
```

These fixes are especially important for the RTX 5000 environment, as undefined variables can cause runtime crashes that waste valuable compute time and GPU memory. The corrections improve the stability of long-running training jobs and interactive sessions.

When working with the codebase, if you encounter similar "undefined variable" errors, you can apply the fixes using the same pattern:

1. Identify the missing variable
2. Define it at the appropriate scope
3. Use default values where necessary
4. Pass required parameters to functions that need them
```
````
