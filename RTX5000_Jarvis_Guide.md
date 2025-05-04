# Jarvis AI Assistant Implementation Guide for RTX 5000 (16GB GPU)

This guide provides optimized commands and best practices for running the Jarvis AI Assistant on a Paperspace Gradient instance with an NVIDIA RTX 5000 GPU.

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

These commands are optimized for the RTX 5000 GPU with 16GB memory. They balance performance with memory constraints.

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
    --batch-size 4 \
    --epochs 3 \
    --learning-rate 2e-5 \
    --warmup-steps 100 \
    --load-in-4bit \
    --sequence-length 1024 \
    --early-stopping 3

# Train text generation models
python src/generative_ai_module/train_models.py \
    --model-type text \
    --datasets writing_prompts persona_chat \
    --batch-size 8 \
    --epochs 3 \
    --learning-rate 3e-5 \
    --early-stopping 3 \
    --sequence-length 512
```

### 2. Fine-tuning DeepSeek Models

The DeepSeek fine-tuning process is optimized for the RTX 5000. We use:

- 4-bit quantization to minimize memory usage
- Gradient accumulation to simulate larger batch sizes
- LoRA for efficient fine-tuning
- Optimized sequence length

```bash
# Install dependencies for Unsloth (if not already installed)
pip install ninja
pip install unsloth

# Run the fine-tuning process
cd /notebooks
python src/generative_ai_module/finetune_deepseek.py \
    --model deepseek-coder-6.7b-instruct \
    --dataset jarvis_code_dataset \
    --batch-size 2 \
    --gradient-accumulation-steps 16 \
    --learning-rate 2e-5 \
    --epochs 3 \
    --lora-alpha 16 \
    --lora-dropout 0.05 \
    --lora-r 16 \
    --max-seq-length 2048 \
    --quantization 4bit \
    --save-steps 200 \
    --eval-steps 200 \
    --warmup-ratio 0.03

# For smaller model (if memory is still an issue)
python src/generative_ai_module/finetune_deepseek.py \
    --model deepseek-coder-1.3b-instruct \
    --dataset jarvis_code_dataset \
    --batch-size 4 \
    --gradient-accumulation-steps 8 \
    --learning-rate 3e-5 \
    --epochs 5 \
    --lora-alpha 16 \
    --lora-dropout 0.05 \
    --lora-r 32 \
    --max-seq-length 2048 \
    --quantization 4bit
```

### 3. Evaluation and Metrics

These commands evaluate model performance with comprehensive metrics on various datasets:

```bash
# Evaluate on code generation tasks
cd /notebooks
python src/generative_ai_module/evaluate_generation.py \
    --model deepseek-coder-6.7b-instruct \
    --dataset jarvis_evaluation_set \
    --num-examples 50 \
    --batch-size 2 \
    --save-results \
    --metrics rouge bleu \
    --visualization-dir /notebooks/Jarvis_AI_Assistant/visualizations \
    --quantization 4bit

# Evaluate on conversational tasks
python src/generative_ai_module/evaluate_generation.py \
    --model deepseek-coder-6.7b-instruct \
    --dataset persona_chat \
    --num-examples 30 \
    --batch-size 2 \
    --save-results \
    --metrics rouge bleu \
    --visualization-dir /notebooks/Jarvis_AI_Assistant/visualizations \
    --quantization 4bit
```

### 4. Advanced Training with UnifiedGenerationPipeline

This pipeline combines multiple steps and provides comprehensive training and visualization:

```bash
# Run the unified generation pipeline with RTX 5000 optimizations
cd /notebooks
python src/generative_ai_module/unified_generation_pipeline.py \
    --model deepseek-coder-6.7b-instruct \
    --dataset jarvis_combined_dataset \
    --batch-size 2 \
    --gradient-accumulation-steps 16 \
    --learning-rate 2e-5 \
    --epochs 3 \
    --quantization 4bit \
    --lora \
    --lora-r 16 \
    --lora-alpha 16 \
    --sequence-length 2048 \
    --visualization \
    --evaluation \
    --save-checkpoints \
    --checkpoint-dir /notebooks/Jarvis_AI_Assistant/checkpoints \
    --visualization-dir /notebooks/Jarvis_AI_Assistant/visualizations
```

### 5. Run Jarvis AI Assistant

Run the Jarvis AI Assistant in a production-like environment:

```bash
# Run the Jarvis AI Assistant with web UI
cd /notebooks
python src/generative_ai_module/run_jarvis.py \
    --model deepseek-coder-6.7b-instruct \
    --quantization 4bit \
    --port 7860 \
    --share \
    --optimize \
    --cache-dir /notebooks/Jarvis_AI_Assistant/models \
    --log-level info
```

## Performance Optimization Tips for RTX 5000

1. **Memory Management:**

   - Use 4-bit quantization for large models
   - Keep batch size ≤ 4 for 6.7B+ models
   - Use gradient accumulation (8-16 steps) to simulate larger batches
   - Monitor memory usage with `nvidia-smi`

2. **Training Efficiency:**

   - Use LoRA for fine-tuning (saves 95%+ memory)
   - Warm up learning rate for first ~10% of steps
   - Use early stopping to prevent overfitting
   - Optimize sequence length (1024-2048 for code)

3. **Model Selection:**

   - DeepSeek-Coder 1.3B works well with larger batches
   - DeepSeek-Coder 6.7B-Instruct has better quality but needs quantization
   - Consider Unsloth optimization for 2x faster training

4. **Data Efficiency:**
   - Use high-quality, domain-specific datasets
   - Apply data preprocessing to remove low-quality samples
   - Consider augmentation techniques

## Troubleshooting

If you encounter memory issues:

```bash
# Check GPU memory usage
watch -n 1 nvidia-smi

# Reduce memory usage by:
1. Decreasing batch size
2. Reducing sequence length
3. Using 4-bit quantization
4. Using a smaller model
5. Increasing gradient accumulation
```

If you encounter import errors:

```bash
# Apply the fix_jarvis_imports.py script
python src/generative_ai_module/fix_jarvis_imports.py --force <problematic_file>.py
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
