#!/bin/bash

echo "===================================================================="
echo "Jarvis AI Assistant - Consolidated Setup Script"
echo "===================================================================="

# Detect environment
IN_COLAB=0
IN_PAPERSPACE=0
if python -c "import google.colab" 2>/dev/null; then
    echo "Running in Google Colab environment"
    IN_COLAB=1
elif [ -d "/notebooks" ] || [ -d "/storage" ]; then
    echo "Running in Paperspace environment"
    IN_PAPERSPACE=1
else
    echo "Running in standard environment"
fi

# Set up Python environment
if [ "$IN_PAPERSPACE" = "0" ] && [ "$IN_COLAB" = "0" ]; then
    echo "Setting up Python environment..."
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
fi

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

# Clean up environment
echo "Performing complete environment cleanup..."

# Uninstall all relevant packages
pip uninstall -y torch torchvision torchaudio
pip uninstall -y numpy scipy matplotlib pandas
pip uninstall -y transformers tokenizers huggingface-hub
pip uninstall -y peft accelerate trl
pip uninstall -y bitsandbytes xformers triton
pip uninstall -y unsloth
pip uninstall -y flash-attn

# Clear cache
pip cache purge
rm -rf ~/.cache/pip
rm -rf ~/.cache/huggingface

# Install dependencies
echo "Installing dependencies..."
chmod +x setup/consolidated_install_dependencies.sh
./setup/consolidated_install_dependencies.sh || {
    echo "⚠️ Warning: Dependencies installation script exited with an error."
    echo "Continuing with setup anyway, but some features may not work correctly."
}

# Configure GPU optimizations
echo "Configuring GPU optimizations..."
mkdir -p ~/.config/accelerate

# Detect GPU type
GPU_TYPE=$(python -c "
try:
    import torch
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
except Exception:
    print('None')
")
echo "Detected GPU: $GPU_TYPE"

if [[ "$GPU_TYPE" == *"A100"* ]]; then
    echo "Applying A100-specific optimizations (BF16)..."
    cat > ~/.config/accelerate/default_config.yaml << EOF
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'yes'
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
mixed_precision: 'bf16'
num_machines: 1
num_processes: 1
use_cpu: false
EOF
else
    echo "Applying RTX-optimized settings (FP16)..."
    cat > ~/.config/accelerate/default_config.yaml << EOF
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'no'
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
mixed_precision: 'fp16'
num_machines: 1
num_processes: 1
use_cpu: false
EOF
fi

# Set and save environment variables
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:256"}
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=true

echo "export PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF" >> ~/.bashrc
echo "export CUDA_LAUNCH_BLOCKING=0" >> ~/.bashrc
echo "export TOKENIZERS_PARALLELISM=true" >> ~/.bashrc

# Fix Unsloth
echo "Setting up minimal Unsloth implementation..."
chmod +x setup/consolidated_fix_unsloth.sh
./setup/consolidated_fix_unsloth.sh

# Fix spaCy
echo "Setting up spaCy with minimal tokenizer..."
chmod +x setup/consolidated_fix_spacy.sh
./setup/consolidated_fix_spacy.sh

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

# Create import_fix.py
echo "Creating import_fix.py..."
mkdir -p src/generative_ai_module
cat > src/generative_ai_module/import_fix.py << 'EOL'
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

# Verify installation
echo "Verifying installation..."
python -c "
import sys
print(f'Python version: {sys.version}')

try:
    import numpy
    print(f'NumPy version: {numpy.__version__}')
except Exception as e:
    print(f'❌ NumPy error: {e}')

try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA version: {torch.version.cuda}')
        print(f'GPU: {torch.cuda.get_device_name(0)}')
except Exception as e:
    print(f'❌ PyTorch error: {e}')

try:
    import transformers
    print(f'transformers version: {transformers.__version__}')
except Exception as e:
    print(f'❌ transformers error: {e}')

try:
    import unsloth
    print(f'unsloth version: {unsloth.__version__ if hasattr(unsloth, \"__version__\") else \"installed\"}')
except Exception as e:
    print(f'❌ unsloth error: {e}')
"

# Final fallback installation for critical dependencies
echo "Performing final fallback installation for critical dependencies..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.36.2 peft==0.6.0 accelerate==0.25.0 bitsandbytes==0.41.0
pip install protobuf\<4.24 werkzeug pandas huggingface-hub markdown

echo "===================================================================="
echo "Consolidated setup complete! Jarvis AI Assistant environment is ready."
echo "===================================================================="
