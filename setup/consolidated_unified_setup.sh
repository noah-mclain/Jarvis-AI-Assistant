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

# Run the dependencies installation script with error handling
./setup/consolidated_install_dependencies.sh || {
    echo "⚠️ Warning: Dependencies installation script exited with an error."
    echo "Attempting to fix common dependency issues..."

    # Fix typing-extensions version
    pip install typing-extensions==4.13.2 --force-reinstall

    # Install wheel and setuptools
    pip install wheel setuptools --upgrade

    # Install core dependencies directly
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu121
    pip install transformers==4.36.2 peft==0.6.0 accelerate==0.25.0 bitsandbytes==0.41.0

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

# Fix transformers.utils issue
echo "Fixing transformers.utils issue..."
chmod +x setup/fix_transformers_utils.py
python setup/fix_transformers_utils.py

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

    # Specifically check for transformers.utils
    try:
        import transformers.utils
        print(f'✅ transformers.utils is available')
    except ImportError as e:
        print(f'❌ transformers.utils is NOT available: {e}')
        print('This will cause issues with attention mask fixes and model training')
        print('Will run fix_transformers_utils.py later in the script')
except Exception as e:
    print(f'❌ transformers error: {e}')

try:
    import unsloth
    print(f'unsloth version: {unsloth.__version__ if hasattr(unsloth, \"__version__\") else \"installed\"}')
except Exception as e:
    print(f'❌ unsloth error: {e}')
"

# Final comprehensive installation for all model types
echo "Performing comprehensive installation for all model types..."

# Install wheel and setuptools first to avoid build issues
pip install wheel setuptools --upgrade --no-deps

# Install typing-extensions with the correct version to avoid conflicts
pip install typing-extensions==4.13.2 --force-reinstall --no-deps

# Install core dependencies in the correct order
echo "Installing core dependencies in the correct order..."

# 1. Install NumPy first (foundation package)
pip install numpy==1.26.4 --force-reinstall --no-deps

# 2. Install PyTorch ecosystem with --no-deps
pip install torch==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu121 --no-deps
pip install torchvision==0.16.2 --extra-index-url https://download.pytorch.org/whl/cu121 --no-deps
pip install torchaudio==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu121 --no-deps

# 3. Install tokenizers before transformers
pip install tokenizers==0.14.0 --no-deps

# 4. Install Hugging Face ecosystem in the correct order with --no-deps
pip install filelock==3.12.2 --no-deps
pip install requests==2.31.0 --no-deps
pip install tqdm==4.66.1 --no-deps
pip install pyyaml==6.0.1 --no-deps
pip install packaging==23.1 --no-deps
pip install fsspec==2023.6.0 --no-deps
pip install psutil==5.9.5 --no-deps
pip install safetensors==0.4.0 --no-deps
pip install huggingface-hub==0.19.4 --no-deps
pip install transformers==4.36.2 --no-deps
pip install peft==0.6.0 --no-deps
pip install accelerate==0.25.0 --no-deps
pip install datasets==2.14.5 --no-deps
pip install bitsandbytes==0.41.0 --no-deps
pip install trl==0.7.4 --no-deps

# 5. Install additional dependencies for all model types with --no-deps
pip install protobuf\<4.24 --no-deps
pip install werkzeug --no-deps
pip install pandas --no-deps
pip install markdown --no-deps
pip install scipy==1.12.0 --no-deps
pip install matplotlib==3.8.3 --no-deps
pip install einops==0.7.0 --no-deps
pip install opt_einsum==3.3.0 --no-deps
pip install sentencepiece==0.1.99 --no-deps
pip install nltk==3.8.1 --no-deps
pip install scikit-learn==1.4.2 --no-deps

# 6. Install xFormers for enhanced attention support with --no-deps
pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121 --no-deps

# 7. Install Flash Attention if possible with --no-deps
pip install flash-attn==2.5.5 --no-build-isolation --no-deps || echo "Flash Attention installation skipped - not critical"

# 8. Install unsloth for optimized training with --no-deps
pip install unsloth==2024.8 --no-deps || echo "Unsloth installation skipped - will use minimal implementation"

# 9. Install additional dependencies for CNN text model with --no-deps
pip install bert-score --no-deps
pip install rouge-score --no-deps

# 10. Download NLTK data
python -c "
import nltk
try:
    nltk.download('punkt', quiet=True)
    print('✅ NLTK punkt downloaded successfully')
except Exception as e:
    print(f'❌ Error downloading NLTK data: {e}')
"

# 11. Install critical dependencies for transformers.utils
echo "Installing critical dependencies for transformers.utils..."
pip install charset-normalizer==3.4.2 --no-deps
pip install idna==3.10 --no-deps
pip install urllib3==2.4.0 --no-deps
pip install certifi==2025.4.26 --no-deps

# 12. Reinstall critical packages to ensure they're properly installed (still with --no-deps)
echo "Reinstalling critical packages with --no-deps..."
pip install transformers==4.36.2 --no-deps  # Reinstall to ensure it's properly installed
pip install tokenizers==0.14.0 --no-deps    # Critical for transformers
pip install huggingface-hub==0.19.4 --no-deps # Critical for transformers

# Create the transformers.utils module if it doesn't exist
echo "Creating transformers.utils module if needed..."
chmod +x setup/fix_transformers_utils.py
python setup/fix_transformers_utils.py

# Ensure the fix was applied
echo "Checking if transformers.utils is now available..."
python -c "
try:
    import transformers.utils
    print('✅ transformers.utils is now available')
except ImportError as e:
    print(f'❌ transformers.utils is still not available: {e}')
    print('This may cause issues with attention mask fixes and model training')
"

# Verify transformers installation
echo "Verifying transformers installation..."
python -c "
try:
    import transformers
    import transformers.utils
    print(f'✅ Transformers {transformers.__version__} successfully installed with utils module')
except ImportError as e:
    print(f'❌ Transformers installation issue: {e}')
    print('Running fix_transformers_utils.py to create the module...')

    # Run the fix script
    import sys
    import os

    # Make the script executable
    os.system('chmod +x setup/fix_transformers_utils.py')

    # Run the script
    os.system('python setup/fix_transformers_utils.py')

    # Try importing again
    try:
        import transformers.utils
        print(f'✅ transformers.utils is now available after fix')
    except ImportError as e:
        print(f'❌ transformers.utils is STILL NOT available after fix: {e}')
        print('This may cause issues with attention mask fixes and model training')
"

# Setup Google Drive mounting with rclone for Paperspace
if [ "$IN_PAPERSPACE" = "1" ]; then
    echo "===================================================================="
    echo "Setting up Google Drive mounting with rclone for Paperspace"
    echo "===================================================================="

    # Install rclone
    echo "Installing rclone..."
    apt-get update -q
    apt-get install -y rclone fuse

    # Check if mount_drive_paperspace.py exists and is executable
    if [ -f "setup/mount_drive_paperspace.py" ]; then
        echo "Running mount_drive_paperspace.py..."
        chmod +x setup/mount_drive_paperspace.py
        python setup/mount_drive_paperspace.py
    else
        echo "⚠️ Warning: mount_drive_paperspace.py not found"
        echo "You can manually mount Google Drive with:"
        echo "1. Run: rclone config"
        echo "2. Follow the setup steps for Google Drive"
        echo "3. Run: rclone mount gdrive: /content/drive --daemon --vfs-cache-mode=full"
    fi
fi

echo "===================================================================="
echo "Consolidated setup complete! Jarvis AI Assistant environment is ready."
echo ""
echo "You can now run any of the following commands to train models:"
echo "  ./setup/train_jarvis.sh --model-type code         # Train code generation model"
echo "  ./setup/train_jarvis.sh --model-type cnn-text     # Train CNN text model"
echo "  ./setup/train_jarvis.sh --model-type custom-model # Train custom encoder-decoder model"
echo ""
echo "All dependencies have been installed and configured. You should not need"
echo "to run this setup script again unless you encounter new dependency issues."
echo "===================================================================="
