#!/bin/bash

echo "===================================================================="
echo "Jarvis AI Assistant - Consolidated Setup Script"
echo "===================================================================="

# Fix all string literal issues first
echo "Fixing all string literal issues in Python files..."
chmod +x setup/fix_all_string_literals.py
python setup/fix_all_string_literals.py

# Copy the new ultimate attention fix to ensure it's available
echo "Setting up new ultimate attention fix..."
cp setup/ultimate_attention_fix_new.py setup/ultimate_attention_fix.py
chmod +x setup/ultimate_attention_fix.py

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

# Clean up invalid distributions (like ~ransformers)
echo "Cleaning up invalid distributions..."
python -c "
import os
import glob
import shutil
import site
import subprocess

# Get site-packages directory
site_packages = site.getsitepackages()[0]
print(f'Site packages directory: {site_packages}')

# Check for common invalid distributions
invalid_dirs = glob.glob(os.path.join(site_packages, '~*'))
if invalid_dirs:
    print(f'Found {len(invalid_dirs)} invalid distribution(s):')
    for d in invalid_dirs:
        print(f'  - {os.path.basename(d)}')
        try:
            if os.path.isdir(d):
                shutil.rmtree(d)
            else:
                os.remove(d)
            print(f'    ✓ Removed successfully')
        except Exception as e:
            print(f'    ✗ Failed to remove: {e}')
            # Try with system command as fallback
            subprocess.run(['rm', '-rf', d])
else:
    print('No invalid distributions found.')

# Specifically check for ~ransformers
ransformers_path = os.path.join(site_packages, '~ransformers')
if os.path.exists(ransformers_path):
    print(f'Found invalid distribution: ~ransformers')
    try:
        if os.path.isdir(ransformers_path):
            shutil.rmtree(ransformers_path)
        else:
            os.remove(ransformers_path)
        print(f'✓ Removed ~ransformers successfully')
    except Exception as e:
        print(f'✗ Failed to remove ~ransformers: {e}')
        # Try with system command as fallback
        subprocess.run(['rm', '-rf', ransformers_path])

# Check for any transformers.dist-info directories that might be corrupted
transformers_info_dirs = glob.glob(os.path.join(site_packages, 'transformers-*.dist-info'))
if transformers_info_dirs:
    print(f'Found {len(transformers_info_dirs)} transformers.dist-info directories:')
    for d in transformers_info_dirs:
        print(f'  - {os.path.basename(d)}')
        try:
            shutil.rmtree(d)
            print(f'    ✓ Removed successfully to prepare for clean reinstall')
        except Exception as e:
            print(f'    ✗ Failed to remove: {e}')
            # Try with system command as fallback
            subprocess.run(['rm', '-rf', d])
"

# Install joblib first to ensure it's available early
echo "Installing joblib first..."
pip install joblib==1.3.2

# Verify joblib installation
python -c "
try:
    import joblib
    print(f'✅ joblib version: {joblib.__version__}')
except ImportError as e:
    print(f'❌ joblib error: {e}')
    print('Installing joblib with pip...')
    import os
    os.system('pip install joblib==1.3.2')
    try:
        import joblib
        print(f'✅ joblib version after reinstall: {joblib.__version__}')
    except ImportError:
        print('❌ Failed to install joblib')
"

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

# Fix TRL/PEFT and spaCy imports
echo "Fixing TRL/PEFT and spaCy import issues..."
chmod +x setup/fix_trl_spacy_imports.py
python setup/fix_trl_spacy_imports.py

# Fix transformers.utils issue
echo "Fixing transformers.utils issue..."
chmod +x setup/fix_transformers_utils.py
python setup/fix_transformers_utils.py

# Apply fixes for DeepSeek models
echo "Applying fixes for DeepSeek models..."

# Fix bitsandbytes version for 4-bit quantization
echo "Checking bitsandbytes version for 4-bit quantization compatibility..."
chmod +x setup/fix_bitsandbytes_version.py
python setup/fix_bitsandbytes_version.py

# Fix unsloth trust_remote_code issue
echo "Fixing unsloth trust_remote_code issue..."
chmod +x setup/fix_unsloth_trust_remote_code.py
python setup/fix_unsloth_trust_remote_code.py

# Apply attention mask fixes
echo "Applying attention mask fixes..."

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

try:
    import joblib
    print(f'joblib version: {joblib.__version__}')
except Exception as e:
    print(f'❌ joblib error: {e}')

try:
    import sklearn
    print(f'scikit-learn version: {sklearn.__version__}')
except Exception as e:
    print(f'❌ scikit-learn error: {e}')
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

# Check for ~ransformers again before installing transformers
python -c "
import os
import site
import shutil
import subprocess

site_packages = site.getsitepackages()[0]
ransformers_path = os.path.join(site_packages, '~ransformers')
if os.path.exists(ransformers_path):
    print(f'Found invalid distribution: ~ransformers before transformers installation')
    try:
        if os.path.isdir(ransformers_path):
            shutil.rmtree(ransformers_path)
        else:
            os.remove(ransformers_path)
        print(f'✓ Removed ~ransformers successfully')
    except Exception as e:
        print(f'✗ Failed to remove ~ransformers: {e}')
        # Try with system command as fallback
        subprocess.run(['rm', '-rf', ransformers_path])
"

# Install transformers with --no-deps
pip install transformers==4.36.2 --no-deps
pip install peft==0.6.0 --no-deps
pip install accelerate==0.25.0 --no-deps
pip install datasets==2.14.5 --no-deps
pip install bitsandbytes==0.43.2 --no-deps  # Install the latest available version for best compatibility with 4-bit quantization
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
pip install joblib==1.3.2 --no-deps  # Required by scikit-learn
pip install scikit-learn==1.4.2 --no-deps
pip install tensorboard==2.15.2 --no-deps  # Required for training visualization

# Verify joblib and scikit-learn installation
python -c "
try:
    import joblib
    print(f'✅ joblib version: {joblib.__version__}')
except ImportError:
    print('❌ joblib not found, reinstalling...')
    import os
    os.system('pip install joblib==1.3.2')
    try:
        import joblib
        print(f'✅ joblib version after reinstall: {joblib.__version__}')
    except ImportError:
        print('❌ Failed to install joblib')

try:
    import sklearn
    print(f'✅ scikit-learn version: {sklearn.__version__}')
except ImportError:
    print('❌ scikit-learn not found')
"

# Fix bitsandbytes installation
echo "Fixing bitsandbytes installation..."
pip uninstall -y bitsandbytes
pip install bitsandbytes --no-deps  # Install the latest available version for best compatibility with 4-bit quantization

# Create a version attribute for bitsandbytes if it doesn't exist
python -c "
import sys
import importlib.util
import os

try:
    import bitsandbytes
    if not hasattr(bitsandbytes, '__version__'):
        print('Adding __version__ attribute to bitsandbytes')
        # Find the bitsandbytes package location
        spec = importlib.util.find_spec('bitsandbytes')
        if spec and spec.origin:
            init_path = os.path.join(os.path.dirname(spec.origin), '__init__.py')

            # Read the current content
            with open(init_path, 'r') as f:
                content = f.read()

            # Add version if not already there
            if '__version__' not in content:
                # Get the installed version from pip
                import subprocess
                try:
                    pip_output = subprocess.check_output([sys.executable, '-m', 'pip', 'show', 'bitsandbytes']).decode('utf-8')
                    version_line = [line for line in pip_output.split('\\n') if line.startswith('Version:')]
                    if version_line:
                        version = version_line[0].split(':', 1)[1].strip()
                    else:
                        version = "0.42.0"  # Default if not found
                except Exception:
                    version = "0.42.0"  # Default if command fails

                with open(init_path, 'a') as f:
                    f.write(f'\\n\\n# Added by setup script\\n__version__ = \"{version}\"\\n')
                print(f'✅ Added __version__ attribute to bitsandbytes: {version}')

                # Reload the module to apply changes
                import importlib
                importlib.reload(bitsandbytes)
                print(f'✅ bitsandbytes version: {bitsandbytes.__version__}')
            else:
                print('__version__ attribute already exists in bitsandbytes')
    else:
        print(f'✅ bitsandbytes version: {bitsandbytes.__version__}')
except Exception as e:
    print(f'❌ Error fixing bitsandbytes: {e}')
"

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

# 11. Install critical dependencies for transformers.utils and scikit-learn
echo "Installing critical dependencies for transformers.utils and scikit-learn..."
pip install charset-normalizer==3.4.2 --no-deps
pip install idna==3.10 --no-deps
pip install urllib3==2.4.0 --no-deps
pip install certifi==2025.4.26 --no-deps
pip install threadpoolctl==3.2.0 --no-deps  # Required by scikit-learn
pip install joblib==1.3.2 --no-deps  # Reinstall joblib to ensure it's available

# 12. Reinstall critical packages to ensure they're properly installed
echo "Reinstalling critical packages..."
pip install transformers==4.36.2 --no-deps  # Reinstall to ensure it's properly installed
pip install tokenizers==0.14.0 --no-deps    # Critical for transformers
pip install huggingface-hub==0.19.4 --no-deps # Critical for transformers

# Install joblib with dependencies to ensure scikit-learn works properly
echo "Installing joblib with dependencies..."
pip install joblib==1.3.2  # Install with dependencies
pip install threadpoolctl==3.2.0  # Required by scikit-learn

# Final check for ~ransformers issue and critical dependencies
echo "Final check for ~ransformers issue and critical dependencies..."
python -c "
import os
import sys
import site
import shutil
import subprocess
import glob

site_packages = site.getsitepackages()[0]

# Check for ~ransformers
ransformers_path = os.path.join(site_packages, '~ransformers')
if os.path.exists(ransformers_path):
    print(f'Found invalid distribution: ~ransformers after installation')
    try:
        if os.path.isdir(ransformers_path):
            shutil.rmtree(ransformers_path)
        else:
            os.remove(ransformers_path)
        print(f'✓ Removed ~ransformers successfully')
    except Exception as e:
        print(f'✗ Failed to remove ~ransformers: {e}')
        # Try with system command as fallback
        subprocess.run(['rm', '-rf', ransformers_path])

# Check for any invalid egg-info or dist-info directories
invalid_info_dirs = glob.glob(os.path.join(site_packages, '~*.egg-info')) + \
                   glob.glob(os.path.join(site_packages, '~*.dist-info'))
if invalid_info_dirs:
    print(f'Found {len(invalid_info_dirs)} invalid info directories:')
    for d in invalid_info_dirs:
        print(f'  - {os.path.basename(d)}')
        try:
            shutil.rmtree(d)
            print(f'    ✓ Removed successfully')
        except Exception as e:
            print(f'    ✗ Failed to remove: {e}')
            # Try with system command as fallback
            subprocess.run(['rm', '-rf', d])

# Final check for critical dependencies
print('\\nFinal check for critical dependencies:')
try:
    import joblib
    print(f'✅ joblib version: {joblib.__version__}')
except ImportError as e:
    print(f'❌ joblib error: {e}')
    print('Installing joblib with pip...')
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'joblib==1.3.2'])

try:
    import sklearn
    from sklearn.utils import _joblib
    print(f'✅ scikit-learn version: {sklearn.__version__}')
    print(f'✅ sklearn.utils._joblib is available')
except ImportError as e:
    print(f'❌ scikit-learn error: {e}')
    print('Installing scikit-learn dependencies...')
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'threadpoolctl==3.2.0', 'joblib==1.3.2'])
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'scikit-learn==1.4.2'])

try:
    import transformers
    import transformers.utils
    print(f'✅ transformers version: {transformers.__version__}')
    print(f'✅ transformers.utils is available')
except ImportError as e:
    print(f'❌ transformers error: {e}')
"

# Create the transformers.utils module if it doesn't exist
echo "Creating transformers.utils module if needed..."
chmod +x setup/fix_transformers_utils.py
python setup/fix_transformers_utils.py

# Fix DeepSeek model in transformers
echo "Fixing DeepSeek model in transformers..."
chmod +x setup/fix_deepseek_model.py
python setup/fix_deepseek_model.py

# Verify DeepSeek model is available
python -c "
try:
    from transformers.models import deepseek
    from transformers.models.deepseek import DeepSeekModel, DeepSeekConfig, DeepSeekAttention
    print('✅ DeepSeek model is available in transformers')
except ImportError as e:
    print(f'❌ DeepSeek model is not available in transformers: {e}')
    print('Please run setup/fix_deepseek_model.py manually')
"

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

    # Interactive rclone configuration
    echo "===================================================================="
    echo "Interactive rclone configuration for Google Drive"
    echo "===================================================================="
    echo "You will now be guided through the rclone configuration process."
    echo "Please follow these steps:"
    echo "1. Select 'n' for New remote"
    echo "2. Enter 'gdrive' as the name"
    echo "3. Select the number for 'Google Drive'"
    echo "4. For client_id and client_secret, just press Enter to use the defaults"
    echo "5. Select 'scope' option 1 (full access)"
    echo "6. For root_folder_id, just press Enter"
    echo "7. For service_account_file, just press Enter"
    echo "8. Select 'y' to edit advanced config if you need to, otherwise 'n'"
    echo "9. Select 'y' to use auto config"
    echo "10. Follow the browser authentication steps when prompted"
    echo "11. Select 'y' to confirm the configuration is correct"
    echo "12. Select 'q' to quit the config process when done"
    echo "===================================================================="
    echo "Starting rclone config now..."
    echo "Press Enter to continue"
    read -p ""

    # Run rclone config interactively
    rclone config

    # Verify rclone configuration
    echo "Verifying rclone configuration..."
    if rclone listremotes | grep -q "gdrive:"; then
        echo "✅ Google Drive remote 'gdrive:' configured successfully"
    else
        echo "⚠️ Google Drive remote 'gdrive:' not found in rclone config"
        echo "You can manually configure it later with: rclone config"
    fi

    # Create mount point for Google Drive
    DRIVE_MOUNT_POINT="/notebooks/drive"
    mkdir -p "$DRIVE_MOUNT_POINT"

    # Mount Google Drive
    echo "Mounting Google Drive..."
    if rclone listremotes | grep -q "gdrive:"; then
        # Mount in the background
        rclone mount gdrive: "$DRIVE_MOUNT_POINT" --daemon --vfs-cache-mode=full --vfs-cache-max-size=1G --dir-cache-time=1h --buffer-size=32M --transfers=4 --checkers=8 --drive-chunk-size=32M --timeout=1h --umask=000

        # Wait for the mount to be ready
        echo "Waiting for Google Drive to be mounted..."
        for i in {1..10}; do
            if mountpoint -q "$DRIVE_MOUNT_POINT"; then
                echo "✅ Google Drive mounted successfully at $DRIVE_MOUNT_POINT"
                break
            fi
            echo "Waiting... ($i/10)"
            sleep 1
        done

        if ! mountpoint -q "$DRIVE_MOUNT_POINT"; then
            echo "⚠️ Google Drive mount not detected. Will try to continue anyway."
        fi
    else
        echo "⚠️ Skipping Google Drive mount as 'gdrive:' remote was not configured"
    fi

    # Define Jarvis AI Assistant directory structure
    JARVIS_DIR="$DRIVE_MOUNT_POINT/My Drive/Jarvis_AI_Assistant"

    # Create Jarvis directory structure in Google Drive
    mkdir -p "$JARVIS_DIR/checkpoints"
    mkdir -p "$JARVIS_DIR/datasets"
    mkdir -p "$JARVIS_DIR/models"
    mkdir -p "$JARVIS_DIR/logs"
    mkdir -p "$JARVIS_DIR/metrics"
    mkdir -p "$JARVIS_DIR/preprocessed_data"
    mkdir -p "$JARVIS_DIR/visualizations"

    # Create symbolic links to the Jarvis directories
    echo "Creating symbolic links to Google Drive directories..."
    ln -sf "$JARVIS_DIR/checkpoints" /notebooks/Jarvis_AI_Assistant/checkpoints
    ln -sf "$JARVIS_DIR/datasets" /notebooks/Jarvis_AI_Assistant/datasets
    ln -sf "$JARVIS_DIR/models" /notebooks/Jarvis_AI_Assistant/models
    ln -sf "$JARVIS_DIR/logs" /notebooks/Jarvis_AI_Assistant/logs
    ln -sf "$JARVIS_DIR/metrics" /notebooks/Jarvis_AI_Assistant/metrics
    ln -sf "$JARVIS_DIR/preprocessed_data" /notebooks/Jarvis_AI_Assistant/preprocessed_data
    ln -sf "$JARVIS_DIR/visualizations" /notebooks/Jarvis_AI_Assistant/visualizations

    # Create a test file to verify the mount is working
    echo "Testing Google Drive mount..." > "$JARVIS_DIR/mount_test.txt"
    if [ -f "$JARVIS_DIR/mount_test.txt" ]; then
        echo "✅ Successfully wrote test file to Google Drive"
    else
        echo "⚠️ Could not write test file to Google Drive"
    fi

    # Check if mount_drive_paperspace.py exists and is executable
    if [ -f "setup/mount_drive_paperspace.py" ]; then
        echo "Running mount_drive_paperspace.py..."
        chmod +x setup/mount_drive_paperspace.py
        python setup/mount_drive_paperspace.py
    else
        echo "Creating mount_drive_paperspace.py..."
        cat > setup/mount_drive_paperspace.py << 'EOL'
#!/usr/bin/env python3
"""
Mount Google Drive in Paperspace using rclone.
This script automates the process of mounting Google Drive in Paperspace.
"""

import os
import subprocess
import time
import sys

def mount_google_drive():
    """Mount Google Drive using rclone."""
    print("Mounting Google Drive...")

    # Check if rclone is configured
    result = subprocess.run(["rclone", "listremotes"], capture_output=True, text=True)
    if "gdrive:" not in result.stdout:
        print("Google Drive remote not found in rclone config.")
        print("Would you like to configure rclone now? (y/n)")
        choice = input().strip().lower()
        if choice == 'y':
            print("\nPlease follow these steps:")
            print("1. Select 'n' for New remote")
            print("2. Enter 'gdrive' as the name")
            print("3. Select the number for 'Google Drive'")
            print("4. For client_id and client_secret, just press Enter to use the defaults")
            print("5. Select 'scope' option 1 (full access)")
            print("6. For root_folder_id, just press Enter")
            print("7. For service_account_file, just press Enter")
            print("8. Select 'y' to edit advanced config if you need to, otherwise 'n'")
            print("9. Select 'y' to use auto config")
            print("10. Follow the browser authentication steps when prompted")
            print("11. Select 'y' to confirm the configuration is correct")
            print("12. Select 'q' to quit the config process when done")
            print("\nStarting rclone config now...\n")

            # Run rclone config
            subprocess.run(["rclone", "config"])

            # Check if configuration was successful
            result = subprocess.run(["rclone", "listremotes"], capture_output=True, text=True)
            if "gdrive:" not in result.stdout:
                print("Google Drive remote still not found in rclone config.")
                print("Please run 'rclone config' manually later.")
                return False
            else:
                print("✅ Google Drive remote 'gdrive:' configured successfully")
        else:
            print("Please run 'rclone config' to set up Google Drive remote.")
            print("Follow the prompts to create a new remote named 'gdrive' for Google Drive.")
            return False

    # Create mount point
    mount_point = "/notebooks/drive"
    os.makedirs(mount_point, exist_ok=True)

    # Check if already mounted
    if os.path.ismount(mount_point):
        print(f"Google Drive is already mounted at {mount_point}")
        return True

    # Mount Google Drive
    cmd = [
        "rclone", "mount",
        "gdrive:", mount_point,
        "--daemon",
        "--vfs-cache-mode=full",
        "--vfs-cache-max-size=1G",
        "--dir-cache-time=1h",
        "--buffer-size=32M",
        "--transfers=4",
        "--checkers=8",
        "--drive-chunk-size=32M",
        "--timeout=1h",
        "--umask=000"
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"Google Drive mounted at {mount_point}")

        # Wait for the mount to be ready
        for _ in range(10):
            if os.path.ismount(mount_point):
                break
            time.sleep(1)

        if os.path.ismount(mount_point):
            print("Mount successful!")

            # Create Jarvis directory structure
            jarvis_dir = os.path.join(mount_point, "My Drive/Jarvis_AI_Assistant")
            os.makedirs(jarvis_dir, exist_ok=True)

            # Create subdirectories
            subdirs = [
                "checkpoints", "datasets", "models", "logs",
                "metrics", "preprocessed_data", "visualizations"
            ]

            for subdir in subdirs:
                os.makedirs(os.path.join(jarvis_dir, subdir), exist_ok=True)

            # Create symbolic links
            for subdir in subdirs:
                source = os.path.join(jarvis_dir, subdir)
                target = os.path.join("/notebooks/Jarvis_AI_Assistant", subdir)

                # Remove existing link or directory
                if os.path.islink(target):
                    os.unlink(target)
                elif os.path.isdir(target):
                    os.system(f"rm -rf {target}")

                # Create symbolic link
                os.symlink(source, target)
                print(f"Created symbolic link: {target} -> {source}")

            # Create a test file
            with open(os.path.join(jarvis_dir, "mount_test.txt"), "w") as f:
                f.write("Google Drive mount test successful!")

            return True
        else:
            print("Failed to mount Google Drive. Mount point is not a mount.")
            return False
    except subprocess.CalledProcessError as e:
        print(f"Failed to mount Google Drive: {e}")
        return False

if __name__ == "__main__":
    success = mount_google_drive()
    sys.exit(0 if success else 1)
EOL

        chmod +x setup/mount_drive_paperspace.py
        python setup/mount_drive_paperspace.py
    fi

    # Verify the mount is working
    if [ -f "$JARVIS_DIR/mount_test.txt" ]; then
        echo "✅ Google Drive mounted successfully!"
    else
        echo "⚠️ Warning: Google Drive mount verification failed"
        echo "You can manually mount Google Drive with:"
        echo "1. Run: rclone config"
        echo "2. Follow the setup steps for Google Drive"
        echo "3. Run: rclone mount gdrive: /notebooks/drive --daemon --vfs-cache-mode=full"
    fi

    # Update environment variables to use Google Drive paths
    echo "export JARVIS_DRIVE_DIR=$JARVIS_DIR" >> ~/.bashrc
    echo "export JARVIS_CHECKPOINTS_DIR=$JARVIS_DIR/checkpoints" >> ~/.bashrc
    echo "export JARVIS_DATASETS_DIR=$JARVIS_DIR/datasets" >> ~/.bashrc
    echo "export JARVIS_MODELS_DIR=$JARVIS_DIR/models" >> ~/.bashrc
    echo "export JARVIS_LOGS_DIR=$JARVIS_DIR/logs" >> ~/.bashrc
    echo "export JARVIS_METRICS_DIR=$JARVIS_DIR/metrics" >> ~/.bashrc
    echo "export JARVIS_PREPROCESSED_DATA_DIR=$JARVIS_DIR/preprocessed_data" >> ~/.bashrc
    echo "export JARVIS_VISUALIZATIONS_DIR=$JARVIS_DIR/visualizations" >> ~/.bashrc

    # Export variables for current session
    export JARVIS_DRIVE_DIR=$JARVIS_DIR
    export JARVIS_CHECKPOINTS_DIR=$JARVIS_DIR/checkpoints
    export JARVIS_DATASETS_DIR=$JARVIS_DIR/datasets
    export JARVIS_MODELS_DIR=$JARVIS_DIR/models
    export JARVIS_LOGS_DIR=$JARVIS_DIR/logs
    export JARVIS_METRICS_DIR=$JARVIS_DIR/metrics
    export JARVIS_PREPROCESSED_DATA_DIR=$JARVIS_DIR/preprocessed_data
    export JARVIS_VISUALIZATIONS_DIR=$JARVIS_DIR/visualizations
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
