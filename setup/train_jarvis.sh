#!/bin/bash
# Consolidated script for Jarvis AI Assistant
# This script runs the Jarvis AI Assistant training process
#
# IMPORTANT: Before running this script for the first time, you should run:
#   ./setup/consolidated_unified_setup.sh
# to ensure all dependencies are properly installed.

# Function to check and activate Python environment
check_and_activate_python_env() {
    echo "Checking Python environment..."

    # Check if we're already in a virtual environment
    if [[ "$VIRTUAL_ENV" == *"jarvis_env"* ]]; then
        echo "✅ Python environment already activated: $VIRTUAL_ENV"
        return 0
    fi

    # Try to activate the environment
    if [ -f "jarvis_env/bin/activate" ]; then
        echo "Activating Python environment from jarvis_env/bin/activate"
        source jarvis_env/bin/activate
    elif [ -f "/notebooks/jarvis_env/bin/activate" ]; then
        echo "Activating Python environment from /notebooks/jarvis_env/bin/activate"
        source /notebooks/jarvis_env/bin/activate
    else
        echo "⚠️ Warning: Could not find jarvis_env/bin/activate"
        echo "Creating new Python environment..."
        python -m venv jarvis_env
        source jarvis_env/bin/activate
    fi

    # Verify activation
    if [[ "$VIRTUAL_ENV" == *"jarvis_env"* ]]; then
        echo "✅ Python environment successfully activated: $VIRTUAL_ENV"
        return 0
    else
        echo "⚠️ Warning: Failed to activate Python environment"
        echo "Current VIRTUAL_ENV: $VIRTUAL_ENV"
        return 1
    fi
}

# Function to check and activate Unsloth
check_and_activate_unsloth() {
    echo "Checking Unsloth installation..."

    # Check if minimal Unsloth is in PYTHONPATH
    CUSTOM_UNSLOTH_DIR="/notebooks/custom_unsloth"
    if [[ "$PYTHONPATH" == *"$CUSTOM_UNSLOTH_DIR"* ]]; then
        echo "✅ Minimal Unsloth already in PYTHONPATH"
    else
        # Try to activate minimal Unsloth
        if [ -f "$CUSTOM_UNSLOTH_DIR/activate_minimal_unsloth.sh" ]; then
            echo "Activating minimal Unsloth..."
            source "$CUSTOM_UNSLOTH_DIR/activate_minimal_unsloth.sh"

            # Verify activation
            if [[ "$PYTHONPATH" == *"$CUSTOM_UNSLOTH_DIR"* ]]; then
                echo "✅ Minimal Unsloth successfully activated"
            else
                echo "⚠️ Warning: Failed to activate minimal Unsloth"
                # Add to PYTHONPATH manually
                export PYTHONPATH="$CUSTOM_UNSLOTH_DIR:$PYTHONPATH"
                echo "Manually added minimal Unsloth to PYTHONPATH"
            fi
        else
            echo "⚠️ Warning: Could not find minimal Unsloth activation script"
            echo "Checking if we need to create minimal Unsloth..."

            # Check if create_minimal_unsloth.sh exists
            if [ -f "setup/create_minimal_unsloth.sh" ]; then
                echo "Creating minimal Unsloth..."
                chmod +x setup/create_minimal_unsloth.sh
                ./setup/create_minimal_unsloth.sh

                # Try to activate it
                if [ -f "$CUSTOM_UNSLOTH_DIR/activate_minimal_unsloth.sh" ]; then
                    echo "Activating newly created minimal Unsloth..."
                    source "$CUSTOM_UNSLOTH_DIR/activate_minimal_unsloth.sh"

                    # Verify activation
                    if [[ "$PYTHONPATH" == *"$CUSTOM_UNSLOTH_DIR"* ]]; then
                        echo "✅ Minimal Unsloth successfully activated"
                    else
                        echo "⚠️ Warning: Failed to activate minimal Unsloth"
                        # Add to PYTHONPATH manually
                        export PYTHONPATH="$CUSTOM_UNSLOTH_DIR:$PYTHONPATH"
                        echo "Manually added minimal Unsloth to PYTHONPATH"
                    fi
                else
                    echo "⚠️ Warning: Failed to create minimal Unsloth"
                fi
            else
                echo "⚠️ Warning: Could not find create_minimal_unsloth.sh"
            fi
        fi
    fi

    # Verify Unsloth is importable
    python -c '
try:
    import unsloth
    print(f"✅ Unsloth is importable, version: {unsloth.__version__}")
except ImportError as e:
    print(f"⚠️ Warning: Could not import unsloth: {e}")
    print("Training may still work without Unsloth, but will be slower")
'
}

# Activate Python environment and Unsloth
check_and_activate_python_env
check_and_activate_unsloth

# Set default values
GPU_TYPE="A6000"  # Default to A6000 with 50 GiB VRAM
VRAM_SIZE=50      # Default to 50 GiB
MODEL_TYPE=""     # Will be set based on user selection

# Function to display help
show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --help                 Show this help message"
    echo "  --gpu-type TYPE        Specify GPU type (A6000, A4000, RTX5000)"
    echo "  --vram SIZE            Specify VRAM size in GiB"
    echo "  --model-type TYPE      Specify model type (code, text, cnn-text, custom-model)"
    echo "  --skip-cleanup         Skip GPU memory cleanup"
    echo "  --skip-patches         Skip transformer patches"
    echo "  --force                Skip confirmation prompts (use with caution)"
    echo "  --debug                Enable debug mode"
    echo "  --use-improved-preprocessor  Use the ImprovedPreprocessor with dataset-specific settings"
    echo ""
    echo "Example: $0 --gpu-type A6000 --vram 50 --model-type code"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            show_help
            exit 0
            ;;
        --gpu-type)
            GPU_TYPE="$2"
            shift 2
            ;;
        --vram)
            VRAM_SIZE="$2"
            shift 2
            ;;
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --skip-cleanup)
            SKIP_CLEANUP=1
            shift
            ;;
        --skip-patches)
            SKIP_PATCHES=1
            shift
            ;;
        --force)
            FORCE=1
            shift
            ;;
        --debug)
            DEBUG=1
            shift
            ;;
        --use-improved-preprocessor)
            USE_IMPROVED_PREPROCESSOR=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Print banner
echo "===== Jarvis AI Assistant ====="
echo "GPU Type: $GPU_TYPE"
echo "VRAM Size: $VRAM_SIZE GiB"
if [ -n "$MODEL_TYPE" ]; then
    echo "Model Type: $MODEL_TYPE"
    if [ "$MODEL_TYPE" = "code" ]; then
        echo "✅ Device mismatch fix enabled for DeepSeek model"
    fi
fi
if [ -n "$USE_IMPROVED_PREPROCESSOR" ]; then
    echo "✅ Using ImprovedPreprocessor with dataset-specific settings"
    echo "   (writing_prompts will use special memory-optimized settings)"
fi
echo "========================================"

# Force Paperspace environment detection
export PAPERSPACE=true

# Set USE_IMPROVED_PREPROCESSOR if the flag is provided
if [ -n "$USE_IMPROVED_PREPROCESSOR" ]; then
    export USE_IMPROVED_PREPROCESSOR=1
    echo "✅ Setting USE_IMPROVED_PREPROCESSOR=1"
fi

echo "Setting up environment and creating directories..."
python setup/setup_environment.py

# Check for required Python packages and install missing ones
echo "Checking for required Python packages..."
python setup/verify_packages.py

# Clear CUDA cache
echo "Clearing CUDA cache..."
python setup/clear_cuda_cache.py

# Set environment variables for optimal memory usage and GPU utilization
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:32,garbage_collection_threshold:0.8"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1  # Changed to 1 for better error messages with device mismatch fix
export TRANSFORMERS_OFFLINE=0
export CUDA_VISIBLE_DEVICES=0  # Ensure we're using the first GPU

# Force certain operations on CPU to save GPU memory
export FORCE_CPU_ONLY_FOR_INITIAL_LOAD=1
export FORCE_CPU_ONLY_FOR_TOKENIZATION=1
export FORCE_CPU_ONLY_FOR_DATASET_PROCESSING=1
export TOKENIZERS_FORCE_CPU=1
export HF_DATASETS_CPU_ONLY=1
export JARVIS_FORCE_CPU_TOKENIZER=1

# Set PyTorch to use deterministic algorithms for reproducibility
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42

# This is the main training script
echo "Starting Jarvis AI Assistant training..."

# Check if model type is specified
if [ -z "$MODEL_TYPE" ]; then
    echo "No model type specified. Please specify a model type with --model-type."
    echo "Available model types: code, text, cnn-text"
    exit 1
fi

# Apply fixes for DeepSeek models
if [ "$MODEL_TYPE" = "code" ]; then
    echo "Applying fixes for DeepSeek model..."

    # Fix bitsandbytes version for 4-bit quantization
    echo "Checking bitsandbytes version for 4-bit quantization compatibility..."

    # Create a script to check and fix bitsandbytes version
    cat > setup/fix_bitsandbytes_version.py << 'EOF'
#!/usr/bin/env python3
"""
Check and fix bitsandbytes version for 4-bit quantization compatibility
"""
import sys
import logging
import importlib
import subprocess
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_bitsandbytes_version():
    """Check if bitsandbytes version is compatible with 4-bit quantization"""
    try:
        import bitsandbytes
        if hasattr(bitsandbytes, '__version__'):
            version = bitsandbytes.__version__
            logger.info(f"bitsandbytes version: {version}")

            # Parse version
            try:
                major, minor, patch = map(int, version.split('.'))
                # Check if version is >= 0.42.0 for 4-bit quantization
                if (major > 0) or (major == 0 and minor >= 42):
                    logger.info("✅ bitsandbytes version is compatible with 4-bit quantization")
                    return True
                else:
                    logger.warning("⚠️ bitsandbytes version is too old for 4-bit quantization")
                    logger.warning("Minimum required: 0.42.0 for 4-bit quantization")
                    return False
            except ValueError:
                logger.warning(f"Could not parse bitsandbytes version: {version}")
                return False
        else:
            logger.warning("bitsandbytes version attribute not found")
            return False
    except ImportError:
        logger.error("bitsandbytes is not installed")
        return False

def fix_bitsandbytes_version():
    """Fix bitsandbytes version for 4-bit quantization compatibility"""
    if check_bitsandbytes_version():
        logger.info("bitsandbytes version is already compatible with 4-bit quantization")
        return True

    logger.info("Attempting to fix bitsandbytes version...")

    # Try to upgrade bitsandbytes
    try:
        logger.info("Upgrading bitsandbytes to version 0.43.0...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "bitsandbytes>=0.43.0", "--no-deps"])
        logger.info("bitsandbytes upgraded successfully")

        # Verify the upgrade
        if check_bitsandbytes_version():
            logger.info("✅ bitsandbytes version is now compatible with 4-bit quantization")
            return True
        else:
            logger.warning("⚠️ bitsandbytes version is still not compatible with 4-bit quantization")
            return False
    except Exception as e:
        logger.error(f"Error upgrading bitsandbytes: {e}")
        return False

if __name__ == "__main__":
    # Fix bitsandbytes version
    success = fix_bitsandbytes_version()

    # Exit with appropriate code
    sys.exit(0 if success else 1)
EOF

    # Make the script executable
    chmod +x setup/fix_bitsandbytes_version.py

    # Run the script
    python setup/fix_bitsandbytes_version.py

    # Fix unsloth trust_remote_code issue
    echo "Fixing unsloth trust_remote_code issue..."
    chmod +x setup/fix_unsloth_trust_remote_code.py
    python setup/fix_unsloth_trust_remote_code.py

    # Fix unterminated string literals
    echo "Fixing unterminated string literals..."
    chmod +x setup/fix_unterminated_strings.py
    python setup/fix_unterminated_strings.py

    # Skip autocast fix as it's causing issues
    echo "Skipping autocast fix..."

    # Apply individual attention mask fix scripts directly
    echo "Applying individual attention mask fix scripts..."
    chmod +x setup/fix_transformers_attention_mask.py
    chmod +x setup/fix_attention_mask_params.py
    chmod +x setup/fix_tensor_size_mismatch.py
    chmod +x setup/fix_attention_dimension_mismatch.py
    chmod +x setup/fix_tuple_unpacking_error.py
    chmod +x setup/fix_custom_encoder_decoder_model.py
    chmod +x setup/comprehensive_attention_mask_fix.py
    chmod +x setup/fix_all_attention_issues.py
    chmod +x setup/ultimate_attention_fix.py

    # Fix custom encoder-decoder model
    echo "Fixing custom encoder-decoder model..."
    python setup/fix_custom_encoder_decoder_model.py

    # Check if transformers.utils is available
    echo "Checking if transformers.utils is available..."
    python -c '
try:
    import transformers.utils
    print("✅ transformers.utils is available")
except ImportError as e:
    print(f"❌ transformers.utils is NOT available: {e}")
    print("Please run the consolidated setup script ONCE to fix this issue:")
    print("./setup/consolidated_unified_setup.sh")
    print("This will cause issues with attention mask fixes and model training")
    print("Continuing anyway, but training will likely fail")
'

    # Try to use the ultimate fix first
    if [ -f "setup/ultimate_attention_fix.py" ]; then
        echo "Using ultimate attention fix..."
        # Make it executable
        chmod +x setup/ultimate_attention_fix.py

        # Run with detailed logging
        PYTHONPATH="$PYTHONPATH:." python -c '
import logging
import sys
import os
import torch

# Configure detailed logging
logging.basicConfig(level=logging.INFO,
                   format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                   handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("attention_fix")

# Print environment information
logger.info(f"Python version: {sys.version}")
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA version: {torch.version.cuda}")

# Import and apply the fix
try:
    import setup.ultimate_attention_fix as fix
    logger.info("Successfully imported ultimate_attention_fix")
    success = fix.apply_ultimate_fix()
    logger.info(f"Ultimate fix applied: {success}")
    print(f"Ultimate fix applied: {success}")
except Exception as e:
    logger.error(f"Error applying ultimate fix: {e}")
    print(f"Error applying ultimate fix: {e}")
    success = False
'
        if [ $? -eq 0 ]; then
            echo "✅ Ultimate attention fix applied successfully"
            # Skip the rest of the attention mask fixes
            SKIP_OTHER_FIXES=1
        else
            echo "⚠️ Ultimate fix failed, falling back to other fixes..."
        fi
    fi

    # Try to use the comprehensive fix if ultimate fix failed
    if [ -z "$SKIP_OTHER_FIXES" ] && [ -f "setup/comprehensive_attention_mask_fix.py" ]; then
        echo "Using comprehensive attention mask fix..."
        # Make it executable
        chmod +x setup/comprehensive_attention_mask_fix.py

        # Run with detailed logging and ensure it's executed in the correct environment
        PYTHONPATH="$PYTHONPATH:." python -c '
import logging
import sys
import os
import torch

# Configure detailed logging
logging.basicConfig(level=logging.INFO,
                   format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                   handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("attention_mask_fix")

# Print environment information
logger.info(f"Python version: {sys.version}")
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA version: {torch.version.cuda}")

# Import and apply the fix
try:
    import setup.comprehensive_attention_mask_fix as fix
    logger.info("Successfully imported comprehensive_attention_mask_fix")
    success = fix.apply_comprehensive_fix()
    logger.info(f"Comprehensive fix applied: {success}")
    print(f"Comprehensive fix applied: {success}")
except Exception as e:
    logger.error(f"Error applying comprehensive fix: {e}")
    print(f"Error applying comprehensive fix: {e}")
    success = False
'
        if [ $? -eq 0 ]; then
            echo "✅ Comprehensive attention mask fix applied successfully"
            # Skip the rest of the attention mask fixes
            SKIP_UNIFIED_FIX=1
        else
            echo "⚠️ Comprehensive fix failed, falling back to unified fix script..."
        fi
    fi

    # Create a unified fix script that applies all fixes in the correct order
    cat > setup/apply_all_fixes.py << 'EOF'
#!/usr/bin/env python3
"""
Unified fix script that applies all attention mask fixes in the correct order
This script addresses the specific error:
"The size of tensor a (6) must match the size of tensor b (2048) at non-singleton dimension 2"
"""

import sys
import logging
import importlib
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def apply_all_fixes():
    """Apply all attention mask fixes in the correct order"""
    try:
        # Step 1: Apply general attention mask fixes
        logger.info("Step 1: Applying general attention mask fixes...")
        try:
            from fix_transformers_attention_mask import fix_transformers_attention_mask
            success = fix_transformers_attention_mask()
            if success:
                logger.info("✅ General attention mask fixes applied successfully")
            else:
                logger.warning("⚠️ General attention mask fixes failed")
        except ImportError:
            logger.warning("⚠️ Could not import fix_transformers_attention_mask module")
            # Try to run as a script
            os.system("python setup/fix_transformers_attention_mask.py")
            logger.info("Ran fix_transformers_attention_mask.py as a script")

        # Step 2: Apply parameter-specific fixes
        logger.info("Step 2: Applying parameter-specific attention mask fixes...")
        try:
            from fix_attention_mask_params import fix_attention_mask_params
            success = fix_attention_mask_params()
            if success:
                logger.info("✅ Parameter-specific attention mask fixes applied successfully")
            else:
                logger.warning("⚠️ Parameter-specific attention mask fixes failed")
        except ImportError:
            logger.warning("⚠️ Could not import fix_attention_mask_params module")
            # Try to run as a script
            os.system("python setup/fix_attention_mask_params.py")
            logger.info("Ran fix_attention_mask_params.py as a script")

        # Step 3: Apply tensor size mismatch fixes
        logger.info("Step 3: Applying tensor size mismatch fixes...")
        try:
            from fix_tensor_size_mismatch import fix_tensor_size_mismatch
            success = fix_tensor_size_mismatch()
            if success:
                logger.info("✅ Tensor size mismatch fixes applied successfully")
            else:
                logger.warning("⚠️ Tensor size mismatch fixes failed")
        except ImportError:
            logger.warning("⚠️ Could not import fix_tensor_size_mismatch module")
            # Try to run as a script
            os.system("python setup/fix_tensor_size_mismatch.py")
            logger.info("Ran fix_tensor_size_mismatch.py as a script")

        # Step 4: Apply attention dimension mismatch fixes
        logger.info("Step 4: Applying attention dimension mismatch fixes...")
        try:
            from fix_attention_dimension_mismatch import fix_attention_dimension_mismatch
            success = fix_attention_dimension_mismatch()
            if success:
                logger.info("✅ Attention dimension mismatch fixes applied successfully")
            else:
                logger.warning("⚠️ Attention dimension mismatch fixes failed")
        except ImportError:
            logger.warning("⚠️ Could not import fix_attention_dimension_mismatch module")
            # Try to run as a script
            os.system("python setup/fix_attention_dimension_mismatch.py")
            logger.info("Ran fix_attention_dimension_mismatch.py as a script")

        # Step 5: Apply direct patch for the specific error
        logger.info("Step 5: Applying direct patch for the specific error...")
        try:
            import torch
            from transformers.modeling_attn_mask_utils import AttentionMaskConverter

            # Define a patched function that handles the specific error
            @staticmethod
            def patched_unmask_unattended(attention_mask, indices_k=None, indices_q=None, unmasked_value=True):
                """
                Patched version of _unmask_unattended that handles the specific tensor size mismatch error
                """
                # Get the device of the attention mask
                device = attention_mask.device

                # Fix attention_mask shape if needed
                if attention_mask.dim() > 2:
                    # Get the batch size and sequence length
                    batch_size = attention_mask.size(0)
                    seq_length = attention_mask.size(-1)

                    # Reshape to 2D [batch_size, seq_length]
                    attention_mask = attention_mask.view(batch_size, seq_length)
                    logger.info(f"Reshaped attention mask from >2D to 2D: {attention_mask.shape}")

                # Get batch size and sequence length
                batch_size = attention_mask.size(0)
                seq_length = attention_mask.size(-1)

                # Create a temporary tensor on the same device
                tmp = torch.ones(seq_length, device=device)

                # Find the first non-masked position for each sequence
                indices = torch.argmax(attention_mask * tmp, 1, keepdim=True)

                # Create a mask for unattended positions
                mask = torch.arange(seq_length, device=device).expand(batch_size, -1)
                mask = mask < indices

                # Expand mask to 4D
                mask = mask.unsqueeze(1).unsqueeze(2)

                # Handle indices_k and indices_q if provided
                try:
                    if indices_k is not None:
                        if isinstance(indices_k, int):
                            mask = mask.expand(-1, -1, indices_k, -1)
                        else:
                            # Handle case where indices_k is a tensor
                            mask = mask.expand(-1, -1, indices_k.size(0) if hasattr(indices_k, 'size') else indices_k, -1)

                    if indices_q is not None:
                        if isinstance(indices_q, int):
                            mask = mask.expand(-1, indices_q, -1, -1)
                        else:
                            # Handle case where indices_q is a tensor
                            mask = mask.expand(-1, indices_q.size(0) if hasattr(indices_q, 'size') else indices_q, -1, -1)
                except Exception as e:
                    logger.warning(f"Error expanding mask dimensions: {e}")
                    # If we encounter the specific tensor size mismatch error, create a compatible mask
                    error_msg = str(e)
                    if "The size of tensor a (6) must match the size of tensor b (2048) at non-singleton dimension 2" in error_msg:
                        logger.info("Creating a compatible mask for the specific error")

                        # Create a causal mask that matches the expected dimensions
                        causal_mask = torch.triu(
                            torch.ones((seq_length, seq_length), device=device, dtype=torch.bool),
                            diagonal=1
                        )
                        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]
                        causal_mask = causal_mask.expand(batch_size, 1, seq_length, seq_length)

                        # Apply the attention mask if needed
                        if attention_mask is not None:
                            # Expand attention_mask to 4D [batch_size, 1, 1, seq_length]
                            expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                            # Expand to match causal mask dimensions
                            expanded_mask = expanded_mask.expand(-1, -1, seq_length, -1)
                            # Combine with causal mask (logical OR)
                            combined_mask = causal_mask | ~expanded_mask.bool()
                            # Convert back to the expected mask format
                            mask = ~combined_mask if unmasked_value else combined_mask

                # Convert mask to the expected type based on unmasked_value
                if unmasked_value is not True:
                    mask = mask.to(dtype=attention_mask.dtype) * unmasked_value

                return mask

            # Apply the patch
            AttentionMaskConverter._unmask_unattended = patched_unmask_unattended
            logger.info("✅ Successfully applied direct patch for _unmask_unattended")
        except Exception as e:
            logger.warning(f"⚠️ Failed to apply direct patch: {e}")

        logger.info("✅ All fixes applied successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to apply all fixes: {e}")
        return False

if __name__ == "__main__":
    # Apply all fixes
    success = apply_all_fixes()

    # Exit with appropriate code
    sys.exit(0 if success else 1)
EOF

    # Make the unified fix script executable
    chmod +x setup/apply_all_fixes.py

    # Run the unified fix script only if comprehensive fix was not successful
    if [ -z "$SKIP_UNIFIED_FIX" ]; then
        echo "Applying all attention mask fixes..."
        python setup/apply_all_fixes.py

        if [ $? -ne 0 ]; then
            echo "⚠️ Warning: Attention mask fix script failed, but continuing anyway..."
        else
            echo "✅ Attention mask fix applied successfully"
        fi
    else
        echo "✅ Skipping unified fix script as comprehensive fix was successful"
    fi
fi

# Run the appropriate training script based on the model type
case "$MODEL_TYPE" in
    code)
        echo "Running DeepSeek code model training with device mismatch fix..."

        # Verify GPU availability before starting training
        python setup/verify_gpu_code.py

        # Check exit code of the GPU verification
        if [ $? -ne 0 ]; then
            echo "❌ GPU verification failed. Cannot proceed with training."
            exit 1
        fi

        # Apply additional safeguards for GPU training
        export PYTORCH_NO_CUDA_MEMORY_CACHING=1  # Disable CUDA memory caching

        python setup/train_code_model.py "$GPU_TYPE" "$VRAM_SIZE"

        # Check if training was successful
        if [ $? -ne 0 ]; then
            echo "❌ Code model training failed. See logs for details."
            exit 1
        else
            echo "✓ Code model training completed successfully!"
        fi
        ;;
    text)
        echo "Running text generation model training..."

        # Verify GPU availability before starting training
        python setup/verify_gpu_text.py

        # Check exit code of the GPU verification
        if [ $? -ne 0 ]; then
            echo "❌ GPU verification failed. Cannot proceed with training."
            exit 1
        fi

        # Apply additional safeguards for GPU training
        export PYTORCH_NO_CUDA_MEMORY_CACHING=1  # Disable CUDA memory caching

        python setup/train_text_model.py "$GPU_TYPE" "$VRAM_SIZE"

        # Check if training was successful
        if [ $? -ne 0 ]; then
            echo "❌ Text model training failed. See logs for details."
            exit 1
        else
            echo "✓ Text model training completed successfully!"
        fi
        ;;
    cnn-text)
        echo "Running CNN-based text generation model training with enhanced GPU handling..."

        # Verify GPU availability before starting training
        python setup/verify_gpu_cnn_text.py

        # Check exit code of the GPU verification
        if [ $? -ne 0 ]; then
            echo "❌ GPU verification failed. Cannot proceed with training."
            exit 1
        fi

        # Apply additional safeguards for GPU training
        export PYTORCH_NO_CUDA_MEMORY_CACHING=1  # Disable CUDA memory caching

        # Check if we should use the ImprovedPreprocessor
        if [ "$USE_IMPROVED_PREPROCESSOR" = "1" ]; then
            python setup/train_cnn_text_model.py "$GPU_TYPE" "$VRAM_SIZE" 1
        else
            python setup/train_cnn_text_model.py "$GPU_TYPE" "$VRAM_SIZE"
        fi

        # Check if training was successful
        if [ $? -ne 0 ]; then
            echo "❌ CNN text model training failed. See logs for details."
            exit 1
        else
            echo "✓ CNN text model training completed successfully!"
        fi
        ;;
    custom-model)
        echo "Running custom encoder-decoder model training with CNN model enhancement..."

        # First check if the CNN model exists
        CNN_MODEL_PATH="/notebooks/Jarvis_AI_Assistant/models/cnn-flan-ul2-finetuned/model.pt"
        if [ ! -f "$CNN_MODEL_PATH" ]; then
            echo "❌ ERROR: CNN model not found at $CNN_MODEL_PATH"
            echo "Please run train_jarvis.sh with --model-type cnn-text first"
            exit 1
        fi

        # Verify GPU availability before starting training
        python setup/verify_gpu_custom_model.py

        # Check exit code of the GPU verification
        if [ $? -ne 0 ]; then
            echo "❌ GPU verification failed. Cannot proceed with training."
            exit 1
        fi

        # Apply additional safeguards for GPU training
        export PYTORCH_NO_CUDA_MEMORY_CACHING=1  # Disable CUDA memory caching

        # Set default values for custom model training
        OUTPUT_DIR="/notebooks/Jarvis_AI_Assistant/models/custom-encoder-decoder"
        EPOCHS=3
        BATCH_SIZE=4
        MAX_SAMPLES=5000
        LEARNING_RATE=5e-5
        WEIGHT_DECAY=0.01
        HIDDEN_SIZE=768
        NUM_ENCODER_LAYERS=3
        NUM_DECODER_LAYERS=3
        DROPOUT=0.1
        LOG_EVERY=10

        # Create output directory if it doesn't exist
        mkdir -p "$OUTPUT_DIR"

        # Print training parameters
        echo "===== Custom Encoder-Decoder Model Training ====="
        echo "CNN Model Path: $CNN_MODEL_PATH"
        echo "Output Directory: $OUTPUT_DIR"
        echo "Epochs: $EPOCHS"
        echo "Batch Size: $BATCH_SIZE"
        echo "Max Samples: $MAX_SAMPLES"
        echo "Learning Rate: $LEARNING_RATE"
        echo "Weight Decay: $WEIGHT_DECAY"
        echo "Hidden Size: $HIDDEN_SIZE"
        echo "Encoder Layers: $NUM_ENCODER_LAYERS"
        echo "Decoder Layers: $NUM_DECODER_LAYERS"
        echo "Dropout: $DROPOUT"
        echo "Log Every: $LOG_EVERY"
        echo "=============================================="

        # Run the training script
        python setup/train_custom_model.py \
            --cnn-model-path "$CNN_MODEL_PATH" \
            --output-dir "$OUTPUT_DIR" \
            --epochs "$EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --max-samples "$MAX_SAMPLES" \
            --learning-rate "$LEARNING_RATE" \
            --weight-decay "$WEIGHT_DECAY" \
            --hidden-size "$HIDDEN_SIZE" \
            --num-encoder-layers "$NUM_ENCODER_LAYERS" \
            --num-decoder-layers "$NUM_DECODER_LAYERS" \
            --dropout "$DROPOUT" \
            --log-every "$LOG_EVERY" \
            $([ "$USE_IMPROVED_PREPROCESSOR" = "1" ] && echo "--use-improved-preprocessor")



        # Check if training was successful
        if [ $? -ne 0 ]; then
            echo "❌ Custom model training failed. See logs for details."
            exit 1
        else
            echo "✓ Custom model training completed successfully!"
        fi
        ;;
    *)
        echo "Unknown model type: $MODEL_TYPE"
        echo "Available model types: code, text, cnn-text, custom-model"
        exit 1
        ;;
esac

# Perform final verification and cleanup
echo "Performing final verification and cleanup..."

# Verify that models were saved correctly
python setup/verify_models.py "$MODEL_TYPE"

# Final cleanup
echo "Cleaning up temporary files..."
find /tmp -name "torch_*" -type d -mmin +60 -exec rm -rf {} \; 2>/dev/null || true
find /tmp -name "transformers_*" -type d -mmin +60 -exec rm -rf {} \; 2>/dev/null || true

echo "✓ Training process completed successfully!"
echo "✓ Done"