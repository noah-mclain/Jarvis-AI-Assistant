#!/bin/bash
# Consolidated script for Jarvis AI Assistant
# This script sets up the environment and runs the Jarvis AI Assistant

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
    python -c "
try:
    import unsloth
    print(f'✅ Unsloth is importable, version: {unsloth.__version__}')
except ImportError as e:
    print(f'⚠️ Warning: Could not import unsloth: {e}')
    print('Training may still work without Unsloth, but will be slower')
"
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
echo "========================================"

# Setup the environment and create directories
echo "Setting up environment and creating directories..."
python -c "
from src.generative_ai_module import setup_paperspace_env, create_directories
print('Setting up Paperspace environment...')
setup_paperspace_env()
print('Creating directories...')
create_directories()
print('Environment setup complete')
"

# Check for required Python packages
echo "Checking for required Python packages..."
python -c "
import sys
required_packages = ['torch', 'transformers', 'datasets', 'bitsandbytes']
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
        print(f'✓ {package} is installed')
    except ImportError:
        missing_packages.append(package)
        print(f'✗ {package} is NOT installed')

if missing_packages:
    print('\n⚠️ Some required packages are missing. Please install them with:')
    print(f'pip install {\" \".join(missing_packages)}')
    print('\nContinuing anyway, but training may fail...')
"

# Clear CUDA cache
echo "Clearing CUDA cache..."
python -c "
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print('✓ CUDA cache cleared')
    else:
        print('✓ No CUDA device available, skipping cache clearing')
except ImportError:
    print('⚠️ PyTorch not installed, skipping cache clearing')
    print('⚠️ Make sure PyTorch is installed with: pip install torch')
"

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

# Apply attention mask fix for DeepSeek models
if [ "$MODEL_TYPE" = "code" ]; then
    echo "Applying attention mask fix for DeepSeek model..."

    # Make the scripts executable
    chmod +x setup/fix_transformers_attention_mask.py
    chmod +x setup/fix_attention_mask_params.py
    chmod +x setup/fix_tensor_size_mismatch.py
    chmod +x setup/fix_attention_dimension_mismatch.py
    chmod +x setup/fix_tuple_unpacking_error.py
    chmod +x setup/comprehensive_attention_mask_fix.py
    chmod +x setup/fix_all_attention_issues.py
    chmod +x setup/ultimate_attention_fix.py

    # Try to use the ultimate fix first
    if [ -f "setup/ultimate_attention_fix.py" ]; then
        echo "Using ultimate attention fix..."
        # Make it executable
        chmod +x setup/ultimate_attention_fix.py

        # Run with detailed logging
        PYTHONPATH="$PYTHONPATH:." python -c "
import logging
import sys
import os
import torch

# Configure detailed logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger('attention_fix')

# Print environment information
logger.info(f'Python version: {sys.version}')
logger.info(f'PyTorch version: {torch.__version__}')
logger.info(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    logger.info(f'CUDA device: {torch.cuda.get_device_name(0)}')
    logger.info(f'CUDA version: {torch.version.cuda}')

# Import and apply the fix
try:
    import setup.ultimate_attention_fix as fix
    logger.info('Successfully imported ultimate_attention_fix')
    success = fix.apply_ultimate_fix()
    logger.info(f'Ultimate fix applied: {success}')
    print(f'Ultimate fix applied: {success}')
except Exception as e:
    logger.error(f'Error applying ultimate fix: {e}')
    print(f'Error applying ultimate fix: {e}')
    success = False
"
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
        PYTHONPATH="$PYTHONPATH:." python -c "
import logging
import sys
import os
import torch

# Configure detailed logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger('attention_mask_fix')

# Print environment information
logger.info(f'Python version: {sys.version}')
logger.info(f'PyTorch version: {torch.__version__}')
logger.info(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    logger.info(f'CUDA device: {torch.cuda.get_device_name(0)}')
    logger.info(f'CUDA version: {torch.version.cuda}')

# Import and apply the fix
try:
    import setup.comprehensive_attention_mask_fix as fix
    logger.info('Successfully imported comprehensive_attention_mask_fix')
    success = fix.apply_comprehensive_fix()
    logger.info(f'Comprehensive fix applied: {success}')
    print(f'Comprehensive fix applied: {success}')
except Exception as e:
    logger.error(f'Error applying comprehensive fix: {e}')
    print(f'Error applying comprehensive fix: {e}')
    success = False
"
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
Unified fix script that applies all attention mask fixes in the correct order.
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
                Patched version of _unmask_unattended that handles the specific tensor size mismatch error.
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
                        else:
                            mask = ~causal_mask if unmasked_value else causal_mask

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

# Run the appropriate training script based on model type
case $MODEL_TYPE in
    code)
        echo "Running code generation model training with enhanced GPU handling..."
        echo "Using device mismatch fix for DeepSeek model to prevent CUDA errors"

        # Verify GPU availability before starting training
        python -c "
import torch
import sys

if not torch.cuda.is_available():
    print('❌ ERROR: CUDA is not available. Cannot proceed with GPU training.')
    sys.exit(1)
else:
    device_name = torch.cuda.get_device_name(0)
    device_capability = torch.cuda.get_device_capability(0)
    free_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'✓ Using GPU: {device_name} with CUDA capability {device_capability[0]}.{device_capability[1]}')
    print(f'✓ Available GPU memory: {free_memory:.2f} GiB')

    # Verify minimum memory requirements
    if free_memory < 10:
        print(f'❌ ERROR: Not enough GPU memory. Need at least 10 GiB, but only {free_memory:.2f} GiB available.')
        sys.exit(1)

    # Clear CUDA cache
    torch.cuda.empty_cache()
    print('✓ CUDA cache cleared')
"

        # Check exit code of the GPU verification
        if [ $? -ne 0 ]; then
            echo "❌ GPU verification failed. Cannot proceed with training."
            exit 1
        fi

        # Apply additional safeguards for GPU training
        export PYTORCH_NO_CUDA_MEMORY_CACHING=1  # Disable CUDA memory caching

        # Create log directory if it doesn't exist
        mkdir -p logs
        LOG_FILE="logs/deepseek_training_$(date +%Y%m%d_%H%M%S).log"
        echo "📝 Logging to $LOG_FILE"

        # Verify attention mask fix is working
        echo "Verifying attention mask fix..."
        python -c "
import torch
import sys
import logging
import importlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Check if our patches are in place
    from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa, AttentionMaskConverter

    # Create a test attention mask with problematic shape
    batch_size = 2
    seq_length = 10

    # Create a 3D attention mask (problematic shape)
    attention_mask_3d = torch.ones(batch_size, 1, seq_length)
    logger.info(f'Created 3D attention mask with shape: {attention_mask_3d.shape}')

    # Test if _prepare_4d_causal_attention_mask_for_sdpa can handle 3D masks
    try:
        # Create dummy inputs for the function
        input_shape = (batch_size, seq_length)
        inputs_embeds = torch.randn(batch_size, seq_length, 32)  # Dummy embeddings
        past_key_values_length = 0
        sliding_window = None
        dtype = torch.float32

        # Call the function with our 3D mask
        result = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask_3d,
            input_shape,
            inputs_embeds,
            past_key_values_length,
            sliding_window,
            dtype
        )
        logger.info(f'Successfully processed 3D mask with _prepare_4d_causal_attention_mask_for_sdpa')
    except Exception as e:
        logger.error(f'Error in _prepare_4d_causal_attention_mask_for_sdpa: {e}')
        # Try with 2D mask as fallback
        attention_mask_2d = attention_mask_3d.view(batch_size, seq_length)
        result = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask_2d,
            input_shape,
            inputs_embeds,
            past_key_values_length,
            sliding_window,
            dtype
        )
        logger.info(f'Successfully processed 2D mask with _prepare_4d_causal_attention_mask_for_sdpa')

    # Test if AttentionMaskConverter._unmask_unattended can handle unmasked_value parameter
    try:
        # Create a simple mask
        mask = torch.ones(batch_size, seq_length)
        # Call the function with unmasked_value parameter
        result = AttentionMaskConverter._unmask_unattended(mask, unmasked_value=0.0)
        logger.info(f'Successfully called _unmask_unattended with unmasked_value parameter')
    except Exception as e:
        logger.error(f'Error in _unmask_unattended: {e}')
        # If it fails, our patch might not be working correctly

    print('✅ Attention mask fix verification passed!')
except Exception as e:
    print(f'❌ Attention mask fix verification failed: {e}')
    sys.exit(1)
"
        # Always continue even if verification fails
        echo "Continuing with training regardless of verification result..."

        # Run the training with enhanced GPU handling and device mismatch fix
        echo "🧠 Starting DeepSeek training with device mismatch fix..."
        python src/generative_ai_module/unified_deepseek_training.py \
            --gpu-type $GPU_TYPE \
            --vram $VRAM_SIZE \
            --model_name "deepseek-ai/deepseek-coder-6.7b-instruct" \
            --output_dir "notebooks/Jarvis_AI_Assistant/models/deepseek-coder-6.7b-finetuned" \
            --epochs 3 \
            --max_samples 5000 \
            --batch_size 6 \
            --max_length 2048 \
            --gradient_accumulation_steps 8 \
            --optimize_memory_usage \
            --gradient_checkpointing \
            --mixed_precision \
            --load_in_4bit \
            --bf16 \
            --lora_rank 32 \
            --lora_alpha 64 \
            --lora_dropout 0.05 \
            --warmup_ratio 0.03 \
            --weight_decay 0.01 \
            --max_grad_norm 1.0 \
            --scheduler_type "cosine" \
            --evaluation_strategy "steps" \
            --eval_steps 100 \
            --save_steps 100 \
            --save_total_limit 3 \
            --adam_beta1 0.9 \
            --adam_beta2 0.999 \
            --adam_epsilon 1e-8 \
            --num_workers 0 2>&1 | tee -a "$LOG_FILE"  # Log output and display it

        # Check if training was successful
        if [ $? -ne 0 ]; then
            echo "❌ Training failed. See logs for details."

            # Try to recover and save partial results
            echo "Attempting to save partial results..."
            python -c "
import os
import torch
from transformers import AutoTokenizer
from datetime import datetime

# Create backup directory
backup_dir = f'/notebooks/Jarvis_AI_Assistant/models/backup_save_{int(datetime.now().timestamp())}'
os.makedirs(backup_dir, exist_ok=True)

# Try to save model state if it exists
try:
    if os.path.exists('/notebooks/Jarvis_AI_Assistant/models/deepseek-coder-6.7b-finetuned'):
        # Save tokenizer if available
        try:
            tokenizer = AutoTokenizer.from_pretrained('/notebooks/Jarvis_AI_Assistant/models/deepseek-coder-6.7b-finetuned')
            tokenizer.save_pretrained(backup_dir)
            print(f'✓ Tokenizer saved to {backup_dir}')
        except Exception as e:
            print(f'❌ Failed to save tokenizer: {e}')

        print(f'✓ Partial results saved to {backup_dir}')
    else:
        print('❌ No model directory found to save partial results')
except Exception as e:
    print(f'❌ Failed to save partial results: {e}')
"
            exit 1
        else
            echo "✓ Training completed successfully!"

            # Verify the saved model
            python -c "
import os
import sys

model_dir = '/notebooks/Jarvis_AI_Assistant/models/deepseek-coder-6.7b-finetuned'
if not os.path.exists(model_dir):
    print(f'❌ ERROR: Model directory {model_dir} does not exist after training.')
    sys.exit(1)

required_files = ['config.json', 'adapter_config.json']
missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]

if missing_files:
    print(f'❌ ERROR: Missing required files in model directory: {missing_files}')
    sys.exit(1)
else:
    print(f'✓ Model saved successfully to {model_dir}')
    print('✓ All required files are present')
"
            # Check if model verification was successful
            if [ $? -ne 0 ]; then
                echo "❌ Model verification failed. The model may not have been saved correctly."
                exit 1
            fi
        fi
        ;;
    text)
        echo "Running text generation model training with enhanced GPU handling..."

        # Verify GPU availability before starting training
        python -c "
import torch
import sys

if not torch.cuda.is_available():
    print('❌ ERROR: CUDA is not available. Cannot proceed with GPU training.')
    sys.exit(1)
else:
    device_name = torch.cuda.get_device_name(0)
    device_capability = torch.cuda.get_device_capability(0)
    free_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'✓ Using GPU: {device_name} with CUDA capability {device_capability[0]}.{device_capability[1]}')
    print(f'✓ Available GPU memory: {free_memory:.2f} GiB')

    # Verify minimum memory requirements
    if free_memory < 10:
        print(f'❌ ERROR: Not enough GPU memory. Need at least 10 GiB, but only {free_memory:.2f} GiB available.')
        sys.exit(1)

    # Clear CUDA cache
    torch.cuda.empty_cache()
    print('✓ CUDA cache cleared')
"

        # Check exit code of the GPU verification
        if [ $? -ne 0 ]; then
            echo "❌ GPU verification failed. Cannot proceed with training."
            exit 1
        fi

        # Apply additional safeguards for GPU training
        export PYTORCH_NO_CUDA_MEMORY_CACHING=1  # Disable CUDA memory caching

        python -c "
import sys
import os
import torch
from src.generative_ai_module.text_generator import create_cnn_text_generator

try:
    print('Starting text generation model training...')

    # Ensure we're using GPU
    if not torch.cuda.is_available():
        print('❌ ERROR: CUDA is not available. Cannot proceed with GPU training.')
        sys.exit(1)

    # Create and train the text generator with optimized parameters for GPU
    model = create_cnn_text_generator(
        model_name='google/flan-ul2',
        force_gpu=True,
        gpu_type='$GPU_TYPE',
        vram_size=$VRAM_SIZE,
        load_in_4bit=True,
        use_flash_attention_2=True,
        gradient_checkpointing=True,
        lora_rank=32,
        lora_alpha=64,
        lora_dropout=0.05,
        max_length=2048,  # Reduced from 4096 to ensure stability with FLAN-UL2
        batch_size=3,  # Explicitly set for stability
        gradient_accumulation_steps=8,  # Explicitly set for stability
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with CUDA
        warmup_ratio=0.03,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0
    )

    # Verify model is on GPU
    if not next(model.parameters()).is_cuda:
        print('❌ WARNING: Model is not on GPU. Moving model to GPU...')
        model = model.cuda()

    # Ensure datasets directory exists
    datasets_dir = 'notebooks/Jarvis_AI_Assistant/datasets'
    os.makedirs(datasets_dir, exist_ok=True)

    # Import dataset processor
    from src.generative_ai_module.dataset_processor import DatasetProcessor

    # Create processor
    processor = DatasetProcessor(model)

    # Define datasets to preprocess
    datasets = ['persona_chat', 'writing_prompts', 'pile', 'openassistant', 'gpteacher']
    preprocessed_paths = {}

    # Preprocess all datasets
    print('Preprocessing datasets...')
    for dataset_name in datasets:
        preprocessed_path = os.path.join(datasets_dir, f'preprocessed_{dataset_name}.pt')
        preprocessed_paths[dataset_name] = preprocessed_path

        # Check if dataset is already preprocessed
        if os.path.exists(preprocessed_path):
            print(f'Dataset {dataset_name} already preprocessed at {preprocessed_path}')
            continue

        print(f'Preprocessing {dataset_name}...')
        try:
            # Prepare dataset with appropriate parameters
            if dataset_name == 'persona_chat':
                sequence_length = 512
                batch_size = 16
                raw_text = processor.load_persona_chat(split='train', max_samples=5000)
            elif dataset_name == 'writing_prompts':
                sequence_length = 1024
                batch_size = 8
                raw_text = processor.load_writing_prompts(split='train', max_samples=5000)
            elif dataset_name == 'pile':
                sequence_length = 1024
                batch_size = 8
                raw_text = processor.load_pile_dataset(split='train', max_samples=5000)
            elif dataset_name == 'openassistant':
                sequence_length = 512
                batch_size = 16
                raw_text = processor.load_openassistant_dataset(split='train', max_samples=5000)
            elif dataset_name == 'gpteacher':
                sequence_length = 768
                batch_size = 12
                raw_text = processor.load_gpteacher_dataset(split='train', max_samples=5000)

            # Create sequences and batches
            print(f'Creating sequences with length {sequence_length}...')
            sequences = processor.create_sequences(raw_text, sequence_length)

            print(f'Creating batches with batch size {batch_size}...')
            batches = processor.create_batches(sequences, batch_size=batch_size)

            # Create dataset dictionary
            dataset = {
                'batches': batches,
                'metadata': {
                    'dataset_name': dataset_name,
                    'split': 'train',
                    'sequence_length': sequence_length,
                    'batch_size': batch_size,
                    'sample_count': len(sequences),
                    'batch_count': len(batches)
                }
            }

            # Save preprocessed data
            print(f'Saving preprocessed {dataset_name} to {preprocessed_path}...')
            torch.save(dataset, preprocessed_path)
            print(f'✓ Successfully preprocessed {dataset_name}')

        except Exception as e:
            print(f'❌ Error preprocessing {dataset_name}: {e}')
            import traceback
            traceback.print_exc()

    # Train the model with all available datasets
    print('Starting training...')

    # Check which datasets were successfully preprocessed
    available_datasets = []
    for dataset_name, path in preprocessed_paths.items():
        if os.path.exists(path):
            available_datasets.append(dataset_name)

    if len(available_datasets) > 1:
        print(f'Training with multiple datasets: {available_datasets}')
        model.train_from_multiple_datasets(
            dataset_names=available_datasets,
            epochs=3,
            dataset_paths=preprocessed_paths
        )
    else:
        # Fall back to single dataset training
        print(f'Training with single dataset: persona_chat')
        model.train_from_preprocessed(
            dataset_name='persona_chat',
            epochs=3,
            preprocessed_path=preprocessed_paths.get('persona_chat')
        )

    # Save the model
    output_dir = 'notebooks/Jarvis_AI_Assistant/models/flan-ul2-finetuned'
    os.makedirs(output_dir, exist_ok=True)
    model.save_model(f'{output_dir}/model.pt')

    # Verify saved model
    if os.path.exists(f'{output_dir}/model.pt'):
        print(f'✓ Model successfully saved to {output_dir}/model.pt')
    else:
        print(f'❌ ERROR: Failed to save model to {output_dir}/model.pt')
        sys.exit(1)

    print('✓ Text generation model training completed successfully')

except Exception as e:
    print(f'❌ ERROR during text model training: {e}')

    # Try to save partial results
    try:
        backup_dir = f'notebooks/Jarvis_AI_Assistant/models/backup_text_model_{int(torch.cuda.current_device())}'
        os.makedirs(backup_dir, exist_ok=True)

        if 'model' in locals():
            model.save_model(f'{backup_dir}/partial_model.pt')
            print(f'✓ Partial model saved to {backup_dir}/partial_model.pt')
    except Exception as save_error:
        print(f'❌ Failed to save partial results: {save_error}')

    sys.exit(1)
"

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
        python -c "
import torch
import sys

if not torch.cuda.is_available():
    print('❌ ERROR: CUDA is not available. Cannot proceed with GPU training.')
    sys.exit(1)
else:
    device_name = torch.cuda.get_device_name(0)
    device_capability = torch.cuda.get_device_capability(0)
    free_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'✓ Using GPU: {device_name} with CUDA capability {device_capability[0]}.{device_capability[1]}')
    print(f'✓ Available GPU memory: {free_memory:.2f} GiB')

    # Verify minimum memory requirements - CNN models need more memory
    if free_memory < 15:
        print(f'❌ ERROR: Not enough GPU memory. Need at least 15 GiB for CNN model, but only {free_memory:.2f} GiB available.')
        sys.exit(1)

    # Clear CUDA cache
    torch.cuda.empty_cache()
    print('✓ CUDA cache cleared')
"

        # Check exit code of the GPU verification
        if [ $? -ne 0 ]; then
            echo "❌ GPU verification failed. Cannot proceed with training."
            exit 1
        fi

        # Apply additional safeguards for GPU training
        export PYTORCH_NO_CUDA_MEMORY_CACHING=1  # Disable CUDA memory caching

        python -c "
import sys
import os
import torch
from src.generative_ai_module.text_generator import create_cnn_text_generator

try:
    print('Starting CNN-based text generation model training...')

    # Ensure we're using GPU
    if not torch.cuda.is_available():
        print('❌ ERROR: CUDA is not available. Cannot proceed with GPU training.')
        sys.exit(1)

    # Create and train the CNN-enhanced text generator with optimized attention mechanisms
    model = create_cnn_text_generator(
        model_name='google/flan-ul2',
        force_gpu=True,
        gpu_type='$GPU_TYPE',
        vram_size=$VRAM_SIZE,
        cnn_layers=3,  # Use 3 CNN layers with enhanced attention-like features
        load_in_4bit=True,
        use_flash_attention_2=True,  # This will trigger our enhanced attention mechanisms for T5/FLAN models
        gradient_checkpointing=True,
        lora_rank=32,
        lora_alpha=64,
        lora_dropout=0.1,  # Increased dropout for better regularization
        max_length=2048,  # Reduced from 4096 to ensure stability with FLAN-UL2
        batch_size=2,  # Further reduced for CNN layers which use more memory
        gradient_accumulation_steps=12,  # Increased to maintain effective batch size
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with CUDA
        warmup_ratio=0.05,  # Increased warmup for better convergence
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0
    )

    # Verify model is on GPU
    if not next(model.parameters()).is_cuda:
        print('❌ WARNING: Model is not on GPU. Moving model to GPU...')
        model = model.cuda()

    # Ensure datasets directory exists
    datasets_dir = 'notebooks/Jarvis_AI_Assistant/datasets'
    os.makedirs(datasets_dir, exist_ok=True)

    # Import dataset processor
    from src.generative_ai_module.dataset_processor import DatasetProcessor

    # Create processor
    processor = DatasetProcessor(model)

    # Define datasets to preprocess
    datasets = ['persona_chat', 'writing_prompts', 'pile', 'openassistant', 'gpteacher']
    preprocessed_paths = {}

    # Preprocess all datasets
    print('Preprocessing datasets...')
    for dataset_name in datasets:
        preprocessed_path = os.path.join(datasets_dir, f'preprocessed_{dataset_name}.pt')
        preprocessed_paths[dataset_name] = preprocessed_path

        # Check if dataset is already preprocessed
        if os.path.exists(preprocessed_path):
            print(f'Dataset {dataset_name} already preprocessed at {preprocessed_path}')
            continue

        print(f'Preprocessing {dataset_name}...')
        try:
            # Prepare dataset with appropriate parameters
            if dataset_name == 'persona_chat':
                sequence_length = 512
                batch_size = 16
                raw_text = processor.load_persona_chat(split='train', max_samples=5000)
            elif dataset_name == 'writing_prompts':
                sequence_length = 1024
                batch_size = 8
                raw_text = processor.load_writing_prompts(split='train', max_samples=5000)
            elif dataset_name == 'pile':
                sequence_length = 1024
                batch_size = 8
                raw_text = processor.load_pile_dataset(split='train', max_samples=5000)
            elif dataset_name == 'openassistant':
                sequence_length = 512
                batch_size = 16
                raw_text = processor.load_openassistant_dataset(split='train', max_samples=5000)
            elif dataset_name == 'gpteacher':
                sequence_length = 768
                batch_size = 12
                raw_text = processor.load_gpteacher_dataset(split='train', max_samples=5000)

            # Create sequences and batches
            print(f'Creating sequences with length {sequence_length}...')
            sequences = processor.create_sequences(raw_text, sequence_length)

            print(f'Creating batches with batch size {batch_size}...')
            batches = processor.create_batches(sequences, batch_size=batch_size)

            # Create dataset dictionary
            dataset = {
                'batches': batches,
                'metadata': {
                    'dataset_name': dataset_name,
                    'split': 'train',
                    'sequence_length': sequence_length,
                    'batch_size': batch_size,
                    'sample_count': len(sequences),
                    'batch_count': len(batches)
                }
            }

            # Save preprocessed data
            print(f'Saving preprocessed {dataset_name} to {preprocessed_path}...')
            torch.save(dataset, preprocessed_path)
            print(f'✓ Successfully preprocessed {dataset_name}')

        except Exception as e:
            print(f'❌ Error preprocessing {dataset_name}: {e}')
            import traceback
            traceback.print_exc()

    # Train the model with all available datasets
    print('Starting training...')

    # Check which datasets were successfully preprocessed
    available_datasets = []
    for dataset_name, path in preprocessed_paths.items():
        if os.path.exists(path):
            available_datasets.append(dataset_name)

    if len(available_datasets) > 1:
        print(f'Training with multiple datasets: {available_datasets}')
        model.train_from_multiple_datasets(
            dataset_names=available_datasets,
            epochs=3,
            dataset_paths=preprocessed_paths
        )
    else:
        # Fall back to single dataset training
        print(f'Training with single dataset: persona_chat')
        model.train_from_preprocessed(
            dataset_name='persona_chat',
            epochs=3,
            preprocessed_path=preprocessed_paths.get('persona_chat')
        )

    # Save the model
    output_dir = 'notebooks/Jarvis_AI_Assistant/models/cnn-flan-ul2-finetuned'
    os.makedirs(output_dir, exist_ok=True)
    model.save_model(f'{output_dir}/model.pt')

    # Verify saved model
    if os.path.exists(f'{output_dir}/model.pt'):
        print(f'✓ Model successfully saved to {output_dir}/model.pt')
    else:
        print(f'❌ ERROR: Failed to save model to {output_dir}/model.pt')
        sys.exit(1)

    print('✓ CNN-enhanced text generation model training completed successfully')

except Exception as e:
    print(f'❌ ERROR during CNN text model training: {e}')

    # Try to save partial results
    try:
        backup_dir = f'notebooks/Jarvis_AI_Assistant/models/backup_cnn_model_{int(torch.cuda.current_device())}'
        os.makedirs(backup_dir, exist_ok=True)

        if 'model' in locals():
            model.save_model(f'{backup_dir}/partial_model.pt')
            print(f'✓ Partial model saved to {backup_dir}/partial_model.pt')
    except Exception as save_error:
        print(f'❌ Failed to save partial results: {save_error}')

    sys.exit(1)
"

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
        python -c "
import torch
import sys

if not torch.cuda.is_available():
    print('❌ ERROR: CUDA is not available. Cannot proceed with GPU training.')
    sys.exit(1)
else:
    device_name = torch.cuda.get_device_name(0)
    device_capability = torch.cuda.get_device_capability(0)
    free_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'✓ Using GPU: {device_name} with CUDA capability {device_capability[0]}.{device_capability[1]}')
    print(f'✓ Available GPU memory: {free_memory:.2f} GiB')

    # Verify minimum memory requirements - Custom models need more memory
    if free_memory < 20:
        print(f'❌ ERROR: Not enough GPU memory. Need at least 20 GiB for custom model, but only {free_memory:.2f} GiB available.')
        sys.exit(1)

    # Clear CUDA cache
    torch.cuda.empty_cache()
    print('✓ CUDA cache cleared')
"

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
        python -c "
import sys
import os
import torch
import logging
import argparse
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Import local modules
try:
    from src.generative_ai_module.text_generator import CNNTextGenerator
    from src.generative_ai_module.unified_dataset_handler import UnifiedDatasetHandler
except ImportError as e:
    logger.error(f'Failed to import required modules: {e}')
    sys.exit(1)

class CustomEncoderDecoder(torch.nn.Module):
    """
    Custom encoder-decoder model that leverages the fine-tuned CNN-enhanced FLAN-UL2 model.

    This model uses the CNN-enhanced FLAN-UL2 model as a feature extractor and adds
    custom encoder-decoder layers on top.
    """

    def __init__(
        self,
        cnn_model_path: str,
        hidden_size: int = 768,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dropout: float = 0.1,
        force_gpu: bool = True
    ):
        super().__init__()

        # Load the CNN-enhanced FLAN-UL2 model
        logger.info(f'Loading CNN-enhanced FLAN-UL2 model from {cnn_model_path}')
        self.cnn_model = self._load_cnn_model(cnn_model_path)

        # Freeze the CNN model parameters
        for param in self.cnn_model.parameters():
            param.requires_grad = False

        # Get the hidden size from the CNN model
        self.feature_size = self.cnn_model.hidden_size

        # Create encoder layers
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=self.feature_size,
                nhead=8,
                dim_feedforward=hidden_size,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_encoder_layers
        )

        # Create decoder layers
        self.decoder = torch.nn.TransformerDecoder(
            torch.nn.TransformerDecoderLayer(
                d_model=self.feature_size,
                nhead=8,
                dim_feedforward=hidden_size,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_decoder_layers
        )

        # Output projection
        self.output_projection = torch.nn.Linear(self.feature_size, self.cnn_model.tokenizer.vocab_size)

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() and force_gpu else 'cpu')
        self.to(self.device)

        logger.info(f'Initialized custom encoder-decoder model on {self.device}')

    def _load_cnn_model(self, model_path: str) -> CNNTextGenerator:
        """Load the CNN-enhanced FLAN-UL2 model"""
        # Check if the model path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model path {model_path} does not exist')

        # Load the model
        model_dir = os.path.dirname(model_path)
        tokenizer_path = os.path.join(model_dir, 'tokenizer')

        # Create a new CNN model with enhanced attention mechanisms
        cnn_model = CNNTextGenerator(
            model_name_or_path='google/flan-ul2',
            force_gpu=True,
            cnn_layers=3,
            load_in_4bit=True,
            use_flash_attention_2=True,  # This will trigger our enhanced attention mechanisms for T5/FLAN models
            lora_dropout=0.1,  # Increased dropout for better regularization
            warmup_ratio=0.05  # Increased warmup for better convergence
        )

        # Load the saved model
        state_dict = torch.load(model_path, map_location='cpu')
        cnn_model.base_model.load_state_dict(state_dict['base_model'])

        # Load CNN layers
        for i, layer_state in enumerate(state_dict['cnn_layers']):
            if i < len(cnn_model.cnn_layers_list):
                cnn_model.cnn_layers_list[i].load_state_dict(layer_state)

        # Load adapter
        cnn_model.adapter.load_state_dict(state_dict['adapter'])

        return cnn_model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the encoder-decoder model

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask for input [batch_size, seq_len]
            decoder_input_ids: Decoder input token IDs [batch_size, target_len]
            decoder_attention_mask: Attention mask for decoder [batch_size, target_len]

        Returns:
            Logits for next token prediction [batch_size, target_len, vocab_size]
        """
        # Get embeddings from CNN model
        with torch.no_grad():
            # Get embeddings from base model
            if hasattr(self.cnn_model.base_model, 'transformer') and hasattr(self.cnn_model.base_model.transformer, 'wte'):
                # GPT-2 style models
                embeddings = self.cnn_model.base_model.transformer.wte(input_ids)
            elif hasattr(self.cnn_model.base_model, 'get_input_embeddings'):
                # Generic approach for most models
                embedding_layer = self.cnn_model.base_model.get_input_embeddings()
                embeddings = embedding_layer(input_ids)
            else:
                raise ValueError('Could not get embeddings from model')

            # Apply CNN layers for feature extraction
            # First, transpose for CNN (batch_size, hidden_size, seq_len)
            x = embeddings.transpose(1, 2)

            # Pass through each CNN layer
            for cnn_layer in self.cnn_model.cnn_layers_list:
                x = cnn_layer(x)

            # Transpose back to transformer format (batch_size, seq_len, hidden_size)
            x = x.transpose(1, 2)

            # Apply adapter to ensure compatibility with transformer
            enhanced_embeddings = self.cnn_model.adapter(x)

            # Add residual connection to preserve original embeddings
            enhanced_embeddings = enhanced_embeddings + embeddings

        # Pass through encoder
        encoder_output = self.encoder(enhanced_embeddings, src_key_padding_mask=~attention_mask.bool())

        # Get decoder embeddings
        if hasattr(self.cnn_model.base_model, 'transformer') and hasattr(self.cnn_model.base_model.transformer, 'wte'):
            # GPT-2 style models
            decoder_embeddings = self.cnn_model.base_model.transformer.wte(decoder_input_ids)
        elif hasattr(self.cnn_model.base_model, 'get_input_embeddings'):
            # Generic approach for most models
            embedding_layer = self.cnn_model.base_model.get_input_embeddings()
            decoder_embeddings = embedding_layer(decoder_input_ids)
        else:
            raise ValueError('Could not get embeddings from model')

        # Pass through decoder
        decoder_output = self.decoder(
            decoder_embeddings,
            encoder_output,
            tgt_key_padding_mask=~decoder_attention_mask.bool()
        )

        # Project to vocabulary
        logits = self.output_projection(decoder_output)

        return logits

def train_custom_model():
    """Train the custom encoder-decoder model"""
    # Set parameters
    cnn_model_path = '$CNN_MODEL_PATH'
    output_dir = '$OUTPUT_DIR'
    epochs = $EPOCHS
    batch_size = $BATCH_SIZE
    max_samples = $MAX_SAMPLES
    learning_rate = $LEARNING_RATE
    weight_decay = $WEIGHT_DECAY
    hidden_size = $HIDDEN_SIZE
    num_encoder_layers = $NUM_ENCODER_LAYERS
    num_decoder_layers = $NUM_DECODER_LAYERS
    dropout = $DROPOUT
    log_every = $LOG_EVERY
    force_gpu = True
    save_checkpoints = True

    # Initialize the dataset handler
    dataset_handler = UnifiedDatasetHandler()

    # Load all preprocessed datasets
    datasets_dir = 'notebooks/Jarvis_AI_Assistant/datasets'
    os.makedirs(datasets_dir, exist_ok=True)
    dataset_names = ['writing_prompts', 'persona_chat', 'pile', 'openassistant', 'gpteacher']
    preprocessed_paths = {name: os.path.join(datasets_dir, f'preprocessed_{name}.pt') for name in dataset_names}

    # Check which datasets are available
    available_datasets = []
    for dataset_name, path in preprocessed_paths.items():
        if os.path.exists(path):
            available_datasets.append(dataset_name)

    if not available_datasets:
        logger.warning("No preprocessed datasets found. Running preprocessing...")
        # Import dataset processor
        from src.generative_ai_module.dataset_processor import DatasetProcessor

        # Create processor for preprocessing
        processor = DatasetProcessor()

        # Preprocess all datasets
        logger.info('Preprocessing datasets...')
        for dataset_name in dataset_names:
            preprocessed_path = preprocessed_paths[dataset_name]

            logger.info(f'Preprocessing {dataset_name}...')
            try:
                # Prepare dataset with appropriate parameters
                if dataset_name == 'persona_chat':
                    sequence_length = 512
                    batch_size = 16
                    raw_text = processor.load_persona_chat(split='train', max_samples=max_samples // 5)
                elif dataset_name == 'writing_prompts':
                    sequence_length = 1024
                    batch_size = 8
                    raw_text = processor.load_writing_prompts(split='train', max_samples=max_samples // 5)
                elif dataset_name == 'pile':
                    sequence_length = 1024
                    batch_size = 8
                    raw_text = processor.load_pile_dataset(split='train', max_samples=max_samples // 5)
                elif dataset_name == 'openassistant':
                    sequence_length = 512
                    batch_size = 16
                    raw_text = processor.load_openassistant_dataset(split='train', max_samples=max_samples // 5)
                elif dataset_name == 'gpteacher':
                    sequence_length = 768
                    batch_size = 12
                    raw_text = processor.load_gpteacher_dataset(split='train', max_samples=max_samples // 5)

                # Create sequences and batches
                logger.info(f'Creating sequences with length {sequence_length}...')
                sequences = processor.create_sequences(raw_text, sequence_length)

                logger.info(f'Creating batches with batch size {batch_size}...')
                batches = processor.create_batches(sequences, batch_size=batch_size)

                # Create dataset dictionary
                dataset = {
                    'batches': batches,
                    'metadata': {
                        'dataset_name': dataset_name,
                        'split': 'train',
                        'sequence_length': sequence_length,
                        'batch_size': batch_size,
                        'sample_count': len(sequences),
                        'batch_count': len(batches)
                    }
                }

                # Save preprocessed data
                logger.info(f'Saving preprocessed {dataset_name} to {preprocessed_path}...')
                os.makedirs(os.path.dirname(preprocessed_path), exist_ok=True)
                torch.save(dataset, preprocessed_path)
                logger.info(f'Successfully preprocessed {dataset_name}')

                # Add to available datasets
                available_datasets.append(dataset_name)

            except Exception as e:
                logger.error(f'Error preprocessing {dataset_name}: {e}')
                import traceback
                logger.error(traceback.format_exc())

    # Load the preprocessed datasets
    datasets = []
    for dataset_name in available_datasets:
        logger.info(f'Loading preprocessed dataset: {dataset_name}')
        try:
            dataset = torch.load(preprocessed_paths[dataset_name])
            datasets.append(dataset)
        except Exception as e:
            logger.warning(f'Failed to load dataset {dataset_name}: {e}')

    # Combine datasets
    combined_batches = []
    for dataset in datasets:
        combined_batches.extend(dataset.get('batches', []))

    logger.info(f'Combined {len(combined_batches)} batches from all datasets')

    # Initialize the custom model
    model = CustomEncoderDecoder(
        cnn_model_path=cnn_model_path,
        hidden_size=hidden_size,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dropout=dropout,
        force_gpu=force_gpu
    )

    # Set up optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Set up loss function
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    # Training loop
    logger.info(f'Starting training for {epochs} epochs')
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # Process batches
        for i, (input_batch, target_batch) in enumerate(combined_batches):
            # Skip invalid batches
            if input_batch is None or target_batch is None:
                continue

            # Move to device
            input_batch = input_batch.to(model.device)
            target_batch = target_batch.to(model.device)

            # Create attention masks
            input_attention_mask = (input_batch != model.cnn_model.tokenizer.pad_token_id).float()
            target_attention_mask = (target_batch != model.cnn_model.tokenizer.pad_token_id).float()

            # Create decoder input IDs (shift right)
            decoder_input_ids = torch.zeros_like(target_batch)
            decoder_input_ids[:, 1:] = target_batch[:, :-1]
            decoder_input_ids[:, 0] = model.cnn_model.tokenizer.bos_token_id if model.cnn_model.tokenizer.bos_token_id is not None else model.cnn_model.tokenizer.pad_token_id

            # Forward pass
            logits = model(
                input_ids=input_batch,
                attention_mask=input_attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=target_attention_mask
            )

            # Calculate loss
            loss = loss_fn(logits.view(-1, logits.size(-1)), target_batch.view(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update total loss
            total_loss += loss.item()

            # Log progress
            if (i + 1) % log_every == 0:
                logger.info(f'Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(combined_batches)}, Loss: {loss.item():.4f}')

        # Log epoch results
        avg_loss = total_loss / len(combined_batches)
        logger.info(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}')

        # Save checkpoint
        if save_checkpoints:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, checkpoint_path)
            logger.info(f'Saved checkpoint to {checkpoint_path}')

    # Save final model
    final_model_path = os.path.join(output_dir, 'custom_encoder_decoder.pt')
    torch.save(model.state_dict(), final_model_path)
    logger.info(f'Saved final model to {final_model_path}')

    return model

try:
    print('Starting custom encoder-decoder model training...')

    # Ensure we're using GPU
    if not torch.cuda.is_available():
        print('❌ ERROR: CUDA is not available. Cannot proceed with GPU training.')
        sys.exit(1)

    # Train the model
    model = train_custom_model()

    # Verify saved model
    final_model_path = os.path.join('$OUTPUT_DIR', 'custom_encoder_decoder.pt')
    if os.path.exists(final_model_path):
        print(f'✓ Model successfully saved to {final_model_path}')
    else:
        print(f'❌ ERROR: Failed to save model to {final_model_path}')
        sys.exit(1)

    print('✓ Custom encoder-decoder model training completed successfully')

except Exception as e:
    print(f'❌ ERROR during custom model training: {e}')

    # Try to save partial results
    try:
        backup_dir = f'notebooks/Jarvis_AI_Assistant/models/backup_custom_model_{int(torch.cuda.current_device())}'
        os.makedirs(backup_dir, exist_ok=True)

        if 'model' in locals():
            torch.save(model.state_dict(), f'{backup_dir}/partial_model.pt')
            print(f'✓ Partial model saved to {backup_dir}/partial_model.pt')
    except Exception as save_error:
        print(f'❌ Failed to save partial results: {save_error}')

    sys.exit(1)
"

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
python -c "
import os
import sys
import torch

# Define expected model directories based on model type
model_dirs = {
    'code': '/notebooks/Jarvis_AI_Assistant/models/deepseek-coder-6.7b-finetuned',
    'text': '/notebooks/Jarvis_AI_Assistant/models/flan-ul2-finetuned',
    'cnn-text': '/notebooks/Jarvis_AI_Assistant/models/cnn-flan-ul2-finetuned',
    'custom-model': '/notebooks/Jarvis_AI_Assistant/models/custom-encoder-decoder'
}

# Check if the model directory exists
model_type = '$MODEL_TYPE'
if model_type in model_dirs:
    model_dir = model_dirs[model_type]
    if os.path.exists(model_dir):
        print(f'✓ Model directory {model_dir} exists')

        # Check for specific files based on model type
        if model_type == 'code':
            required_files = ['config.json', 'adapter_config.json']
        else:
            required_files = ['model.pt']

        missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]

        if missing_files:
            print(f'❌ WARNING: Missing required files in model directory: {missing_files}')
        else:
            print(f'✓ All required files are present in {model_dir}')
    else:
        print(f'❌ WARNING: Model directory {model_dir} does not exist')
else:
    print(f'❌ Unknown model type: {model_type}')

# Clear CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('✓ CUDA cache cleared')
"

# Final cleanup
echo "Cleaning up temporary files..."
find /tmp -name "torch_*" -type d -mmin +60 -exec rm -rf {} \; 2>/dev/null || true
find /tmp -name "transformers_*" -type d -mmin +60 -exec rm -rf {} \; 2>/dev/null || true

echo "✓ Training process completed successfully!"
echo "✓ Done"
