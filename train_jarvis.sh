#!/bin/bash
# Consolidated training script for Jarvis AI Assistant
# This script combines functionality from:
# - fix_and_run_training.sh
# - direct_train_deepseek.sh
# - reset_gpu_and_train.sh
# - run_deepseek_training.sh
# - run_fixed_training.sh

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
    echo "  --model-type TYPE      Specify model type (code, text, cnn-text)"
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
echo "===== Jarvis AI Assistant Training ====="
echo "GPU Type: $GPU_TYPE"
echo "VRAM Size: $VRAM_SIZE GiB"
if [ -n "$MODEL_TYPE" ]; then
    echo "Model Type: $MODEL_TYPE"
fi
echo "========================================"

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
    print('\\n⚠️ Some required packages are missing. Please install them with:')
    print(f'pip install {\" \".join(missing_packages)}')
    print('\\nContinuing anyway, but training may fail...')
"

# Step 1: Clear CUDA cache
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

# Step 2: Set environment variables to prevent memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.8"
echo "✓ Set PYTORCH_CUDA_ALLOC_CONF to prevent memory fragmentation"

# Enable benchmarking for faster training
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDNN_V8_API_ENABLED=1
export TORCH_CUDNN_BENCHMARK=1
echo "✓ Enabled CUDNN benchmarking for faster training"

# Step 3: Monitor GPU usage in the background
echo "Starting GPU monitoring..."
python gpu_utils.py monitor --interval 5 --log-file gpu_memory_log.txt &
MONITOR_PID=$!

# Display GPU info
echo "===================================================================="
echo "GPU Information:"
nvidia-smi
echo "===================================================================="

# Clean GPU memory
if [ -z "$SKIP_CLEANUP" ]; then
    echo "Performing GPU memory cleanup..."

    # First, try to identify only GPU-using Python processes
    echo "Checking for Python processes using GPU memory..."
    THIS_PID=$$
    GPU_PIDS=()

    # Try to use nvidia-smi to find GPU processes
    if command -v nvidia-smi &> /dev/null; then
        echo "Using nvidia-smi to identify GPU processes..."
        # Get PIDs of processes using GPU
        NVIDIA_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)

        if [ -n "$NVIDIA_PIDS" ]; then
            for pid in $NVIDIA_PIDS; do
                # Check if it's a Python process
                if ps -p $pid -o comm= | grep -q "python"; then
                    if [ "$pid" != "$THIS_PID" ] && [ "$pid" != "$MONITOR_PID" ]; then
                        GPU_PIDS+=($pid)
                        echo "Found Python process $pid using GPU"
                    fi
                fi
            done
        else
            echo "No GPU processes found via nvidia-smi"
        fi
    else
        echo "nvidia-smi not found, using alternative method"
    fi

    # If no GPU processes found with nvidia-smi, use a more conservative approach
    if [ ${#GPU_PIDS[@]} -eq 0 ]; then
        echo "No GPU-specific processes identified. Using safer approach..."
        # Look for Python processes with 'torch' or 'tensorflow' in their command line
        for pid in $(ps aux | grep -E 'python.*torch|python.*tensorflow' | grep -v grep | awk '{print $2}'); do
            if [ "$pid" != "$THIS_PID" ] && [ "$pid" != "$MONITOR_PID" ]; then
                GPU_PIDS+=($pid)
                echo "Found potential GPU-using process $pid"
            fi
        done
    fi

    # If we found GPU processes, ask for confirmation before killing
    if [ ${#GPU_PIDS[@]} -gt 0 ]; then
        echo "Found ${#GPU_PIDS[@]} Python processes potentially using GPU memory."
        echo "These processes might be using GPU resources needed for training."

        # Add a --force flag check to skip confirmation
        if [ -n "$FORCE" ]; then
            CONFIRM="y"
        else
            read -p "Do you want to terminate these processes? (y/N): " CONFIRM
        fi

        if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
            for pid in "${GPU_PIDS[@]}"; do
                echo "Terminating Python process $pid (using SIGTERM first)..."
                # Try SIGTERM first (graceful shutdown)
                kill $pid 2>/dev/null

                # Wait a moment to see if it terminates
                sleep 1

                # Check if process still exists
                if ps -p $pid > /dev/null; then
                    echo "Process $pid still running, using SIGKILL..."
                    kill -9 $pid 2>/dev/null
                fi
            done
            echo "Waiting for processes to terminate..."
            sleep 2
        else
            echo "Skipping process termination as requested."
        fi
    else
        echo "No GPU-using Python processes found to terminate."
    fi

    # Clear GPU memory with our utility
    echo "Clearing GPU memory cache..."
    python gpu_utils.py clear
else
    echo "Skipping GPU memory cleanup (--skip-cleanup flag set)"
fi

# Set environment variables for optimal memory usage
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:32,garbage_collection_threshold:0.8"
export TOKENIZERS_PARALLELISM=false
export FORCE_CPU_ONLY_FOR_INITIAL_LOAD=1
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1
export FORCE_CPU_ONLY_FOR_TOKENIZATION=1
export FORCE_CPU_ONLY_FOR_DATASET_PROCESSING=1
export TRANSFORMERS_OFFLINE=0
export TOKENIZERS_FORCE_CPU=1
export HF_DATASETS_CPU_ONLY=1
export JARVIS_FORCE_CPU_TOKENIZER=1

# Apply transformer patches if not skipped
if [ -z "$SKIP_PATCHES" ]; then
    echo "Applying transformer patches..."
    python fix_transformer_issues.py

    # Apply attention mask fix for DeepSeek models
    echo "Applying attention mask fix for DeepSeek models..."
    if [ -f "fix_attention_mask.py" ]; then
        python fix_attention_mask.py
        echo "✓ Applied attention mask fix"
    else
        echo "⚠️ fix_attention_mask.py not found, creating it..."
        # Create the fix_attention_mask.py file
        cat > fix_attention_mask.py << 'EOL'
#!/usr/bin/env python3
"""
Fix for the transformers attention mask issue in DeepSeek-Coder models.

This script patches the LlamaModel.forward method to properly handle attention masks,
fixing the 'Attention mask should be of size (batch_size, 1, seq_length, seq_length)' error.
"""

import torch
import sys
import os
import inspect
from typing import Optional, Tuple, Union, List, Dict, Any

def debug_function_signature(func):
    """Print detailed information about a function's signature"""
    sig = inspect.signature(func)
    print(f"Function: {func.__name__}")
    print(f"Signature: {sig}")
    print(f"Parameters:")
    for name, param in sig.parameters.items():
        print(f"  - {name}: {param.kind} (default: {param.default if param.default is not param.empty else 'required'})")
    print()

def patch_llama_model_forward():
    """
    Patch the LlamaModel.forward method to properly handle attention masks.
    This fixes the 'Attention mask should be of size (batch_size, 1, seq_length, seq_length)' error.
    """
    try:
        from transformers.models.llama.modeling_llama import LlamaModel

        # Store the original forward method
        original_forward = LlamaModel.forward

        # Debug the original forward method
        print("\n===== Original LlamaModel.forward Method =====")
        debug_function_signature(original_forward)

        # Define a patched forward method that properly handles attention masks
        def patched_forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ):
            """
            Patched forward method for LlamaModel that properly handles attention masks.
            """
            # Force use_cache to False when using gradient checkpointing
            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                if use_cache:
                    print("use_cache=True is incompatible with gradient checkpointing. Setting use_cache=False...")
                use_cache = False

            # Fix attention mask shape if needed
            if attention_mask is not None and attention_mask.dim() == 2:
                # Get the device and dtype
                device = attention_mask.device
                dtype = attention_mask.dtype

                # Get sequence length
                seq_length = attention_mask.size(1)
                batch_size = attention_mask.size(0)

                # Convert attention_mask from [batch_size, seq_length] to [batch_size, 1, seq_length, seq_length]
                # First, expand to [batch_size, 1, 1, seq_length]
                expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)

                # Create a causal mask of shape [1, 1, seq_length, seq_length]
                causal_mask = torch.triu(
                    torch.ones((seq_length, seq_length), device=device, dtype=dtype),
                    diagonal=1
                ).unsqueeze(0).unsqueeze(0)

                # Convert masks to proper format (-inf for masked positions, 0 for attended positions)
                expanded_mask = (1.0 - expanded_mask) * -10000.0
                causal_mask = (causal_mask > 0) * -10000.0

                # Combine the masks
                combined_mask = expanded_mask + causal_mask

                # Replace the original attention_mask with our fixed version
                attention_mask = combined_mask

                print(f"Fixed attention mask shape: {attention_mask.shape}")

            # Call the original forward method with the fixed attention mask
            return original_forward(
                self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # Replace the original forward method with our patched version
        LlamaModel.forward = patched_forward

        print("Successfully patched LlamaModel.forward method")
        return True

    except ImportError as e:
        print(f"Error importing transformers: {e}")
        return False
    except Exception as e:
        print(f"Error patching LlamaModel.forward: {e}")
        return False

def patch_attention_mask_in_dataset_collator():
    """
    Patch the data collator to ensure attention masks have the correct shape.
    """
    try:
        from transformers import DataCollatorForLanguageModeling

        # Store the original __call__ method
        original_call = DataCollatorForLanguageModeling.__call__

        # Define a patched __call__ method
        def patched_call(self, features, return_tensors=None):
            batch = original_call(self, features, return_tensors)

            # Fix attention mask shape if needed
            if "attention_mask" in batch and batch["attention_mask"].dim() == 2:
                print("Fixing attention mask shape in data collator...")

                # Get dimensions
                batch_size, seq_length = batch["attention_mask"].shape
                device = batch["attention_mask"].device

                # Reshape to 4D
                batch["attention_mask"] = batch["attention_mask"].unsqueeze(1).unsqueeze(2).expand(
                    batch_size, 1, seq_length, seq_length
                )

                print(f"Attention mask shape after fix: {batch['attention_mask'].shape}")

            return batch

        # Replace the original __call__ method with our patched version
        DataCollatorForLanguageModeling.__call__ = patched_call

        print("Successfully patched DataCollatorForLanguageModeling.__call__ method")
        return True

    except ImportError as e:
        print(f"Error importing transformers: {e}")
        return False
    except Exception as e:
        print(f"Error patching data collator: {e}")
        return False

def main():
    """Main function to fix the attention mask error"""
    print("=" * 50)
    print("FIXING ATTENTION MASK ERROR")
    print("=" * 50)

    # Patch the LlamaModel.forward method
    success1 = patch_llama_model_forward()

    # Patch the data collator
    success2 = patch_attention_mask_in_dataset_collator()

    if success1 and success2:
        print("\nSuccessfully applied all patches!")
        print("The attention mask error should now be fixed.")
    else:
        print("\nSome patches failed to apply. The error might still occur.")

    print("=" * 50)

if __name__ == "__main__":
    main()
EOL
        chmod +x fix_attention_mask.py
        python fix_attention_mask.py
        echo "✓ Created and applied attention mask fix"
    fi
else
    echo "Skipping transformer patches (--skip-patches flag set)"
fi

# Set BF16 capability - always enabled for high-VRAM setups
echo "High-VRAM setup detected ($VRAM_SIZE GiB) - enabling BF16 mixed precision"
BF16_FLAG="--bf16"

# Ensure directories exist
echo "Ensuring directories exist..."
# Check if we're in Paperspace environment
if [ -d "/notebooks" ]; then
    # Paperspace environment
    BASE_DIR="/notebooks/Jarvis_AI_Assistant"
else
    # Local environment
    BASE_DIR="./Jarvis_AI_Assistant"
fi

mkdir -p "$BASE_DIR/models"
mkdir -p "$BASE_DIR/metrics"
mkdir -p "$BASE_DIR/checkpoints"
mkdir -p "$BASE_DIR/logs"
mkdir -p "$BASE_DIR/evaluation_metrics"
mkdir -p "$BASE_DIR/preprocessed_data"
mkdir -p "$BASE_DIR/visualization"

# Set output directory variable for later use
OUTPUT_DIR="$BASE_DIR"

# If model type not specified, ask user
if [ -z "$MODEL_TYPE" ]; then
    echo "Which model would you like to train?"
    echo "1) Text model (optimized for story/writing generation)"
    echo "2) Code model (optimized for code generation)"
    echo "3) CNN-enhanced text model"
    read -p "Enter choice (1-3): " choice

    case $choice in
        1) MODEL_TYPE="text" ;;
        2)
            # Use a unified code training approach
            MODEL_TYPE="code-unified"
            ;;
        3) MODEL_TYPE="cnn-text" ;;
        *) echo "Invalid choice. Exiting."; exit 1 ;;
    esac
fi

# Set training parameters based on GPU type and VRAM size
case $GPU_TYPE in
    "A6000")
        if [ $VRAM_SIZE -ge 48 ]; then
            # A6000 with 48+ GiB VRAM
            BATCH_SIZE=8
            MAX_LENGTH=2048
            GRAD_ACCUM=8
            NUM_WORKERS=8
            QUANT="--use_8bit"
        elif [ $VRAM_SIZE -ge 24 ]; then
            # A6000 with 24-48 GiB VRAM
            BATCH_SIZE=4
            MAX_LENGTH=1024
            GRAD_ACCUM=16
            NUM_WORKERS=4
            QUANT="--use_8bit"
        else
            # A6000 with <24 GiB VRAM
            BATCH_SIZE=2
            MAX_LENGTH=512
            GRAD_ACCUM=32
            NUM_WORKERS=2
            QUANT="--use_4bit"
        fi
        ;;
    "A4000")
        # A4000 with ~16 GiB VRAM
        BATCH_SIZE=1
        MAX_LENGTH=512
        GRAD_ACCUM=32
        NUM_WORKERS=1
        QUANT="--use_4bit"
        ;;
    "RTX5000")
        # RTX5000 with ~16 GiB VRAM
        BATCH_SIZE=1
        MAX_LENGTH=512
        GRAD_ACCUM=32
        NUM_WORKERS=1
        QUANT="--use_4bit"
        ;;
    *)
        echo "Unknown GPU type: $GPU_TYPE. Using conservative settings."
        BATCH_SIZE=1
        MAX_LENGTH=512
        GRAD_ACCUM=32
        NUM_WORKERS=1
        QUANT="--use_4bit"
        ;;
esac

# Run the appropriate training based on model type
case $MODEL_TYPE in
    "code")
        echo "Starting DeepSeek Coder training with $GPU_TYPE optimizations ($VRAM_SIZE GiB VRAM)..."
        echo "Executing the following command:"
        echo "python -m src.generative_ai_module.train_models \
    --model_type code \
    --model_name_or_path deepseek-ai/deepseek-coder-6.7b-instruct \
    --dataset \"code-search-net/code_search_net\" \
    --batch_size $BATCH_SIZE \
    --max_length $MAX_LENGTH \
    --gradient_accumulation_steps $GRAD_ACCUM \
    $QUANT \
    --use_qlora \
    --gradient_checkpointing \
    --optim adamw_bnb_8bit \
    --learning_rate 2e-5 \
    --weight_decay 0.05 \
    $BF16_FLAG \
    --num_workers $NUM_WORKERS \
    --cache_dir .cache \
    --force_gpu \
    --pad_token_id 50256 \
    --dataset_subset \"python\" \
    --skip_layer_freezing \
    --fim_rate 0.7 \
    --evaluation_strategy \"steps\" \
    --eval_steps 250 \
    --save_steps 500 \
    --logging_steps 50 \
    --epochs 5 \
    --output_dir \"$OUTPUT_DIR/models/deepseek-coder-6.7b-finetuned\""

        # Run the training command
        python -m src.generative_ai_module.train_models \
            --model_type code \
            --model_name_or_path deepseek-ai/deepseek-coder-6.7b-instruct \
            --dataset "code-search-net/code_search_net" \
            --batch_size $BATCH_SIZE \
            --max_length $MAX_LENGTH \
            --gradient_accumulation_steps $GRAD_ACCUM \
            $QUANT \
            --use_qlora \
            --gradient_checkpointing \
            --optim adamw_bnb_8bit \
            --learning_rate 2e-5 \
            --weight_decay 0.05 \
            $BF16_FLAG \
            --num_workers $NUM_WORKERS \
            --cache_dir .cache \
            --force_gpu \
            --pad_token_id 50256 \
            --dataset_subset "python" \
            --skip_layer_freezing \
            --fim_rate 0.7 \
            --evaluation_strategy "steps" \
            --eval_steps 250 \
            --save_steps 500 \
            --logging_steps 50 \
            --epochs 5 \
            --output_dir "$OUTPUT_DIR/models/deepseek-coder-6.7b-finetuned"
        ;;

    "code-unified")
        echo "Starting unified DeepSeek Coder training with $GPU_TYPE optimizations ($VRAM_SIZE GiB VRAM)..."
        echo "This method combines the best aspects of all training approaches with attention mask fix"

        # Check if unsloth is installed - it's optional but recommended
        UNSLOTH_AVAILABLE=0
        if python -c "import unsloth" &>/dev/null; then
            echo "✓ Unsloth optimization available"
            UNSLOTH_AVAILABLE=1
        else
            echo "⚠️ Unsloth not found. For faster training, consider installing it with: pip install unsloth"
            echo "Continuing with standard optimization..."
        fi

        # Adjust batch size for unified training
        UNIFIED_BATCH_SIZE=$((BATCH_SIZE * 2))
        if [ $UNIFIED_BATCH_SIZE -gt 16 ]; then
            UNIFIED_BATCH_SIZE=16
        fi

        # Create a unified training script that combines the best aspects
        echo "Creating unified training script..."
        cat > unified_deepseek_training.py << 'EOL'
#!/usr/bin/env python3
"""
Unified DeepSeek-Coder training script that combines the best aspects of all training approaches.
This script:
1. Applies the attention mask fix
2. Uses Unsloth optimization if available
3. Falls back to standard optimization if Unsloth is not available
4. Optimizes memory usage for different GPU types
5. Supports both 4-bit and 8-bit quantization
"""

import os
import sys
import time
import torch
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"deepseek_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Check for Unsloth availability
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
    logger.info("Unsloth optimization available")
except ImportError:
    UNSLOTH_AVAILABLE = False
    logger.info("Unsloth not available. Using standard optimization.")

# Apply attention mask fix
def apply_attention_mask_fix():
    """Apply the attention mask fix for DeepSeek models"""
    try:
        from transformers.models.llama.modeling_llama import LlamaModel

        # Store the original forward method
        original_forward = LlamaModel.forward

        # Define a patched forward method that properly handles attention masks
        def patched_forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        ):
            """Patched forward method for LlamaModel that properly handles attention masks."""
            # Force use_cache to False when using gradient checkpointing
            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                if use_cache:
                    logger.info("use_cache=True is incompatible with gradient checkpointing. Setting use_cache=False...")
                use_cache = False

            # Fix attention mask shape if needed
            if attention_mask is not None and attention_mask.dim() == 2:
                # Get the device and dtype
                device = attention_mask.device
                dtype = attention_mask.dtype

                # Get sequence length
                seq_length = attention_mask.size(1)
                batch_size = attention_mask.size(0)

                # Convert attention_mask from [batch_size, seq_length] to [batch_size, 1, seq_length, seq_length]
                # First, expand to [batch_size, 1, 1, seq_length]
                expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)

                # Create a causal mask of shape [1, 1, seq_length, seq_length]
                causal_mask = torch.triu(
                    torch.ones((seq_length, seq_length), device=device, dtype=dtype),
                    diagonal=1
                ).unsqueeze(0).unsqueeze(0)

                # Convert masks to proper format (-inf for masked positions, 0 for attended positions)
                expanded_mask = (1.0 - expanded_mask) * -10000.0
                causal_mask = (causal_mask > 0) * -10000.0

                # Combine the masks
                combined_mask = expanded_mask + causal_mask

                # Replace the original attention_mask with our fixed version
                attention_mask = combined_mask

                logger.debug(f"Fixed attention mask shape: {attention_mask.shape}")

            # Call the original forward method with the fixed attention mask
            return original_forward(
                self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # Replace the original forward method with our patched version
        LlamaModel.forward = patched_forward

        logger.info("Successfully applied attention mask fix")
        return True
    except Exception as e:
        logger.error(f"Error applying attention mask fix: {e}")
        return False

def main():
    """Main function to run the unified training"""
    parser = argparse.ArgumentParser(description="Unified DeepSeek-Coder training")
    parser.add_argument("--model_name", type=str, default="deepseek-ai/deepseek-coder-6.7b-instruct",
                        help="Model name or path")
    parser.add_argument("--output_dir", type=str, default="models/deepseek-coder-6.7b-finetuned",
                        help="Output directory")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--max_samples", type=int, default=5000,
                        help="Maximum number of samples to use")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Load model in 4-bit precision")
    parser.add_argument("--load_in_8bit", action="store_true",
                        help="Load model in 8-bit precision")
    parser.add_argument("--use_flash_attn", action="store_true",
                        help="Use Flash Attention for faster training")
    parser.add_argument("--dataset_subset", type=str, default="python",
                        help="Dataset subset to use")
    parser.add_argument("--all_subsets", action="store_true",
                        help="Use all language subsets")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Number of warmup steps")
    parser.add_argument("--bf16", action="store_true",
                        help="Use bfloat16 precision")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")

    args = parser.parse_args()

    # Apply attention mask fix
    logger.info("Applying attention mask fix...")
    apply_attention_mask_fix()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Set default quantization if none specified
    if not args.load_in_4bit and not args.load_in_8bit:
        args.load_in_4bit = True

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    # Choose the appropriate training method
    if UNSLOTH_AVAILABLE:
        logger.info("Using Unsloth-optimized training")
        train_with_unsloth(args)
    else:
        logger.info("Using standard training with attention mask fix")
        train_with_standard_method(args)

    logger.info(f"Training completed. Model saved to {args.output_dir}")

def train_with_unsloth(args):
    """Train using Unsloth optimization"""
    from unsloth import FastLanguageModel
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    from datasets import load_dataset

    logger.info("Loading and preprocessing dataset...")

    # Load dataset
    if args.all_subsets:
        logger.info("Loading all language subsets")
        from src.generative_ai_module.code_preprocessing import load_and_preprocess_all_subsets
        train_dataset, eval_dataset = load_and_preprocess_all_subsets(
            max_samples=args.max_samples,
            sequence_length=args.max_length,
            return_raw=True
        )
    else:
        logger.info(f"Loading {args.dataset_subset} subset")
        from src.generative_ai_module.code_preprocessing import load_and_preprocess_dataset
        train_dataset, eval_dataset = load_and_preprocess_dataset(
            max_samples=args.max_samples,
            sequence_length=args.max_length,
            subset=args.dataset_subset,
            all_subsets=False,
            return_raw=True
        )

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")

    # Load model with Unsloth
    logger.info(f"Loading model: {args.model_name}")
    print("🦥 Loading model", args.model_name, "with minimal unsloth")
    if args.load_in_4bit:
        print("Loading model in 4-bit quantization")
    elif args.load_in_8bit:
        print("Loading model in 8-bit quantization")

    # Note: FastLanguageModel.from_pretrained already sets trust_remote_code=True internally
    # and also sets device_map="auto" by default, so we don't need to pass these explicitly
    # Don't pass max_seq_length directly to avoid TypeError with LlamaForCausalLM
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit
    )

    # Set max sequence length after model is loaded
    model.config.max_position_embeddings = args.max_length
    tokenizer.model_max_length = args.max_length

    # Set up LoRA
    logger.info("Configuring LoRA adapters")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        use_gradient_checkpointing=True,
        random_state=42,
    )

    # Tokenize dataset
    logger.info("Tokenizing dataset")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
            return_tensors="pt"
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=args.warmup_steps,
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="steps",
        eval_steps=100,
        save_total_limit=3,
        bf16=args.bf16,
        fp16=not args.bf16 and torch.cuda.is_available(),
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        group_by_length=True,
    )

    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train model
    logger.info("Starting training...")
    trainer.train()

    # Save model
    logger.info(f"Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

def train_with_standard_method(args):
    """Train using standard method with attention mask fix"""
    logger.info("Using standard training method with attention mask fix")

    # Import the finetune_deepseek module
    try:
        from src.generative_ai_module.finetune_deepseek import main as finetune_main

        # Prepare arguments for finetune_deepseek
        sys.argv = [
            "finetune_deepseek.py",
            f"--epochs={args.epochs}",
            f"--batch-size={args.batch_size}",
            f"--learning-rate={args.learning_rate}",
            f"--sequence-length={args.max_length}",
            f"--max-samples={args.max_samples}",
            f"--warmup-steps={args.warmup_steps}",
            f"--output-dir={args.output_dir}"
        ]

        if args.load_in_4bit:
            sys.argv.append("--load-in-4bit")
        elif args.load_in_8bit:
            sys.argv.append("--load-in-8bit")

        if args.all_subsets:
            sys.argv.append("--all-subsets")
        else:
            sys.argv.append(f"--subset={args.dataset_subset}")

        # Run the finetune_deepseek main function
        logger.info("Starting finetune_deepseek with attention mask fix")
        finetune_main()

    except ImportError:
        logger.error("Could not import finetune_deepseek module. Falling back to CodeGenerator.")

        # Import CodeGenerator and related modules
        from src.generative_ai_module.code_generator import CodeGenerator
        from src.generative_ai_module.code_preprocessing import load_and_preprocess_dataset

        # Load dataset
        logger.info("Loading and preprocessing dataset...")
        train_dataset, eval_dataset = load_and_preprocess_dataset(
            max_samples=args.max_samples,
            sequence_length=args.max_length,
            subset=args.dataset_subset,
            all_subsets=args.all_subsets
        )

        # Initialize CodeGenerator
        logger.info("Initializing CodeGenerator...")
        code_gen = CodeGenerator(
            use_deepseek=True,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit
        )

        # Fine-tune the model
        logger.info("Starting fine-tuning...")
        code_gen.fine_tune_deepseek(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            sequence_length=args.max_length,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            subset=args.dataset_subset,
            all_subsets=args.all_subsets
        )

if __name__ == "__main__":
    main()
EOL
        chmod +x unified_deepseek_training.py

        # Run the unified training script
        echo "Starting unified training..."
        python unified_deepseek_training.py \
            --model_name "deepseek-ai/deepseek-coder-6.7b-instruct" \
            --output_dir "$OUTPUT_DIR/models/deepseek-coder-6.7b-finetuned" \
            --batch_size $UNIFIED_BATCH_SIZE \
            --max_length $MAX_LENGTH \
            --gradient_accumulation_steps $GRAD_ACCUM \
            --learning_rate 2e-5 \
            --epochs 3 \
            --max_samples 5000 \
            --load_in_4bit \
            --use_flash_attn \
            --dataset_subset "python" \
            --all_subsets \
            --warmup_steps 100 \
            --num_workers $NUM_WORKERS \
            $BF16_FLAG
        ;;
    "text")
        echo "Starting text model training with $GPU_TYPE optimizations ($VRAM_SIZE GiB VRAM)..."
        python -m src.generative_ai_module.train_models \
            --model_type text \
            --model_name_or_path "google/flan-ul2" \
            --dataset "euclaise/writingprompts,google/Synthetic-Persona-Chat,EleutherAI/pile,teknium/GPTeacher-General-Instruct,agie-ai/OpenAssistant-oasst1" \
            --batch_size $BATCH_SIZE \
            --max_length $MAX_LENGTH \
            --gradient_accumulation_steps $GRAD_ACCUM \
            $QUANT \
            --use_qlora \
            --gradient_checkpointing \
            --optim adamw_bnb_8bit \
            --learning_rate 1e-5 \
            --weight_decay 0.01 \
            $BF16_FLAG \
            --num_workers $NUM_WORKERS \
            --cache_dir .cache \
            --force_gpu \
            --pad_token_id 0 \
            --dataset_subset "all" \
            --skip_layer_freezing \
            --lora_r 32 \
            --lora_alpha 64 \
            --lora_dropout 0.05 \
            --evaluation_strategy "steps" \
            --eval_steps 250 \
            --save_steps 500 \
            --logging_steps 50 \
            --epochs 3 \
            --output_dir "$OUTPUT_DIR/models/flan-ul2-finetuned"
        ;;
    "cnn-text")
        echo "Starting CNN-enhanced text model training with $GPU_TYPE optimizations ($VRAM_SIZE GiB VRAM)..."
        python -m src.generative_ai_module.train_models \
            --model_type text \
            --use_cnn \
            --cnn_layers 6 \
            --dataset "agie-ai/OpenAssistant-oasst1,teknium/GPTeacher-General-Instruct,google/Synthetic-Persona-Chat,euclaise/writingprompts" \
            --model_name_or_path "google/flan-ul2" \
            --batch_size $BATCH_SIZE \
            --gradient_accumulation_steps $GRAD_ACCUM \
            --max_length $MAX_LENGTH \
            --learning_rate 3e-5 \
            --weight_decay 0.05 \
            $QUANT \
            --use_qlora \
            --gradient_checkpointing \
            --optim adamw_bnb_8bit \
            --eval_steps 250 \
            --save_steps 500 \
            --epochs 5 \
            --save_strategy steps \
            --logging_steps 50 \
            --evaluation_strategy steps \
            --sequence_packing \
            --output_dir "$OUTPUT_DIR/models/flan-ul2-cnn-finetuned" \
            --visualize_metrics \
            --use_unsloth \
            --num_workers $NUM_WORKERS \
            --cache_dir .cache \
            --force_gpu
        ;;
    *)
        echo "Invalid model type: $MODEL_TYPE. Exiting."
        exit 1
        ;;
esac

# Kill the background monitoring process
kill $MONITOR_PID

echo "Training complete!"
echo "Check gpu_memory_log.txt for memory usage during training."

# Memory usage analysis
echo "Peak GPU memory usage:"
grep -o "[0-9]\+\.[0-9]\+" gpu_memory_log.txt | sort -nr | head -1

echo "✓ Done"
