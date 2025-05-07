#!/bin/bash
# Optimized script for running DeepSeek Coder training with memory efficiency
# and device handling fixes for GPU memory constraints

# Set error handling
set -e
trap 'echo "Error occurred at line $LINENO. Command: $BASH_COMMAND"' ERR

# Print banner
echo "============================================================"
echo "  OPTIMIZED DEEPSEEK CODER TRAINING"
echo "  Memory-efficient with CPU-first loading and GPU training"
echo "============================================================"

# Step 1: Clear GPU memory aggressively
echo "Clearing GPU memory..."
python -c "
import torch
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    initial_mem = torch.cuda.memory_allocated() / (1024**3)
    print(f'Initial GPU memory: {initial_mem:.2f} GB')

    # Force a second round of cleanup
    torch.cuda.empty_cache()
    gc.collect()

    # Check memory again
    current_mem = torch.cuda.memory_allocated() / (1024**3)
    print(f'After cleanup: {current_mem:.2f} GB')
    print(f'Freed: {initial_mem - current_mem:.2f} GB')
else:
    print('No GPU available, will use CPU only')
"

# Step 2: Set environment variables for optimal memory usage
echo "Setting environment variables for optimal memory usage..."
export FORCE_CPU_ONLY_FOR_INITIAL_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:32,garbage_collection_threshold:0.8"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1

# Step 3: Check GPU availability and memory
echo "Checking GPU availability and memory..."
python -c "
import torch
if torch.cuda.is_available():
    device = torch.device('cuda')
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    free_mem = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / (1024**3)
    print(f'GPU available: {torch.cuda.get_device_name(0)}')
    print(f'Total memory: {total_mem:.2f} GB')
    print(f'Free memory: {free_mem:.2f} GB')

    # Set BF16 flag based on GPU capability
    if free_mem > 2.0:
        print('Sufficient memory for BF16 mixed precision')
        BF16_CAPABLE=true
    else:
        print('Limited memory, disabling BF16 mixed precision')
        BF16_CAPABLE=false
else:
    print('No GPU available, will use CPU only')
    BF16_CAPABLE=false
"

# Step 4: Create the patch script for transformers library
echo "Creating transformers patch script..."
cat > patch_transformers.py << 'EOF'
#!/usr/bin/env python3
"""
Patch for transformers library to fix device mismatch issues.
This specifically targets the _unmask_unattended function that's using .cpu()
"""
import torch
import transformers.modeling_attn_mask_utils as attn_utils

# Store the original function
original_unmask_unattended = attn_utils.AttentionMaskConverter._unmask_unattended

# Define our patched version that doesn't use .cpu()
def patched_unmask_unattended(attention_mask, unmasked_value=0.0):
    """Patched version that doesn't force CPU conversion"""
    # Get the device of the attention mask
    device = attention_mask.device

    # Create a temporary tensor on the same device
    tmp = torch.ones_like(attention_mask) * unmasked_value

    # Use argmax without forcing CPU
    indices = torch.argmax(attention_mask * tmp, 1, keepdim=True)

    # Create a range tensor on the same device
    range_tensor = torch.arange(attention_mask.shape[1], device=device).expand_as(attention_mask)

    # Create the expanded mask on the same device
    expanded_mask = (range_tensor <= indices).to(attention_mask.dtype)

    return expanded_mask

# Apply the patch
attn_utils.AttentionMaskConverter._unmask_unattended = patched_unmask_unattended
print("Successfully patched transformers attention mask function")
EOF

# Make the script executable
chmod +x patch_transformers.py

# Step 5: Apply the transformers patch
echo "Applying transformers patch..."
python -c "
import sys
sys.path.insert(0, '.')
import patch_transformers
"

# Step 6: Ensure directories exist
echo "Ensuring directories exist..."
mkdir -p /notebooks/Jarvis_AI_Assistant/models
mkdir -p /notebooks/Jarvis_AI_Assistant/metrics
mkdir -p /notebooks/Jarvis_AI_Assistant/checkpoints
mkdir -p /notebooks/Jarvis_AI_Assistant/logs
mkdir -p /notebooks/Jarvis_AI_Assistant/evaluation_metrics
mkdir -p /notebooks/Jarvis_AI_Assistant/preprocessed_data
mkdir -p /notebooks/Jarvis_AI_Assistant/visualization

# Step 7: Start GPU monitoring
echo "Starting GPU monitoring..."
(nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu --format=csv -l 10 > gpu_memory_log.txt) &
MONITOR_PID=$!

# Step 8: Run the training with optimal parameters
echo "Starting DeepSeek Coder training with memory optimizations..."

# Check if we're running on a CUDA device and set BF16 flag accordingly
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    if [ "$BF16_CAPABLE" = true ]; then
        echo "CUDA device detected with sufficient memory, enabling BF16 mixed precision"
        BF16_FLAG="--bf16"
    else
        echo "CUDA device detected with limited memory, disabling BF16 mixed precision"
        BF16_FLAG=""
    fi
else
    echo "No CUDA device detected, disabling BF16 mixed precision"
    BF16_FLAG=""
fi

# Run the training with memory-efficient settings
python -m src.generative_ai_module.train_models \
    --model_type code \
    --model_name_or_path deepseek-ai/deepseek-coder-5.7b-instruct \
    --dataset "code-search-net/code_search_net" \
    --batch_size 1 \
    --max_length 512 \
    --gradient_accumulation_steps 64 \
    --use_4bit \
    --use_qlora \
    --gradient_checkpointing \
    --optim adamw_bnb_8bit \
    --learning_rate 1.5e-5 \
    --weight_decay 0.05 \
    $BF16_FLAG \
    --num_workers 1 \
    --cache_dir .cache \
    --force_gpu \
    --pad_token_id 50256 \
    --dataset_subset "python" \
    --skip_layer_freezing \
    --fim_rate 0.6 \
    --evaluation_strategy "steps" \
    --eval_steps 500 \
    --save_steps 1000 \
    --logging_steps 50 \
    --output_dir "/notebooks/Jarvis_AI_Assistant/models/deepseek-coder-finetuned"

# Step 9: Kill the background monitoring process
kill $MONITOR_PID

echo "Training complete!"
echo "Check gpu_memory_log.txt for memory usage during training."

# Memory usage analysis
echo "Peak GPU memory usage:"
grep -o "[0-9]\+\.[0-9]\+" gpu_memory_log.txt | sort -nr | head -1

echo "✓ Done"
