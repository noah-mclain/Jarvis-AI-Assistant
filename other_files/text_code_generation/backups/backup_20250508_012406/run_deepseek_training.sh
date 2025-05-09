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

# Step 1: Run diagnostic to check GPU memory usage
echo "Running GPU memory diagnostic..."
python diagnose_gpu_memory.py

# Step 2: Fix tokenizer to use CPU memory only
echo "Fixing tokenizer to use CPU memory only..."
python fix_tokenizer_memory.py

# Step 3: Clear GPU memory aggressively
echo "Clearing GPU memory again..."
python -c "
import torch
import gc
import os
import psutil

# Kill any zombie Python processes that might be using GPU memory
def kill_zombie_processes():
    try:
        current_process = os.getpid()
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            # Skip the current process
            if proc.info['pid'] == current_process:
                continue

            # Look for Python processes that might be using GPU
            if 'python' in proc.info['name'].lower():
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'torch' in cmdline or 'tensorflow' in cmdline or 'cuda' in cmdline:
                    print(f'Found potential GPU-using process: {proc.info}')
                    try:
                        # Try to terminate gracefully first
                        proc.terminate()
                        print(f'Terminated process {proc.info}')
                    except Exception as e:
                        print(f'Error terminating process: {e}')
    except Exception as e:
        print(f'Error in kill_zombie_processes: {e}')
        # Continue with memory cleanup even if process killing fails

# Try to kill zombie processes first
try:
    import psutil
    print('Checking for zombie processes using GPU memory...')
    kill_zombie_processes()
except ImportError:
    print('psutil not available, skipping zombie process check')

if torch.cuda.is_available():
    # Get initial memory usage
    initial_mem = torch.cuda.memory_allocated() / (1024**3)
    initial_reserved = torch.cuda.memory_reserved() / (1024**3)
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    print(f'Initial GPU memory: {initial_mem:.2f} GB allocated, {initial_reserved:.2f} GB reserved')
    print(f'Total GPU memory: {total_mem:.2f} GB')

    # First round of cleanup
    torch.cuda.empty_cache()
    gc.collect()

    # Force a second round of cleanup
    torch.cuda.empty_cache()
    gc.collect()

    # Check memory again
    current_mem = torch.cuda.memory_allocated() / (1024**3)
    current_reserved = torch.cuda.memory_reserved() / (1024**3)

    print(f'After cleanup: {current_mem:.2f} GB allocated, {current_reserved:.2f} GB reserved')
    print(f'Freed: {initial_mem - current_mem:.2f} GB allocated, {initial_reserved - current_reserved:.2f} GB reserved')
    print(f'Free GPU memory: {total_mem - current_mem:.2f} GB')

    # Set memory fraction to avoid OOM
    try:
        torch.cuda.set_per_process_memory_fraction(0.9)
        print('Set GPU memory fraction to 90% to avoid OOM errors')
    except:
        print('Could not set memory fraction, continuing anyway')
else:
    print('No GPU available, will use CPU only')
"

# Step 4: Set environment variables for optimal memory usage
echo "Setting environment variables for optimal memory usage..."
export FORCE_CPU_ONLY_FOR_INITIAL_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:32,garbage_collection_threshold:0.8"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1
export FORCE_CPU_ONLY_FOR_TOKENIZATION=1
export FORCE_CPU_ONLY_FOR_DATASET_PROCESSING=1
export TRANSFORMERS_OFFLINE=0  # Ensure we're not using cached models that might be on GPU
export TOKENIZERS_FORCE_CPU=1  # Force tokenizers to use CPU
export HF_DATASETS_CPU_ONLY=1  # Force datasets to use CPU
export JARVIS_FORCE_CPU_TOKENIZER=1  # Custom environment variable for our code

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

    # Check if there's sufficient memory for BF16
    if free_mem > 2.0:
        print('Sufficient memory for BF16 mixed precision')
    else:
        print('Limited memory, disabling BF16 mixed precision')
else:
    print('No GPU available, will use CPU only')
"

# Set BF16 capability based on available memory
if python -c "import torch; exit(0 if torch.cuda.is_available() and (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / (1024**3) > 2.0 else 1)"; then
    echo "Setting BF16_CAPABLE=true"
    BF16_CAPABLE=true
else
    echo "Setting BF16_CAPABLE=false"
    BF16_CAPABLE=false
fi

# Step 4: Apply a more robust patch for the attention mask error
echo "Applying a more robust patch for the attention mask error..."
python fix_attention_mask_error.py

# Step 5: Ensure directories exist
echo "Ensuring directories exist..."
mkdir -p /notebooks/Jarvis_AI_Assistant/models
mkdir -p /notebooks/Jarvis_AI_Assistant/metrics
mkdir -p /notebooks/Jarvis_AI_Assistant/checkpoints
mkdir -p /notebooks/Jarvis_AI_Assistant/logs
mkdir -p /notebooks/Jarvis_AI_Assistant/evaluation_metrics
mkdir -p /notebooks/Jarvis_AI_Assistant/preprocessed_data
mkdir -p /notebooks/Jarvis_AI_Assistant/visualization

# Step 6: Start GPU monitoring
echo "Starting GPU monitoring..."
(nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu --format=csv -l 10 > gpu_memory_log.txt) &
MONITOR_PID=$!

# Step 7: Run the training with optimal parameters
echo "Starting DeepSeek Coder training with memory optimizations..."

# Check if we're running on a CUDA device and set BF16 flag accordingly
if [ "$BF16_CAPABLE" = true ]; then
    echo "GPU detected with sufficient memory, enabling BF16 mixed precision"
    BF16_FLAG="--bf16"
else
    echo "GPU with limited memory or no GPU detected, disabling BF16 mixed precision"
    BF16_FLAG=""
fi

# Verify the command before execution
echo "Executing the following command:"
echo "python -m src.generative_ai_module.train_models \
    --model_type code \
    --model_name_or_path deepseek-ai/deepseek-coder-5.7b-instruct \
    --dataset \"code-search-net/code_search_net\" \
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
    --dataset_subset \"python\" \
    --skip_layer_freezing \
    --fim_rate 0.6 \
    --evaluation_strategy \"steps\" \
    --eval_steps 500 \
    --save_steps 1000 \
    --logging_steps 50 \
    --output_dir \"/notebooks/Jarvis_AI_Assistant/models/deepseek-coder-finetuned\""

# Check for unsupported arguments
# Check if the command contains unsupported arguments
if grep -q -- "--force_cpu_tokenizer\|--cpu_offload" <<< "$(cat <<EOF
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
EOF
)"; then
    echo "ERROR: Command contains unsupported arguments!"
    echo "Please remove the --force_cpu_tokenizer and --cpu_offload arguments."
    exit 1
else
    echo "✅ Command verification successful. No unsupported arguments found."
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

# Step 8: Kill the background monitoring process
kill $MONITOR_PID

echo "Training complete!"
echo "Check gpu_memory_log.txt for memory usage during training."

# Memory usage analysis
echo "Peak GPU memory usage:"
grep -o "[0-9]\+\.[0-9]\+" gpu_memory_log.txt | sort -nr | head -1

echo "✓ Done"
