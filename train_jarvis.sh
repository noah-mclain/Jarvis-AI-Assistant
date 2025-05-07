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
    echo "Performing aggressive GPU memory cleanup..."
    # Kill all Python processes except this one
    echo "Killing all other Python processes to free GPU memory..."
    THIS_PID=$$
    for pid in $(ps aux | grep python | grep -v grep | awk '{print $2}'); do
        if [ "$pid" != "$THIS_PID" ] && [ "$pid" != "$MONITOR_PID" ]; then
            echo "Killing Python process $pid"
            kill -9 $pid 2>/dev/null
        fi
    done
    sleep 2

    # Clear GPU memory with our utility
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
        2) MODEL_TYPE="code" ;;
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
