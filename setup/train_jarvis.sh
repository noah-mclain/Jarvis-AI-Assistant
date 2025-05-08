#!/bin/bash
# Consolidated script for Jarvis AI Assistant
# This script sets up the environment and runs the Jarvis AI Assistant

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
echo "===== Jarvis AI Assistant ====="
echo "GPU Type: $GPU_TYPE"
echo "VRAM Size: $VRAM_SIZE GiB"
if [ -n "$MODEL_TYPE" ]; then
    echo "Model Type: $MODEL_TYPE"
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

# This is the main training script
echo "Starting Jarvis AI Assistant training..."

# Check if model type is specified
if [ -z "$MODEL_TYPE" ]; then
    echo "No model type specified. Please specify a model type with --model-type."
    echo "Available model types: code, text, cnn-text"
    exit 1
fi

# Run the appropriate training script based on model type
case $MODEL_TYPE in
    code)
        echo "Running code generation model training..."
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
            --num_workers 8
        ;;
    text)
        echo "Running text generation model training..."
        python -c "
from src.generative_ai_module.text_generator import create_cnn_text_generator
import torch

# Create and train the text generator with optimized parameters for A6000 GPU
model = create_cnn_text_generator(
    model_name='google/flan-ul2-20b',
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
    num_workers=8,  # Match your 8 CPU cores
    warmup_ratio=0.03,
    weight_decay=0.01,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    max_grad_norm=1.0
)

# Train the model
model.train_from_preprocessed(
    dataset_name='persona_chat',
    epochs=3,
    preprocessed_path='notebooks/Jarvis_AI_Assistant/datasets/preprocessed_persona_chat.pt'
)

# Save the model
model.save_model('notebooks/Jarvis_AI_Assistant/models/flan-ul2-20b-finetuned/model.pt')
print('Text generation model training completed')
"
        ;;
    cnn-text)
        echo "Running CNN-based text generation model training..."
        python -c "
from src.generative_ai_module.text_generator import create_cnn_text_generator
import torch

# Create and train the CNN-enhanced text generator with optimized parameters for A6000 GPU
model = create_cnn_text_generator(
    model_name='google/flan-ul2-20b',
    force_gpu=True,
    gpu_type='$GPU_TYPE',
    vram_size=$VRAM_SIZE,
    cnn_layers=3,  # Use 3 CNN layers for enhanced pattern recognition
    load_in_4bit=True,
    use_flash_attention_2=True,
    gradient_checkpointing=True,
    lora_rank=32,
    lora_alpha=64,
    lora_dropout=0.05,
    max_length=2048,  # Reduced from 4096 to ensure stability with FLAN-UL2
    batch_size=2,  # Further reduced for CNN layers which use more memory
    gradient_accumulation_steps=12,  # Increased to maintain effective batch size
    num_workers=8,  # Match your 8 CPU cores
    warmup_ratio=0.03,
    weight_decay=0.01,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    max_grad_norm=1.0
)

# Train the model
model.train_from_preprocessed(
    dataset_name='persona_chat',
    epochs=3,
    preprocessed_path='notebooks/Jarvis_AI_Assistant/datasets/preprocessed_persona_chat.pt'
)

# Save the model
model.save_model('notebooks/Jarvis_AI_Assistant/models/cnn-flan-ul2-20b-finetuned/model.pt')
print('CNN-enhanced text generation model training completed')
"
        ;;
    *)
        echo "Unknown model type: $MODEL_TYPE"
        echo "Available model types: code, text, cnn-text"
        exit 1
        ;;
esac

echo "✓ Done"
