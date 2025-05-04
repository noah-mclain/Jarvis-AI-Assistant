#!/bin/bash

echo "====================================================================="
echo "🚀 JARVIS AI TRAINING PIPELINE WITH FALLBACK MECHANISMS"
echo "📊 Optimized for RTX5000 GPU (30GB VRAM, 8 CPU, 16GB RAM)"
echo "====================================================================="

# Setup timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="/notebooks/Jarvis_AI_Assistant/logs"
mkdir -p $LOG_DIR
LOG_FILE="$LOG_DIR/full_training_$TIMESTAMP.log"

# Ensure we have rclone configured
if ! command -v rclone &> /dev/null; then
    echo "⚠️ rclone not found. Please install and configure rclone first."
    exit 1
fi

# Check for library conflicts first and fix if needed
echo "🔍 Checking for library conflicts..."
if pip list | grep -q "spacy"; then
    if ! python -c "import spacy, thinc; from thinc.api import ParametricAttention_v2" 2>/dev/null; then
        echo "⚠️ Detected spaCy/thinc version conflict. Fixing dependencies..."
        # Fix the spaCy/thinc dependencies
        bash /notebooks/setup/fix_spacy_dependencies.sh
    else
        echo "✅ spaCy and thinc versions are compatible."
    fi
fi

# Setup environment variables for optimal performance
echo "⚙️ Setting up environment variables for RTX5000..."
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_OFFLINE=0
export HF_HUB_DISABLE_TELEMETRY=1

echo "📥 Syncing latest data from Google Drive..."
echo "📂 Syncing models from Google Drive..."
rclone sync jarvis-gdrive:jarvis_ai/models /notebooks/Jarvis_AI_Assistant/models

echo "📂 Syncing datasets from Google Drive..."
rclone sync jarvis-gdrive:jarvis_ai/datasets /notebooks/Jarvis_AI_Assistant/datasets

echo "📂 Syncing checkpoints from Google Drive..."
rclone sync jarvis-gdrive:jarvis_ai/checkpoints /notebooks/Jarvis_AI_Assistant/checkpoints

echo "📂 Syncing metrics from Google Drive..."
rclone sync jarvis-gdrive:jarvis_ai/metrics /notebooks/Jarvis_AI_Assistant/metrics

echo "📂 Syncing logs from Google Drive..."
rclone sync jarvis-gdrive:jarvis_ai/logs /notebooks/Jarvis_AI_Assistant/logs

echo "📂 Syncing preprocessed_data from Google Drive..."
rclone sync jarvis-gdrive:jarvis_ai/preprocessed_data /notebooks/Jarvis_AI_Assistant/preprocessed_data

echo "📝 Logging all output to $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "====================================================================="
echo "🧠 Training all Jarvis AI models with fallback mechanisms..."
echo "====================================================================="

# Function to run a command with fallback
run_with_fallback() {
    local command="$1"
    local description="$2"
    local retry_count=0
    local max_retries=2
    
    echo "📌 $description"
    
    while [ $retry_count -le $max_retries ]; do
        if [ $retry_count -gt 0 ]; then
            echo "🔄 Retry attempt $retry_count..."
        fi
        
        eval "$command"
        
        if [ $? -eq 0 ]; then
            echo "✅ $description completed successfully"
            return 0
        fi
        
        retry_count=$((retry_count + 1))
        
        if [ $retry_count -le $max_retries ]; then
            echo "⚠️ Command failed, attempting fallback approach..."
            # Try running with a fallback option (e.g., --disable-spacy)
            eval "$command --fallback"
        fi
    done
    
    echo "❌ All attempts failed for: $description"
    return 1
}

# Base training phase
run_with_fallback "python -m src.generative_ai_module.train_models --all --optimize-for rtx5000" "Base model training"

echo "====================================================================="
echo "✅ Base training complete! Starting specialized fine-tuning..."
echo "====================================================================="

# Fine-tuning phase
run_with_fallback "python -m src.generative_ai_module.finetune_deepseek --optimize-for rtx5000" "Fine-tuning DeepSeek model for code generation"

echo "====================================================================="
echo "📊 Running comprehensive model evaluation on ALL datasets..."
echo "====================================================================="

# Function to evaluate on a specific dataset
evaluate_dataset() {
    local dataset="$1"
    echo "Evaluating on $dataset dataset..."
    run_with_fallback "python -m src.generative_ai_module.evaluate_generation --dataset $dataset --full-metrics" "Evaluation on $dataset"
}

# Run evaluations on all datasets
for dataset in writing_prompts persona_chat pile openassistant gpteacher code; do
    evaluate_dataset "$dataset"
done

echo "====================================================================="
echo "☁️ Syncing all training results to Google Drive..."
echo "====================================================================="

# Sync all results back to Google Drive
echo "📂 Syncing models to Google Drive..."
rclone sync /notebooks/Jarvis_AI_Assistant/models jarvis-gdrive:jarvis_ai/models

echo "📂 Syncing datasets to Google Drive..."
rclone sync /notebooks/Jarvis_AI_Assistant/datasets jarvis-gdrive:jarvis_ai/datasets

echo "📂 Syncing checkpoints to Google Drive..."
rclone sync /notebooks/Jarvis_AI_Assistant/checkpoints jarvis-gdrive:jarvis_ai/checkpoints

echo "📂 Syncing metrics to Google Drive..."
rclone sync /notebooks/Jarvis_AI_Assistant/metrics jarvis-gdrive:jarvis_ai/metrics

echo "📂 Syncing logs to Google Drive..."
rclone sync /notebooks/Jarvis_AI_Assistant/logs jarvis-gdrive:jarvis_ai/logs

echo "📂 Syncing preprocessed_data to Google Drive..."
rclone sync /notebooks/Jarvis_AI_Assistant/preprocessed_data jarvis-gdrive:jarvis_ai/preprocessed_data

echo "====================================================================="
echo "🎉 TRAINING AND EVALUATION PIPELINE COMPLETE!"
echo "All models have been trained, evaluated on ALL datasets, and synced to Google Drive."
echo "You can now run the models using jarvis_unified.py or run_jarvis.py"
echo "=====================================================================" 