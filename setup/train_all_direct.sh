#!/bin/bash
# Complete Jarvis AI training pipeline without problematic module imports
# Optimized for RTX5000 GPU

# Set environment variables for optimal GPU performance
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_OFFLINE=0
export HF_HUB_DISABLE_TELEMETRY=1

echo "====================================================================="
echo "🚀 STARTING DIRECT JARVIS AI TRAINING PIPELINE"
echo "📊 Optimized for RTX5000 GPU (30GB VRAM, 8 CPU, 16GB RAM)"
echo "====================================================================="

# Sync directly using rclone instead of Python module
echo "📥 Syncing latest data from Google Drive..."
# Ensure local directories exist
mkdir -p /notebooks/Jarvis_AI_Assistant/{models,datasets,checkpoints,metrics,logs,preprocessed_data}

# Sync all folders from Google Drive
for FOLDER in "models" "datasets" "checkpoints" "metrics" "logs" "preprocessed_data"; do
    echo "📂 Syncing $FOLDER from Google Drive..."
    rclone sync "gdrive:Jarvis_AI_Assistant/$FOLDER" "/notebooks/Jarvis_AI_Assistant/$FOLDER" -v
done

# Set log file
LOG_DIR="/notebooks/Jarvis_AI_Assistant/logs"
mkdir -p $LOG_DIR
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/full_training_${TIMESTAMP}.log"

echo "📝 Logging all output to $LOG_FILE"
echo "====================================================================="

# Train all model types with direct Python script call
echo "🧠 Training all Jarvis AI models in sequence..."
python /notebooks/src/generative_ai_module/train_models.py \
    --model-type all \
    --datasets all \
    --batch-size 16 \
    --epochs 3 \
    --learning-rate 2e-5 \
    --max-samples 5000 \
    --sequence-length 2048 \
    --load-in-4bit \
    --warmup-steps 100 \
    --use-deepseek \
    --deepseek-batch-size 8 \
    --code-subset python \
    --all-code-subsets \
    --validation-split 0.15 \
    --test-split 0.1 \
    --early-stopping 2 2>&1 | tee -a "$LOG_FILE"

echo "====================================================================="
echo "✅ Base training complete! Starting specialized fine-tuning..."
echo "====================================================================="

# Run DeepSeek fine-tuning with direct Python script call
echo "🔧 Fine-tuning DeepSeek model for code generation..."
python /notebooks/src/generative_ai_module/finetune_deepseek.py \
    --epochs 2 \
    --batch-size 8 \
    --learning-rate 1e-5 \
    --sequence-length 2048 \
    --max-samples 3000 \
    --all-subsets \
    --load-in-4bit \
    --warmup-steps 100 2>&1 | tee -a "$LOG_FILE"

echo "====================================================================="
echo "📊 Running comprehensive model evaluation on ALL datasets..."
echo "====================================================================="

# Evaluate on all text datasets
for DATASET in "writing_prompts" "persona_chat" "pile" "openassistant" "gpteacher"; do
    echo "Evaluating on $DATASET dataset..."
    python /notebooks/src/generative_ai_module/evaluate_generation.py \
        --model-name jarvis_unified \
        --dataset-name $DATASET \
        --batch-size 12 \
        --num-samples 500 \
        --save-results 2>&1 | tee -a "$LOG_FILE"
done

# Evaluate on code dataset
echo "Evaluating on code dataset..."
python /notebooks/src/generative_ai_module/evaluate_generation.py \
    --model-name deepseek-ai/deepseek-coder-6.7b-instruct \
    --dataset-name code_search_net \
    --batch-size 8 \
    --language python \
    --num-samples 50 \
    --save-results 2>&1 | tee -a "$LOG_FILE"

# Sync all trained models and results back to Google Drive
echo "====================================================================="
echo "☁️ Syncing all training results to Google Drive..."
echo "====================================================================="

# Sync all folders to Google Drive
for FOLDER in "models" "datasets" "checkpoints" "metrics" "logs" "preprocessed_data"; do
    echo "📂 Syncing $FOLDER to Google Drive..."
    rclone sync "/notebooks/Jarvis_AI_Assistant/$FOLDER" "gdrive:Jarvis_AI_Assistant/$FOLDER" -v
done

echo "====================================================================="
echo "🎉 TRAINING AND EVALUATION PIPELINE COMPLETE!"
echo "All models have been trained, evaluated on ALL datasets, and synced to Google Drive."
echo "You can now run the models using jarvis_unified.py or run_jarvis.py"
echo "====================================================================="
