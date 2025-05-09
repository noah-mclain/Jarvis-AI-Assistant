#!/bin/bash

# Script to remove redundant files after consolidation in src/generative_ai_module

echo "Cleaning up redundant files in src/generative_ai_module/"

# Create backup directory
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR/src/generative_ai_module"
echo "Created backup directory: $BACKUP_DIR"

# Function to backup and remove a file
backup_and_remove() {
    if [ -f "$1" ]; then
        echo "Backing up and removing: $1"
        mkdir -p "$(dirname "$BACKUP_DIR/$1")"
        cp "$1" "$BACKUP_DIR/$1"
        rm "$1"
    else
        echo "File not found: $1"
    fi
}

# Google Drive and storage files (consolidated into storage_manager.py)
echo "Removing Google Drive and storage files..."
backup_and_remove "src/generative_ai_module/google_drive_storage.py"
backup_and_remove "src/generative_ai_module/sync_gdrive.py"
backup_and_remove "src/generative_ai_module/optimize_deepseek_gdrive.py"
backup_and_remove "src/generative_ai_module/optimize_deepseek_storage.py"
backup_and_remove "src/generative_ai_module/storage_optimization.py"

# DeepSeek training files (consolidated into deepseek_training.py)
echo "Removing DeepSeek training files..."
backup_and_remove "src/generative_ai_module/deepseek_handler.py"
backup_and_remove "src/generative_ai_module/unsloth_deepseek.py"
backup_and_remove "src/generative_ai_module/finetune_deepseek.py"
backup_and_remove "src/generative_ai_module/finetune_deepseek_examples.py"
backup_and_remove "src/generative_ai_module/finetune_on_mini.py"
backup_and_remove "src/generative_ai_module/fixed_run_finetune.py"
backup_and_remove "src/generative_ai_module/run_finetune.py"
backup_and_remove "src/generative_ai_module/unified_deepseek_training.py"

# Evaluation files (consolidated into evaluation.py)
echo "Removing evaluation files..."
backup_and_remove "src/generative_ai_module/evaluate_generation.py"
backup_and_remove "src/generative_ai_module/evaluation_example.py"
backup_and_remove "src/generative_ai_module/evaluation_metrics.py"

# Dataset Processing Files (consolidated into consolidated_dataset_processor.py)
echo "Removing dataset processing files..."
backup_and_remove "src/generative_ai_module/dataset_processor.py"
backup_and_remove "src/generative_ai_module/dataset_processor_fixed.py"
backup_and_remove "src/generative_ai_module/improved_preprocessing.py"
backup_and_remove "src/generative_ai_module/unified_dataset_handler.py"
backup_and_remove "src/generative_ai_module/dataset_demo.py"
backup_and_remove "src/generative_ai_module/test_dataset.py"
backup_and_remove "src/generative_ai_module/use_improved_preprocessor.py"
backup_and_remove "src/generative_ai_module/use_new_datasets.py"

# Generation Pipeline Files (consolidated into consolidated_generation_pipeline.py)
echo "Removing generation pipeline files..."
backup_and_remove "src/generative_ai_module/text_generator.py"
backup_and_remove "src/generative_ai_module/code_generator.py"
backup_and_remove "src/generative_ai_module/unified_generation_pipeline.py"

# Backup Files
echo "Removing backup files..."
backup_and_remove "src/generative_ai_module/finetune_deepseek.py.bak"
backup_and_remove "src/generative_ai_module/train_models.py.bak"
backup_and_remove "src/generative_ai_module/unified_deepseek_training.py.bak3"

# One-time Fix Files
echo "Removing one-time fix files..."
backup_and_remove "src/generative_ai_module/fix_tuple_unpacking.py"
backup_and_remove "src/generative_ai_module/direct_model_fix.py"

# Utility files (functionality in utils.py or other consolidated files)
echo "Removing utility files..."
backup_and_remove "src/generative_ai_module/sync_finetune_log.py"

echo "Cleanup complete! All removed files have been backed up to $BACKUP_DIR"
