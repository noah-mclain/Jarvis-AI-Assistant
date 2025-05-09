#!/bin/bash

# Script to remove remaining redundant files in src/generative_ai_module

echo "Cleaning up remaining redundant files in src/generative_ai_module/"

# Create backup directory
BACKUP_DIR="backup_final_$(date +%Y%m%d_%H%M%S)"
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

# Training files (consolidated into consolidated_training.py and deepseek_training.py)
echo "Removing redundant training files..."
backup_and_remove "src/generative_ai_module/train_models.py"
backup_and_remove "src/generative_ai_module/train_with_cnn_model.py"
backup_and_remove "src/generative_ai_module/train_with_preprocessor.py"

# Storage files (consolidated into storage_manager.py)
echo "Removing redundant storage files..."
backup_and_remove "src/generative_ai_module/manage_storage.py"

# Inference files (consolidated into consolidated_generation_pipeline.py)
echo "Removing redundant inference files..."
backup_and_remove "src/generative_ai_module/inference_with_preprocessor.py"

# Integration files (consolidated into jarvis_unified.py)
echo "Removing redundant integration files..."
backup_and_remove "src/generative_ai_module/integration.py"

# Import utilities (consolidated into __init__.py and utils.py)
echo "Removing redundant import utilities..."
backup_and_remove "src/generative_ai_module/import_utilities.py"

# Sample data files
echo "Removing sample data files..."
backup_and_remove "src/generative_ai_module/mini_dataset.txt"

echo "Cleanup complete! All removed files have been backed up to $BACKUP_DIR"
