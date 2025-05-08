#!/bin/bash
# Script to clean up redundant files in the Jarvis AI Assistant codebase

echo "===== Cleaning up redundant files ====="

# Create backup directory
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
echo "Created backup directory: $BACKUP_DIR"

# Function to backup and remove a file
backup_and_remove() {
    if [ -f "$1" ]; then
        echo "Backing up and removing: $1"
        cp "$1" "$BACKUP_DIR/"
        rm "$1"
    else
        echo "File not found: $1"
    fi
}

# Attention mask fix scripts (replaced by fix_transformer_issues.py)
echo "Removing redundant attention mask fix scripts..."
backup_and_remove "fix_attention_mask.py"
backup_and_remove "fix_attention_mask_error.py"
backup_and_remove "fix_attention_mask_size.py"
backup_and_remove "fix_tokenizer_memory.py"

# GPU memory management scripts (replaced by gpu_utils.py)
echo "Removing redundant GPU memory management scripts..."
backup_and_remove "clear_gpu_memory.py"
backup_and_remove "diagnose_gpu_memory.py"
backup_and_remove "monitor_gpu.py"

# Training scripts (replaced by train_jarvis.sh)
echo "Removing redundant training scripts..."
backup_and_remove "fix_and_run_training_fixed.sh"
backup_and_remove "direct_train_deepseek.sh"
backup_and_remove "reset_gpu_and_train.sh"
backup_and_remove "run_deepseek_training.sh"
backup_and_remove "run_fixed_training.sh"
backup_and_remove "run_memory_efficient_training.sh"
backup_and_remove "run_patched_training.sh"
backup_and_remove "run_gpu_training_test.sh"
backup_and_remove "free_gpu_and_train.sh"
backup_and_remove "run_patched_training.py"
backup_and_remove "test_gpu_training.py"

# Keep fix_and_run_training.sh for backward compatibility
# but rename it to indicate it's deprecated
if [ -f "fix_and_run_training.sh" ]; then
    echo "Renaming fix_and_run_training.sh to fix_and_run_training.sh.deprecated"
    cp "fix_and_run_training.sh" "$BACKUP_DIR/"
    mv "fix_and_run_training.sh" "fix_and_run_training.sh.deprecated"
fi

# Other redundant files
echo "Removing other redundant files..."
backup_and_remove "cpu_first_patch.py"
backup_and_remove "paperspace_deepseek_train.py"

echo "===== Cleanup complete ====="
echo "All removed files have been backed up to: $BACKUP_DIR"
echo "To restore any file, use: cp $BACKUP_DIR/filename ."
