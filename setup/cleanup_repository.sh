#!/bin/bash
# Comprehensive cleanup script for Jarvis AI Assistant repository
# This script removes redundant files and organizes the repository

echo "===== Jarvis AI Assistant Repository Cleanup ====="

# Files to be removed (redundant fix scripts that have been merged)
REDUNDANT_FILES=(
  "fix_all_unsloth_issues.py"
  "fix_and_run_deepseek.sh"
  "fix_custom_unsloth.py"
  "fix_dataset_issues.py"
  "fix_dataset_processing.py"
  "fix_lora_config.py"
  "fix_random_state.py"
  "fix_tokenization.py"
  "fix_unsloth_deepseek.py"
  "fix_unsloth_issues.sh"
  "direct_edit.py"
  "edit_existing_files.sh"
  "check_unsloth_implementation.py"
  "fix_and_run_training.sh.deprecated"
)

# Create a backup directory for redundant files
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
echo "Created backup directory: $BACKUP_DIR"

# Move redundant files to backup directory
for file in "${REDUNDANT_FILES[@]}"; do
  if [ -f "$file" ]; then
    echo "Moving $file to $BACKUP_DIR/"
    mv "$file" "$BACKUP_DIR/"
  else
    echo "File $file not found, skipping"
  fi
done

# Check for duplicate run scripts
if [ -L "run_jarvis.sh" ] && [ -f "train_jarvis.sh" ]; then
  echo "run_jarvis.sh is already a symlink to train_jarvis.sh, keeping it"
else
  echo "Creating symlink run_jarvis.sh -> train_jarvis.sh"
  ln -sf train_jarvis.sh run_jarvis.sh
fi

# Check for redundant run scripts
if [ -f "run_deepseek_training.sh" ] && [ -f "run_training.sh" ]; then
  echo "Moving redundant run scripts to backup directory"
  mv run_deepseek_training.sh "$BACKUP_DIR/"
  mv run_training.sh "$BACKUP_DIR/"
fi

# Organize setup scripts
echo "Organizing setup scripts..."

# Check for redundant setup scripts in the root directory
ROOT_SETUP_SCRIPTS=(
  "cleanup_redundant_files.sh"
)

for script in "${ROOT_SETUP_SCRIPTS[@]}"; do
  if [ -f "$script" ]; then
    echo "Moving $script to $BACKUP_DIR/"
    mv "$script" "$BACKUP_DIR/"
  fi
done

echo "===== Cleanup Complete ====="
echo "Redundant files have been moved to $BACKUP_DIR/"
echo "The repository is now cleaner and better organized."
echo ""
echo "Main scripts to use:"
echo "- train_jarvis.sh: Main training script"
echo "- fix_deepseek_training.py: Comprehensive fix for DeepSeek training issues"
echo "- fix_transformer_issues.py: Fix for transformer model issues"
echo "- fix_attention_mask.py: Fix for attention mask issues"
echo "- gpu_utils.py: GPU monitoring and management utilities"
echo ""
echo "To run training:"
echo "./train_jarvis.sh --model-type code-unified"
echo ""
echo "If you encounter issues, run the fix script:"
echo "./fix_deepseek_training.py"
