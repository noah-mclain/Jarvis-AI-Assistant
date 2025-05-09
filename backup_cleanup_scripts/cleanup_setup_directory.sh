#!/bin/bash

# Script to remove redundant files after consolidation in setup directory

echo "Cleaning up redundant files in setup directory..."

# Create backup directory
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR/setup"
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

# Attention Mask Fix Files
echo "Removing redundant attention mask fix files..."
backup_and_remove "setup/attention_mask_fix.py"
backup_and_remove "setup/comprehensive_attention_mask_fix.py"
backup_and_remove "setup/fix_all_attention_issues.py"
backup_and_remove "setup/fix_attention_dimension_mismatch.py"
backup_and_remove "setup/fix_attention_mask_params.py"
backup_and_remove "setup/fix_attention_mask.py"
backup_and_remove "setup/fix_attention_mask.sh"
backup_and_remove "setup/fix_attention_without_deepseek.py"
backup_and_remove "setup/fix_tensor_size_mismatch.py"
backup_and_remove "setup/fix_transformers_attention_mask.py"
backup_and_remove "setup/fix_ultimate_attention_fix.py"
backup_and_remove "setup/ultimate_attention_fix_new.py"
# Keep setup/ultimate_attention_fix.py as it's referenced in consolidated_unified_setup.sh

# DeepSeek Model Fix Files
echo "Removing redundant DeepSeek model fix files..."
backup_and_remove "setup/bypass_deepseek.py"
backup_and_remove "setup/debug_unsloth.py"
backup_and_remove "setup/direct_fix_deepseek.py"
backup_and_remove "setup/fix_deepseek_init.py"
backup_and_remove "setup/fix_deepseek_model.py"
backup_and_remove "setup/fix_deepseek_training.py"
backup_and_remove "setup/manual_fix_deepseek.py"
backup_and_remove "setup/manual_fix_deepseek.sh"

# Training Files
echo "Removing redundant training files..."
backup_and_remove "setup/train_cnn_text_model.py"
backup_and_remove "setup/train_code_model.py"
backup_and_remove "setup/train_custom_model.py"
backup_and_remove "setup/train_custom_model.sh"
backup_and_remove "setup/train_text_model.py"
backup_and_remove "setup/run_training.sh"
# Keep setup/train_jarvis.sh as it's the main entry point

# Verification Files
echo "Removing redundant verification files..."
backup_and_remove "setup/verify_gpu_cnn_text.py"
backup_and_remove "setup/verify_gpu_code.py"
backup_and_remove "setup/verify_gpu_custom_model.py"
backup_and_remove "setup/verify_gpu_text.py"
backup_and_remove "setup/verify_models.py"
backup_and_remove "setup/verify_packages.py"

# Google Drive Integration Files
echo "Removing redundant Google Drive integration files..."
backup_and_remove "setup/fix_google_drive_mount.sh"
backup_and_remove "setup/mount_drive_paperspace.py"
backup_and_remove "setup/mount_drive.sh"
backup_and_remove "setup/sync_google_drive.py"
backup_and_remove "setup/test_mount.py"

# Utility and Miscellaneous Files
echo "Removing redundant utility and miscellaneous files..."
backup_and_remove "setup/adjust_python_imports.py"
backup_and_remove "setup/apply_all_fixes.py"
backup_and_remove "setup/clear_cuda_cache.py"
backup_and_remove "setup/fix_all_setup_scripts.py"
backup_and_remove "setup/fix_all_string_literals.py"
backup_and_remove "setup/fix_bitsandbytes_version.py"
backup_and_remove "setup/fix_dependencies.py"
backup_and_remove "setup/fix_docstrings.py"
backup_and_remove "setup/fix_imports.py"
backup_and_remove "setup/fix_jarvis_unified.py"
backup_and_remove "setup/fix_joblib.py"
backup_and_remove "setup/fix_models_init.py"
backup_and_remove "setup/fix_syntax_errors.py"
backup_and_remove "setup/fix_tensorboard_callback.py"
backup_and_remove "setup/fix_transformer_issues.py"
backup_and_remove "setup/fix_transformers_utils.py"
backup_and_remove "setup/fix_trl_peft_imports.py"
backup_and_remove "setup/fix_trl_spacy_imports.py"
backup_and_remove "setup/fix_tuple_unpacking_error.py"
backup_and_remove "setup/fix_unsloth_trust_remote_code.py"
backup_and_remove "setup/fix_unterminated_strings.py"
backup_and_remove "setup/gpu_utils.py"
backup_and_remove "setup/optimize_memory_usage.py"
backup_and_remove "setup/setup_environment.py"

echo "Cleanup complete! All removed files have been backed up to $BACKUP_DIR"
