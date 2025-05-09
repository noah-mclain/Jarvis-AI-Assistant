#!/bin/bash

# Script to remove redundant files after consolidation

echo "Cleaning up redundant files in the root directory and setup directory..."

# Attention Mask Fix Files
echo "Removing redundant attention mask fix files..."
rm -f setup/attention_mask_fix.py
rm -f setup/comprehensive_attention_mask_fix.py
rm -f setup/fix_all_attention_issues.py
rm -f setup/fix_attention_dimension_mismatch.py
rm -f setup/fix_attention_mask_params.py
rm -f setup/fix_attention_mask.py
rm -f setup/fix_attention_mask.sh
rm -f setup/fix_attention_without_deepseek.py
rm -f setup/fix_tensor_size_mismatch.py
rm -f setup/fix_transformers_attention_mask.py
rm -f setup/fix_ultimate_attention_fix.py
rm -f setup/ultimate_attention_fix_new.py
# Keep setup/ultimate_attention_fix.py as it's referenced in consolidated_unified_setup.sh

# DeepSeek Model Fix Files
echo "Removing redundant DeepSeek model fix files..."
rm -f setup/bypass_deepseek.py
rm -f setup/debug_unsloth.py
rm -f setup/direct_fix_deepseek.py
rm -f setup/fix_deepseek_init.py
rm -f setup/fix_deepseek_model.py
rm -f setup/fix_deepseek_training.py
rm -f setup/manual_fix_deepseek.py
rm -f setup/manual_fix_deepseek.sh

# Training Files
echo "Removing redundant training files..."
rm -f setup/train_cnn_text_model.py
rm -f setup/train_code_model.py
rm -f setup/train_custom_model.py
rm -f setup/train_custom_model.sh
rm -f setup/train_text_model.py
rm -f setup/run_training.sh
# Keep setup/train_jarvis.sh as it's the main entry point

# Verification Files
echo "Removing redundant verification files..."
rm -f setup/verify_gpu_cnn_text.py
rm -f setup/verify_gpu_code.py
rm -f setup/verify_gpu_custom_model.py
rm -f setup/verify_gpu_text.py
rm -f setup/verify_models.py
rm -f setup/verify_packages.py

# Google Drive Integration Files
echo "Removing redundant Google Drive integration files..."
rm -f setup/fix_google_drive_mount.sh
rm -f setup/mount_drive_paperspace.py
rm -f setup/mount_drive.sh
rm -f setup/sync_google_drive.py
rm -f setup/test_mount.py

# Utility and Miscellaneous Files
echo "Removing redundant utility and miscellaneous files..."
rm -f setup/adjust_python_imports.py
rm -f setup/apply_all_fixes.py
rm -f setup/clear_cuda_cache.py
rm -f setup/fix_all_setup_scripts.py
rm -f setup/fix_all_string_literals.py
rm -f setup/fix_bitsandbytes_version.py
rm -f setup/fix_dependencies.py
rm -f setup/fix_docstrings.py
rm -f setup/fix_imports.py
rm -f setup/fix_jarvis_unified.py
rm -f setup/fix_joblib.py
rm -f setup/fix_models_init.py
rm -f setup/fix_syntax_errors.py
rm -f setup/fix_tensorboard_callback.py
rm -f setup/fix_transformer_issues.py
rm -f setup/fix_transformers_utils.py
rm -f setup/fix_trl_peft_imports.py
rm -f setup/fix_trl_spacy_imports.py
rm -f setup/fix_tuple_unpacking_error.py
rm -f setup/fix_unsloth_trust_remote_code.py
rm -f setup/fix_unterminated_strings.py
rm -f setup/gpu_utils.py
rm -f setup/optimize_memory_usage.py
rm -f setup/setup_environment.py

# Test Files
echo "Removing redundant test files..."
rm -f test_attention_mask_fix.py
rm -f test_fix_integration.py
rm -f test_fix_simple.py
rm -f test_fixes.py
rm -f test_peft_fix_simple.py
rm -f test_peft_fix.py
rm -f test_ultimate_fix.py
rm -f test_unified_deepseek.py

echo "Cleanup complete!"
