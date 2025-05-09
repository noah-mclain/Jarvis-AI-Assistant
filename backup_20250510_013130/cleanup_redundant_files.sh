#!/bin/bash

# Script to remove redundant files after consolidation

echo "Cleaning up redundant files in src/generative_ai_module/"

# Google Drive and storage files (consolidated into storage_manager.py)
rm -f src/generative_ai_module/google_drive_storage.py
rm -f src/generative_ai_module/sync_gdrive.py
rm -f src/generative_ai_module/optimize_deepseek_gdrive.py
rm -f src/generative_ai_module/optimize_deepseek_storage.py
rm -f src/generative_ai_module/storage_optimization.py

# DeepSeek training files (consolidated into deepseek_training.py)
rm -f src/generative_ai_module/deepseek_handler.py
rm -f src/generative_ai_module/unsloth_deepseek.py
rm -f src/generative_ai_module/finetune_deepseek.py
rm -f src/generative_ai_module/finetune_deepseek_examples.py
rm -f src/generative_ai_module/finetune_on_mini.py
rm -f src/generative_ai_module/fixed_run_finetune.py
rm -f src/generative_ai_module/run_finetune.py
rm -f src/generative_ai_module/unified_deepseek_training.py

# Evaluation files (consolidated into evaluation.py)
rm -f src/generative_ai_module/evaluate_generation.py
rm -f src/generative_ai_module/evaluation_example.py
rm -f src/generative_ai_module/evaluation_metrics.py

# Preprocessing files (functionality in consolidated_dataset_processor.py)
rm -f src/generative_ai_module/use_improved_preprocessor.py
rm -f src/generative_ai_module/use_new_datasets.py

# Utility files (functionality in utils.py or other consolidated files)
rm -f src/generative_ai_module/sync_finetune_log.py

echo "Cleanup complete!"
