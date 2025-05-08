#!/bin/bash
# Script to clean up redundant files after refactoring

echo "Jarvis AI Assistant - Redundant Files Cleanup"
echo "This script will remove files that are no longer needed after refactoring."
echo

# Ask for confirmation before proceeding
read -p "Are you sure you want to delete redundant files? (y/n): " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Operation cancelled."
    exit 0
fi

# Create a backup directory
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
echo "Creating backup directory: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

# Files that have been consolidated into evaluation_metrics.py
EVAL_FILES=(
    "src/generative_ai_module/evaluate_model.py"
    "src/generative_ai_module/evaluation.py"
    "src/generative_ai_module/consolidated_evaluation.py"
)

# Files that have been consolidated into nlp_utils.py
NLP_FILES=(
    "src/generative_ai_module/check_spacy.py"
    "src/generative_ai_module/paperspace_spacy_fix.py"
    "src/generative_ai_module/fix_spacy_paperspace.py"
    "src/generative_ai_module/set_paperspace_env.py"
)

# Files that have been consolidated into import_utilities.py
IMPORT_FILES=(
    "src/generative_ai_module/fix_jarvis_imports.py"
    "src/generative_ai_module/import_fix.py"
    "src/generative_ai_module/final_fix.py"
    "src/generative_ai_module/fix_imports.py"
)

# Function to process each file
process_file() {
    local file=$1
    if [[ -f "$file" ]]; then
        echo "  - Backing up: $file"
        # Create directory structure in backup folder
        mkdir -p "$BACKUP_DIR/$(dirname "$file")"
        # Copy the file to backup
        cp "$file" "$BACKUP_DIR/$file"
        # Remove the original file
        rm "$file"
        echo "    ✓ Removed: $file"
    else
        echo "  - Skipping (not found): $file"
    fi
}

# Process evaluation files
echo "Processing evaluation files..."
for file in "${EVAL_FILES[@]}"; do
    process_file "$file"
done

# Process NLP files
echo "Processing NLP files..."
for file in "${NLP_FILES[@]}"; do
    process_file "$file"
done

# Process import files
echo "Processing import files..."
for file in "${IMPORT_FILES[@]}"; do
    process_file "$file"
done

# Create a message about the backup
echo
echo "All redundant files have been backed up to: $BACKUP_DIR"
echo "If you need to restore any files, you can find them in the backup directory."
echo
echo "Cleanup completed successfully!" 