#!/bin/bash

# Script to remove redundant files after consolidation in root directory

echo "Cleaning up redundant files in root directory..."

# Create backup directory
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
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

# Test Files
echo "Removing redundant test files..."
backup_and_remove "test_attention_mask_fix.py"
backup_and_remove "test_fix_integration.py"
backup_and_remove "test_fix_simple.py"
backup_and_remove "test_fixes.py"
backup_and_remove "test_peft_fix_simple.py"
backup_and_remove "test_peft_fix.py"
backup_and_remove "test_ultimate_fix.py"
backup_and_remove "test_unified_deepseek.py"

# Cleanup scripts (now redundant since we've run them)
echo "Removing redundant cleanup scripts..."
backup_and_remove "cleanup_redundant_files.sh"
backup_and_remove "cleanup_repository.sh"
backup_and_remove "cleanup_root_redundant_files.sh"

# Fix scripts
echo "Removing redundant fix scripts..."
backup_and_remove "fix_deepseek_training.py"

echo "Cleanup complete! All removed files have been backed up to $BACKUP_DIR"
