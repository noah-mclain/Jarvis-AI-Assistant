#!/bin/bash

echo "================================================================"
echo "Applying Minimal Unsloth to Python Files"
echo "================================================================"

# Directory paths
SRC_DIR="/notebooks/src/generative_ai_module"
CUSTOM_UNSLOTH_DIR="/notebooks/custom_unsloth"

# Make sure custom unsloth directory exists
if [ ! -d "$CUSTOM_UNSLOTH_DIR" ]; then
    echo "❌ Error: Custom unsloth directory not found at $CUSTOM_UNSLOTH_DIR"
    echo "Please run setup/create_minimal_unsloth.sh first"
    exit 1
fi

# Make sure the src directory exists
if [ ! -d "$SRC_DIR" ]; then
    echo "⚠️ Warning: Source directory not found at $SRC_DIR"
    echo "Please enter the path to your code directory:"
    read -p "> " SRC_DIR
    
    if [ ! -d "$SRC_DIR" ]; then
        echo "❌ Error: Directory not found at $SRC_DIR"
        exit 1
    fi
fi

echo "Adjusting Python files in $SRC_DIR to use minimal unsloth..."

# Run the Python script to adjust import statements
python setup/adjust_python_imports.py "$SRC_DIR" --custom-path "$CUSTOM_UNSLOTH_DIR"

echo "================================================================"
echo "Setup Complete!"
echo ""
echo "Remember to activate the minimal unsloth environment before running your code:"
echo "source $CUSTOM_UNSLOTH_DIR/activate_minimal_unsloth.sh"
echo ""
echo "If you need to restore the original Python files, you can use the .bak backups"
echo "================================================================" 