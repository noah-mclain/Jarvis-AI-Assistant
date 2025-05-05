#!/bin/bash

# Get the absolute path of the PySide6 installation
PYSIDE_PATH=$(python -c "import PySide6; import os; print(os.path.dirname(PySide6.__file__))")
QT_PATH="$PYSIDE_PATH/Qt"

echo "PySide6 path: $PYSIDE_PATH"
echo "Qt path: $QT_PATH"

# Set environment variables
export QT_DEBUG_PLUGINS=1
export QT_PLUGIN_PATH="$QT_PATH/plugins"
export QT_QPA_PLATFORM_PLUGIN_PATH="$QT_PATH/plugins/platforms"
export DYLD_FRAMEWORK_PATH="$QT_PATH/lib"
export DYLD_PRINT_LIBRARIES=1
export DYLD_PRINT_LIBRARIES_POST_LAUNCH=1

# Enable macOS library loading debug
echo "Running with debug enabled..."
python main.py 