#!/bin/bash
VENV_PATH="/Users/nadamohamed/Documents/GitHub/Jarvis-AI-Assistant/.env"

# Set critical paths
export QT_PLUGIN_PATH="$VENV_PATH/lib/python3.11/site-packages/PySide6/Qt/plugins"
export QT_QPA_PLATFORM_PLUGIN_PATH="$QT_PLUGIN_PATH/platforms"

# Force macOS to use the cocoa plugin
export QT_QPA_PLATFORM="cocoa"

# Fix macOS library resolution
export DYLD_FALLBACK_LIBRARY_PATH="$VENV_PATH/lib/python3.11/site-packages/PySide6/Qt/lib:/usr/lib:/usr/local/lib"

# Code sign ONLY the cocoa plugin
codesign --force --deep --sign - "$QT_PLUGIN_PATH/platforms/libqcocoa.dylib"

# Run with debug logging
"$VENV_PATH/bin/python" src/main.py