#!/bin/bash
VENV_PATH="/Users/nadamohamed/Documents/GitHub/Jarvis-AI-Assistant/.env"

export QT_DEBUG_PLUGINS=1
export QT_QPA_PLATFORM="cocoa"
export QT_PLUGIN_PATH="$VENV_PATH/lib/python3.11/site-packages/PySide6/Qt/plugins"
export QT_QPA_PLATFORM_PLUGIN_PATH="$QT_PLUGIN_PATH/platforms"
export DYLD_FALLBACK_LIBRARY_PATH="$VENV_PATH/lib/python3.11/site-packages/PySide6/Qt/lib:/usr/lib:/usr/local/lib"

# Sign all frameworks
codesign --force --deep --sign - \
  "$VENV_PATH/lib/python3.11/site-packages/PySide6/Qt/plugins/platforms/libqcocoa.dylib" \
  "$VENV_PATH/lib/python3.11/site-packages/PySide6/Qt/lib/QtCore.framework/Versions/A/QtCore" \
  "$VENV_PATH/lib/python3.11/site-packages/PySide6/Qt/lib/QtGui.framework/Versions/A/QtGui" \
  "$VENV_PATH/lib/python3.11/site-packages/PySide6/Qt/lib/QtWidgets.framework/Versions/A/QtWidgets"

# Run the app
"$VENV_PATH/bin/python" src/unified_launcher.py