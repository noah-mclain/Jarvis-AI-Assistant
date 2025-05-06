#!/usr/bin/env python3
"""
macOS-specific launcher for Jarvis AI Assistant.
This script works around System Integrity Protection (SIP) restrictions
that can prevent DYLD environment variables from working correctly.
"""

import sys
import os
import platform
from pathlib import Path
import shutil

# Check platform
if platform.system() != "Darwin":
    print("This script is for macOS only.")
    sys.exit(1)

# Set up local plugins directory
cwd = os.path.abspath(os.path.dirname(__file__))
local_plugins_dir = os.path.join(cwd, "platforms")
os.makedirs(local_plugins_dir, exist_ok=True)

# Import PySide6 to get Qt paths
try:
    import PySide6
    from PySide6.QtCore import QLibraryInfo, QCoreApplication
    
    # Find plugins
    qt_plugin_path = QLibraryInfo.path(QLibraryInfo.PluginsPath)
    platform_plugins_path = os.path.join(qt_plugin_path, "platforms")
    
    # Copy platform plugins to local directory
    if os.path.exists(platform_plugins_path):
        for plugin in os.listdir(platform_plugins_path):
            src = os.path.join(platform_plugins_path, plugin)
            dst = os.path.join(local_plugins_dir, plugin)
            print(f"Copying {src} to {dst}")
            shutil.copy2(src, dst)
    else:
        # Fallback to manual search
        pyside_dir = os.path.dirname(PySide6.__file__)
        alt_path = os.path.join(pyside_dir, "Qt", "plugins", "platforms")
        if os.path.exists(alt_path):
            for plugin in os.listdir(alt_path):
                src = os.path.join(alt_path, plugin)
                dst = os.path.join(local_plugins_dir, plugin)
                print(f"Copying {src} to {dst}")
                shutil.copy2(src, dst)
        else:
            print(f"Warning: Could not find platform plugins in {platform_plugins_path} or {alt_path}")

    # Run the application with Qt paths set programmatically
    QCoreApplication.addLibraryPath(cwd)  # Look in current directory first
    QCoreApplication.addLibraryPath(qt_plugin_path)
    
    # Now run the app
    from PySide6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Load style sheet with error handling
    try:
        with open("src/styles/theme.qss", "r") as f:
            style = f.read()
            app.setStyleSheet(style)
    except FileNotFoundError:
        print("Warning: theme.qss file not found. Using default styles.")
    
    from main_window import MainWindow
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
    
except ImportError as e:
    print(f"Error importing PySide6: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)