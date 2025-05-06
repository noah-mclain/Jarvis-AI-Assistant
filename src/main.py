import sys
import os
os.environ["QT_MAC_WANTS_LAYER"] = "1" 
from pathlib import Path
import site

# Configure Qt plugin paths before importing any PySide6 modules
def configure_qt_paths():
    # Find PySide6 installation
    pyside_dir = None
    qt_dir = None
    
    # Method 1: Search in site-packages
    for site_dir in site.getsitepackages():
        candidate = Path(site_dir) / "PySide6"
        if candidate.exists():
            pyside_dir = candidate
            qt_dir = pyside_dir / "Qt"
            break
    
    # Method 2: Try relative to current directory (virtual env)
    if not pyside_dir:
        candidate = Path(".env") / "lib" / "python3.11" / "site-packages" / "PySide6"
        if candidate.exists():
            pyside_dir = candidate
            qt_dir = pyside_dir / "Qt"
    
    if not qt_dir or not qt_dir.exists():
        print("Error: Could not find PySide6 Qt directory")
        return False
    
    print(f"Found Qt directory: {qt_dir}")
    
    # Set environment variables
    plugin_path = qt_dir / "plugins"
    platform_path = plugin_path / "platforms"
    lib_path = qt_dir / "lib"
    
    os.environ["QT_PLUGIN_PATH"] = str(plugin_path)
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platform_path)
    
    if sys.platform == "darwin":
        os.environ["QTWEBENGINE_DISABLE_SANDBOX"] = "1"  # Disable sandboxing (critical for macOS)
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = f"{lib_path}:/usr/lib:/usr/local/lib"  # Safer library resolution
        
    return True

# Configure Qt paths before importing PySide6
if not configure_qt_paths():
    print("Failed to configure Qt paths. Application may not start correctly.")

# Import PySide6 modules after configuring paths
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QCoreApplication, QLibraryInfo

# Also set paths programmatically (as a fallback)
plugin_dir = QLibraryInfo.path(QLibraryInfo.PluginsPath)
QCoreApplication.addLibraryPath(plugin_dir)

# Check if paths are set correctly
print(f"Library paths: {QCoreApplication.libraryPaths()}")
print(f"Plugin path from QLibraryInfo: {plugin_dir}")
print(f"Plugin path from env: {os.environ.get('QT_PLUGIN_PATH', 'Not set')}")

# Finally, import MainWindow and start app
from main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Load style sheet
    with open("src/styles/theme.qss", "r") as f:
        style = f.read()
        app.setStyleSheet(style)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())