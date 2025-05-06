import sys
import os
import site
from pathlib import Path

# Find PySide6 installation to set environment variables
pyside6_locations = []
for site_dir in site.getsitepackages():
    qt_plugin_path = Path(site_dir) / "PySide6" / "Qt" / "plugins"
    qt_lib_path = Path(site_dir) / "PySide6" / "Qt" / "lib"
    
    if qt_plugin_path.exists():
        pyside6_locations.append((str(qt_plugin_path), str(qt_lib_path)))

# Set environment variables if we found PySide6
if pyside6_locations:
    plugin_path, lib_path = pyside6_locations[0]
    
    # Set platform plugin paths
    os.environ["QT_PLUGIN_PATH"] = plugin_path
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(plugin_path, "platforms")
    
    # Set library path
    if sys.platform == "darwin":
        os.environ["DYLD_FRAMEWORK_PATH"] = lib_path
    elif sys.platform == "linux":
        os.environ["LD_LIBRARY_PATH"] = lib_path
    
    print(f"Set QT_PLUGIN_PATH to {os.environ.get('QT_PLUGIN_PATH')}")
    print(f"Set QT_QPA_PLATFORM_PLUGIN_PATH to {os.environ.get('QT_QPA_PLATFORM_PLUGIN_PATH')}")

# Now import PySide6
from PySide6.QtWidgets import QApplication
from main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Load style sheet with error handling
    try:
        with open("src/styles/theme.qss", "r") as f:
            style = f.read()
            app.setStyleSheet(style)
    except FileNotFoundError:
        print("Warning: theme.qss file not found. Using default styles.")

    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 