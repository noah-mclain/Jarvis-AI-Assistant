import sys
import os
import platform
from pathlib import Path

# Set environment variables before importing PySide6
pyside_path = Path.cwd() / '.env' / 'lib' / 'python3.11' / 'site-packages' / 'PySide6'
if pyside_path.exists():
    os.environ['DYLD_FRAMEWORK_PATH'] = str(pyside_path / 'Qt' / 'lib')
    os.environ['QT_PLUGIN_PATH'] = str(pyside_path / 'Qt' / 'plugins')
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(pyside_path / 'Qt' / 'plugins' / 'platforms')

from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton
from PySide6.QtCore import QLibraryInfo

def main():
    # Print diagnostic information
    print(f"Python version: {sys.version}")
    print(f"Operating system: {platform.platform()}")
    print(f"PySide6 path: {os.path.dirname(os.path.abspath(sys.modules['PySide6'].__file__))}")
    
    for var in ['DYLD_FRAMEWORK_PATH', 'QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH']:
        print(f"{var}: {os.environ.get(var, 'Not set')}")
    
    print(f"Qt plugins directory: {QLibraryInfo.path(QLibraryInfo.PluginsPath)}")
    
    # Check if the platform plugins exist
    platform_plugin_dir = Path(QLibraryInfo.path(QLibraryInfo.PluginsPath)) / 'platforms'
    print(f"Platform plugin directory: {platform_plugin_dir}")
    print(f"Does directory exist: {platform_plugin_dir.exists()}")
    if platform_plugin_dir.exists():
        print(f"Contents: {list(platform_plugin_dir.glob('*'))}")
    
    # Create a simple application
    app = QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle("PySide6 Test")
    window.setGeometry(100, 100, 400, 200)
    
    button = QPushButton("Click me", window)
    button.setGeometry(150, 80, 100, 40)
    
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 