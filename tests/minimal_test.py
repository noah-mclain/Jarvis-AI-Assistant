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

try:
    from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
    from PySide6.QtCore import Qt
    
    class MinimalWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("PySide6 Minimal Test")
            self.setMinimumSize(400, 300)
            
            # Create central widget
            central_widget = QWidget()
            layout = QVBoxLayout(central_widget)
            
            # Add a button
            button = QPushButton("Test Button")
            button.clicked.connect(lambda: print("Button clicked!"))
            layout.addWidget(button)
            
            self.setCentralWidget(central_widget)
    
    if __name__ == "__main__":
        print("Creating QApplication...")
        app = QApplication(sys.argv)
        
        print("Creating window...")
        window = MinimalWindow()
        
        print("Showing window...")
        window.show()
        
        print("Starting event loop...")
        sys.exit(app.exec())
        
except Exception as e:
    print(f"Error: {e}")
    print(f"Type: {type(e).__name__}")
    import traceback
    traceback.print_exc() 