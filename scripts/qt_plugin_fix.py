import sys
import os
import site
import importlib
from pathlib import Path

def find_pyside6_paths():
    """Find PySide6 installation paths using multiple methods."""
    paths = []
    
    # Method 1: Use site-packages
    for site_dir in site.getsitepackages():
        qt_path = Path(site_dir) / "PySide6" / "Qt"
        if qt_path.exists():
            paths.append(qt_path)
    
    # Method 2: Check virtual environment
    venv_path = Path('.env') / 'lib' / 'python3.11' / 'site-packages' / 'PySide6' / 'Qt'
    if venv_path.exists():
        paths.append(venv_path)
    
    # Method 3: Use PySide6 module path
    try:
        if importlib.util.find_spec("PySide6"):
            import PySide6
            module_dir = Path(PySide6.__file__).parent / "Qt"
            if module_dir.exists():
                paths.append(module_dir)
    except (ImportError, AttributeError):
        pass
    
    return paths

def fix_qt_plugin_paths():
    """Apply all known fixes for Qt plugin path issues."""
    pyside_paths = find_pyside6_paths()
    
    if not pyside_paths:
        print("Error: Could not find PySide6 installation.")
        return False
    
    qt_path = pyside_paths[0]
    plugin_path = qt_path / "plugins"
    lib_path = qt_path / "lib"
    
    # Fix 1: Set environment variables
    os.environ["QT_PLUGIN_PATH"] = str(plugin_path)
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(plugin_path / "platforms")
    
    # macOS specific environment variables
    if sys.platform == "darwin":
        os.environ["DYLD_FRAMEWORK_PATH"] = str(lib_path)
        os.environ["DYLD_LIBRARY_PATH"] = str(lib_path)
        
        # Extra path that sometimes helps
        fw_path = lib_path / "QtCore.framework"
        if fw_path.exists():
            os.environ["DYLD_FRAMEWORK_PATH"] = f"{os.environ.get('DYLD_FRAMEWORK_PATH', '')}:{fw_path}"
    
    print(f"Environment variables set:")
    print(f"  QT_PLUGIN_PATH: {os.environ.get('QT_PLUGIN_PATH')}")
    print(f"  QT_QPA_PLATFORM_PLUGIN_PATH: {os.environ.get('QT_QPA_PLATFORM_PLUGIN_PATH')}")
    if sys.platform == "darwin":
        print(f"  DYLD_FRAMEWORK_PATH: {os.environ.get('DYLD_FRAMEWORK_PATH')}")
    
    # Fix 2: Programmatically add library paths when QApplication is created
    def create_patched_qapp(*args, **kwargs):
        """Create a QApplication with library paths fixed."""
        try:
            from PySide6.QtWidgets import QApplication
            from PySide6.QtCore import QCoreApplication

            # Add library paths programmatically
            QCoreApplication.addLibraryPath(str(plugin_path))
            QCoreApplication.addLibraryPath(str(plugin_path / "platforms"))
            
            # Create the application
            app = QApplication(*args, **kwargs)
            return app
            
        except ImportError as e:
            print(f"Error importing PySide6: {e}")
            raise
    
    return create_patched_qapp

# For direct use in other modules
create_qapp = fix_qt_plugin_paths()

if __name__ == "__main__":
    # Test the fix
    try:
        qapp = create_qapp(sys.argv)
        
        from PySide6.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget
        from PySide6.QtCore import Qt
        
        # Create a simple test window
        window = QMainWindow()
        window.setWindowTitle("PySide6 Plugin Fix Test")
        window.resize(400, 200)
        
        # Set up central widget
        central = QWidget()
        layout = QVBoxLayout(central)
        
        # Add a label
        label = QLabel("If you can see this, the plugin fix worked!")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        
        window.setCentralWidget(central)
        window.show()
        
        sys.exit(qapp.exec())
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc() 