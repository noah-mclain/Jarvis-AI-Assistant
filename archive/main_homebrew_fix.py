import sys
import os
import subprocess
from pathlib import Path

def find_homebrew_qt():
    """Find Qt installation from Homebrew on macOS."""
    try:
        # Check if homebrew is installed
        result = subprocess.run(['which', 'brew'], capture_output=True, text=True)
        if result.returncode != 0:
            return None
        
        # Find Qt path using homebrew
        result = subprocess.run(['brew', '--prefix', 'qt'], capture_output=True, text=True)
        if result.returncode != 0:
            return None
        
        qt_path = Path(result.stdout.strip())
        if not qt_path.exists():
            return None
            
        return qt_path
    except Exception:
        return None

def setup_paths():
    """Set up environment variables for Qt."""
    # First try homebrew Qt
    qt_path = find_homebrew_qt()
    if qt_path:
        print(f"Found Homebrew Qt at: {qt_path}")
        
        # Set environment variables
        os.environ["QT_PLUGIN_PATH"] = str(qt_path / "plugins")
        os.environ["DYLD_FRAMEWORK_PATH"] = str(qt_path / "lib")
        
        print(f"Set QT_PLUGIN_PATH to {os.environ.get('QT_PLUGIN_PATH')}")
        print(f"Set DYLD_FRAMEWORK_PATH to {os.environ.get('DYLD_FRAMEWORK_PATH')}")
        return True
    
    # Fallback to PySide6 paths
    try:
        import PySide6
        pyside_path = Path(PySide6.__file__).parent
        qt_path = pyside_path / "Qt"
        
        if qt_path.exists():
            print(f"Using PySide6 Qt at: {qt_path}")
            
            # Set environment variables
            os.environ["QT_PLUGIN_PATH"] = str(qt_path / "plugins")
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(qt_path / "plugins" / "platforms")
            os.environ["DYLD_FRAMEWORK_PATH"] = str(qt_path / "lib")
            
            print(f"Set QT_PLUGIN_PATH to {os.environ.get('QT_PLUGIN_PATH')}")
            print(f"Set QT_QPA_PLATFORM_PLUGIN_PATH to {os.environ.get('QT_QPA_PLATFORM_PLUGIN_PATH')}")
            print(f"Set DYLD_FRAMEWORK_PATH to {os.environ.get('DYLD_FRAMEWORK_PATH')}")
            return True
    except ImportError:
        pass
    
    print("Error: Could not find Qt installation")
    return False

if __name__ == "__main__":
    if setup_paths():
        # Now import PySide6 after setting environment variables
        from PySide6.QtWidgets import QApplication
        from src.main_window import MainWindow
        
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        
        # Load style sheet
        with open("styles/theme.qss", "r") as f:
            style = f.read()
            app.setStyleSheet(style)
        
        window = MainWindow()
        window.show()
        
        sys.exit(app.exec())
    else:
        print("Failed to set up Qt environment. Please check Qt installation.") 