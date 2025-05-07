"""
Cross-Platform Launcher for Jarvis AI Assistant
This script sets up the environment for PySide6 and launches the application.
"""

import sys
import os
import shutil
import subprocess
from pathlib import Path
import importlib.util

def find_pyside6():
    """Find PySide6 installation and return paths."""
    try:
        if not importlib.util.find_spec("PySide6"):
            print("Error: PySide6 is not installed. Please install with: pip install PySide6")
            return None
        
        import PySide6
        
        pyside_dir = Path(PySide6.__file__).parent
        qt_dir = pyside_dir / "Qt"
        
        return {
            "pyside_dir": pyside_dir,
            "qt_dir": qt_dir,
            "lib_dir": qt_dir / "lib",
            "plugins_dir": qt_dir / "plugins",
            "platforms_dir": qt_dir / "plugins" / "platforms"
        }
    except Exception as e:
        print(f"Error finding PySide6: {e}")
        return None

def set_environment():
    """Set all necessary environment variables based on the OS."""
    paths = find_pyside6()
    if not paths:
        return False
    
    # Set environment variables for Qt
    os.environ["QT_PLUGIN_PATH"] = str(paths["plugins_dir"])
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(paths["platforms_dir"])
    
    # Set library paths based on the OS
    if sys.platform == "darwin":  # macOS
        os.environ["DYLD_FRAMEWORK_PATH"] = str(paths["lib_dir"])
        os.environ["DYLD_LIBRARY_PATH"] = str(paths["lib_dir"])
    elif sys.platform == "linux":  # Linux
        os.environ["LD_LIBRARY_PATH"] = str(paths["lib_dir"])
    elif sys.platform == "win32":  # Windows
        os.environ["PATH"] += os.pathsep + str(paths["lib_dir"])
    
    print("\nEnvironment variables set:")
    for var in ["QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH", "DYLD_FRAMEWORK_PATH", "DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH"]:
        if var in os.environ:
            print(f"  {var}={os.environ[var]}")
    
    return True

def launch_app():
    """Launch the application."""
    try:
        if not set_environment():
            print("Warning: Failed to set environment variables")
            return 1  # Exit with error code
        
        print("\nLaunching Jarvis AI Assistant...")
        
        python_exe = sys.executable
        result = subprocess.run([python_exe, "src/main.py"], env=os.environ)
        
        if result.returncode != 0:
            print("Application exited with an error.")
            return result.returncode
        
        return 0  # Successful launch
        
    except Exception as e:
        print(f"Error launching application: {e}")
        return 1  # Exit with error code

if __name__ == "__main__":
    print("=== Jarvis AI Assistant Launcher ===")
    print(f"Python {sys.version} on {sys.platform}")
    
    # Launch the app
    sys.exit(launch_app())
