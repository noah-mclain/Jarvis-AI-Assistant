#!/usr/bin/env python3
"""
Jarvis AI Assistant Launcher
This script provides a robust way to launch the application with proper Qt plugin configuration.
"""

import sys
import os
import shutil
import site
from pathlib import Path
import importlib.util
import subprocess

def find_pyside6():
    """Find PySide6 installation and return paths."""
    try:
        # Check if PySide6 is installed
        if not importlib.util.find_spec("PySide6"):
            print("Error: PySide6 is not installed. Please install with: pip install PySide6")
            return None
        
        # Import PySide6 to get its location
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

def ensure_plugins_copied():
    """Ensure platform plugins are copied to local directory."""
    paths = find_pyside6()
    if not paths:
        return False
    
    # Create local platforms directory
    local_platforms_dir = Path("platforms")
    local_platforms_dir.mkdir(exist_ok=True)
    
    # Copy platform plugins
    platforms_dir = paths["platforms_dir"]
    if not platforms_dir.exists():
        print(f"Error: Platform plugins directory not found at {platforms_dir}")
        return False
    
    plugins_copied = False
    for plugin in platforms_dir.glob("*"):
        dest = local_platforms_dir / plugin.name
        print(f"Copying {plugin.name} to {dest}")
        try:
            shutil.copy2(plugin, dest)
            plugins_copied = True
        except Exception as e:
            print(f"Error copying {plugin.name}: {e}")
    
    if not plugins_copied:
        print("No plugins were copied!")
        return False
        
    return True

def set_environment():
    """Set all necessary environment variables."""
    paths = find_pyside6()
    if not paths:
        return False
    
    # Set environment variables for Qt
    os.environ["QT_PLUGIN_PATH"] = str(paths["plugins_dir"])
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(paths["platforms_dir"])
    
    # Also include the local platforms directory in the plugin path
    local_platforms = os.path.abspath("platforms")
    if os.path.exists(local_platforms):
        os.environ["QT_PLUGIN_PATH"] = f"{os.environ['QT_PLUGIN_PATH']}:{local_platforms}"
        # On macOS, explicitly set the platform plugin path to include both
        if sys.platform == "darwin":
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = f"{local_platforms}:{os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']}"
    
    # Set library path for dylib/so/dll files
    if sys.platform == "darwin":
        os.environ["DYLD_FRAMEWORK_PATH"] = str(paths["lib_dir"])
        os.environ["DYLD_LIBRARY_PATH"] = str(paths["lib_dir"])
    elif sys.platform == "linux":
        os.environ["LD_LIBRARY_PATH"] = str(paths["lib_dir"])
    
    # Debug flag to help troubleshoot plugin issues
    os.environ["QT_DEBUG_PLUGINS"] = "1"
    
    print("\nEnvironment variables set:")
    for var in ["QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH", "DYLD_FRAMEWORK_PATH", "DYLD_LIBRARY_PATH"]:
        if var in os.environ:
            print(f"  {var}={os.environ[var]}")
    
    return True

def launch_app():
    """Launch the application."""
    try:
        # Copy plugins locally first
        if not ensure_plugins_copied():
            print("Warning: Failed to copy plugins locally")
        
        # Set environment variables
        if not set_environment():
            print("Warning: Failed to set environment variables")
            
        # Launch the main application
        print("\nLaunching Jarvis AI Assistant...")
        
        # Use the python executable that's running this script
        python_exe = sys.executable
        result = subprocess.run([python_exe, "main.py"], env=os.environ)
        
        return result.returncode
        
    except Exception as e:
        print(f"Error launching application: {e}")
        return 1

if __name__ == "__main__":
    print("=== Jarvis AI Assistant Launcher ===")
    print(f"Python {sys.version} on {sys.platform}")
    
    # Launch the app
    sys.exit(launch_app()) 