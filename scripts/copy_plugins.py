import sys
import os
import shutil
import importlib
from pathlib import Path

def find_plugins():
    """Find Qt platform plugins in PySide6 installation."""
    try:
        # Try to import PySide6 to get its location
        import PySide6
        from PySide6.QtCore import QLibraryInfo
        
        # Get plugin directory from QLibraryInfo
        plugins_dir = Path(QLibraryInfo.path(QLibraryInfo.PluginsPath))
        platforms_dir = plugins_dir / "platforms"
        
        if platforms_dir.exists():
            return platforms_dir
        
        # Fallback: look relative to the PySide6 module
        pyside_dir = Path(PySide6.__file__).parent
        alt_platforms_dir = pyside_dir / "Qt" / "plugins" / "platforms"
        
        if alt_platforms_dir.exists():
            return alt_platforms_dir
            
        print(f"Could not find platforms directory in {plugins_dir} or {alt_platforms_dir}")
        return None
        
    except ImportError:
        print("Error: PySide6 is not installed")
        return None
    except Exception as e:
        print(f"Error finding plugins: {e}")
        return None

def copy_platform_plugins():
    """Copy platform plugins to current directory."""
    platforms_dir = find_plugins()
    if not platforms_dir:
        return False
    
    # Create local platforms directory if it doesn't exist
    local_platforms_dir = Path("platforms")
    local_platforms_dir.mkdir(exist_ok=True)
    
    # Find all platform plugins (like libqcocoa.dylib on macOS)
    platform_plugins = list(platforms_dir.glob("*"))
    
    if not platform_plugins:
        print(f"No platform plugins found in {platforms_dir}")
        return False
    
    # Copy each plugin
    for plugin in platform_plugins:
        target = local_platforms_dir / plugin.name
        print(f"Copying {plugin} to {target}")
        try:
            shutil.copy2(plugin, target)
        except Exception as e:
            print(f"Error copying {plugin}: {e}")
    
    # Set QT_PLUGIN_PATH to include the local directory
    os.environ["QT_PLUGIN_PATH"] = str(os.path.abspath("platforms"))
    print(f"Set QT_PLUGIN_PATH to {os.environ['QT_PLUGIN_PATH']}")
    
    # Also set QT_QPA_PLATFORM_PLUGIN_PATH for some versions
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(os.path.abspath("platforms"))
    print(f"Set QT_QPA_PLATFORM_PLUGIN_PATH to {os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']}")
    
    return True

if __name__ == "__main__":
    success = copy_platform_plugins()
    
    if success:
        print("Platform plugins copied successfully.")
        print("Now you can run your application with 'python main.py'")
    else:
        print("Failed to copy platform plugins. Please check your PySide6 installation.") 