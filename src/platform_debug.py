import sys
import os
import platform
from pathlib import Path
import importlib.util

def check_module_installed(module_name):
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return False, None
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return True, module.__version__ if hasattr(module, "__version__") else "Unknown"

print(f"Python version: {platform.python_version()}")
print(f"System: {platform.system()} {platform.release()}")
print(f"Architecture: {platform.machine()}")
print()

# Check if PySide6 is installed
pyside6_installed, pyside6_version = check_module_installed("PySide6")
print(f"PySide6 installed: {pyside6_installed}")
print(f"PySide6 version: {pyside6_version}")

if pyside6_installed:
    try:
        from PySide6.QtCore import QLibraryInfo, QCoreApplication
        
        # Print Qt version
        print(f"Qt version: {QLibraryInfo.version().toString()}")
        
        # Print QLibraryInfo paths
        print("\nQt Library Paths:")
        print(f"Plugins: {QLibraryInfo.path(QLibraryInfo.PluginsPath)}")
        print(f"Libraries: {QLibraryInfo.path(QLibraryInfo.LibrariesPath)}")
        print(f"Binaries: {QLibraryInfo.path(QLibraryInfo.BinariesPath)}")
        print(f"Data: {QLibraryInfo.path(QLibraryInfo.DataPath)}")
        
        # Print environment variables
        print("\nEnvironment Variables:")
        for var in ['QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH', 'DYLD_FRAMEWORK_PATH', 'LD_LIBRARY_PATH', 'PATH']:
            print(f"{var}: {os.environ.get(var, 'Not set')}")
        
        # Try to locate the cocoa plugin
        plugins_dir = QLibraryInfo.path(QLibraryInfo.PluginsPath)
        cocoa_paths = list(Path(plugins_dir).glob("**/libqcocoa.*"))
        print("\nSearching for cocoa plugin:")
        if cocoa_paths:
            for path in cocoa_paths:
                print(f"Found: {path}")
                print(f"Exists: {path.exists()}")
        else:
            print("No cocoa plugin found in Qt plugin paths")
            
            # Try to find it in site-packages
            import site
            for site_dir in site.getsitepackages():
                site_cocoa_paths = list(Path(site_dir).glob("**/libqcocoa.*"))
                for path in site_cocoa_paths:
                    print(f"Found in site-packages: {path}")
                    print(f"Exists: {path.exists()}")
        
    except ImportError as e:
        print(f"Error importing from PySide6: {e}")
    except Exception as e:
        print(f"Error checking Qt paths: {e}")

print("\nApplication would initialize with:")
print(f"Current working directory: {os.getcwd()}")
print(f"sys.path[0]: {sys.path[0]}") 