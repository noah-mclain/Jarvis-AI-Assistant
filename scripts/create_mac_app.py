import os
import sys
import shutil
import subprocess
from pathlib import Path
import stat

# Define paths
app_name = "JarvisAI"
bundle_dir = Path(f"{app_name}.app")
contents_dir = bundle_dir / "Contents"
macos_dir = contents_dir / "MacOS"
resources_dir = contents_dir / "Resources"
frameworks_dir = contents_dir / "Frameworks"
plugins_dir = contents_dir / "PlugIns"
platforms_dir = plugins_dir / "platforms"

def find_pyside6_paths():
    """Find PySide6 Qt libraries and plugins."""
    try:
        import PySide6
        from PySide6.QtCore import QLibraryInfo
        
        pyside_dir = Path(PySide6.__file__).parent
        qt_dir = pyside_dir / "Qt"
        
        lib_dir = qt_dir / "lib"
        plugins_dir = qt_dir / "plugins"
        platform_plugins_dir = plugins_dir / "platforms"
        
        return {
            'pyside_dir': pyside_dir,
            'qt_dir': qt_dir,
            'lib_dir': lib_dir,
            'plugins_dir': plugins_dir,
            'platform_plugins_dir': platform_plugins_dir
        }
    except ImportError:
        print("Error: PySide6 is not installed")
        sys.exit(1)

def create_app_bundle():
    """Create macOS app bundle structure."""
    print(f"Creating {app_name}.app bundle...")
    
    # Create directory structure
    for directory in [macos_dir, resources_dir, frameworks_dir, platforms_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Create Info.plist
    info_plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDisplayName</key>
    <string>{app_name}</string>
    <key>CFBundleExecutable</key>
    <string>jarvis_launcher</string>
    <key>CFBundleIdentifier</key>
    <string>com.example.{app_name.lower()}</string>
    <key>CFBundleName</key>
    <string>{app_name}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
"""
    
    with open(contents_dir / "Info.plist", "w") as f:
        f.write(info_plist)
    
    # Create launcher script
    launcher_script = f"""#!/bin/bash
# Get directory where this script is located
DIR="$( cd "$( dirname "${{BASH_SOURCE[0]}}" )" && pwd )"
CONTENTS="$( dirname "$DIR" )"
ROOT="$( dirname "$CONTENTS" )"

# Set environment variables
export PYTHONPATH="$ROOT/../.."
export QT_PLUGIN_PATH="$CONTENTS/PlugIns"
export DYLD_FRAMEWORK_PATH="$CONTENTS/Frameworks"

# Launch the app
cd "$ROOT/../.."
"$PYTHON_PATH" main.py
"""
    
    # Try to find Python interpreter path
    python_path = sys.executable
    launcher_script = launcher_script.replace("$PYTHON_PATH", python_path)
    
    launcher_path = macos_dir / "jarvis_launcher"
    with open(launcher_path, "w") as f:
        f.write(launcher_script)
    
    # Make launcher executable
    launcher_path.chmod(launcher_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    
    # Copy Qt frameworks and plugins
    paths = find_pyside6_paths()
    
    # Copy platform plugins
    platform_plugins = list(paths['platform_plugins_dir'].glob("*"))
    for plugin in platform_plugins:
        if plugin.name.endswith('.dylib'):
            print(f"Copying {plugin.name} to {platforms_dir}")
            shutil.copy2(plugin, platforms_dir)
    
    # Copy essential frameworks
    essential_frameworks = ["QtCore", "QtGui", "QtWidgets"]
    for framework in essential_frameworks:
        framework_dir = paths['lib_dir'] / f"{framework}.framework"
        if framework_dir.exists():
            target_dir = frameworks_dir / f"{framework}.framework"
            print(f"Copying {framework}.framework to bundle")
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(framework_dir, target_dir)
    
    print(f"\nApp bundle created at: {bundle_dir.absolute()}")
    print(f"Launch with: open {bundle_dir}")

if __name__ == "__main__":
    if sys.platform != "darwin":
        print("This script is only for macOS")
        sys.exit(1)
        
    create_app_bundle() 