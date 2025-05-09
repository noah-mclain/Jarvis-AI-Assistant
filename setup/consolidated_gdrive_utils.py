#!/usr/bin/env python3
"""
Consolidated Google Drive Utilities

This module consolidates all Google Drive integration functionality for:
- Mounting Google Drive in Paperspace and other environments
- Syncing files between local storage and Google Drive
- Testing Google Drive mounts
- Setting up directory structure for Jarvis AI Assistant

This consolidates functionality from:
- mount_drive_paperspace.py
- sync_google_drive.py
- test_mount.py
- fix_google_drive_mount.sh
- mount_drive.sh
"""

import os
import sys
import time
import logging
import subprocess
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to the path so we can import from the root
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def is_paperspace_environment():
    """Check if running in Paperspace Gradient environment"""
    return os.path.exists("/notebooks") or os.path.exists("/storage")

def is_colab_environment():
    """Check if running in Google Colab environment"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def check_rclone_installed():
    """Check if rclone is installed"""
    try:
        result = subprocess.run(["rclone", "version"], capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False

def install_rclone():
    """Install rclone if not already installed"""
    if check_rclone_installed():
        logger.info("rclone is already installed.")
        return True
    
    logger.info("Installing rclone...")
    try:
        # Install rclone
        if sys.platform == "linux" or sys.platform == "darwin":
            subprocess.run(["curl", "https://rclone.org/install.sh", "|", "sudo", "bash"], check=True, shell=True)
        elif sys.platform == "win32":
            logger.error("Automatic rclone installation on Windows is not supported.")
            logger.error("Please install rclone manually from https://rclone.org/downloads/")
            return False
        
        # Verify installation
        if check_rclone_installed():
            logger.info("rclone installed successfully.")
            return True
        else:
            logger.error("Failed to install rclone.")
            return False
    except Exception as e:
        logger.error(f"Error installing rclone: {e}")
        return False

def check_rclone_config():
    """Check if rclone is configured with a Google Drive remote"""
    try:
        result = subprocess.run(["rclone", "listremotes"], capture_output=True, text=True)
        return "gdrive:" in result.stdout
    except Exception:
        return False

def configure_rclone_interactive():
    """Configure rclone interactively"""
    logger.info("\nPlease follow these steps to configure rclone:")
    logger.info("1. Select 'n' for New remote")
    logger.info("2. Enter 'gdrive' as the name")
    logger.info("3. Select the number for 'Google Drive'")
    logger.info("4. For client_id and client_secret, just press Enter to use the defaults")
    logger.info("5. Select 'scope' option 1 (full access)")
    logger.info("6. For root_folder_id, just press Enter")
    logger.info("7. For service_account_file, just press Enter")
    logger.info("8. Select 'y' to edit advanced config if you need to, otherwise 'n'")
    logger.info("9. Select 'y' to use auto config")
    logger.info("10. Follow the browser authentication steps when prompted")
    logger.info("11. Select 'y' to confirm the configuration is correct")
    logger.info("12. Select 'q' to quit the config process when done")
    logger.info("\nStarting rclone config now...\n")
    
    # Run rclone config
    subprocess.run(["rclone", "config"])
    
    # Check if configuration was successful
    if check_rclone_config():
        logger.info("✅ Google Drive remote 'gdrive:' configured successfully")
        return True
    else:
        logger.error("Google Drive remote 'gdrive:' not found in rclone config.")
        logger.error("Please run 'rclone config' manually to set up Google Drive remote.")
        return False

def mount_google_drive(mount_point=None):
    """
    Mount Google Drive using rclone.
    
    Args:
        mount_point (str): Directory to mount Google Drive (default: auto-detect based on environment)
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Check if rclone is installed
    if not check_rclone_installed():
        if not install_rclone():
            return False
    
    # Check if rclone is configured
    if not check_rclone_config():
        if not configure_rclone_interactive():
            return False
    
    # Determine mount point
    if mount_point is None:
        if is_paperspace_environment():
            mount_point = "/notebooks/drive"
        elif is_colab_environment():
            mount_point = "/content/drive"
        else:
            mount_point = os.path.expanduser("~/gdrive")
    
    # Create mount point
    os.makedirs(mount_point, exist_ok=True)
    
    # Check if already mounted
    if os.path.ismount(mount_point):
        logger.info(f"Google Drive is already mounted at {mount_point}")
        return True
    
    # Mount Google Drive
    logger.info(f"Mounting Google Drive at {mount_point}...")
    
    cmd = [
        "rclone", "mount",
        "gdrive:", mount_point,
        "--daemon",
        "--vfs-cache-mode=full",
        "--vfs-cache-max-size=1G",
        "--dir-cache-time=1h",
        "--buffer-size=32M",
        "--transfers=4",
        "--checkers=8",
        "--drive-chunk-size=32M",
        "--timeout=1h",
        "--umask=000"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Google Drive mounted at {mount_point}")
        
        # Wait for the mount to be ready
        for _ in range(10):
            if os.path.ismount(mount_point):
                break
            time.sleep(1)
        
        if os.path.ismount(mount_point):
            logger.info("Mount successful!")
            return True
        else:
            logger.error("Failed to mount Google Drive. Mount point is not a mount.")
            return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to mount Google Drive: {e}")
        return False

def setup_jarvis_directory_structure(mount_point=None):
    """
    Set up Jarvis AI Assistant directory structure in Google Drive.
    
    Args:
        mount_point (str): Directory where Google Drive is mounted (default: auto-detect based on environment)
        
    Returns:
        dict: Directory paths
    """
    # Determine mount point
    if mount_point is None:
        if is_paperspace_environment():
            mount_point = "/notebooks/drive"
        elif is_colab_environment():
            mount_point = "/content/drive"
        else:
            mount_point = os.path.expanduser("~/gdrive")
    
    # Check if Google Drive is mounted
    if not os.path.ismount(mount_point):
        logger.error(f"Google Drive is not mounted at {mount_point}")
        return None
    
    # Define Jarvis directory structure
    if is_colab_environment():
        jarvis_dir = os.path.join(mount_point, "MyDrive/Jarvis_AI_Assistant")
    else:
        jarvis_dir = os.path.join(mount_point, "My Drive/Jarvis_AI_Assistant")
    
    # Create Jarvis directory
    os.makedirs(jarvis_dir, exist_ok=True)
    
    # Create subdirectories
    subdirs = [
        "checkpoints",
        "datasets",
        "models",
        "logs",
        "metrics",
        "preprocessed_data",
        "visualizations"
    ]
    
    dir_paths = {"root": jarvis_dir}
    
    for subdir in subdirs:
        subdir_path = os.path.join(jarvis_dir, subdir)
        os.makedirs(subdir_path, exist_ok=True)
        dir_paths[subdir] = subdir_path
    
    # Create a test file to verify the mount is working
    test_file = os.path.join(jarvis_dir, "mount_test.txt")
    try:
        with open(test_file, "w") as f:
            f.write("Google Drive mount test successful!")
        
        logger.info("✅ Successfully wrote test file to Google Drive")
    except Exception as e:
        logger.error(f"❌ Could not write test file to Google Drive: {e}")
    
    return dir_paths

def create_symbolic_links(dir_paths, local_base_dir=None):
    """
    Create symbolic links to Google Drive directories.
    
    Args:
        dir_paths (dict): Directory paths from setup_jarvis_directory_structure
        local_base_dir (str): Local base directory (default: auto-detect based on environment)
        
    Returns:
        bool: True if successful, False otherwise
    """
    if dir_paths is None:
        logger.error("Directory paths not provided.")
        return False
    
    # Determine local base directory
    if local_base_dir is None:
        if is_paperspace_environment():
            local_base_dir = "/notebooks/Jarvis_AI_Assistant"
        elif is_colab_environment():
            local_base_dir = "/content/Jarvis_AI_Assistant"
        else:
            local_base_dir = os.path.join(os.getcwd(), "Jarvis_AI_Assistant")
    
    # Create local base directory
    os.makedirs(local_base_dir, exist_ok=True)
    
    # Create symbolic links
    for subdir, remote_path in dir_paths.items():
        if subdir == "root":
            continue
        
        local_path = os.path.join(local_base_dir, subdir)
        
        # Remove existing link or directory
        if os.path.islink(local_path):
            os.unlink(local_path)
        elif os.path.isdir(local_path):
            shutil.rmtree(local_path)
        
        # Create symbolic link
        os.symlink(remote_path, local_path)
        logger.info(f"Created symbolic link: {local_path} -> {remote_path}")
    
    return True

def sync_to_gdrive(local_dir, remote_dir=None):
    """
    Sync files from local directory to Google Drive.
    
    Args:
        local_dir (str): Local directory to sync
        remote_dir (str): Remote directory in Google Drive (default: same name as local_dir)
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Check if rclone is installed
    if not check_rclone_installed():
        if not install_rclone():
            return False
    
    # Check if rclone is configured
    if not check_rclone_config():
        if not configure_rclone_interactive():
            return False
    
    # Determine remote directory
    if remote_dir is None:
        remote_dir = f"gdrive:Jarvis_AI_Assistant/{os.path.basename(local_dir)}"
    
    # Sync files
    logger.info(f"Syncing {local_dir} to {remote_dir}...")
    
    try:
        cmd = ["rclone", "copy", local_dir, remote_dir, "--progress"]
        subprocess.run(cmd, check=True)
        logger.info(f"Successfully synced {local_dir} to {remote_dir}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error syncing to Google Drive: {e}")
        return False

def sync_from_gdrive(remote_dir, local_dir=None):
    """
    Sync files from Google Drive to local directory.
    
    Args:
        remote_dir (str): Remote directory in Google Drive
        local_dir (str): Local directory to sync (default: same name as remote_dir)
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Check if rclone is installed
    if not check_rclone_installed():
        if not install_rclone():
            return False
    
    # Check if rclone is configured
    if not check_rclone_config():
        if not configure_rclone_interactive():
            return False
    
    # Determine local directory
    if local_dir is None:
        local_dir = os.path.basename(remote_dir)
    
    # Create local directory
    os.makedirs(local_dir, exist_ok=True)
    
    # Sync files
    logger.info(f"Syncing {remote_dir} to {local_dir}...")
    
    try:
        cmd = ["rclone", "copy", remote_dir, local_dir, "--progress"]
        subprocess.run(cmd, check=True)
        logger.info(f"Successfully synced {remote_dir} to {local_dir}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error syncing from Google Drive: {e}")
        return False

def test_google_drive_mount(mount_point=None):
    """
    Test Google Drive mount.
    
    Args:
        mount_point (str): Directory where Google Drive is mounted (default: auto-detect based on environment)
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Determine mount point
    if mount_point is None:
        if is_paperspace_environment():
            mount_point = "/notebooks/drive"
        elif is_colab_environment():
            mount_point = "/content/drive"
        else:
            mount_point = os.path.expanduser("~/gdrive")
    
    # Check if Google Drive is mounted
    if not os.path.ismount(mount_point):
        logger.error(f"Google Drive is not mounted at {mount_point}")
        return False
    
    # Create a test file
    test_file = os.path.join(mount_point, "test_mount.txt")
    try:
        with open(test_file, "w") as f:
            f.write("Google Drive mount test successful!")
        
        logger.info(f"✅ Successfully wrote test file to {test_file}")
        
        # Read the test file
        with open(test_file, "r") as f:
            content = f.read()
        
        logger.info(f"✅ Successfully read test file: {content}")
        
        # Remove the test file
        os.remove(test_file)
        logger.info(f"✅ Successfully removed test file")
        
        return True
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def setup_google_drive():
    """
    Set up Google Drive integration.
    
    This function:
    1. Installs rclone if needed
    2. Configures rclone if needed
    3. Mounts Google Drive
    4. Sets up Jarvis directory structure
    5. Creates symbolic links
    6. Tests the mount
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Install rclone if needed
    if not check_rclone_installed():
        if not install_rclone():
            return False
    
    # Configure rclone if needed
    if not check_rclone_config():
        if not configure_rclone_interactive():
            return False
    
    # Mount Google Drive
    if not mount_google_drive():
        return False
    
    # Set up Jarvis directory structure
    dir_paths = setup_jarvis_directory_structure()
    if dir_paths is None:
        return False
    
    # Create symbolic links
    if not create_symbolic_links(dir_paths):
        return False
    
    # Test the mount
    if not test_google_drive_mount():
        return False
    
    logger.info("Google Drive integration set up successfully!")
    return True

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Google Drive utilities for Jarvis AI Assistant.")
    parser.add_argument("--action", type=str, default="setup",
                        choices=["setup", "mount", "test", "sync-to", "sync-from"],
                        help="Action to perform")
    parser.add_argument("--local-dir", type=str, help="Local directory for sync operations")
    parser.add_argument("--remote-dir", type=str, help="Remote directory for sync operations")
    args = parser.parse_args()
    
    # Perform the requested action
    if args.action == "setup":
        setup_google_drive()
    elif args.action == "mount":
        mount_google_drive()
    elif args.action == "test":
        test_google_drive_mount()
    elif args.action == "sync-to":
        if args.local_dir:
            sync_to_gdrive(args.local_dir, args.remote_dir)
        else:
            logger.error("Local directory not specified.")
    elif args.action == "sync-from":
        if args.remote_dir:
            sync_from_gdrive(args.remote_dir, args.local_dir)
        else:
            logger.error("Remote directory not specified.")
