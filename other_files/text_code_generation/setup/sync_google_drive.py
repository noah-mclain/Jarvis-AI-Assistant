#!/usr/bin/env python3
"""
Sync files with Google Drive using rclone without mounting.

This script provides functions to sync files between local directories and Google Drive
using rclone's sync command instead of mounting.'
"""

import os
import sys
import subprocess
import logging
import time
import argparse
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Define constants
GDRIVE_REMOTE = "gdrive:"
LOCAL_BASE_DIR = "/notebooks/Jarvis_AI_Assistant"
GDRIVE_BASE_DIR = "My Drive/Jarvis_AI_Assistant"
SUBDIRS = [
    "checkpoints",
    "datasets",
    "models",
    "logs",
    "metrics",
    "preprocessed_data",
    "visualizations"
]

def run_command(cmd, check=True):
    """Run a command and return the result."""
    logger.info(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if check and result.returncode != 0:
        logger.error(f"Command failed with exit code {result.returncode}")
        logger.error(f"STDOUT: {result.stdout}")
        logger.error(f"STDERR: {result.stderr}")
        return False
    
    return result

def check_rclone_config():
    """Check if rclone is configured for Google Drive."""
    result = run_command(["rclone", "listremotes"], check=False)
    
    if result and result.returncode == 0:
        if "gdrive:" in result.stdout:
            logger.info("✅ Google Drive remote 'gdrive:' is configured")
            return True
        else:
            logger.warning("⚠️ Google Drive remote 'gdrive:' is not configured")
            return False
    else:
        logger.error("❌ Failed to run rclone listremotes")
        return False

def setup_rclone():
    """Install and configure rclone if needed."""
    # Check if rclone is installed
    result = run_command(["which", "rclone"], check=False)
    
    if result and result.returncode == 0:
        logger.info("✅ rclone is already installed")
    else:
        logger.info("Installing rclone...")
        subprocess.run(["apt-get", "update", "-q"], check=True)
        subprocess.run(["apt-get", "install", "-y", "rclone"], check=True)
        logger.info("✅ rclone installed")
    
    # Check if rclone is configured
    if not check_rclone_config():
        logger.info("Configuring rclone for Google Drive...")
        logger.info("Please follow the interactive setup process:")
        
        print("\n" + "="*70)
        print("Interactive rclone configuration for Google Drive")
        print("="*70)
        print("Please follow these steps:")
        print("1. Select 'n' for New remote")
        print("2. Enter 'gdrive' as the name")
        print("3. Select the number for 'Google Drive'")
        print("4. For client_id and client_secret, just press Enter to use the defaults")
        print("5. Select 'scope' option 1 (full access)")
        print("6. For root_folder_id, just press Enter")
        print("7. For service_account_file, just press Enter")
        print("8. Select 'y' to edit advanced config if you need to, otherwise 'n'")
        print("9. Select 'y' to use auto config")
        print("10. Follow the browser authentication steps when prompted")
        print("11. Select 'y' to confirm the configuration is correct")
        print("12. Select 'q' to quit the config process when done")
        print("="*70 + "\n")
        
        # Run rclone config interactively
        subprocess.run(["rclone", "config"])
        
        # Verify configuration
        if not check_rclone_config():
            logger.error("❌ Failed to configure rclone for Google Drive")
            return False
    
    return True

def create_directories():
    """Create local directories for Jarvis AI Assistant."""
    os.makedirs(LOCAL_BASE_DIR, exist_ok=True)
    
    for subdir in SUBDIRS:
        dir_path = os.path.join(LOCAL_BASE_DIR, subdir)
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")
    
    # Create a test file
    test_file = os.path.join(LOCAL_BASE_DIR, "sync_test.txt")
    with open(test_file, "w") as f:
        f.write(f"Sync test file created at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    logger.info(f"Created test file: {test_file}")
    
    return True

def create_gdrive_directories():
    """Create directories in Google Drive."""
    gdrive_path = f"{GDRIVE_REMOTE}{GDRIVE_BASE_DIR}"
    
    # Check if the base directory exists
    result = run_command(["rclone", "lsf", f"{GDRIVE_REMOTE}My Drive"], check=False)
    
    if result and result.returncode == 0:
        if "Jarvis_AI_Assistant/" not in result.stdout:
            # Create the base directory
            run_command(["rclone", "mkdir", gdrive_path])
            logger.info(f"Created directory in Google Drive: {gdrive_path}")
        else:
            logger.info(f"Directory already exists in Google Drive: {gdrive_path}")
    else:
        logger.warning("⚠️ Could not check if directory exists in Google Drive")
        # Try to create it anyway
        run_command(["rclone", "mkdir", gdrive_path])
    
    # Create subdirectories
    for subdir in SUBDIRS:
        subdir_path = f"{gdrive_path}/{subdir}"
        run_command(["rclone", "mkdir", subdir_path])
        logger.info(f"Created directory in Google Drive: {subdir_path}")
    
    return True

def sync_to_gdrive(local_path=None, gdrive_path=None):
    """Sync files from local to Google Drive."""
    if local_path is None:
        local_path = LOCAL_BASE_DIR
    
    if gdrive_path is None:
        gdrive_path = f"{GDRIVE_REMOTE}{GDRIVE_BASE_DIR}"
    
    logger.info(f"Syncing from local ({local_path}) to Google Drive ({gdrive_path})...")
    
    result = run_command([
        "rclone", "sync",
        local_path,
        gdrive_path,
        "--progress",
        "--update",
        "--verbose"
    ])
    
    if result:
        logger.info("✅ Successfully synced to Google Drive")
        return True
    else:
        logger.error("❌ Failed to sync to Google Drive")
        return False

def sync_from_gdrive(gdrive_path=None, local_path=None):
    """Sync files from Google Drive to local."""
    if local_path is None:
        local_path = LOCAL_BASE_DIR
    
    if gdrive_path is None:
        gdrive_path = f"{GDRIVE_REMOTE}{GDRIVE_BASE_DIR}"
    
    logger.info(f"Syncing from Google Drive ({gdrive_path}) to local ({local_path})...")
    
    result = run_command([
        "rclone", "sync",
        gdrive_path,
        local_path,
        "--progress",
        "--update",
        "--verbose"
    ])
    
    if result:
        logger.info("✅ Successfully synced from Google Drive")
        return True
    else:
        logger.error("❌ Failed to sync from Google Drive")
        return False

def setup_environment_variables():
    """Set up environment variables for Jarvis AI Assistant."""
    # Add to .bashrc
    bashrc_path = os.path.expanduser("~/.bashrc")
    
    with open(bashrc_path, "a") as f:
        f.write("\n# Jarvis AI Assistant environment variables\n")
        f.write(f"export JARVIS_STORAGE_BASE={LOCAL_BASE_DIR}\n")
        
        for subdir in SUBDIRS:
            env_var_name = f"JARVIS_{subdir.upper()}_DIR"
            env_var_value = os.path.join(LOCAL_BASE_DIR, subdir)
            f.write(f"export {env_var_name}={env_var_value}\n")
        
        # Add sync aliases
        f.write("\n# Jarvis AI Assistant sync aliases\n")
        f.write(f"alias jarvis-sync-to-drive='python {os.path.abspath(__file__)} --to-drive'\n")
        f.write(f"alias jarvis-sync-from-drive='python {os.path.abspath(__file__)} --from-drive'\n")
        f.write(f"alias jarvis-sync='python {os.path.abspath(__file__)} --both'\n")
    
    logger.info(f"Added environment variables and sync aliases to {bashrc_path}")
    
    # Set for current session
    os.environ["JARVIS_STORAGE_BASE"] = LOCAL_BASE_DIR
    
    for subdir in SUBDIRS:
        env_var_name = f"JARVIS_{subdir.upper()}_DIR"
        env_var_value = os.path.join(LOCAL_BASE_DIR, subdir)
        os.environ[env_var_name] = env_var_value
    
    return True

def create_sync_script():
    """Create a sync script that can be run periodically."""
    script_path = os.path.join(LOCAL_BASE_DIR, "sync_drive.sh")
    
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("# Sync script for Jarvis AI Assistant\n\n")
        f.write(f"python {os.path.abspath(__file__)} --both\n")
    
    os.chmod(script_path, 0o755)
    logger.info(f"Created sync script: {script_path}")
    
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Sync files with Google Drive using rclone")
    parser.add_argument("--to-drive", action="store_true", help="Sync from local to Google Drive")
    parser.add_argument("--from-drive", action="store_true", help="Sync from Google Drive to local")
    parser.add_argument("--both", action="store_true", help="Sync in both directions")
    parser.add_argument("--setup", action="store_true", help="Set up directories and environment variables")
    args = parser.parse_args()
    
    # Default to setup if no arguments are provided
    if not (args.to_drive or args.from_drive or args.both or args.setup):
        args.setup = True
    
    # Set up rclone
    if not setup_rclone():
        logger.error("❌ Failed to set up rclone")
        return False
    
    # Set up directories and environment variables
    if args.setup:
        logger.info("Setting up directories and environment variables...")
        
        if not create_directories():
            logger.error("❌ Failed to create local directories")
            return False
        
        if not create_gdrive_directories():
            logger.warning("⚠️ Failed to create Google Drive directories")
            # Continue anyway
        
        if not setup_environment_variables():
            logger.error("❌ Failed to set up environment variables")
            return False
        
        if not create_sync_script():
            logger.warning("⚠️ Failed to create sync script")
            # Continue anyway
        
        # Initial sync from Google Drive
        sync_from_gdrive()
    
    # Sync to Google Drive
    if args.to_drive or args.both:
        if not sync_to_gdrive():
            logger.error("❌ Failed to sync to Google Drive")
            return False
    
    # Sync from Google Drive
    if args.from_drive or args.both:
        if not sync_from_gdrive():
            logger.error("❌ Failed to sync from Google Drive")
            return False
    
    logger.info("✅ All operations completed successfully")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
