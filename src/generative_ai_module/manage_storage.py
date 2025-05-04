#!/usr/bin/env python3
"""
Utility script to manage storage between Paperspace local storage and Google Drive.
Helps users sync data to Google Drive and then clear local storage to save space.
"""

import os
import sys
import argparse
import logging
import shutil
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fix the import path
if __name__ == "__main__":
    # Add the parent directory to the path to make the module importable
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
    from src.generative_ai_module.utils import sync_to_gdrive, sync_from_gdrive, is_paperspace_environment
    from src.generative_ai_module.google_drive_storage import GoogleDriveSync
else:
    # When imported as a module, use relative imports
    from .utils import sync_to_gdrive, sync_from_gdrive, is_paperspace_environment
    from .google_drive_storage import GoogleDriveSync

def sync_everything_to_gdrive():
    """Sync all data to Google Drive."""
    logger.info("Syncing all data to Google Drive...")
    sync_to_gdrive()  # None means sync all folders
    logger.info("Sync complete!")
    return True

def clear_local_storage(folder_type=None, confirm=False):
    """
    Clear local storage for the specified folder type or all folders.
    
    Args:
        folder_type (str, optional): One of "metrics", "models", "datasets", "checkpoints",
                                    "preprocessed_data", "logs", or None for all folders
        confirm (bool): Whether the operation is confirmed
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not confirm:
        logger.warning("Operation not confirmed. Use --confirm to execute deletion.")
        return False
    
    if not is_paperspace_environment():
        logger.warning("Not running in Paperspace environment. This is intended for Paperspace.")
        return False
    
    # Get the folders to clear
    if folder_type:
        if folder_type not in GoogleDriveSync.SYNC_FOLDERS:
            logger.error(f"Invalid folder_type: {folder_type}")
            return False
        folders = [folder_type]
    else:
        folders = GoogleDriveSync.SYNC_FOLDERS.keys()
    
    for folder in folders:
        local_path = os.path.join(GoogleDriveSync.LOCAL_BASE, folder)
        
        if os.path.exists(local_path):
            try:
                # Sync to Google Drive first to ensure data is not lost
                sync_to_gdrive(folder)
                logger.info(f"Synced {folder} to Google Drive")
                
                # Remove the directory contents but keep the directory itself
                for item in os.listdir(local_path):
                    item_path = os.path.join(local_path, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                        logger.info(f"Removed directory: {item_path}")
                    else:
                        os.remove(item_path)
                        logger.info(f"Removed file: {item_path}")
                
                logger.info(f"Cleared local storage for {folder}")
            except Exception as e:
                logger.error(f"Error clearing {folder}: {str(e)}")
                return False
        else:
            logger.info(f"Directory {local_path} does not exist, nothing to clear")
    
    return True

def show_storage_status():
    """
    Show the current storage status for both local and Google Drive storage.
    """
    if not is_paperspace_environment():
        logger.warning("Not running in Paperspace environment. This is intended for Paperspace.")
        return False
    
    logger.info("Storage Status:")
    logger.info("---------------")
    
    # Local storage status
    logger.info("Local Storage:")
    total_local_size = 0
    
    for folder in GoogleDriveSync.SYNC_FOLDERS.keys():
        local_path = os.path.join(GoogleDriveSync.LOCAL_BASE, folder)
        
        if os.path.exists(local_path):
            folder_size = 0
            file_count = 0
            
            for dirpath, dirnames, filenames in os.walk(local_path):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    folder_size += os.path.getsize(file_path)
                    file_count += 1
            
            folder_size_mb = folder_size / (1024 * 1024)
            total_local_size += folder_size
            
            logger.info(f"  {folder}: {folder_size_mb:.2f} MB, {file_count} files")
        else:
            logger.info(f"  {folder}: Directory does not exist")
    
    total_local_size_mb = total_local_size / (1024 * 1024)
    logger.info(f"Total Local Storage: {total_local_size_mb:.2f} MB")
    
    # Google Drive storage status (using rclone)
    logger.info("\nGoogle Drive Storage:")
    try:
        for folder in GoogleDriveSync.SYNC_FOLDERS.keys():
            gdrive_path = GoogleDriveSync.SYNC_FOLDERS[folder]
            
            # Use rclone to list files and get size info
            import subprocess
            cmd = ["rclone", "size", gdrive_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"  {folder}:\n{result.stdout.strip()}")
            else:
                logger.error(f"Error getting size for {folder}: {result.stderr}")
    except Exception as e:
        logger.error(f"Error getting Google Drive storage status: {str(e)}")
    
    return True

def main():
    """Command-line interface for storage management."""
    parser = argparse.ArgumentParser(description="Manage storage between Paperspace and Google Drive")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Sync to Google Drive command
    sync_parser = subparsers.add_parser("sync", help="Sync all data to Google Drive")
    
    # Clear local storage command
    clear_parser = subparsers.add_parser("clear", help="Clear local storage after syncing to Google Drive")
    clear_parser.add_argument("--folder", choices=list(GoogleDriveSync.SYNC_FOLDERS.keys()),
                            help="Specific folder to clear (all folders if not specified)")
    clear_parser.add_argument("--confirm", action="store_true",
                            help="Confirm the deletion operation")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show storage status")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if running in Paperspace environment
    if not is_paperspace_environment():
        logger.error("This script can only be run in a Paperspace Gradient environment with Google Drive access.")
        sys.exit(1)
    
    # Execute the appropriate command
    if args.command == "sync":
        sync_everything_to_gdrive()
    
    elif args.command == "clear":
        if clear_local_storage(args.folder, args.confirm):
            logger.info("Local storage cleared successfully.")
        else:
            logger.error("Failed to clear local storage.")
            sys.exit(1)
    
    elif args.command == "status":
        show_storage_status()
    
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() 