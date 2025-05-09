#!/usr/bin/env python3
"""
Utility script to sync the finetune_output.log file to Google Drive.
This allows users to archive logs and free up local Paperspace storage.
"""

import os
import sys
import argparse
import logging
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fix the import path
if __name__ == "__main__":
    # Add the parent directory to the path to make the module importable
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
    from src.generative_ai_module.utils import save_log_file, is_paperspace_environment, get_storage_path
else:
    # When imported as a module, use relative imports
    from .utils import save_log_file, is_paperspace_environment, get_storage_path

def sync_finetune_log(local_log_path, delete_after_sync=False):
    """
    Sync the finetune_output.log file to Google Drive.
    
    Args:
        local_log_path (str): Path to the finetune_output.log file
        delete_after_sync (bool): Whether to delete the local file after syncing
        
    Returns:
        bool: True if sync was successful, False otherwise
    """
    if not is_paperspace_environment():
        logger.warning("Not running in Paperspace environment. This script is intended for use in Paperspace.")
        return False
    
    if not os.path.exists(local_log_path):
        logger.error(f"Log file not found: {local_log_path}")
        return False
    
    try:
        # Read the log file content
        with open(local_log_path, 'r') as f:
            log_content = f.read()
        
        # Get the filename from the path
        filename = os.path.basename(local_log_path)
        
        # Save to Google Drive synced location
        remote_path = save_log_file(log_content, filename)
        
        logger.info(f"Synced {local_log_path} to Google Drive at {remote_path}")
        
        # Delete the local file if requested
        if delete_after_sync:
            os.remove(local_log_path)
            logger.info(f"Deleted local file: {local_log_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Failed to sync log file: {str(e)}")
        return False

def main():
    """Command-line interface for syncing finetune_output.log"""
    parser = argparse.ArgumentParser(description="Sync finetune_output.log to Google Drive")
    
    parser.add_argument("--log-file", type=str, default="finetune_output.log",
                        help="Path to the log file (default: finetune_output.log)")
    
    parser.add_argument("--delete-after-sync", action="store_true",
                        help="Delete the local log file after syncing to Google Drive")
    
    args = parser.parse_args()
    
    # Check if running in Paperspace environment
    if not is_paperspace_environment():
        logger.error("This script can only be run in a Paperspace Gradient environment with Google Drive access.")
        sys.exit(1)
    
    # Find the log file - check both the current directory and the project root
    log_path = args.log_file
    if not os.path.exists(log_path):
        # Try the project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        log_path = os.path.join(project_root, args.log_file)
    
    # Also check in the logs directory
    if not os.path.exists(log_path):
        logs_dir = get_storage_path("logs")
        log_path = os.path.join(logs_dir, args.log_file)
    
    if not os.path.exists(log_path):
        logger.error(f"Log file not found: {args.log_file}")
        logger.error("Please provide the correct path to the log file.")
        sys.exit(1)
    
    # Sync the log file
    success = sync_finetune_log(log_path, args.delete_after_sync)
    
    if success:
        logger.info("Log sync completed successfully.")
        sys.exit(0)
    else:
        logger.error("Log sync failed.")
        sys.exit(1)

if __name__ == "__main__":
    main() 