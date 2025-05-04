#!/usr/bin/env python3
"""
Utility script to manually sync data between local Paperspace storage and Google Drive.
This can be run directly as a script to perform sync operations.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fix the import path
if __name__ == "__main__":
    # Add the parent directory to the path to make the module importable
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
    from src.generative_ai_module.utils import sync_to_gdrive, sync_from_gdrive, is_paperspace_environment
else:
    # When imported as a module, use relative imports
    from .utils import sync_to_gdrive, sync_from_gdrive, is_paperspace_environment

def sync_all_to_gdrive():
    """Sync all data folders to Google Drive."""
    if not is_paperspace_environment():
        logger.warning("Not running in Paperspace environment. Sync operations are only supported in Paperspace.")
        return False
    
    logger.info("Syncing all data to Google Drive...")
    sync_to_gdrive(None)  # None means sync all folders
    logger.info("Sync complete!")
    return True

def sync_all_from_gdrive():
    """Sync all data folders from Google Drive to local storage."""
    if not is_paperspace_environment():
        logger.warning("Not running in Paperspace environment. Sync operations are only supported in Paperspace.")
        return False
    
    logger.info("Syncing all data from Google Drive...")
    sync_from_gdrive(None)  # None means sync all folders
    logger.info("Sync complete!")
    return True

def main():
    """Command-line interface for sync operations."""
    parser = argparse.ArgumentParser(description="Sync data between Paperspace and Google Drive")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # To Google Drive command
    to_gdrive_parser = subparsers.add_parser("to-gdrive", help="Sync data to Google Drive")
    to_gdrive_parser.add_argument("--folder", choices=["models", "datasets", "metrics", "checkpoints", "preprocessed_data", "logs"],
                                help="Specific folder to sync (all folders if not specified)")
    
    # From Google Drive command
    from_gdrive_parser = subparsers.add_parser("from-gdrive", help="Sync data from Google Drive")
    from_gdrive_parser.add_argument("--folder", choices=["models", "datasets", "metrics", "checkpoints", "preprocessed_data", "logs"],
                                  help="Specific folder to sync (all folders if not specified)")
    
    # All command (syncs in both directions)
    all_parser = subparsers.add_parser("all", help="Sync in both directions")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if running in Paperspace environment
    if not is_paperspace_environment():
        logger.error("This script can only be run in a Paperspace Gradient environment with Google Drive access.")
        sys.exit(1)
    
    # Execute the appropriate command
    if args.command == "to-gdrive":
        logger.info(f"Syncing {'all folders' if args.folder is None else args.folder} to Google Drive")
        sync_to_gdrive(args.folder)
        logger.info("Sync complete!")
    
    elif args.command == "from-gdrive":
        logger.info(f"Syncing {'all folders' if args.folder is None else args.folder} from Google Drive")
        sync_from_gdrive(args.folder)
        logger.info("Sync complete!")
    
    elif args.command == "all":
        logger.info("Syncing all folders in both directions")
        sync_from_gdrive(None)
        sync_to_gdrive(None)
        logger.info("Sync complete!")
    
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() 