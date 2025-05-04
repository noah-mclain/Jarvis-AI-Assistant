import zipfile
import os
import sys
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_project_root():
    """Returns the absolute path to the project root directory."""
    # Adjust this path based on your project structure
    current_file = Path(__file__).resolve()
    # Go up two levels: from utils.py to generative_ai_module to src
    return str(current_file.parent.parent.parent)

def is_paperspace_environment():
    """Checks if code is running in Paperspace Gradient environment."""
    return os.path.exists('/notebooks') and (
        os.environ.get('PAPERSPACE') == 'true' or 
        os.path.exists('/etc/paperspace') or
        os.path.exists('/notebooks/Jarvis_AI_Assistant')
    )

def get_storage_path(folder_type, relative_path=""):
    """
    Get the appropriate storage path based on environment.
    On Paperspace, this will return the local path but also ensures Google Drive sync is configured.
    
    Args:
        folder_type (str): One of "metrics", "models", "datasets", "checkpoints", "preprocessed_data"
        relative_path (str, optional): Relative path within the folder
        
    Returns:
        str: Full path to the storage location
    """
    # Import here to avoid circular imports
    if is_paperspace_environment():
        try:
            from .google_drive_storage import GoogleDriveSync
            return GoogleDriveSync.get_local_path(folder_type, relative_path)
        except ImportError:
            # Fallback if GoogleDriveSync is not available
            return os.path.join('/notebooks/Jarvis_AI_Assistant', folder_type, relative_path)
    else:
        # For local development or other environments
        root_dir = get_project_root()
        return os.path.join(root_dir, folder_type, relative_path)

def sync_to_gdrive(folder_type=None):
    """
    Sync data to Google Drive if running in Paperspace environment.
    
    Args:
        folder_type (str, optional): One of "metrics", "models", "datasets", "checkpoints", "preprocessed_data"
                                     or None to sync all folders
    """
    if is_paperspace_environment():
        try:
            from .google_drive_storage import GoogleDriveSync
            GoogleDriveSync.sync_to_gdrive(folder_type)
        except ImportError:
            logger.warning("GoogleDriveSync module not available. No sync performed.")
    else:
        logger.info("Not running in Paperspace environment, skipping Google Drive sync.")

def sync_from_gdrive(folder_type=None):
    """
    Sync data from Google Drive if running in Paperspace environment.
    
    Args:
        folder_type (str, optional): One of "metrics", "models", "datasets", "checkpoints", "preprocessed_data"
                                     or None to sync all folders
    """
    if is_paperspace_environment():
        try:
            from .google_drive_storage import GoogleDriveSync
            GoogleDriveSync.sync_from_gdrive(folder_type)
        except ImportError:
            logger.warning("GoogleDriveSync module not available. No sync performed.")
    else:
        logger.info("Not running in Paperspace environment, skipping Google Drive sync.")

def is_zipfile(filepath):
    return zipfile.is_zipfile(filepath)

def process_zip(zip_path):
    texts = []
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                if not file_info.is_dir():
                    with zip_ref.open(file_info) as file:
                        try:
                            texts.append(file.read().decode('utf-8'))
                        except UnicodeDecodeError:
                            continue
    except zipfile.BadZipFile as e:
       raise ValueError("Invalid zip file") from e
    return texts
    