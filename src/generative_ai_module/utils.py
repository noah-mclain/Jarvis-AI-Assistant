import zipfile
import os
import sys
import json
import logging
from pathlib import Path
import datetime

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

def ensure_directory_exists(folder_type, relative_path=""):
    """
    Ensure that a directory exists, syncing from Google Drive first if needed.
    This handles cases where local directories have been deleted.
    
    Args:
        folder_type (str): One of "metrics", "models", "datasets", "checkpoints", 
                           "preprocessed_data", "logs"
        relative_path (str, optional): Relative path within the folder
        
    Returns:
        str: Full path to the directory
    """
    # Get the appropriate path
    directory = get_storage_path(folder_type, relative_path)
    
    # If the directory doesn't exist and we're in Paperspace, try to sync from Google Drive
    if not os.path.exists(directory) and is_paperspace_environment():
        try:
            # Create parent directory if it doesn't exist
            os.makedirs(os.path.dirname(directory), exist_ok=True)
            
            # Try to sync from Google Drive to get the directory contents
            logger.info(f"Directory {directory} doesn't exist. Attempting to sync from Google Drive...")
            sync_from_gdrive(folder_type)
        except Exception as e:
            logger.warning(f"Error syncing from Google Drive: {str(e)}")
    
    # Create the directory if it still doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    return directory

def get_storage_path(folder_type, relative_path=""):
    """
    Get the appropriate storage path based on environment.
    On Paperspace, this will return the local path but also ensures Google Drive sync is configured.
    
    Args:
        folder_type (str): One of "metrics", "models", "datasets", "checkpoints", 
                           "preprocessed_data", "logs"
        relative_path (str, optional): Relative path within the folder
        
    Returns:
        str: Full path to the storage location
    """
    # Import here to avoid circular imports
    if is_paperspace_environment():
        try:
            from .google_drive_storage import GoogleDriveSync
            path = GoogleDriveSync.get_local_path(folder_type, relative_path)
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            return path
        except ImportError:
            # Fallback if GoogleDriveSync is not available
            path = os.path.join('/notebooks/Jarvis_AI_Assistant', folder_type, relative_path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            return path
    else:
        # For local development or other environments
        root_dir = get_project_root()
        path = os.path.join(root_dir, folder_type, relative_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

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

def setup_logging(log_filename=None):
    """
    Set up logging configuration to log to both console and a file.
    If running in Paperspace, this will store logs in the logs directory
    that syncs to Google Drive.
    
    Args:
        log_filename (str, optional): Name of the log file. If not provided,
                                     a timestamp-based name will be used.
    
    Returns:
        str: Path to the log file
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = log_filename or f"jarvis_log_{timestamp}.log"
    
    # Get the logs directory path and ensure it exists
    log_dir = ensure_directory_exists("logs")
    log_path = os.path.join(log_dir, log_filename)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_path}")
    
    return log_path

def sync_logs():
    """
    Sync log files to Google Drive.
    """
    if is_paperspace_environment():
        try:
            from .google_drive_storage import GoogleDriveSync
            GoogleDriveSync.sync_to_gdrive("logs")
            logger = logging.getLogger(__name__)
            logger.info("Synced logs to Google Drive")
        except ImportError:
            logger = logging.getLogger(__name__)
            logger.warning("GoogleDriveSync module not available. No sync performed.")
    else:
        logger = logging.getLogger(__name__)
        logger.info("Not running in Paperspace environment, skipping Google Drive sync for logs.")

def save_log_file(content, filename=None):
    """
    Save content to a log file and sync to Google Drive.
    
    Args:
        content (str): Content to save
        filename (str, optional): Name of the log file. If not provided,
                                 a timestamp-based name will be used.
    
    Returns:
        str: Path to the log file
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = filename or f"custom_log_{timestamp}.log"
    
    # Get the logs directory path and ensure it exists
    log_dir = ensure_directory_exists("logs")
    log_path = os.path.join(log_dir, filename)
    
    # Save content to log file
    with open(log_path, 'w') as f:
        f.write(content)
    
    # Sync logs to Google Drive
    sync_logs()
    
    return log_path
    