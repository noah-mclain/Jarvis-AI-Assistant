import zipfile
import os
import sys
import json
import logging
from pathlib import Path
import datetime
import shutil
from typing import Optional, List, Dict, Any, Union

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

def is_zipfile(file_path: str) -> bool:
    """
    Check if a file is a valid ZIP file.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        bool: True if the file is a valid ZIP file, False otherwise
    """
    return zipfile.is_zipfile(file_path) if os.path.exists(file_path) else False

def process_zip(zip_path: str, extract_to: str) -> bool:
    """
    Process a ZIP file by extracting its contents.
    
    Args:
        zip_path: Path to the ZIP file
        extract_to: Directory to extract the files to
        
    Returns:
        bool: True if extraction was successful, False otherwise
    """
    try:
        if not is_zipfile(zip_path):
            logger.error(f"File {zip_path} is not a valid ZIP file")
            return False
            
        # Create extraction directory if it doesn't exist
        os.makedirs(extract_to, exist_ok=True)
        
        # Extract the ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            
        logger.info(f"Successfully extracted {zip_path} to {extract_to}")
        return True
    except Exception as e:
        logger.error(f"Error extracting ZIP file {zip_path}: {e}")
        return False

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

# Add spaCy utilities
def is_spacy_available():
    """Check if spaCy is available in the current environment"""
    try:
        import spacy
        return True, spacy.__version__
    except ImportError:
        return False, None

def is_spacy_model_loaded(model_name="en_core_web_sm"):
    """Check if a specific spaCy model is available and can be loaded"""
    try:
        import spacy
        nlp = spacy.load(model_name)
        # Test the model with a simple sentence
        doc = nlp("This is a test sentence.")
        return True, f"Model {model_name} loaded successfully"
    except ImportError:
        return False, "spaCy not installed"
    except OSError:
        return False, f"Model {model_name} not found"
    except Exception as e:
        return False, f"Error loading model: {str(e)}"

def initialize_spacy(fallback_to_basic=True, log_errors=True):
    """
    Initialize spaCy with the en_core_web_sm model if available.
    
    Args:
        fallback_to_basic: If True, will not raise errors but return None if spaCy is unavailable
        log_errors: If True, will log errors and warnings
        
    Returns:
        nlp: The spaCy NLP object if available, None otherwise
    """
    import logging
    
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
            if log_errors:
                logging.info("spaCy initialized with en_core_web_sm model")
            return nlp
        except OSError as e:
            if log_errors:
                logging.warning(f"spaCy model not found: {str(e)}")
                logging.warning("To install the model, run: python -m spacy download en_core_web_sm")
            if not fallback_to_basic:
                raise e
            return None
        except Exception as e:
            if log_errors:
                logging.error(f"Error initializing spaCy: {str(e)}")
            if not fallback_to_basic:
                raise e
            return None
    except ImportError:
        if log_errors:
            logging.warning("spaCy not installed")
            logging.warning("For better text processing, install spaCy with: pip install spacy")
        if not fallback_to_basic:
            raise ImportError("spaCy not installed")
        return None

def process_text_with_spacy_or_fallback(text, nlp=None):
    """
    Process text with spaCy if available, or use a simple fallback method.
    
    Args:
        text: The text to process
        nlp: Optional spaCy NLP object. If None, will try to initialize
        
    Returns:
        dict: Processed text information (entities, tokens, etc.)
    """
    # If nlp is not provided, try to initialize spaCy
    if nlp is None:
        nlp = initialize_spacy(fallback_to_basic=True, log_errors=False)
    
    # If spaCy is available, use it for processing
    if nlp is not None:
        try:
            doc = nlp(text)
            return {
                "entities": [(ent.text, ent.label_) for ent in doc.ents],
                "tokens": [token.text for token in doc],
                "pos_tags": [(token.text, token.pos_) for token in doc],
                "nouns": [token.text for token in doc if token.pos_ == "NOUN"],
                "verbs": [token.text for token in doc if token.pos_ == "VERB"],
                "adjectives": [token.text for token in doc if token.pos_ == "ADJ"],
                "dependency_tree": [(token.text, token.dep_, token.head.text) for token in doc],
                "sentences": [sent.text for sent in doc.sents]
            }
        except Exception:
            # Fall back to basic processing if spaCy fails
            pass
    
    # Basic fallback processing
    import re
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Basic POS guesses - very rudimentary!
    nouns = [word for word in words if word[0].isupper() and len(word) > 3]
    potential_verbs = [word for word in words if word.lower() in {
        'is', 'are', 'was', 'were', 'have', 'has', 'had', 'do', 'does', 'did',
        'can', 'could', 'will', 'would', 'shall', 'should', 'may', 'might',
        'make', 'create', 'generate', 'write', 'build', 'design', 'develop'
    }]
    
    return {
        "entities": [],  # Empty since we can't detect entities
        "tokens": words,
        "pos_tags": [],  # Empty since we can't determine POS
        "nouns": nouns,
        "verbs": potential_verbs,
        "adjectives": [],  # Empty since we can't determine adjectives
        "dependency_tree": [],  # Empty since we can't determine dependencies
        "sentences": sentences
    }
    