"""
Google Drive Storage Integration for DeepSeek-Coder Fine-tuning on Paperspace Gradient

This module provides a dedicated implementation for saving and loading model checkpoints
and datasets using Google Drive, specifically optimized for Paperspace Gradient's 15GB
storage limitations.
"""

import os
import json
import time
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# For Google Drive integration
try:
    import gdown
    from google.oauth2 import service_account
    from google.auth.transport.requests import Request
    GDRIVE_AVAILABLE = True
except ImportError:
    GDRIVE_AVAILABLE = False
    print("Google Drive integration requires additional packages.")
    print("Install with: pip install gdown google-auth google-auth-oauthlib google-auth-httplib2")

class CheckpointStrategy(str, Enum):
    """Strategy for saving checkpoints during training"""
    IMPROVEMENT = "improvement"  # Save only checkpoints that improve on validation metrics
    REGULAR = "regular"          # Save checkpoint at regular intervals
    HYBRID = "hybrid"            # Save best checkpoint + latest checkpoint
    ALL = "all"                  # Save all checkpoints (not recommended for limited storage)

class GoogleDriveStorage:
    """Google Drive storage manager optimized for Paperspace Gradient"""
    
    def __init__(
        self, 
        folder_id: Optional[str] = None,
        service_account_file: Optional[str] = None,
        create_subfolders: bool = True
    ):
        """
        Initialize Google Drive storage
        
        Args:
            folder_id: Google Drive folder ID to use for storage
            service_account_file: Path to service account credentials JSON file
            create_subfolders: Whether to create subfolders for models, data, etc.
        """
        if not GDRIVE_AVAILABLE:
            raise ImportError(
                "Google Drive integration requires gdown and google-auth libraries. "
                "Install with: pip install gdown google-auth google-auth-oauthlib google-auth-httplib2"
            )
        
        self.folder_id = folder_id
        self.service_account_file = service_account_file
        self.create_subfolders = create_subfolders
        self.authenticated = False
        self.subfolders = {}
        
        # Set up authentication
        self.setup_auth()
        
        # Create standard subfolders if needed
        if create_subfolders and self.folder_id:
            self._create_standard_subfolders()
    
    def setup_auth(self) -> bool:
        """
        Setup authentication for Google Drive
        
        Returns:
            success: True if authentication succeeded, False otherwise
        """
        if self.service_account_file and os.path.exists(self.service_account_file):
            try:
                # Use service account for authentication
                credentials = service_account.Credentials.from_service_account_file(
                    self.service_account_file,
                    scopes=['https://www.googleapis.com/auth/drive']
                )
                Request().refresh(credentials)
                self.authenticated = True
                print("Successfully authenticated with Google Drive using service account")
                return True
            except Exception as e:
                print(f"Error authenticating with service account: {e}")
                print("Falling back to gdown for authentication")
        
        # Without service account file, we'll use gdown which uses browser-based auth
        # Let the user know they need to authenticate manually when requested
        print("Note: You may need to authenticate via browser when uploading/downloading files")
        self.authenticated = True  # We'll assume success for now
        return True
    
    def _create_standard_subfolders(self):
        """Create standard subfolders for organizing files"""
        standard_folders = ["models", "datasets", "checkpoints", "logs"]
        
        for folder in standard_folders:
            print(f"Note: To create {folder} folder in Google Drive:")
            print(f"1. Go to https://drive.google.com/drive/folders/{self.folder_id}")
            print(f"2. Create a folder named '{folder}'")
            print(f"3. Enter the folder and copy the folder ID from the URL")
            
            folder_id = input(f"Enter the folder ID for '{folder}' (leave empty to skip): ").strip()
            if folder_id:
                self.subfolders[folder] = folder_id
    
    def upload_file(self, 
                   local_path: str, 
                   remote_name: Optional[str] = None,
                   folder_type: Optional[str] = None) -> Optional[str]:
        """
        Upload a file to Google Drive
        
        Args:
            local_path: Path to local file
            remote_name: Name to use in Google Drive (defaults to local filename)
            folder_type: Type of folder to upload to ('models', 'datasets', etc.)
            
        Returns:
            file_id: ID of the uploaded file, or None if upload failed
        """
        if not os.path.exists(local_path):
            print(f"Error: Local file {local_path} does not exist")
            return None
        
        # Determine the folder ID to use
        folder_id = None
        if folder_type and folder_type in self.subfolders:
            folder_id = self.subfolders[folder_type]
        else:
            folder_id = self.folder_id
        
        if not folder_id:
            print("Error: No folder ID provided for Google Drive upload")
            return None
        
        remote_name = remote_name or os.path.basename(local_path)
        
        try:
            # Upload to Google Drive using gdown
            url = f"https://drive.google.com/drive/folders/{folder_id}"
            gdown.upload(local_path, url, folder=True)
            print(f"Successfully uploaded {local_path} to Google Drive folder {folder_id}")
            
            # Return a placeholder ID - gdown doesn't return the specific file ID
            return f"uploaded_file_{time.time()}"
        except Exception as e:
            print(f"Error uploading to Google Drive: {e}")
            return None
    
    def download_file(self, 
                     file_id: str, 
                     local_path: str) -> bool:
        """
        Download a file from Google Drive
        
        Args:
            file_id: Google Drive file ID
            local_path: Path to save the file locally
            
        Returns:
            success: True if download succeeded, False otherwise
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
            
            # Download from Google Drive
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, local_path, quiet=False)
            
            if os.path.exists(local_path):
                print(f"Successfully downloaded to {local_path}")
                return True
            else:
                print(f"Error: Download completed but file {local_path} does not exist")
                return False
        except Exception as e:
            print(f"Error downloading from Google Drive: {e}")
            return False
    
    def compress_directory(self, directory: str, output_file: str) -> str:
        """
        Compress a directory into a ZIP archive
        
        Args:
            directory: Directory to compress
            output_file: Path to output ZIP file
            
        Returns:
            output_file: Path to created ZIP file
        """
        if not os.path.exists(directory):
            raise ValueError(f"Directory {directory} does not exist")
        
        output_file = output_file if output_file.endswith('.zip') else f"{output_file}.zip"
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        try:
            # Create a ZIP archive
            shutil.make_archive(
                output_file.replace('.zip', ''),  # Base name (without .zip)
                'zip',                           # Format
                directory                        # Source directory
            )
            print(f"Successfully compressed {directory} to {output_file}")
            return output_file
        except Exception as e:
            print(f"Error compressing directory: {e}")
            return ""
    
    def decompress_archive(self, archive_file: str, output_dir: str) -> str:
        """
        Decompress a ZIP archive
        
        Args:
            archive_file: Path to ZIP archive
            output_dir: Directory to extract to
            
        Returns:
            output_dir: Path to extraction directory
        """
        if not os.path.exists(archive_file):
            raise ValueError(f"Archive file {archive_file} does not exist")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Extract the ZIP archive
            shutil.unpack_archive(archive_file, output_dir)
            print(f"Successfully extracted {archive_file} to {output_dir}")
            return output_dir
        except Exception as e:
            print(f"Error extracting archive: {e}")
            return ""
    
    def upload_model(self, 
                    model_dir: str, 
                    model_name: str) -> Optional[str]:
        """
        Upload a model directory to Google Drive
        
        Args:
            model_dir: Path to model directory
            model_name: Name for the model
            
        Returns:
            file_id: ID of the uploaded model ZIP, or None if upload failed
        """
        if not os.path.exists(model_dir):
            print(f"Error: Model directory {model_dir} does not exist")
            return None
        
        # Create a temporary directory for compression
        with tempfile.TemporaryDirectory() as temp_dir:
            # Compress the model directory
            zip_file = os.path.join(temp_dir, f"{model_name}.zip")
            self.compress_directory(model_dir, zip_file)
            
            # Upload to the models subfolder if available, otherwise main folder
            folder_type = "models" if "models" in self.subfolders else None
            file_id = self.upload_file(zip_file, f"{model_name}.zip", folder_type)
            
            # Create and upload a metadata file
            metadata = {
                "model_name": model_name,
                "timestamp": time.time(),
                "upload_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "file_id": file_id
            }
            
            metadata_file = os.path.join(temp_dir, f"{model_name}_metadata.json")
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            
            self.upload_file(metadata_file, f"{model_name}_metadata.json", folder_type)
            
            return file_id
    
    def download_model(self, 
                      file_id: str, 
                      output_dir: str, 
                      model_name: Optional[str] = None) -> Optional[str]:
        """
        Download a model from Google Drive
        
        Args:
            file_id: Google Drive file ID for the model ZIP
            output_dir: Directory to extract the model to
            model_name: Name for the model (optional)
            
        Returns:
            model_dir: Path to the extracted model directory, or None if download failed
        """
        # Create a temporary directory for download
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download the ZIP file
            model_name = model_name or f"model_{time.time()}"
            zip_file = os.path.join(temp_dir, f"{model_name}.zip")
            
            success = self.download_file(file_id, zip_file)
            if not success:
                print(f"Error downloading model {model_name}")
                return None
            
            # Extract to the output directory
            model_dir = os.path.join(output_dir, model_name)
            self.decompress_archive(zip_file, model_dir)
            
            return model_dir
    
    def upload_checkpoint(self, 
                         checkpoint_dir: str, 
                         checkpoint_name: str,
                         metrics: Dict[str, float],
                         step: int) -> Optional[str]:
        """
        Upload a training checkpoint to Google Drive
        
        Args:
            checkpoint_dir: Directory containing checkpoint files
            checkpoint_name: Base name for the checkpoint
            metrics: Evaluation metrics for the checkpoint
            step: Training step number
            
        Returns:
            file_id: ID of the uploaded checkpoint ZIP, or None if upload failed
        """
        if not os.path.exists(checkpoint_dir):
            print(f"Error: Checkpoint directory {checkpoint_dir} does not exist")
            return None
        
        # Create a temporary directory for compression
        with tempfile.TemporaryDirectory() as temp_dir:
            # Compress the checkpoint directory
            zip_file = os.path.join(temp_dir, f"{checkpoint_name}.zip")
            self.compress_directory(checkpoint_dir, zip_file)
            
            # Upload to the checkpoints subfolder if available, otherwise main folder
            folder_type = "checkpoints" if "checkpoints" in self.subfolders else None
            file_id = self.upload_file(zip_file, f"{checkpoint_name}.zip", folder_type)
            
            # Create and upload a metadata file
            metadata = {
                "checkpoint_name": checkpoint_name,
                "step": step,
                "metrics": metrics,
                "timestamp": time.time(),
                "upload_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "file_id": file_id
            }
            
            metadata_file = os.path.join(temp_dir, f"{checkpoint_name}_metadata.json")
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            
            self.upload_file(metadata_file, f"{checkpoint_name}_metadata.json", folder_type)
            
            return file_id
    
    def save_checkpoints_with_strategy(self,
                                      checkpoint_dir: str,
                                      checkpoint_name: str,
                                      metrics: Dict[str, float],
                                      step: int,
                                      strategy: CheckpointStrategy = CheckpointStrategy.IMPROVEMENT,
                                      max_checkpoints: int = 3,
                                      checkpoints_info: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Save checkpoints according to a strategy and maintain a list of checkpoint info
        
        Args:
            checkpoint_dir: Directory containing checkpoint files
            checkpoint_name: Base name for the checkpoint
            metrics: Evaluation metrics for the checkpoint
            step: Training step number
            strategy: Checkpoint saving strategy
            max_checkpoints: Maximum number of checkpoints to keep
            checkpoints_info: List of existing checkpoint info dictionaries
            
        Returns:
            checkpoints_info: Updated list of checkpoint info dictionaries
        """
        checkpoints_info = checkpoints_info or []
        
        # Prepare metadata for this checkpoint
        checkpoint_info = {
            "name": checkpoint_name,
            "step": step,
            "metrics": metrics,
            "timestamp": time.time(),
        }
        
        # Check if this checkpoint should be saved based on strategy
        if strategy == CheckpointStrategy.ALL:
            # Save all checkpoints
            save_checkpoint = True
        elif strategy == CheckpointStrategy.IMPROVEMENT:
            # Only save if this is the first checkpoint or metrics improved
            if not checkpoints_info:
                save_checkpoint = True
            else:
                # Check if eval_loss improved (lower is better)
                best_loss = min(c["metrics"].get("eval_loss", float("inf")) for c in checkpoints_info)
                current_loss = metrics.get("eval_loss", float("inf"))
                save_checkpoint = current_loss < best_loss
        elif strategy == CheckpointStrategy.REGULAR:
            # Save at regular intervals (every 100 steps by default)
            save_interval = 100
            save_checkpoint = step % save_interval == 0
        elif strategy == CheckpointStrategy.HYBRID:
            # Save if metrics improved OR this is the latest checkpoint
            if not checkpoints_info:
                save_checkpoint = True
            else:
                # Check if eval_loss improved (lower is better)
                best_loss = min(c["metrics"].get("eval_loss", float("inf")) for c in checkpoints_info)
                current_loss = metrics.get("eval_loss", float("inf"))
                # Save if improved or if this is significantly later than last checkpoint
                last_step = max(c["step"] for c in checkpoints_info)
                save_checkpoint = (current_loss < best_loss) or (step - last_step >= 100)
        else:
            # Default - save all checkpoints
            save_checkpoint = True
        
        if not save_checkpoint:
            print(f"Skipping checkpoint at step {step} based on {strategy} strategy")
            return checkpoints_info
        
        # Upload the checkpoint
        file_id = self.upload_checkpoint(checkpoint_dir, checkpoint_name, metrics, step)
        
        # If upload successful, add to checkpoints list
        if file_id:
            checkpoint_info["file_id"] = file_id
            checkpoints_info.append(checkpoint_info)
            
            # Enforce maximum number of checkpoints if needed
            if len(checkpoints_info) > max_checkpoints:
                # If using improvement strategy, remove worst checkpoint
                if strategy == CheckpointStrategy.IMPROVEMENT:
                    # Sort by loss (higher is worse)
                    sorted_checkpoints = sorted(
                        checkpoints_info,
                        key=lambda c: c["metrics"].get("eval_loss", float("inf"))
                    )
                    # Remove worst checkpoint
                    checkpoints_info = [c for c in checkpoints_info if c != sorted_checkpoints[-1]]
                # For other strategies, remove oldest checkpoint
                else:
                    # Sort by step (lower is older)
                    sorted_checkpoints = sorted(checkpoints_info, key=lambda c: c["step"])
                    # Remove oldest checkpoint
                    checkpoints_info = [c for c in checkpoints_info if c != sorted_checkpoints[0]]
            
            # Save updated checkpoints info to a JSON file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(checkpoints_info, f, indent=2)
                temp_path = f.name
            
            # Upload the checkpoints info file
            folder_type = "checkpoints" if "checkpoints" in self.subfolders else None
            self.upload_file(temp_path, "checkpoints_info.json", folder_type)
            
            # Remove temporary file
            os.unlink(temp_path)
        
        return checkpoints_info


def get_gdrive_checkpoint_callback(gdrive_storage, strategy=CheckpointStrategy.IMPROVEMENT, max_checkpoints=3):
    """
    Create a checkpoint callback that saves to Google Drive
    
    Args:
        gdrive_storage: GoogleDriveStorage instance
        strategy: Checkpoint saving strategy
        max_checkpoints: Maximum number of checkpoints to keep
        
    Returns:
        callback: A callback for use with transformers.Trainer
    """
    from transformers import TrainerCallback
    
    class GDriveCheckpointCallback(TrainerCallback):
        def __init__(self, gdrive_storage, strategy, max_checkpoints):
            self.gdrive_storage = gdrive_storage
            self.strategy = strategy
            self.max_checkpoints = max_checkpoints
            self.checkpoints_info = []
        
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if not metrics:
                return
            
            # Save checkpoint using Google Drive storage
            checkpoint_dir = args.output_dir
            checkpoint_name = f"checkpoint-{state.global_step}"
            
            self.checkpoints_info = self.gdrive_storage.save_checkpoints_with_strategy(
                checkpoint_dir=checkpoint_dir,
                checkpoint_name=checkpoint_name,
                metrics=metrics,
                step=state.global_step,
                strategy=self.strategy,
                max_checkpoints=self.max_checkpoints,
                checkpoints_info=self.checkpoints_info
            )
    
    return GDriveCheckpointCallback(gdrive_storage, strategy, max_checkpoints)

class GoogleDriveSync:
    """
    Handles syncing files between local Paperspace storage and Google Drive
    using the established rclone connection.
    """
    
    GDRIVE_BASE = "gdrive:Jarvis_AI_Assistant"
    LOCAL_BASE = "/notebooks/Jarvis_AI_Assistant"
    
    SYNC_FOLDERS = {
        "metrics": f"{GDRIVE_BASE}/metrics",
        "models": f"{GDRIVE_BASE}/models",
        "datasets": f"{GDRIVE_BASE}/datasets",
        "checkpoints": f"{GDRIVE_BASE}/checkpoints",
        "preprocessed_data": f"{GDRIVE_BASE}/preprocessed_data",
        "logs": f"{GDRIVE_BASE}/logs"
    }
    
    @classmethod
    def ensure_local_dirs(cls):
        """Ensure that all local directories needed for syncing exist."""
        directories = [
            cls.LOCAL_MODELS_DIR,
            cls.LOCAL_DATASETS_DIR,
            cls.LOCAL_METRICS_DIR,
            cls.LOCAL_EVALS_DIR,
            cls.LOCAL_LOGS_DIR,
            cls.LOCAL_CHECKPOINTS_DIR,
            cls.LOCAL_VISUALIZATIONS_DIR
        ]
        
        for local_path in directories:
            try:
                os.makedirs(local_path, exist_ok=True)
                logger.info(f"Ensured directory exists: {local_path}")
            except (OSError, PermissionError) as e:
                # Handle read-only filesystem errors gracefully
                logger.warning(f"Could not create directory {local_path}: {e}")
                logger.warning("Continuing without this directory. Some functionality may be limited.")
    
    @classmethod
    def sync_to_gdrive(cls, folder_type=None):
        """
        Syncs the specified folder type (or all folders if None) to Google Drive.
        
        Args:
            folder_type (str, optional): One of "metrics", "models", "datasets", "checkpoints"
                                         or None to sync all folders
        """
        if folder_type and folder_type not in cls.SYNC_FOLDERS:
            raise ValueError(f"Invalid folder_type: {folder_type}. Must be one of {list(cls.SYNC_FOLDERS.keys())} or None")
        
        folders_to_sync = [folder_type] if folder_type else cls.SYNC_FOLDERS.keys()
        
        for folder in folders_to_sync:
            local_path = os.path.join(cls.LOCAL_BASE, folder)
            gdrive_path = cls.SYNC_FOLDERS[folder]
            
            # Always ensure local directory exists
            os.makedirs(local_path, exist_ok=True)
            
            logger.info(f"Syncing {local_path} to {gdrive_path}")
            try:
                cmd = ["rclone", "sync", local_path, gdrive_path]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info(f"Successfully synced {folder} to Google Drive")
                else:
                    logger.error(f"Failed to sync {folder} to Google Drive: {result.stderr}")
            except Exception as e:
                logger.error(f"Error syncing {folder} to Google Drive: {str(e)}")
    
    @classmethod
    def sync_from_gdrive(cls, folder_type=None):
        """
        Syncs the specified folder type (or all folders if None) from Google Drive to local storage.
        Creates local directories if they don't exist.
        
        Args:
            folder_type (str, optional): One of "metrics", "models", "datasets", "checkpoints"
                                         or None to sync all folders
        """
        if folder_type and folder_type not in cls.SYNC_FOLDERS:
            raise ValueError(f"Invalid folder_type: {folder_type}. Must be one of {list(cls.SYNC_FOLDERS.keys())} or None")
        
        folders_to_sync = [folder_type] if folder_type else cls.SYNC_FOLDERS.keys()
        
        for folder in folders_to_sync:
            local_path = os.path.join(cls.LOCAL_BASE, folder)
            gdrive_path = cls.SYNC_FOLDERS[folder]
            
            # Ensure local directory exists
            os.makedirs(local_path, exist_ok=True)
                
            logger.info(f"Syncing {gdrive_path} to {local_path}")
            try:
                cmd = ["rclone", "sync", gdrive_path, local_path]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info(f"Successfully synced {folder} from Google Drive")
                else:
                    logger.error(f"Failed to sync {folder} from Google Drive: {result.stderr}")
            except Exception as e:
                logger.error(f"Error syncing {folder} from Google Drive: {str(e)}")
    
    @classmethod
    def get_local_path(cls, folder_type, relative_path=""):
        """
        Get the local path for a specific folder type.
        Also ensures the directory exists.
        
        Args:
            folder_type (str): One of "metrics", "models", "datasets", "checkpoints"
            relative_path (str, optional): Relative path within the folder
            
        Returns:
            str: Full local path
        """
        if folder_type not in cls.SYNC_FOLDERS:
            raise ValueError(f"Invalid folder_type: {folder_type}. Must be one of {list(cls.SYNC_FOLDERS.keys())}")
        
        path = os.path.join(cls.LOCAL_BASE, folder_type, relative_path)
        
        # Ensure the parent directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        return path
    
    @classmethod
    def get_gdrive_path(cls, folder_type, relative_path=""):
        """
        Get the Google Drive path for a specific folder type.
        
        Args:
            folder_type (str): One of "metrics", "models", "datasets", "checkpoints"
            relative_path (str, optional): Relative path within the folder
            
        Returns:
            str: Full Google Drive path
        """
        if folder_type not in cls.SYNC_FOLDERS:
            raise ValueError(f"Invalid folder_type: {folder_type}. Must be one of {list(cls.SYNC_FOLDERS.keys())}")
        
        base_path = cls.SYNC_FOLDERS[folder_type]
        return f"{base_path}/{relative_path}" if relative_path else base_path

# Initialize directories on module import
GoogleDriveSync.ensure_local_dirs() 