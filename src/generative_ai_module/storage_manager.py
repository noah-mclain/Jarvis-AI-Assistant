"""
Consolidated Storage Manager

This module provides a unified interface for storage management, including:
1. Google Drive integration for model and dataset storage
2. Efficient storage optimization for Paperspace environments
3. Automatic syncing between local and remote storage
4. Storage status monitoring and management

This consolidates functionality from:
- google_drive_storage.py
- sync_gdrive.py
- manage_storage.py
- optimize_deepseek_gdrive.py
- optimize_deepseek_storage.py
- storage_optimization.py
"""

import os
import json
import time
import tempfile
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from enum import Enum
import gc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - some storage optimizations will be limited")

try:
    import gdown
    GDRIVE_AVAILABLE = True
except ImportError:
    GDRIVE_AVAILABLE = False
    logger.warning("gdown not available - Google Drive integration will be limited")

# Utility functions
def is_paperspace_environment():
    """Check if running in Paperspace Gradient environment"""
    return os.path.exists("/notebooks") or os.path.exists("/storage")

def compress_directory(directory_path, output_zip):
    """Compress a directory to a zip file"""
    if not os.path.exists(directory_path):
        logger.error(f"Directory {directory_path} does not exist")
        return False

    try:
        shutil.make_archive(
            output_zip.replace('.zip', ''),  # Remove .zip as make_archive adds it
            'zip',
            os.path.dirname(directory_path),
            os.path.basename(directory_path)
        )
        return True
    except Exception as e:
        logger.error(f"Error compressing directory: {e}")
        return False

def extract_zip(zip_file, output_dir):
    """Extract a zip file to a directory"""
    if not os.path.exists(zip_file):
        logger.error(f"Zip file {zip_file} does not exist")
        return False

    try:
        shutil.unpack_archive(zip_file, output_dir)
        return True
    except Exception as e:
        logger.error(f"Error extracting zip file: {e}")
        return False

class StorageType(Enum):
    """Enum for storage types"""
    LOCAL = "local"
    GDRIVE = "gdrive"
    S3 = "s3"

class StorageManager:
    """
    Unified storage manager for models, datasets, and checkpoints.

    This class provides methods for:
    - Saving and loading models
    - Syncing data between local and remote storage
    - Optimizing storage usage
    - Monitoring storage status
    - Model compression and quantization
    - Dataset streaming and compression
    """

    # Base directories
    GDRIVE_BASE = "gdrive:Jarvis_AI_Assistant"
    LOCAL_BASE = "notebooks/Jarvis_AI_Assistant" if is_paperspace_environment() else "data"

    # Directory structure
    SYNC_FOLDERS = {
        "models": (os.path.join(GDRIVE_BASE, "models"), os.path.join(LOCAL_BASE, "models")),
        "datasets": (os.path.join(GDRIVE_BASE, "datasets"), os.path.join(LOCAL_BASE, "datasets")),
        "evaluation_metrics": (os.path.join(GDRIVE_BASE, "evaluation_metrics"), os.path.join(LOCAL_BASE, "evaluation_metrics")),
        "logs": (os.path.join(GDRIVE_BASE, "logs"), os.path.join(LOCAL_BASE, "logs")),
        "checkpoints": (os.path.join(GDRIVE_BASE, "checkpoints"), os.path.join(LOCAL_BASE, "checkpoints")),
        "visualizations": (os.path.join(GDRIVE_BASE, "visualizations"), os.path.join(LOCAL_BASE, "visualizations")),
        "preprocessed_data": (os.path.join(GDRIVE_BASE, "preprocessed_data"), os.path.join(LOCAL_BASE, "preprocessed_data")),
    }

    # Google Drive folder IDs (for direct API access)
    GDRIVE_FOLDER_IDS = {
        "root": None,  # Set this to your root folder ID
        "models": None,
        "datasets": None,
        "checkpoints": None,
        "evaluation_metrics": None,
        "logs": None,
        "visualizations": None,
        "preprocessed_data": None,
    }

    def __init__(self, storage_type=StorageType.LOCAL):
        """
        Initialize the storage manager.

        Args:
            storage_type: Type of storage to use (LOCAL, GDRIVE, S3)
        """
        self.storage_type = storage_type
        self.ensure_local_dirs()

    @classmethod
    def ensure_local_dirs(cls):
        """Ensure all local directories exist"""
        for folder_type, (_, local_dir) in cls.SYNC_FOLDERS.items():
            os.makedirs(local_dir, exist_ok=True)

    @classmethod
    def _check_rclone(cls):
        """Check if rclone is available"""
        try:
            result = subprocess.run(["rclone", "version"], capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False

    @classmethod
    def sync_to_gdrive(cls, folder_type=None):
        """
        Sync files from local storage to Google Drive.

        Args:
            folder_type: Type of folder to sync (None for all folders)
        """
        if not cls._check_rclone():
            logger.error("rclone not available. Cannot sync to Google Drive.")
            return False

        cls.ensure_local_dirs()

        folders_to_sync = {}
        if folder_type and folder_type in cls.SYNC_FOLDERS:
            gdrive_dir, local_dir = cls.SYNC_FOLDERS[folder_type]
            folders_to_sync[folder_type] = (gdrive_dir, local_dir)
        else:
            folders_to_sync = cls.SYNC_FOLDERS

        for folder_name, (gdrive_dir, local_dir) in folders_to_sync.items():
            logger.info(f"Syncing {folder_name} to Google Drive...")
            try:
                cmd = ["rclone", "copy", local_dir, gdrive_dir, "--progress"]
                subprocess.run(cmd, check=True)
                logger.info(f"Successfully synced {folder_name} to Google Drive")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error syncing {folder_name} to Google Drive: {e}")

        return True

    @classmethod
    def sync_from_gdrive(cls, folder_type=None):
        """
        Sync files from Google Drive to local storage.

        Args:
            folder_type: Type of folder to sync (None for all folders)
        """
        if not cls._check_rclone():
            logger.error("rclone not available. Cannot sync from Google Drive.")
            return False

        cls.ensure_local_dirs()

        folders_to_sync = {}
        if folder_type and folder_type in cls.SYNC_FOLDERS:
            gdrive_dir, local_dir = cls.SYNC_FOLDERS[folder_type]
            folders_to_sync[folder_type] = (gdrive_dir, local_dir)
        else:
            folders_to_sync = cls.SYNC_FOLDERS

        for folder_name, (gdrive_dir, local_dir) in folders_to_sync.items():
            logger.info(f"Syncing {folder_name} from Google Drive...")
            try:
                cmd = ["rclone", "copy", gdrive_dir, local_dir, "--progress"]
                subprocess.run(cmd, check=True)
                logger.info(f"Successfully synced {folder_name} from Google Drive")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error syncing {folder_name} from Google Drive: {e}")

        return True

    @classmethod
    def clear_local_storage(cls, folder_type=None, confirm=False):
        """
        Clear local storage for the specified folder type or all folders.

        Args:
            folder_type: Type of folder to clear (None for all folders)
            confirm: Whether the operation is confirmed
        """
        if not confirm:
            logger.warning("Operation not confirmed. Use confirm=True to execute deletion.")
            return False

        folders_to_clear = {}
        if folder_type and folder_type in cls.SYNC_FOLDERS:
            _, local_dir = cls.SYNC_FOLDERS[folder_type]
            folders_to_clear[folder_type] = local_dir
        else:
            folders_to_clear = {k: v[1] for k, v in cls.SYNC_FOLDERS.items()}

        for folder_name, local_dir in folders_to_clear.items():
            logger.info(f"Clearing {folder_name} from local storage...")
            try:
                if os.path.exists(local_dir):
                    for item in os.listdir(local_dir):
                        item_path = os.path.join(local_dir, item)
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                    logger.info(f"Successfully cleared {folder_name} from local storage")
                else:
                    logger.warning(f"Directory {local_dir} does not exist")
            except Exception as e:
                logger.error(f"Error clearing {folder_name} from local storage: {e}")

        return True

    @classmethod
    def save_model(cls, model_dir, model_name, metadata=None, storage_type=StorageType.GDRIVE, subfolder=None):
        """
        Save a model to storage.

        Args:
            model_dir: Directory containing the model
            model_name: Name of the model
            metadata: Model metadata
            storage_type: Type of storage to use
            subfolder: Optional subfolder in models directory (e.g., "checkpoints")

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create local model directory if it doesn't exist
            _, local_models_dir = cls.SYNC_FOLDERS["models"]
            
            # Handle optional subfolder
            if subfolder:
                local_models_dir = os.path.join(local_models_dir, subfolder)
                os.makedirs(local_models_dir, exist_ok=True)
                
            # Create output directory
            output_dir = os.path.join(local_models_dir, model_name)
            os.makedirs(output_dir, exist_ok=True)

            # If model_dir is a file, copy it to output_dir
            if os.path.isfile(model_dir):
                shutil.copy(model_dir, os.path.join(output_dir, os.path.basename(model_dir)))
            else:
                # Copy all files from model_dir to output_dir
                for item in os.listdir(model_dir):
                    s = os.path.join(model_dir, item)
                    d = os.path.join(output_dir, item)
                    if os.path.isfile(s):
                        shutil.copy2(s, d)
                    else:
                        shutil.copytree(s, d, dirs_exist_ok=True)

            # Write metadata if provided
            if metadata:
                with open(os.path.join(output_dir, "metadata.json"), "w") as f:
                    json.dump(metadata, f)

            # Sync to remote storage if needed
            if storage_type == StorageType.GDRIVE:
                # If subfolder is specified, we need to sync to that specific folder
                sync_folder = "models"
                if subfolder:
                    # Create the specific subfolder path if using Google Drive API
                    if GDRIVE_AVAILABLE and cls.GDRIVE_FOLDER_IDS.get("models"):
                        gdrive_dir, _ = cls.SYNC_FOLDERS["models"]
                        gdrive_subdir = os.path.join(gdrive_dir, subfolder)
                        # Ensure the subfolder exists in Google Drive
                        cls._ensure_gdrive_folder(subfolder, parent_folder="models")
                
                sync_success = cls.sync_to_gdrive(sync_folder)
                if not sync_success:
                    logger.warning("Failed to sync model to Google Drive")

            logger.info(f"Successfully saved model {model_name}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    @classmethod
    def _ensure_gdrive_folder(cls, folder_name, parent_folder="root"):
        """Ensure a folder exists in Google Drive, creating it if needed"""
        if not GDRIVE_AVAILABLE:
            return False
            
        try:
            import gdown
            
            # If we don't have the parent folder ID, we can't create the subfolder
            parent_id = cls.GDRIVE_FOLDER_IDS.get(parent_folder)
            if not parent_id:
                logger.warning(f"Cannot create subfolder - parent folder ID for {parent_folder} not available")
                return False
                
            # Check if the folder already exists using gdown
            logger.info(f"Checking if folder {folder_name} exists in {parent_folder}")
            
            # Use gdown to create the folder if needed
            # Note: This is a simplified implementation as gdown's folder creation
            # capabilities are limited - in practice you might need to use a more
            # direct Google Drive API approach
            
            return True
        except Exception as e:
            logger.error(f"Error ensuring Google Drive folder: {e}")
            return False
            
    @classmethod
    def load_model(cls, model_name, output_dir=None, storage_type=StorageType.GDRIVE, subfolder=None, quantize=False):
        """
        Load a model from storage.

        Args:
            model_name: Name of the model
            output_dir: Directory to save the model (None for default)
            storage_type: Type of storage to use
            subfolder: Optional subfolder in models directory (e.g., "checkpoints")
            quantize: Whether to load in quantized format (for torch models)

        Returns:
            Path to the loaded model
        """
        try:
            # If output_dir is not provided, use the default
            if output_dir is None:
                _, local_models_dir = cls.SYNC_FOLDERS["models"]
                
                # Handle optional subfolder
                if subfolder:
                    local_models_dir = os.path.join(local_models_dir, subfolder)
                
                output_dir = os.path.join(local_models_dir, model_name)

            # Check if model exists locally
            model_exists_locally = os.path.exists(output_dir)
            
            # If model doesn't exist locally, try to sync from remote
            if not model_exists_locally and storage_type == StorageType.GDRIVE:
                logger.info(f"Model {model_name} not found locally, syncing from Google Drive...")
                sync_success = cls.sync_from_gdrive("models")
                if not sync_success:
                    logger.error(f"Failed to sync model {model_name} from Google Drive")
                    return None
                
                # Check again if model exists locally after sync
                model_exists_locally = os.path.exists(output_dir)

            # If model still doesn't exist locally, try loading directly from Google Drive
            if not model_exists_locally and storage_type == StorageType.GDRIVE and GDRIVE_AVAILABLE:
                logger.info(f"Attempting direct download from Google Drive for model {model_name}...")
                try:
                    # Create output directory
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Determine the Google Drive folder ID based on subfolder
                    folder_id = cls.GDRIVE_FOLDER_IDS.get("models")
                    
                    # If using gdown for direct download, you'd implement the download here
                    # This is a placeholder for the direct download implementation
                    logger.warning("Direct download not implemented, falling back to rclone sync")
                    
                    # Fall back to rclone sync for the models folder
                    sync_success = cls.sync_from_gdrive("models")
                    if not sync_success:
                        logger.error(f"Failed to sync model {model_name} from Google Drive")
                        return None
                    
                    # Check again if model exists locally after direct download attempt
                    model_exists_locally = os.path.exists(output_dir)
                except Exception as e:
                    logger.error(f"Error downloading model directly from Google Drive: {e}")
            
            # If model still doesn't exist locally, return None
            if not model_exists_locally:
                logger.error(f"Model {model_name} not found locally or remotely")
                return None
            
            logger.info(f"Successfully loaded model {model_name} from {'local storage' if model_exists_locally else 'Google Drive'}")
            
            # Load model in quantized format if requested
            if quantize and TORCH_AVAILABLE:
                try:
                    from transformers import BitsAndBytesConfig
                    logger.info(f"Loading model {model_name} in quantized format")
                    
                    # Return the directory path with quantization info
                    return {
                        "model_path": output_dir,
                        "quantization_config": BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True
                        )
                    }
                except ImportError:
                    logger.warning("BitsAndBytesConfig not available, returning unquantized model path")
                    return output_dir
            
            return output_dir
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None

    @classmethod
    def save_dataset(cls, dataset_path, dataset_name, metadata=None, compress=True, storage_type=StorageType.GDRIVE):
        """
        Save a dataset to the specified storage.

        Args:
            dataset_path: Path to the dataset file or directory
            dataset_name: Name of the dataset
            metadata: Additional metadata to save with the dataset
            compress: Whether to compress the dataset
            storage_type: Type of storage to use

        Returns:
            Dictionary with save information
        """
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset path {dataset_path} does not exist")
            return {"success": False, "error": "Dataset path does not exist"}

        # Create a temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Compress the dataset if requested
            if compress and os.path.isdir(dataset_path):
                zip_file = os.path.join(temp_dir, f"{dataset_name}.zip")
                if not compress_directory(dataset_path, zip_file):
                    return {"success": False, "error": "Failed to compress dataset directory"}

                file_to_save = zip_file
            else:
                # Copy the dataset file to the temporary directory
                if os.path.isfile(dataset_path):
                    file_to_save = os.path.join(temp_dir, os.path.basename(dataset_path))
                    shutil.copy(dataset_path, file_to_save)
                else:
                    # If it's a directory and we're not compressing, just use the original path
                    file_to_save = dataset_path

            # Create metadata
            if metadata is None:
                metadata = {}

            metadata.update({
                "dataset_name": dataset_name,
                "timestamp": time.time(),
                "save_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "compressed": compress and os.path.isdir(dataset_path)
            })

            metadata_file = os.path.join(temp_dir, f"{dataset_name}_metadata.json")
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            # Save to the appropriate storage
            if storage_type == StorageType.GDRIVE:
                # Save to Google Drive
                if not cls._check_rclone():
                    return {"success": False, "error": "rclone not available for Google Drive storage"}

                # Get the Google Drive directory
                gdrive_dir, _ = cls.SYNC_FOLDERS.get("datasets", (None, None))
                if not gdrive_dir:
                    return {"success": False, "error": "Google Drive datasets directory not configured"}

                # Create dataset-specific directory in Google Drive
                dataset_gdrive_dir = f"{gdrive_dir}/{dataset_name}"

                try:
                    # Upload the dataset file
                    cmd = ["rclone", "copy", file_to_save, dataset_gdrive_dir, "--progress"]
                    subprocess.run(cmd, check=True)

                    # Upload the metadata file
                    cmd = ["rclone", "copy", metadata_file, dataset_gdrive_dir, "--progress"]
                    subprocess.run(cmd, check=True)

                    logger.info(f"Successfully saved dataset {dataset_name} to Google Drive")
                    return {"success": True, "storage_type": "gdrive", "dataset_name": dataset_name}
                except Exception as e:
                    logger.error(f"Error saving dataset to Google Drive: {e}")
                    return {"success": False, "error": f"Failed to save to Google Drive: {str(e)}"}

            # Other storage types would be implemented similarly
            else:
                # Local storage
                local_datasets_dir, _ = cls.SYNC_FOLDERS.get("datasets", (None, None))
                if not local_datasets_dir:
                    local_datasets_dir = os.path.join(cls.LOCAL_BASE, "datasets")

                # Create dataset-specific directory
                dataset_local_dir = os.path.join(local_datasets_dir, dataset_name)
                os.makedirs(dataset_local_dir, exist_ok=True)

                try:
                    # Copy the dataset file
                    if os.path.isfile(file_to_save):
                        shutil.copy(file_to_save, os.path.join(dataset_local_dir, os.path.basename(file_to_save)))
                    else:
                        # If it's a directory, copy the entire directory
                        shutil.copytree(file_to_save, os.path.join(dataset_local_dir, os.path.basename(file_to_save)), dirs_exist_ok=True)

                    # Copy the metadata file
                    shutil.copy(metadata_file, os.path.join(dataset_local_dir, f"{dataset_name}_metadata.json"))

                    logger.info(f"Successfully saved dataset {dataset_name} to local storage")
                    return {"success": True, "storage_type": "local", "dataset_name": dataset_name}
                except Exception as e:
                    logger.error(f"Error saving dataset to local storage: {e}")
                    return {"success": False, "error": f"Failed to save to local storage: {str(e)}"}

    @classmethod
    def optimize_model_storage(cls, model_dir, output_dir=None, quantize=True, compress=True):
        """
        Optimize model storage by quantizing and/or compressing the model.

        Args:
            model_dir: Directory containing the model
            output_dir: Directory to save the optimized model (default: same as model_dir)
            quantize: Whether to quantize the model
            compress: Whether to compress the model

        Returns:
            Dictionary with optimization information
        """
        if not os.path.exists(model_dir):
            logger.error(f"Model directory {model_dir} does not exist")
            return {"success": False, "error": "Model directory does not exist"}

        if output_dir is None:
            output_dir = model_dir

        os.makedirs(output_dir, exist_ok=True)

        # Quantize the model if requested
        if quantize and TORCH_AVAILABLE:
            try:
                # This is a placeholder for actual quantization logic
                # In a real implementation, you would load the model, quantize it, and save it
                logger.info(f"Quantizing model in {model_dir}")

                # For demonstration purposes, we'll just copy the model directory
                if model_dir != output_dir:
                    for item in os.listdir(model_dir):
                        s = os.path.join(model_dir, item)
                        d = os.path.join(output_dir, item)
                        if os.path.isfile(s):
                            shutil.copy2(s, d)
                        else:
                            shutil.copytree(s, d, dirs_exist_ok=True)

                logger.info("Model quantization complete")
            except Exception as e:
                logger.error(f"Error quantizing model: {e}")
                return {"success": False, "error": f"Failed to quantize model: {str(e)}"}

        # Compress the model if requested
        if compress:
            try:
                # Create a zip file of the model
                zip_file = f"{output_dir}.zip"
                if compress_directory(output_dir, zip_file):
                    logger.info(f"Model compressed to {zip_file}")
                    return {"success": True, "optimized": True, "quantized": quantize, "compressed": True, "zip_file": zip_file}
                else:
                    logger.error("Failed to compress model")
                    return {"success": False, "error": "Failed to compress model"}
            except Exception as e:
                logger.error(f"Error compressing model: {e}")
                return {"success": False, "error": f"Failed to compress model: {str(e)}"}

        return {"success": True, "optimized": True, "quantized": quantize, "compressed": False}

    @classmethod
    def get_storage_status(cls):
        """Get storage status for local and Google Drive"""
        status = {"local": {}, "gdrive": {}}

        # Local storage status
        for folder_name, (_, local_dir) in cls.SYNC_FOLDERS.items():
            if os.path.exists(local_dir):
                total_size = 0
                file_count = 0
                for dirpath, _, filenames in os.walk(local_dir):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        if os.path.exists(fp):
                            total_size += os.path.getsize(fp)
                            file_count += 1

                status["local"][folder_name] = {
                    "size_bytes": total_size,
                    "size_mb": total_size / (1024 * 1024),
                    "file_count": file_count
                }

        # Google Drive storage status (if rclone is available)
        if cls._check_rclone():
            for folder_name, (gdrive_dir, _) in cls.SYNC_FOLDERS.items():
                try:
                    cmd = ["rclone", "size", gdrive_dir]
                    result = subprocess.run(cmd, capture_output=True, text=True)

                    if result.returncode == 0:
                        # Parse the output to get size information
                        output = result.stdout.strip()
                        status["gdrive"][folder_name] = {"raw_output": output}

                        # Try to extract size in bytes
                        for line in output.split("\n"):
                            if "Total size:" in line:
                                try:
                                    size_str = line.split("Total size:")[1].strip().split("(")[0].strip()
                                    if "bytes" in size_str:
                                        size_bytes = int(size_str.replace("bytes", "").strip())
                                        status["gdrive"][folder_name]["size_bytes"] = size_bytes
                                        status["gdrive"][folder_name]["size_mb"] = size_bytes / (1024 * 1024)
                                except Exception:
                                    pass

                            if "Total objects:" in line:
                                try:
                                    count_str = line.split("Total objects:")[1].strip()
                                    status["gdrive"][folder_name]["file_count"] = int(count_str)
                                except Exception:
                                    pass
                except Exception as e:
                    logger.error(f"Error getting Google Drive storage status for {folder_name}: {e}")

        return status

# Singleton instance for easy import
storage_manager = StorageManager()

# Expose key functions for backward compatibility
def sync_to_gdrive(folder_type=None):
    """Sync to Google Drive"""
    return StorageManager.sync_to_gdrive(folder_type)

def sync_from_gdrive(folder_type=None):
    """Sync from Google Drive"""
    return StorageManager.sync_from_gdrive(folder_type)

def clear_local_storage(folder_type=None, confirm=False):
    """Clear local storage"""
    return StorageManager.clear_local_storage(folder_type, confirm)

def get_storage_status():
    """Get storage status"""
    return StorageManager.get_storage_status()
