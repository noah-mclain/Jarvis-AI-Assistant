"""
Storage Optimization Module for DeepSeek-Coder

This module provides utilities for optimizing storage usage when fine-tuning 
DeepSeek-Coder models on Paperspace Gradient with Google Drive integration.

Features:
- Model quantization (4-bit/8-bit)
- External cloud storage integration (Google Drive/S3)
- Efficient checkpointing strategies
- Dataset streaming and compression
"""

import os
import json
import shutil
import tempfile
import torch
from typing import Optional, List, Dict, Any, Union, Tuple
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset
from peft import PeftModel, PeftConfig
import gzip
import logging
import time
from dataclasses import dataclass
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# For Google Drive integration
try:
    import gdown
    import google.auth
    import google.auth.transport.requests
    from google.oauth2 import service_account
    GDRIVE_AVAILABLE = True
except ImportError:
    GDRIVE_AVAILABLE = False

# For S3 integration
try:
    import boto3
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

class StorageType(str, Enum):
    """Storage types for model checkpoints and datasets"""
    LOCAL = "local"
    GOOGLE_DRIVE = "gdrive"
    S3 = "s3"

class CheckpointStrategy(str, Enum):
    """Strategy for saving checkpoints during training"""
    ALL = "all"           # Save all checkpoints (not recommended for limited storage)
    IMPROVEMENT = "improvement"  # Save only checkpoints that improve on validation metrics
    REGULAR = "regular"    # Save checkpoint at regular intervals
    HYBRID = "hybrid"      # Save best checkpoint + latest checkpoint

@dataclass
class StorageConfig:
    """Configuration for storage optimization"""
    storage_type: StorageType = StorageType.LOCAL
    # Google Drive specific
    gdrive_folder_id: Optional[str] = None
    service_account_file: Optional[str] = None
    # S3 specific
    s3_bucket: Optional[str] = None
    s3_prefix: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: Optional[str] = "us-east-1"
    # Local specific
    local_dir: Optional[str] = "./models"
    # Common configuration
    checkpoint_strategy: CheckpointStrategy = CheckpointStrategy.IMPROVEMENT
    max_checkpoints: int = 3
    compress_checkpoints: bool = True
    

class GoogleDriveStorage:
    """Google Drive storage manager optimized for Paperspace Gradient"""
    
    def __init__(
        self, 
        folder_id: Optional[str] = None,
        service_account_file: Optional[str] = None,
    ):
        """
        Initialize Google Drive storage
        
        Args:
            folder_id: Google Drive folder ID to use for storage
            service_account_file: Path to service account credentials JSON file
        """
        if not GDRIVE_AVAILABLE:
            raise ImportError(
                "Google Drive integration requires gdown and google-auth libraries. "
                "Install with: pip install gdown google-auth"
            )
        
        self.folder_id = folder_id
        self.service_account_file = service_account_file
        self.authenticated = False
        self.setup_auth()
    
    def setup_auth(self):
        """Setup authentication for Google Drive"""
        if self.service_account_file and os.path.exists(self.service_account_file):
            try:
                # Use service account for authentication
                credentials = service_account.Credentials.from_service_account_file(
                    self.service_account_file,
                    scopes=['https://www.googleapis.com/auth/drive']
                )
                auth_req = google.auth.transport.requests.Request()
                credentials.refresh(auth_req)
                self.authenticated = True
                print("Successfully authenticated with Google Drive using service account")
            except Exception as e:
                print(f"Error authenticating with service account: {e}")
                print("Falling back to gdown for authentication")
        
        # Even without explicit authentication, gdown will handle browser auth if needed
        self.authenticated = True
    
    def create_folder(self, folder_name: str) -> Optional[str]:
        """
        Create a folder in Google Drive
        
        Args:
            folder_name: Name of the folder to create
            
        Returns:
            folder_id: ID of the created folder, or None if creation failed
        """
        # This is a simplified implementation that relies on gdown
        # For more complex operations, consider using the full Google Drive API
        if not self.folder_id:
            print("Warning: Parent folder ID not provided. Creating folder in root directory.")
        
        # For now, users will need to create folders manually and provide folder IDs
        print(f"Please create folder '{folder_name}' manually in Google Drive and provide the folder ID")
        return None
    
    def upload_file(self, local_path: str, remote_name: Optional[str] = None) -> Optional[str]:
        """
        Upload a file to Google Drive
        
        Args:
            local_path: Path to local file
            remote_name: Name to use in Google Drive (defaults to local filename)
            
        Returns:
            file_id: ID of the uploaded file, or None if upload failed
        """
        if not os.path.exists(local_path):
            print(f"Error: Local file {local_path} does not exist")
            return None
        
        if not self.folder_id:
            print("Error: No folder ID provided for Google Drive upload")
            return None
        
        remote_name = remote_name or os.path.basename(local_path)
        
        try:
            # Upload to Google Drive using gdown
            url = f"https://drive.google.com/drive/folders/{self.folder_id}"
            gdown.upload(local_path, url, folder=True)
            print(f"Successfully uploaded {local_path} to Google Drive folder {self.folder_id}")
            
            # Return a placeholder ID - gdown doesn't return the specific file ID
            return "uploaded_via_gdown"
        except Exception as e:
            print(f"Error uploading to Google Drive: {e}")
            return None
    
    def download_file(self, file_id: str, local_path: str) -> bool:
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
    
    def list_files(self, folder_id: Optional[str] = None) -> List[Dict[str, str]]:
        """
        List files in a Google Drive folder
        
        Args:
            folder_id: Google Drive folder ID (defaults to initialized folder_id)
            
        Returns:
            files: List of dictionaries with file information
        """
        folder_id = folder_id or self.folder_id
        if not folder_id:
            print("Error: No folder ID provided")
            return []
        
        try:
            # Use gdown to list folder contents
            url = f"https://drive.google.com/drive/folders/{folder_id}"
            print(f"Please check folder contents manually at: {url}")
            
            # This is a limitation of gdown - it doesn't have a Python API for listing files
            # Return an empty list but prompt user to check the URL
            return []
        except Exception as e:
            print(f"Error listing files: {e}")
            return []
    
    def delete_file(self, file_id: str) -> bool:
        """
        Delete a file from Google Drive
        
        Args:
            file_id: Google Drive file ID
            
        Returns:
            success: True if deletion succeeded, False otherwise
        """
        # Not supported by gdown - would require full Drive API
        print("File deletion not supported through gdown API")
        print(f"Please delete file {file_id} manually from Google Drive")
        return False


class StorageOptimization:
    """Storage optimization utilities for DeepSeek-Coder fine-tuning"""
    
    def __init__(self, config: StorageConfig):
        """
        Initialize storage optimization
        
        Args:
            config: Storage configuration
        """
        self.config = config
        self.storage_handler = self._initialize_storage()
        self.checkpoints = []
        
    def _initialize_storage(self):
        """Initialize appropriate storage handler based on configuration"""
        if self.config.storage_type == StorageType.GOOGLE_DRIVE:
            if not GDRIVE_AVAILABLE:
                raise ImportError(
                    "Google Drive integration requires gdown and google-auth libraries. "
                    "Install with: pip install gdown google-auth"
                )
            return GoogleDriveStorage(
                folder_id=self.config.gdrive_folder_id,
                service_account_file=self.config.service_account_file,
            )
        elif self.config.storage_type == StorageType.S3:
            if not S3_AVAILABLE:
                raise ImportError("S3 integration requires boto3. Install with: pip install boto3")
            # S3 initialization would go here
            return None
        else:
            # Local storage doesn't need special initialization
            return None
    
    def compress_directory(self, directory: str, output_file: str) -> str:
        """
        Compress a directory into a ZIP archive
        
        Args:
            directory: Directory to compress
            output_file: Path to output ZIP file
            
        Returns:
            output_file: Path to created ZIP file
        """
        import shutil
        
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
        import shutil
        
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
    
    def save_checkpoint(
        self, 
        checkpoint_dir: str, 
        checkpoint_name: str,
        metrics: Dict[str, float],
        step: int,
    ) -> bool:
        """
        Save a training checkpoint based on the configured strategy
        
        Args:
            checkpoint_dir: Directory containing checkpoint files
            checkpoint_name: Base name for the checkpoint
            metrics: Evaluation metrics for the checkpoint
            step: Training step number
            
        Returns:
            success: True if the checkpoint was saved, False otherwise
        """
        if not os.path.exists(checkpoint_dir):
            print(f"Error: Checkpoint directory {checkpoint_dir} does not exist")
            return False
        
        # Prepare metadata
        checkpoint_info = {
            "name": checkpoint_name,
            "step": step,
            "metrics": metrics,
            "timestamp": time.time(),
        }
        
        # Check if this checkpoint should be saved based on strategy
        if self.config.checkpoint_strategy == CheckpointStrategy.ALL:
            # Save all checkpoints
            save_checkpoint = True
        elif self.config.checkpoint_strategy == CheckpointStrategy.IMPROVEMENT:
            # Only save if this is the first checkpoint or metrics improved
            if not self.checkpoints:
                save_checkpoint = True
            else:
                # Check if eval_loss improved (lower is better)
                best_loss = min(c["metrics"].get("eval_loss", float("inf")) for c in self.checkpoints)
                current_loss = metrics.get("eval_loss", float("inf"))
                save_checkpoint = current_loss < best_loss
        elif self.config.checkpoint_strategy == CheckpointStrategy.REGULAR:
            # Save at regular intervals (every N steps)
            save_interval = 100  # Save every 100 steps by default
            save_checkpoint = step % save_interval == 0
        elif self.config.checkpoint_strategy == CheckpointStrategy.HYBRID:
            # Save if metrics improved OR this is the latest checkpoint
            if not self.checkpoints:
                save_checkpoint = True
            else:
                # Check if eval_loss improved (lower is better)
                best_loss = min(c["metrics"].get("eval_loss", float("inf")) for c in self.checkpoints)
                current_loss = metrics.get("eval_loss", float("inf"))
                # Save if improved or if this is significantly later than last checkpoint
                last_step = max(c["step"] for c in self.checkpoints)
                save_checkpoint = (current_loss < best_loss) or (step - last_step >= 100)
        else:
            # Default - save all checkpoints
            save_checkpoint = True
        
        if not save_checkpoint:
            print(f"Skipping checkpoint at step {step} based on {self.config.checkpoint_strategy} strategy")
            return False
        
        # Add to checkpoints list
        self.checkpoints.append(checkpoint_info)
        
        # Enforce maximum number of checkpoints if needed
        if len(self.checkpoints) > self.config.max_checkpoints:
            # If using improvement strategy, remove worst checkpoint
            if self.config.checkpoint_strategy == CheckpointStrategy.IMPROVEMENT:
                # Sort by loss (higher is worse)
                sorted_checkpoints = sorted(
                    self.checkpoints,
                    key=lambda c: c["metrics"].get("eval_loss", float("inf"))
                )
                # Remove worst checkpoint
                checkpoint_to_remove = sorted_checkpoints[-1]
                self._remove_checkpoint(checkpoint_to_remove)
                self.checkpoints.remove(checkpoint_to_remove)
            # For other strategies, remove oldest checkpoint
            else:
                # Sort by step (lower is older)
                sorted_checkpoints = sorted(self.checkpoints, key=lambda c: c["step"])
                # Remove oldest checkpoint
                checkpoint_to_remove = sorted_checkpoints[0]
                self._remove_checkpoint(checkpoint_to_remove)
                self.checkpoints.remove(checkpoint_to_remove)
        
        # Save the current checkpoint
        if self.config.storage_type == StorageType.LOCAL:
            # For local storage, no need to do anything - checkpoint is already saved
            # Just save a metadata file with checkpoint information
            metadata_file = os.path.join(checkpoint_dir, f"{checkpoint_name}_metadata.json")
            with open(metadata_file, "w") as f:
                json.dump(checkpoint_info, f, indent=2)
            return True
        elif self.config.storage_type == StorageType.GOOGLE_DRIVE and self.storage_handler:
            # For Google Drive, compress and upload the checkpoint
            return self._upload_checkpoint_to_gdrive(checkpoint_dir, checkpoint_name, checkpoint_info)
        elif self.config.storage_type == StorageType.S3 and self.storage_handler:
            # S3 implementation would go here
            pass
        
        return False
    
    def _upload_checkpoint_to_gdrive(self, checkpoint_dir: str, checkpoint_name: str, checkpoint_info: Dict) -> bool:
        """
        Upload a checkpoint to Google Drive
        
        Args:
            checkpoint_dir: Directory containing checkpoint files
            checkpoint_name: Base name for the checkpoint
            checkpoint_info: Metadata for the checkpoint
            
        Returns:
            success: True if upload succeeded, False otherwise
        """
        # Create a temporary directory for compression
        with tempfile.TemporaryDirectory() as temp_dir:
            # Compress the checkpoint if needed
            if self.config.compress_checkpoints:
                zip_file = os.path.join(temp_dir, f"{checkpoint_name}.zip")
                self.compress_directory(checkpoint_dir, zip_file)
                
                # Upload the compressed file
                file_id = self.storage_handler.upload_file(zip_file, f"{checkpoint_name}.zip")
                success = file_id is not None
            else:
                # Upload all files in the checkpoint directory individually
                success = True
                for filename in os.listdir(checkpoint_dir):
                    file_path = os.path.join(checkpoint_dir, filename)
                    if os.path.isfile(file_path):
                        file_id = self.storage_handler.upload_file(file_path, filename)
                        if file_id is None:
                            success = False
            
            # Save checkpoint metadata
            if success:
                # Add upload information to metadata
                checkpoint_info["storage"] = {
                    "type": self.config.storage_type,
                    "folder_id": self.storage_handler.folder_id,
                    "compressed": self.config.compress_checkpoints,
                }
                
                # Save metadata locally
                metadata_file = os.path.join(temp_dir, f"{checkpoint_name}_metadata.json")
                with open(metadata_file, "w") as f:
                    json.dump(checkpoint_info, f, indent=2)
                
                # Upload metadata file
                self.storage_handler.upload_file(metadata_file, f"{checkpoint_name}_metadata.json")
            
            return success
    
    def _remove_checkpoint(self, checkpoint_info: Dict) -> bool:
        """
        Remove a checkpoint based on the storage type
        
        Args:
            checkpoint_info: Metadata for the checkpoint to remove
            
        Returns:
            success: True if removal succeeded, False otherwise
        """
        checkpoint_name = checkpoint_info["name"]
        
        if self.config.storage_type == StorageType.LOCAL:
            # For local storage, simply delete the checkpoint directory
            checkpoint_dir = os.path.join(self.config.local_dir, checkpoint_name)
            if os.path.exists(checkpoint_dir):
                try:
                    shutil.rmtree(checkpoint_dir)
                    print(f"Removed checkpoint: {checkpoint_name}")
                    return True
                except Exception as e:
                    print(f"Error removing checkpoint {checkpoint_name}: {e}")
                    return False
            return False
        elif self.config.storage_type == StorageType.GOOGLE_DRIVE and self.storage_handler:
            # For Google Drive, file deletion is not supported by gdown
            # Print a warning and let the user know they need to delete manually
            print(f"Warning: Cannot automatically delete checkpoint {checkpoint_name} from Google Drive")
            print("Please delete the file manually to save space")
            return False
        elif self.config.storage_type == StorageType.S3 and self.storage_handler:
            # S3 implementation would go here
            pass
        
        return False
    
    def load_best_checkpoint(self, output_dir: str) -> Optional[str]:
        """
        Load the best checkpoint based on evaluation metrics
        
        Args:
            output_dir: Directory to load the checkpoint into
            
        Returns:
            checkpoint_dir: Path to the loaded checkpoint directory, or None if loading failed
        """
        if not self.checkpoints:
            print("No checkpoints available to load")
            return None
        
        # Find the best checkpoint based on eval_loss
        best_checkpoint = min(
            self.checkpoints,
            key=lambda c: c["metrics"].get("eval_loss", float("inf"))
        )
        
        checkpoint_name = best_checkpoint["name"]
        print(f"Loading best checkpoint: {checkpoint_name}")
        
        if self.config.storage_type == StorageType.LOCAL:
            # For local storage, just return the path
            checkpoint_dir = os.path.join(self.config.local_dir, checkpoint_name)
            if os.path.exists(checkpoint_dir):
                return checkpoint_dir
            else:
                print(f"Error: Checkpoint directory {checkpoint_dir} does not exist")
                return None
        elif self.config.storage_type == StorageType.GOOGLE_DRIVE and self.storage_handler:
            # For Google Drive, download and extract the checkpoint
            return self._download_checkpoint_from_gdrive(best_checkpoint, output_dir)
        elif self.config.storage_type == StorageType.S3 and self.storage_handler:
            # S3 implementation would go here
            pass
        
        return None
    
    def _download_checkpoint_from_gdrive(self, checkpoint_info: Dict, output_dir: str) -> Optional[str]:
        """
        Download a checkpoint from Google Drive
        
        Args:
            checkpoint_info: Metadata for the checkpoint
            output_dir: Directory to extract the checkpoint into
            
        Returns:
            checkpoint_dir: Path to the extracted checkpoint directory, or None if download failed
        """
        checkpoint_name = checkpoint_info["name"]
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get storage information
        storage = checkpoint_info.get("storage", {})
        compressed = storage.get("compressed", self.config.compress_checkpoints)
        
        if compressed:
            # Download the compressed file
            local_zip = os.path.join(output_dir, f"{checkpoint_name}.zip")
            # Note: For real implementation, you would need to store and retrieve the actual file_id
            file_id = input(f"Please enter the Google Drive file ID for {checkpoint_name}.zip: ")
            
            success = self.storage_handler.download_file(file_id, local_zip)
            if not success:
                print(f"Error downloading checkpoint {checkpoint_name}")
                return None
            
            # Extract the ZIP file
            checkpoint_dir = os.path.join(output_dir, checkpoint_name)
            self.decompress_archive(local_zip, checkpoint_dir)
            
            # Clean up the ZIP file
            os.remove(local_zip)
            
            return checkpoint_dir
        else:
            # We would need to download individual files
            # This is more complex and requires knowledge of the checkpoint structure
            print("Error: Uncompressed checkpoint download not implemented")
            return None


def configure_optimizer_for_4bit(model, lr=2e-5, weight_decay=0.01):
    """
    Configure optimizer specifically for 4-bit training
    
    Args:
        model: The model to optimize
        lr: Learning rate
        weight_decay: Weight decay factor
        
    Returns:
        optimizer: Configured optimizer
    """
    from torch.optim import AdamW
    from transformers.optimization import get_scheduler
    
    # Separate weight decay between LoRA and non-LoRA parameters
    # This is important for 4-bit quantized models
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if "lora" in n and p.requires_grad],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if "lora" not in n and p.requires_grad],
            "weight_decay": 0.0,  # No weight decay for non-LoRA params with 4-bit
        },
    ]
    
    # Use AdamW optimizer
    optimizer = AdamW(param_groups, lr=lr)
    return optimizer


def setup_checkpoint_callback(trainer, storage_optimizer):
    """
    Set up a checkpoint callback that uses storage optimization
    
    Args:
        trainer: Hugging Face Trainer
        storage_optimizer: StorageOptimization instance
        
    Returns:
        callback: Checkpoint callback
    """
    from transformers import TrainerCallback
    
    class OptimizedCheckpointCallback(TrainerCallback):
        def __init__(self, storage_optimizer):
            self.storage_optimizer = storage_optimizer
        
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if not metrics:
                return
            
            # Save checkpoint using storage optimization
            checkpoint_dir = args.output_dir
            checkpoint_name = f"checkpoint-{state.global_step}"
            
            self.storage_optimizer.save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                checkpoint_name=checkpoint_name,
                metrics=metrics,
                step=state.global_step,
            )
    
    return OptimizedCheckpointCallback(storage_optimizer)


def get_default_paperspace_config():
    """
    Get default storage configuration for Paperspace Gradient
    
    Returns:
        config: StorageConfig with Paperspace-specific defaults
    """
    # Check if running on Paperspace Gradient
    on_gradient = os.environ.get("GRADIENT", "0") == "1" or os.path.exists("/storage")
    
    if on_gradient:
        # Configure for Paperspace Gradient environment
        storage_dir = "/storage/models" if os.path.exists("/storage") else "./models"
        return StorageConfig(
            storage_type=StorageType.LOCAL,
            local_dir=storage_dir,
            checkpoint_strategy=CheckpointStrategy.IMPROVEMENT,
            max_checkpoints=2,  # Keep only 2 checkpoints to save space
            compress_checkpoints=True,
        )
    else:
        # Standard configuration for other environments
        return StorageConfig(
            storage_type=StorageType.LOCAL,
            local_dir="./models",
            checkpoint_strategy=CheckpointStrategy.IMPROVEMENT,
            max_checkpoints=3,
            compress_checkpoints=False,
        )

# ========== MODEL QUANTIZATION ==========

def quantize_model(
    model_path: str, 
    output_path: str, 
    bits: int = 8,
    device: str = "auto"
) -> str:
    """
    Quantize a DeepSeek model to reduce its size.
    
    Args:
        model_path: Path to the model to quantize
        output_path: Path to save the quantized model
        bits: Quantization precision (4 or 8)
        device: Device to use for quantization
        
    Returns:
        Path to the quantized model
    """
    from transformers import BitsAndBytesConfig
    
    logger.info(f"Quantizing model from {model_path} to {bits}-bit precision")
    
    if bits not in [4, 8]:
        raise ValueError(f"Bits must be 4 or 8, got {bits}")
    
    # Create quantization config
    if bits == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    else:  # 8-bit
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map=device,
        trust_remote_code=True
    )
    
    # Save tokenizer and model config
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Save the quantized model
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    logger.info(f"Quantized model saved to {output_path}")
    
    # Calculate and log size reduction
    original_size = get_directory_size(model_path)
    quantized_size = get_directory_size(output_path)
    reduction = (1 - quantized_size / original_size) * 100 if original_size > 0 else 0
    
    logger.info(f"Size reduction: {reduction:.2f}% (from {original_size/1e9:.2f}GB to {quantized_size/1e9:.2f}GB)")
    
    return output_path

def get_directory_size(path: str) -> int:
    """Calculate the total size of a directory in bytes"""
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

# ========== EXTERNAL STORAGE INTEGRATION ==========

def setup_google_drive():
    """
    Set up Google Drive for external storage.
    Requires the user to authenticate with their Google account.
    
    Returns:
        The Google Drive client
    """
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        logger.info("Google Drive mounted at /content/drive")
        return True
    except ImportError:
        try:
            # For non-Colab environments
            import gdown
            logger.info("Using gdown for Google Drive access")
            return gdown
        except ImportError:
            logger.error("Failed to set up Google Drive. Install gdown package: pip install gdown")
            return None

def setup_s3_storage(aws_access_key_id=None, aws_secret_access_key=None):
    """
    Set up Amazon S3 for external storage.
    
    Args:
        aws_access_key_id: AWS access key ID
        aws_secret_access_key: AWS secret access key
        
    Returns:
        S3 client
    """
    try:
        import boto3
        from botocore.exceptions import NoCredentialsError
        
        # Get credentials from environment if not provided
        if aws_access_key_id is None:
            aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
        if aws_secret_access_key is None:
            aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        
        if aws_access_key_id and aws_secret_access_key:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key
            )
            logger.info("S3 client initialized successfully")
            return s3_client
        else:
            logger.error("AWS credentials not provided and not found in environment variables")
            return None
    except ImportError:
        logger.error("Failed to set up S3 storage. Install boto3 package: pip install boto3")
        return None
    except NoCredentialsError:
        logger.error("AWS credentials not found or invalid")
        return None

def upload_to_gdrive(source_path: str, drive_folder: str = "DeepSeek_Models"):
    """
    Upload a file or directory to Google Drive
    
    Args:
        source_path: Local path to file or directory
        drive_folder: Folder in Google Drive to upload to
        
    Returns:
        Path in Google Drive where the file/directory was uploaded
    """
    try:
        from google.colab import drive
        drive_root = "/content/drive/MyDrive"
        
        # Create folder if it doesn't exist
        os.makedirs(f"{drive_root}/{drive_folder}", exist_ok=True)
        
        if os.path.isfile(source_path):
            # Upload single file
            destination = f"{drive_root}/{drive_folder}/{os.path.basename(source_path)}"
            shutil.copy2(source_path, destination)
            logger.info(f"Uploaded {source_path} to {destination}")
            return destination
        elif os.path.isdir(source_path):
            # Upload directory
            destination = f"{drive_root}/{drive_folder}/{os.path.basename(source_path)}"
            if os.path.exists(destination):
                shutil.rmtree(destination)
            shutil.copytree(source_path, destination)
            logger.info(f"Uploaded directory {source_path} to {destination}")
            return destination
        else:
            logger.error(f"Source path {source_path} does not exist")
            return None
    except ImportError:
        try:
            import gdown
            logger.warning("Google Colab drive not available. Using gdown to upload to Google Drive")
            logger.error("Automatic upload with gdown not yet implemented. Please use `setup_google_drive()` in a Colab environment.")
            return None
        except ImportError:
            logger.error("Failed to upload to Google Drive. Install gdown package: pip install gdown")
            return None

def download_from_gdrive(drive_path: str, local_destination: str = None):
    """
    Download a file or directory from Google Drive
    
    Args:
        drive_path: Path in Google Drive
        local_destination: Local path to save to (default: current directory)
        
    Returns:
        Local path where the file/directory was downloaded
    """
    try:
        from google.colab import drive
        if local_destination is None:
            local_destination = os.path.basename(drive_path)
        
        if os.path.isfile(drive_path):
            # Download single file
            shutil.copy2(drive_path, local_destination)
            logger.info(f"Downloaded {drive_path} to {local_destination}")
        elif os.path.isdir(drive_path):
            # Download directory
            if os.path.exists(local_destination):
                shutil.rmtree(local_destination)
            shutil.copytree(drive_path, local_destination)
            logger.info(f"Downloaded directory {drive_path} to {local_destination}")
        else:
            logger.error(f"Source path {drive_path} does not exist")
            return None
        
        return local_destination
    except ImportError:
        try:
            import gdown
            logger.warning("Google Colab drive not available. Using gdown to download from Google Drive")
            
            # Handle file ID or URL
            if "/" not in drive_path and len(drive_path) == 33:
                # Looks like a file ID
                gdown.download(id=drive_path, output=local_destination)
                logger.info(f"Downloaded file ID {drive_path} to {local_destination}")
                return local_destination
            elif drive_path.startswith("https://drive.google.com"):
                # Handle Google Drive URL
                gdown.download(url=drive_path, output=local_destination)
                logger.info(f"Downloaded {drive_path} to {local_destination}")
                return local_destination
            else:
                logger.error(f"Invalid Google Drive path: {drive_path}")
                return None
        except ImportError:
            logger.error("Failed to download from Google Drive. Install gdown package: pip install gdown")
            return None

def upload_to_s3(s3_client, source_path: str, bucket_name: str, s3_prefix: str = "deepseek_models/"):
    """
    Upload a file or directory to Amazon S3
    
    Args:
        s3_client: Initialized S3 client
        source_path: Local path to file or directory
        bucket_name: S3 bucket name
        s3_prefix: Prefix for S3 object keys
        
    Returns:
        S3 URI of uploaded content
    """
    if s3_client is None:
        logger.error("S3 client not initialized")
        return None
    
    if os.path.isfile(source_path):
        # Upload single file
        s3_key = f"{s3_prefix}{os.path.basename(source_path)}"
        s3_client.upload_file(source_path, bucket_name, s3_key)
        logger.info(f"Uploaded {source_path} to s3://{bucket_name}/{s3_key}")
        return f"s3://{bucket_name}/{s3_key}"
    elif os.path.isdir(source_path):
        # Upload directory
        uploaded_files = []
        for root, _, files in os.walk(source_path):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, source_path)
                s3_key = f"{s3_prefix}{os.path.basename(source_path)}/{relative_path}"
                s3_client.upload_file(local_path, bucket_name, s3_key)
                uploaded_files.append(s3_key)
        
        logger.info(f"Uploaded directory {source_path} to s3://{bucket_name}/{s3_prefix}{os.path.basename(source_path)}/")
        return f"s3://{bucket_name}/{s3_prefix}{os.path.basename(source_path)}/"
    else:
        logger.error(f"Source path {source_path} does not exist")
        return None

def download_from_s3(s3_client, s3_uri: str, local_destination: str = None):
    """
    Download a file or directory from Amazon S3
    
    Args:
        s3_client: Initialized S3 client
        s3_uri: S3 URI to download from
        local_destination: Local path to save to
        
    Returns:
        Local path where the file/directory was downloaded
    """
    if s3_client is None:
        logger.error("S3 client not initialized")
        return None
    
    # Parse S3 URI
    if not s3_uri.startswith("s3://"):
        logger.error(f"Invalid S3 URI: {s3_uri}")
        return None
    
    s3_path = s3_uri[5:]  # Remove "s3://"
    bucket_name = s3_path.split("/")[0]
    s3_key = "/".join(s3_path.split("/")[1:])
    
    if local_destination is None:
        local_destination = os.path.basename(s3_key)
    
    try:
        # Check if this is a directory (has trailing slash)
        if s3_key.endswith("/"):
            os.makedirs(local_destination, exist_ok=True)
            
            # Get all objects with this prefix
            response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=s3_key)
            
            for obj in response.get('Contents', []):
                obj_key = obj['Key']
                if obj_key != s3_key:  # Skip the directory object itself
                    rel_path = obj_key[len(s3_key):]
                    local_path = os.path.join(local_destination, rel_path)
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    s3_client.download_file(bucket_name, obj_key, local_path)
            
            logger.info(f"Downloaded directory from s3://{bucket_name}/{s3_key} to {local_destination}")
        else:
            # Download single file
            os.makedirs(os.path.dirname(local_destination), exist_ok=True)
            s3_client.download_file(bucket_name, s3_key, local_destination)
            logger.info(f"Downloaded s3://{bucket_name}/{s3_key} to {local_destination}")
        
        return local_destination
    except Exception as e:
        logger.error(f"Error downloading from S3: {e}")
        return None

# ========== EFFICIENT CHECKPOINTING ==========

def create_checkpoint_strategy(
    total_steps: int, 
    save_mode: str = "improvement", 
    save_interval: int = 100,
    patience: int = 5,
    max_checkpoints: int = 3,
) -> Dict[str, Any]:
    """
    Create a storage-efficient checkpointing strategy
    
    Args:
        total_steps: Total training steps
        save_mode: Strategy mode - "regular" (periodic), "improvement" (on evaluation improvement only),
                  or "hybrid" (combination of both with fewer regular checkpoints)
        save_interval: Steps between regular checkpoints (for "regular" and "hybrid" modes)
        patience: Number of evaluations without improvement before overwriting checkpoint (for "improvement" mode)
        max_checkpoints: Maximum number of checkpoints to keep
        
    Returns:
        Dictionary with checkpointing strategy
    """
    if save_mode not in ["regular", "improvement", "hybrid"]:
        raise ValueError(f"Invalid save_mode: {save_mode}. Must be 'regular', 'improvement', or 'hybrid'")
    
    # Calculate steps for "hybrid" mode to space checkpoints more efficiently
    if save_mode == "hybrid":
        # Keep checkpoints at beginning, ~ 1/3, ~ 2/3, and end of training
        checkpoint_steps = [0, total_steps // 3, 2 * total_steps // 3, total_steps - 1]
    else:
        checkpoint_steps = []
    
    return {
        "mode": save_mode,
        "save_interval": save_interval,
        "patience": patience,
        "max_checkpoints": max_checkpoints,
        "checkpoint_steps": checkpoint_steps,
        "best_metric": float('inf'),
        "patience_counter": 0,
        "checkpoints": []
    }

def manage_checkpoints(
    strategy: Dict[str, Any],
    current_step: int,
    output_dir: str,
    metric_value: Optional[float] = None,
    remote_storage_func=None
) -> Tuple[bool, Optional[str]]:
    """
    Manage checkpoints according to strategy to minimize storage usage
    
    Args:
        strategy: Checkpoint strategy from create_checkpoint_strategy
        current_step: Current training step
        output_dir: Directory where checkpoints are saved
        metric_value: Value of evaluation metric (lower is better) for "improvement" mode
        remote_storage_func: Optional function to upload checkpoint to remote storage
        
    Returns:
        Tuple of (should_save, checkpoint_path)
    """
    should_save = False
    checkpoint_path = None
    
    mode = strategy["mode"]
    
    # Regular checkpointing at fixed intervals
    if mode in ["regular", "hybrid"]:
        if current_step % strategy["save_interval"] == 0 or current_step in strategy["checkpoint_steps"]:
            checkpoint_path = os.path.join(output_dir, f"checkpoint-{current_step}")
            should_save = True
            
            # Add to list of checkpoints
            strategy["checkpoints"].append(checkpoint_path)
            
            # Remove oldest checkpoint if exceeding max
            if len(strategy["checkpoints"]) > strategy["max_checkpoints"]:
                oldest_checkpoint = strategy["checkpoints"].pop(0)
                if os.path.exists(oldest_checkpoint):
                    logger.info(f"Removing oldest checkpoint: {oldest_checkpoint}")
                    shutil.rmtree(oldest_checkpoint)
    
    # Save on evaluation improvement
    if mode in ["improvement", "hybrid"] and metric_value is not None:
        if metric_value < strategy["best_metric"]:
            checkpoint_path = os.path.join(output_dir, "checkpoint-best")
            should_save = True
            
            # Update best metric and reset patience
            strategy["best_metric"] = metric_value
            strategy["patience_counter"] = 0
            
            logger.info(f"New best model with metric value {metric_value}")
        else:
            # Increment patience counter
            strategy["patience_counter"] += 1
            
            if strategy["patience_counter"] >= strategy["patience"]:
                logger.info(f"No improvement for {strategy['patience']} evaluations")
                
                # Optional: create early stopping mechanism
                # return False, "EARLY_STOP"
    
    # Upload to remote storage if provided
    if should_save and checkpoint_path and remote_storage_func:
        try:
            remote_path = remote_storage_func(checkpoint_path)
            logger.info(f"Uploaded checkpoint to remote storage: {remote_path}")
        except Exception as e:
            logger.error(f"Failed to upload checkpoint to remote storage: {e}")
    
    return should_save, checkpoint_path

# ========== EFFICIENT DATA HANDLING ==========

def compress_dataset(dataset_path: str, output_path: Optional[str] = None) -> str:
    """
    Compress a dataset file using gzip to save storage space
    
    Args:
        dataset_path: Path to dataset file
        output_path: Path to save compressed dataset (default: {dataset_path}.gz)
        
    Returns:
        Path to compressed file
    """
    if output_path is None:
        output_path = f"{dataset_path}.gz"
    
    with open(dataset_path, 'rb') as f_in:
        with gzip.open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    # Check compression ratio
    original_size = os.path.getsize(dataset_path)
    compressed_size = os.path.getsize(output_path)
    ratio = original_size / compressed_size if compressed_size > 0 else 0
    
    logger.info(f"Compressed {dataset_path} from {original_size/1e6:.2f}MB to {compressed_size/1e6:.2f}MB (ratio: {ratio:.2f}x)")
    
    return output_path

def decompress_dataset(compressed_path: str, output_path: Optional[str] = None) -> str:
    """
    Decompress a gzip-compressed dataset file
    
    Args:
        compressed_path: Path to compressed dataset file
        output_path: Path to save decompressed dataset
        
    Returns:
        Path to decompressed file
    """
    if output_path is None:
        output_path = compressed_path.replace('.gz', '')
    
    with gzip.open(compressed_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    logger.info(f"Decompressed {compressed_path} to {output_path}")
    
    return output_path

def setup_streaming_dataset(
    dataset_name: str,
    split: str = "train",
    streaming: bool = True,
    **kwargs
) -> Dataset:
    """
    Load a dataset in streaming mode to minimize storage usage
    
    Args:
        dataset_name: Name of the dataset in Hugging Face datasets
        split: Split to load (train, validation, test)
        streaming: Whether to stream the dataset
        **kwargs: Additional arguments to pass to load_dataset
        
    Returns:
        Dataset object
    """
    try:
        dataset = load_dataset(dataset_name, split=split, streaming=streaming, **kwargs)
        logger.info(f"Loaded dataset {dataset_name} in {'streaming' if streaming else 'standard'} mode")
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_name}: {e}")
        return None

# ========== MAIN UTILITY FUNCTIONS ==========

def optimize_storage_for_model(
    model_name: str = "deepseek-ai/deepseek-coder-6.7b-base",
    output_dir: str = "models/deepseek_optimized",
    quantize_bits: int = 8,
    use_external_storage: bool = False,
    storage_type: str = "gdrive",  # "gdrive" or "s3"
    remote_path: str = "DeepSeek_Models",
    aws_access_key_id: str = None,
    aws_secret_access_key: str = None,
    s3_bucket: str = None,
) -> Dict[str, Any]:
    """
    Optimize storage for a DeepSeek model by applying quantization and external storage
    
    Args:
        model_name: DeepSeek model name or path
        output_dir: Directory to save optimized model
        quantize_bits: Quantization precision (4 or 8)
        use_external_storage: Whether to use external storage
        storage_type: Type of external storage (gdrive or s3)
        remote_path: Path in external storage
        aws_access_key_id: AWS access key ID (for S3 storage)
        aws_secret_access_key: AWS secret access key (for S3 storage)
        s3_bucket: S3 bucket name (for S3 storage)
        
    Returns:
        Dictionary with optimization results
    """
    results = {
        "model_name": model_name,
        "optimized_path": None,
        "quantized": False,
        "external_storage_used": False,
        "external_storage_path": None,
        "size_original": None,
        "size_optimized": None,
    }
    
    # Create temporary directory for model download
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Downloading model {model_name} to temporary directory")
        
        # Download model
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        
        temp_model_path = os.path.join(temp_dir, "model")
        os.makedirs(temp_model_path, exist_ok=True)
        
        model.save_pretrained(temp_model_path)
        tokenizer.save_pretrained(temp_model_path)
        
        results["size_original"] = get_directory_size(temp_model_path)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Apply quantization
        quantized_path = quantize_model(temp_model_path, output_dir, bits=quantize_bits)
        results["optimized_path"] = quantized_path
        results["quantized"] = True
        results["size_optimized"] = get_directory_size(quantized_path)
        
        # Use external storage if requested
        if use_external_storage:
            if storage_type == "gdrive":
                # Set up Google Drive
                drive_client = setup_google_drive()
                if drive_client:
                    # Upload to Google Drive
                    remote_storage_path = upload_to_gdrive(quantized_path, drive_folder=remote_path)
                    results["external_storage_used"] = True
                    results["external_storage_path"] = remote_storage_path
                else:
                    logger.error("Failed to set up Google Drive")
            elif storage_type == "s3":
                # Set up S3
                s3_client = setup_s3_storage(aws_access_key_id, aws_secret_access_key)
                if s3_client and s3_bucket:
                    # Upload to S3
                    remote_storage_path = upload_to_s3(
                        s3_client, quantized_path, s3_bucket, 
                        s3_prefix=f"{remote_path}/"
                    )
                    results["external_storage_used"] = True
                    results["external_storage_path"] = remote_storage_path
                else:
                    logger.error("Failed to set up S3 storage or missing S3 bucket")
            else:
                logger.error(f"Unsupported storage type: {storage_type}")
    
    # Calculate storage savings
    if results["size_original"] and results["size_optimized"]:
        savings = (1 - results["size_optimized"] / results["size_original"]) * 100
        logger.info(f"Storage optimization complete. Saved {savings:.2f}% of original size")
        results["storage_savings_percent"] = savings
    
    return results

def create_training_pipeline_with_storage_optimization(
    model_name: str = "deepseek-ai/deepseek-coder-6.7b-base",
    output_dir: str = "models/deepseek_optimized",
    dataset_name: str = "code_search_net",
    max_samples: int = 5000,
    use_streaming: bool = True,
    quantize_bits: int = 8,
    use_external_storage: bool = False,
    storage_type: str = "gdrive",
    checkpoint_strategy: str = "improvement",
    max_checkpoints: int = 2,
    compress_datasets: bool = True
) -> Dict[str, Any]:
    """
    Create a complete training pipeline with storage optimizations
    
    Args:
        model_name: DeepSeek model name or path
        output_dir: Directory to save optimized model
        dataset_name: Name of dataset to use
        max_samples: Maximum number of samples to use
        use_streaming: Whether to use streaming for datasets
        quantize_bits: Quantization precision (4 or 8)
        use_external_storage: Whether to use external storage
        storage_type: Type of external storage (gdrive or s3)
        checkpoint_strategy: Checkpointing strategy (regular, improvement, hybrid)
        max_checkpoints: Maximum number of checkpoints to keep
        compress_datasets: Whether to compress datasets
        
    Returns:
        Dictionary with pipeline configuration
    """
    # Create configuration
    config = {
        "model": {
            "name": model_name,
            "quantize_bits": quantize_bits,
        },
        "dataset": {
            "name": dataset_name,
            "max_samples": max_samples,
            "use_streaming": use_streaming,
            "compress": compress_datasets,
        },
        "storage": {
            "output_dir": output_dir,
            "use_external_storage": use_external_storage,
            "storage_type": storage_type,
        },
        "checkpointing": {
            "strategy": checkpoint_strategy,
            "max_checkpoints": max_checkpoints,
        },
    }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(output_dir, "storage_optimization_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Created training pipeline with storage optimizations. Config saved to {config_path}")
    
    return config

# Example usage function
def example_usage():
    # 1. Optimize model storage
    optimization_results = optimize_storage_for_model(
        model_name="deepseek-ai/deepseek-coder-6.7b-base",
        output_dir="models/deepseek_optimized",
        quantize_bits=8,
        use_external_storage=True,
        storage_type="gdrive"
    )
    
    # 2. Create optimized training pipeline
    pipeline_config = create_training_pipeline_with_storage_optimization(
        model_name="deepseek-ai/deepseek-coder-6.7b-base",
        output_dir="models/deepseek_optimized",
        dataset_name="code_search_net",
        max_samples=5000,
        use_streaming=True,
        quantize_bits=8,
        use_external_storage=True,
        storage_type="gdrive",
        checkpoint_strategy="improvement",
        max_checkpoints=2,
        compress_datasets=True
    )
    
    return {
        "optimization_results": optimization_results,
        "pipeline_config": pipeline_config
    }

if __name__ == "__main__":
    example_usage() 