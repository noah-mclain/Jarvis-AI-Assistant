"""
Storage Optimization Utilities for DeepSeek Fine-tuning

This module provides utilities to optimize storage usage when fine-tuning DeepSeek models,
especially in environments with limited storage like Gradient Pro's 15GB persistent storage.

Features:
- Model quantization (4-bit and 8-bit)
- External cloud storage integration (Google Drive, S3)
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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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