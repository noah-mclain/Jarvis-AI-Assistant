#!/usr/bin/env python3
"""
Setup Environment for Jarvis AI Assistant

This script sets up the environment for Jarvis AI Assistant by:
1. Creating necessary directories
2. Setting environment variables
3. Checking for GPU availability
4. Setting up logging
"""

import os
import sys
import logging
import torch
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Set up the environment for Jarvis AI Assistant"""
    logger.info("Setting up environment for Jarvis AI Assistant")
    
    # Create necessary directories
    create_directories()
    
    # Set environment variables
    set_environment_variables()
    
    # Check for GPU availability
    check_gpu()
    
    logger.info("Environment setup complete")

def create_directories():
    """Create necessary directories for Jarvis AI Assistant"""
    logger.info("Creating necessary directories")
    
    # Define directories to create
    directories = [
        "models",
        "datasets",
        "checkpoints",
        "logs",
        "metrics",
        "preprocessed_data",
        "visualizations"
    ]
    
    # Create directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    # Create subdirectories for different model types
    model_types = ["code", "text", "cnn-text", "custom-model"]
    for model_type in model_types:
        model_dir = os.path.join("models", model_type)
        os.makedirs(model_dir, exist_ok=True)
        logger.info(f"Created model directory: {model_dir}")

def set_environment_variables():
    """Set environment variables for Jarvis AI Assistant"""
    logger.info("Setting environment variables")
    
    # Set environment variables for optimal memory usage and GPU utilization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,garbage_collection_threshold:0.8"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Force certain operations on CPU to save GPU memory
    os.environ["FORCE_CPU_ONLY_FOR_INITIAL_LOAD"] = "1"
    os.environ["FORCE_CPU_ONLY_FOR_TOKENIZATION"] = "1"
    os.environ["FORCE_CPU_ONLY_FOR_DATASET_PROCESSING"] = "1"
    os.environ["TOKENIZERS_FORCE_CPU"] = "1"
    os.environ["HF_DATASETS_CPU_ONLY"] = "1"
    os.environ["JARVIS_FORCE_CPU_TOKENIZER"] = "1"
    
    # Set PyTorch to use deterministic algorithms for reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = "42"
    
    # Set Paperspace environment variable
    os.environ["PAPERSPACE"] = "true"
    
    # Log environment variables
    logger.info("Environment variables set:")
    logger.info(f"PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
    logger.info(f"TOKENIZERS_PARALLELISM: {os.environ.get('TOKENIZERS_PARALLELISM')}")
    logger.info(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")
    logger.info(f"CUDA_LAUNCH_BLOCKING: {os.environ.get('CUDA_LAUNCH_BLOCKING')}")
    logger.info(f"TRANSFORMERS_OFFLINE: {os.environ.get('TRANSFORMERS_OFFLINE')}")
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

def check_gpu():
    """Check for GPU availability"""
    logger.info("Checking for GPU availability")
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"GPU available: {device_count} device(s)")
        
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            device_capability = torch.cuda.get_device_capability(i)
            logger.info(f"Device {i}: {device_name} (CUDA Capability {device_capability[0]}.{device_capability[1]})")
        
        # Get GPU memory
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            logger.info(f"GPU memory: {gpu_memory / (1024 ** 3):.2f} GB")
        except Exception as e:
            logger.warning(f"Could not get GPU memory: {e}")
    else:
        logger.warning("No GPU available. Using CPU.")
        
        # Check if MPS (Metal Performance Shaders) is available for Apple Silicon
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Apple MPS (Metal Performance Shaders) is available.")
        else:
            logger.warning("Neither CUDA nor MPS is available. Using CPU only.")

if __name__ == "__main__":
    setup_environment()
