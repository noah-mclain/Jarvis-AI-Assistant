#!/usr/bin/env python3
"""
Generative AI Module - Jarvis AI Assistant

This module provides functionality for training and using generative AI models.
"""
import os
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Force GPU usage for RTX5000 on Paperspace early at import time
try:
    import torch

    # Force CUDA visibility
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        # Log GPU information
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Initializing with GPU: {gpu_name}")

        # Check if we're on Paperspace and have an RTX5000
        is_paperspace = os.path.exists("/notebooks") or os.path.exists("/storage")
        is_rtx5000 = "RTX5000" in gpu_name or "RTX 5000" in gpu_name

        if is_paperspace and is_rtx5000:
            # Set RTX5000-specific optimizations at module import time
            logger.info("RTX5000 GPU detected - applying system-wide optimizations")

            # Memory optimizations
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
            os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

            # Performance optimizations
            torch.backends.cudnn.benchmark = True

            # Set default tensor type
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
except ImportError:
    logger.warning("PyTorch not available - GPU optimizations skipped")
except Exception as e:
    logger.warning(f"Failed to initialize GPU settings: {str(e)}")

# Import key components with proper error handling
try:
    from .jarvis_unified import JarvisAI
except ImportError as e:
    logger.warning(f"Unable to import JarvisAI: {e}")
    class JarvisAI:
        def __init__(self, *args, **kwargs):
            logger.error("JarvisAI class not properly loaded")
            raise ImportError("JarvisAI could not be loaded")

try:
    from .text_generator import TextGenerator
except ImportError as e:
    logger.warning(f"Unable to import TextGenerator: {e}")
    class TextGenerator:
        def __init__(self, *args, **kwargs):
            logger.error("TextGenerator class not properly loaded")
            raise ImportError("TextGenerator could not be loaded")

try:
    from .code_generator import CodeGenerator
except ImportError as e:
    logger.warning(f"Unable to import CodeGenerator: {e}")
    class CodeGenerator:
        def __init__(self, *args, **kwargs):
            logger.error("CodeGenerator class not properly loaded")
            raise ImportError("CodeGenerator could not be loaded")

try:
    from .utils import (
        get_storage_path,
        sync_to_gdrive,
        sync_from_gdrive,
        setup_logging,
        setup_gpu_for_training,
        force_cuda_device
    )
    from .set_paperspace_env import (
        create_directories,
        setup_paperspace_env,
        get_storage_base_path
    )
except ImportError as e:
    logger.warning(f"Unable to import utils functions: {e}")
    # Provide stub implementations for essential functions
    def get_storage_path(path_type): return f"./data/{path_type}"
    def sync_to_gdrive(path): logger.error("sync_to_gdrive not available")
    def sync_from_gdrive(path): logger.error("sync_from_gdrive not available")
    def setup_logging(): pass
    def setup_gpu_for_training(): return "cpu"
    def force_cuda_device(): return "cpu"
    def create_directories(): pass
    def setup_paperspace_env(): pass
    def get_storage_base_path(): return "./data"

# Optional imports based on availability
try:
    from .evaluation_metrics import EvaluationMetrics
except ImportError as e:
    logger.warning(f"EvaluationMetrics not available: {e}")
    class EvaluationMetrics:
        def __init__(self, *args, **kwargs):
            logger.error("EvaluationMetrics class not properly loaded")

try:
    from .unified_generation_pipeline import UnifiedGenerationPipeline
except ImportError as e:
    logger.warning(f"UnifiedGenerationPipeline not available: {e}")
    class UnifiedGenerationPipeline:
        def __init__(self, *args, **kwargs):
            logger.error("UnifiedGenerationPipeline class not properly loaded")

try:
    from .unified_dataset_handler import UnifiedDatasetHandler
except ImportError as e:
    logger.warning(f"UnifiedDatasetHandler not available: {e}")
    class UnifiedDatasetHandler:
        def __init__(self, *args, **kwargs):
            logger.error("UnifiedDatasetHandler class not properly loaded")

# Expose key classes and functions
__all__ = [
    'JarvisAI',
    'TextGenerator',
    'CodeGenerator',
    'EvaluationMetrics',
    'UnifiedGenerationPipeline',
    'UnifiedDatasetHandler',
    'setup_gpu_for_training',
    'force_cuda_device',
    'get_storage_path',
    'sync_to_gdrive',
    'sync_from_gdrive',
    'setup_logging',
    'create_directories',
    'setup_paperspace_env',
    'get_storage_base_path'
]

# Version information
__version__ = '0.3.0'
__author__ = 'Jarvis AI Team'
__description__ = 'Generative AI Module for Jarvis Assistant'
