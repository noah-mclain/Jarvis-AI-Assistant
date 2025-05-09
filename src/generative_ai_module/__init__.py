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

# Import consolidated modules
try:
    from .consolidated_generation_pipeline import ConsolidatedGenerationPipeline
except ImportError as e:
    logger.warning(f"Unable to import ConsolidatedGenerationPipeline: {e}")
    class ConsolidatedGenerationPipeline:
        def __init__(self, *args, **kwargs):
            logger.error("ConsolidatedGenerationPipeline class not properly loaded")
            raise ImportError("ConsolidatedGenerationPipeline could not be loaded")

try:
    from .consolidated_dataset_processor import ConsolidatedDatasetProcessor, ImprovedTokenizer, ConversationContext
except ImportError as e:
    logger.warning(f"Unable to import ConsolidatedDatasetProcessor: {e}")
    class ConsolidatedDatasetProcessor:
        def __init__(self, *args, **kwargs):
            logger.error("ConsolidatedDatasetProcessor class not properly loaded")
            raise ImportError("ConsolidatedDatasetProcessor could not be loaded")
    class ImprovedTokenizer:
        def __init__(self, *args, **kwargs):
            logger.error("ImprovedTokenizer class not properly loaded")
            raise ImportError("ImprovedTokenizer could not be loaded")
    class ConversationContext:
        def __init__(self, *args, **kwargs):
            logger.error("ConversationContext class not properly loaded")
            raise ImportError("ConversationContext could not be loaded")

# For backward compatibility
try:
    from .consolidated_generation_pipeline import ConsolidatedGenerationPipeline as TextGenerator
    from .consolidated_generation_pipeline import ConsolidatedGenerationPipeline as CodeGenerator
    from .consolidated_generation_pipeline import ConsolidatedGenerationPipeline as UnifiedGenerationPipeline
    from .consolidated_dataset_processor import ConsolidatedDatasetProcessor as UnifiedDatasetHandler
except ImportError as e:
    logger.warning(f"Unable to set up backward compatibility classes: {e}")
    class TextGenerator:
        def __init__(self, *args, **kwargs):
            logger.error("TextGenerator class not properly loaded")
            raise ImportError("TextGenerator could not be loaded")
    class CodeGenerator:
        def __init__(self, *args, **kwargs):
            logger.error("CodeGenerator class not properly loaded")
            raise ImportError("CodeGenerator could not be loaded")
    class UnifiedGenerationPipeline:
        def __init__(self, *args, **kwargs):
            logger.error("UnifiedGenerationPipeline class not properly loaded")
    class UnifiedDatasetHandler:
        def __init__(self, *args, **kwargs):
            logger.error("UnifiedDatasetHandler class not properly loaded")

try:
    from .utils import (
        get_storage_path,
        setup_logging,
        setup_gpu_for_training,
        force_cuda_device
    )
    from .storage_manager import (
        sync_to_gdrive,
        sync_from_gdrive,
        StorageManager,
        save_model,
        load_model,
        save_dataset,
        optimize_model_storage,
        get_storage_status
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
    def sync_to_gdrive(path=None): logger.error("sync_to_gdrive not available")
    def sync_from_gdrive(path=None): logger.error("sync_from_gdrive not available")
    def save_model(*args, **kwargs): logger.error("save_model not available")
    def load_model(*args, **kwargs): logger.error("load_model not available")
    def save_dataset(*args, **kwargs): logger.error("save_dataset not available")
    def optimize_model_storage(*args, **kwargs): logger.error("optimize_model_storage not available")
    def get_storage_status(*args, **kwargs): logger.error("get_storage_status not available")
    def setup_logging(): pass
    def setup_gpu_for_training(): return "cpu"
    def force_cuda_device(): return "cpu"
    def create_directories(): pass
    def setup_paperspace_env(): pass
    def get_storage_base_path(): return "./data"
    class StorageManager:
        @classmethod
        def sync_to_gdrive(cls, folder_type=None):
            logger.error("StorageManager.sync_to_gdrive not available")

        @classmethod
        def sync_from_gdrive(cls, folder_type=None):
            logger.error("StorageManager.sync_from_gdrive not available")

        @classmethod
        def save_model(cls, *args, **kwargs):
            logger.error("StorageManager.save_model not available")

        @classmethod
        def load_model(cls, *args, **kwargs):
            logger.error("StorageManager.load_model not available")

        @classmethod
        def save_dataset(cls, *args, **kwargs):
            logger.error("StorageManager.save_dataset not available")

        @classmethod
        def optimize_model_storage(cls, *args, **kwargs):
            logger.error("StorageManager.optimize_model_storage not available")

        @classmethod
        def get_storage_status(cls):
            logger.error("StorageManager.get_storage_status not available")
            return {}

# Optional imports based on availability
try:
    from .evaluation import (
        EvaluationMetrics,
        evaluate_generation,
        calculate_bleu,
        calculate_rouge,
        evaluate_code_generation,
        compare_models
    )
except ImportError as e:
    logger.warning(f"EvaluationMetrics not available: {e}")
    class EvaluationMetrics:
        def __init__(self, *args, **kwargs):
            logger.error("EvaluationMetrics class not properly loaded")
    def evaluate_generation(*args, **kwargs):
        logger.error("evaluate_generation not available")
    def calculate_bleu(*args, **kwargs):
        logger.error("calculate_bleu not available")
    def calculate_rouge(*args, **kwargs):
        logger.error("calculate_rouge not available")
    def evaluate_code_generation(*args, **kwargs):
        logger.error("evaluate_code_generation not available")
    def compare_models(*args, **kwargs):
        logger.error("compare_models not available")

# Import DeepSeek training functionality
try:
    from .deepseek_training import DeepSeekTrainer, fine_tune_deepseek, load_deepseek_model, generate_with_deepseek
except ImportError as e:
    logger.warning(f"DeepSeekTrainer not available: {e}")
    class DeepSeekTrainer:
        def __init__(self, *args, **kwargs):
            logger.error("DeepSeekTrainer class not properly loaded")
    def fine_tune_deepseek(*args, **kwargs):
        logger.error("fine_tune_deepseek not available")
    def load_deepseek_model(*args, **kwargs):
        logger.error("load_deepseek_model not available")
    def generate_with_deepseek(*args, **kwargs):
        logger.error("generate_with_deepseek not available")

# Expose key classes and functions
__all__ = [
    'JarvisAI',
    # New consolidated modules
    'ConsolidatedGenerationPipeline',
    'ConsolidatedDatasetProcessor',
    'ImprovedTokenizer',
    'ConversationContext',
    # Backward compatibility
    'TextGenerator',
    'CodeGenerator',
    'EvaluationMetrics',
    'UnifiedGenerationPipeline',
    'UnifiedDatasetHandler',
    # Evaluation functions
    'evaluate_generation',
    'calculate_bleu',
    'calculate_rouge',
    'evaluate_code_generation',
    'compare_models',
    # DeepSeek training functions
    'DeepSeekTrainer',
    'fine_tune_deepseek',
    'load_deepseek_model',
    'generate_with_deepseek',
    # Storage functions
    'StorageManager',
    'sync_to_gdrive',
    'sync_from_gdrive',
    'save_model',
    'load_model',
    'save_dataset',
    'optimize_model_storage',
    'get_storage_status',
    # Utility functions
    'setup_gpu_for_training',
    'force_cuda_device',
    'get_storage_path',
    'setup_logging',
    'create_directories',
    'setup_paperspace_env',
    'get_storage_base_path'
]

# Version information
__version__ = '0.4.0'  # Updated for consolidated modules
__author__ = 'Jarvis AI Team'
__description__ = 'Generative AI Module for Jarvis Assistant'
