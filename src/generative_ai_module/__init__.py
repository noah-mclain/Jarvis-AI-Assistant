"""
Generative AI Module for the Jarvis AI Assistant

This module provides the core functionality for generative AI tasks,
including text generation, code generation, and fine-tuning.
"""

# Import core components
from .jarvis_unified import JarvisAI as UnifiedModel
from .text_generator import TextGenerator
from .code_generator import CodeGenerator
from .evaluation_metrics import calculate_metrics, evaluate_generation
from .dataset_processor import DatasetProcessor
from .prompt_enhancer import PromptEnhancer, analyze_prompt

# Import Google Drive sync functionality
from .utils import (
    sync_to_gdrive, 
    sync_from_gdrive, 
    get_storage_path, 
    setup_logging, 
    sync_logs, 
    save_log_file,
    ensure_directory_exists,
    is_paperspace_environment
)
from .sync_gdrive import sync_all_to_gdrive, sync_all_from_gdrive
try:
    from .manage_storage import sync_everything_to_gdrive, clear_local_storage, show_storage_status
except ImportError:
    pass  # Not critical if missing

# Define public API
__all__ = [
    # Core components
    'UnifiedModel',
    'TextGenerator',
    'CodeGenerator',
    'DatasetProcessor',
    'PromptEnhancer',
    'calculate_metrics',
    'evaluate_generation',
    'analyze_prompt',
    
    # Google Drive sync functionality
    'sync_to_gdrive',
    'sync_from_gdrive',
    'sync_all_to_gdrive',
    'sync_all_from_gdrive',
    'get_storage_path',
    'setup_logging',
    'sync_logs',
    'save_log_file',
    
    # Storage management
    'ensure_directory_exists',
    'is_paperspace_environment',
    'sync_everything_to_gdrive',
    'clear_local_storage',
    'show_storage_status'
]

# Version information
__version__ = '0.3.0'
__author__ = 'Jarvis AI Team'
__description__ = 'Generative AI Module for Jarvis Assistant'
