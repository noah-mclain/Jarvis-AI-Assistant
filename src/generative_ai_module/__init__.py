"""
Generative AI Module for the Jarvis AI Assistant

This module provides the core functionality for generative AI tasks,
including text generation, code generation, and fine-tuning.
"""

# Import core components
from .jarvis_unified import UnifiedModel
from .text_generator import TextGenerator
from .code_generator import CodeGenerator
from .evaluation_metrics import calculate_metrics, evaluate_generation
from .dataset_processor import DatasetProcessor
from .prompt_enhancer import PromptEnhancer, analyze_prompt

# Import Google Drive sync functionality
from .utils import sync_to_gdrive, sync_from_gdrive, get_storage_path
from .sync_gdrive import sync_all_to_gdrive, sync_all_from_gdrive

# Define public API
__all__ = [
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
    'get_storage_path'
]

# Version information
__version__ = '0.2.0'
__author__ = 'Jarvis AI Team'
__description__ = 'Generative AI Module for Jarvis Assistant'
