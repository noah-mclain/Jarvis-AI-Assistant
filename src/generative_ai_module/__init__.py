"""
Generative AI Module

A module for generating text and code using neural networks.
"""

# Core components
from .text_generator import TextGenerator, CombinedModel
from .code_generator import CodeGenerator
from .dataset_processor import DatasetProcessor
from .prompt_enhancer import PromptEnhancer
from .unified_dataset_handler import UnifiedDatasetHandler, ConversationContext

# Utilities
from .utils import is_zipfile, process_zip

# Preprocessing tools
from .improved_preprocessing import ImprovedCharTokenizer, ImprovedPreprocessor, clean_and_normalize_text
from .code_preprocessing import load_and_preprocess_dataset, save_preprocessing_metrics

# Pipeline
from .unified_generation_pipeline import (
    train_text_generator,
    train_code_generator,
    preprocess_data,
    calculate_metrics,
    TrainingVisualizer,
    main as run_pipeline
)

__version__ = "1.0.0"
