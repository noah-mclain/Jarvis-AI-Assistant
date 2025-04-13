"""
Generative AI Module

A module for generating text and code using neural networks.
"""

# Core components
from .text_generator import TextGenerator, CombinedModel
from .code_generator import CodeGenerator
from .dataset_processor import DatasetProcessor
from .prompt_enhancer import PromptEnhancer

# Utilities
from .utils import is_zipfile, process_zip

# Preprocessing tools
from .improved_preprocessing import ImprovedCharTokenizer, ImprovedPreprocessor, clean_and_normalize_text

__version__ = "1.0.0"
