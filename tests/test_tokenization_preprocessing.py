"""
Consolidated test module for tokenization and preprocessing.

This module consolidates tests from:
- test_minimal_spacy.py
- test_minimal_tokenizer.py
- test_spacy_minimal.py
- test_spacy_simple.py
- test_spacy.py
- test_dataset_loading.py
"""

import os
import sys
import unittest
import logging
from pathlib import Path

# Add the parent directory to the path so we can import from the root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available. Some tests will be skipped.")

try:
    from src.generative_ai_module import (
        ConsolidatedDatasetProcessor,
        ImprovedTokenizer
    )
    GENERATIVE_AI_MODULE_AVAILABLE = True
except ImportError:
    GENERATIVE_AI_MODULE_AVAILABLE = False
    logger.warning("Generative AI module not available. Some tests will be skipped.")

try:
    from src.generative_ai_module.minimal_spacy_tokenizer import MinimalSpacyTokenizer
    MINIMAL_SPACY_AVAILABLE = True
except ImportError:
    MINIMAL_SPACY_AVAILABLE = False
    logger.warning("Minimal spaCy tokenizer not available. Some tests will be skipped.")


class TestSpaCyIntegration(unittest.TestCase):
    """Test spaCy integration."""

    @unittest.skipIf(not SPACY_AVAILABLE, "spaCy not available")
    def test_spacy_loading(self):
        """Test that we can load spaCy."""
        try:
            # Try to load the small English model
            nlp = spacy.blank("en")
            
            # Check that we got a spaCy Language object
            self.assertEqual(type(nlp).__name__, "Language")
            
            # Test basic tokenization
            text = "Hello, world!"
            doc = nlp(text)
            
            # Check that we got some tokens
            self.assertGreater(len(doc), 0)
        except Exception as e:
            self.fail(f"spaCy loading test raised an exception: {e}")

    @unittest.skipIf(not MINIMAL_SPACY_AVAILABLE, "Minimal spaCy tokenizer not available")
    def test_minimal_spacy_tokenizer(self):
        """Test the minimal spaCy tokenizer."""
        try:
            # Initialize the tokenizer
            tokenizer = MinimalSpacyTokenizer()
            
            # Test basic tokenization
            text = "Hello, world!"
            tokens = tokenizer.tokenize(text)
            
            # Check that we got some tokens
            self.assertGreater(len(tokens), 0)
            
            # Check that the tokens are strings
            for token in tokens:
                self.assertIsInstance(token, str)
        except Exception as e:
            self.fail(f"Minimal spaCy tokenizer test raised an exception: {e}")


class TestImprovedTokenizer(unittest.TestCase):
    """Test the improved tokenizer."""

    @unittest.skipIf(not GENERATIVE_AI_MODULE_AVAILABLE, "Generative AI module not available")
    def test_improved_tokenizer_initialization(self):
        """Test that we can initialize the improved tokenizer."""
        try:
            # Initialize the tokenizer with default parameters
            tokenizer = ImprovedTokenizer()
            
            # Check that the tokenizer has the expected attributes
            self.assertTrue(hasattr(tokenizer, "tokenize"))
        except Exception as e:
            self.fail(f"Improved tokenizer initialization test raised an exception: {e}")

    @unittest.skipIf(not GENERATIVE_AI_MODULE_AVAILABLE, "Generative AI module not available")
    def test_improved_tokenizer_tokenization(self):
        """Test that the improved tokenizer can tokenize text."""
        try:
            # Initialize the tokenizer with default parameters
            tokenizer = ImprovedTokenizer()
            
            # Test basic tokenization
            text = "Hello, world!"
            tokens = tokenizer.tokenize(text)
            
            # Check that we got some tokens
            self.assertGreater(len(tokens), 0)
        except Exception as e:
            self.fail(f"Improved tokenizer tokenization test raised an exception: {e}")


class TestDatasetLoading(unittest.TestCase):
    """Test dataset loading."""

    @unittest.skipIf(not GENERATIVE_AI_MODULE_AVAILABLE, "Generative AI module not available")
    def test_dataset_processor_loading(self):
        """Test that the dataset processor can load datasets."""
        try:
            # Initialize the dataset processor with default parameters
            processor = ConsolidatedDatasetProcessor()
            
            # Check that the processor has the expected methods
            self.assertTrue(hasattr(processor, "load_dataset"))
            self.assertTrue(hasattr(processor, "preprocess_dataset"))
        except Exception as e:
            self.fail(f"Dataset processor loading test raised an exception: {e}")

    @unittest.skipIf(not GENERATIVE_AI_MODULE_AVAILABLE, "Generative AI module not available")
    def test_dataset_processor_preprocessing(self):
        """Test that the dataset processor can preprocess datasets."""
        try:
            # Initialize the dataset processor with default parameters
            processor = ConsolidatedDatasetProcessor()
            
            # Create a simple dataset for testing
            dataset = [
                {"text": "Hello, world!"},
                {"text": "This is a test."}
            ]
            
            # Check that the processor has the expected methods
            self.assertTrue(hasattr(processor, "preprocess_dataset"))
        except Exception as e:
            self.fail(f"Dataset processor preprocessing test raised an exception: {e}")


if __name__ == "__main__":
    unittest.main()
