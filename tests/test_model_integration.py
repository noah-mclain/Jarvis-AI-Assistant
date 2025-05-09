"""
Consolidated test module for model integration.

This module consolidates tests from:
- test_fix_integration.py
- test_unified_deepseek.py
- test_model_loading.py
- test_refactored_modules.py
"""

import os
import sys
import unittest
import torch
import logging
from pathlib import Path

# Add the parent directory to the path so we can import from the root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Some tests will be skipped.")

try:
    from src.generative_ai_module import (
        ConsolidatedGenerationPipeline,
        ConsolidatedDatasetProcessor,
        DeepSeekTrainer,
        StorageManager
    )
    GENERATIVE_AI_MODULE_AVAILABLE = True
except ImportError:
    GENERATIVE_AI_MODULE_AVAILABLE = False
    logger.warning("Generative AI module not available. Some tests will be skipped.")


class TestModelLoading(unittest.TestCase):
    """Test model loading functionality."""

    @unittest.skipIf(not TRANSFORMERS_AVAILABLE, "Transformers not available")
    def test_tokenizer_loading(self):
        """Test that we can load a tokenizer."""
        try:
            # Use a small model for testing
            model_name = "gpt2"  # Small model for testing
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Test basic tokenization
            text = "Hello, world!"
            tokens = tokenizer(text, return_tensors="pt")
            
            # Check that the tokens have the expected keys
            self.assertIn("input_ids", tokens)
            self.assertIn("attention_mask", tokens)
            
            # Check that the tokens have the expected shape
            self.assertEqual(tokens["input_ids"].dim(), 2)
            self.assertEqual(tokens["attention_mask"].dim(), 2)
            
            # Check that the first dimension is 1 (batch size)
            self.assertEqual(tokens["input_ids"].shape[0], 1)
            self.assertEqual(tokens["attention_mask"].shape[0], 1)
        except Exception as e:
            self.fail(f"Tokenizer loading test raised an exception: {e}")

    @unittest.skipIf(not TRANSFORMERS_AVAILABLE, "Transformers not available")
    def test_model_loading(self):
        """Test that we can load a model."""
        try:
            # Use a small model for testing
            model_name = "gpt2"  # Small model for testing
            
            # Load the model
            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Test a simple forward pass
            text = "Hello, world!"
            inputs = tokenizer(text, return_tensors="pt")
            
            # Generate a short sequence
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_length=20,
                    num_return_sequences=1
                )
            
            # Decode the outputs
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Check that we got some text back
            self.assertIsInstance(generated_text, str)
            self.assertGreater(len(generated_text), len(text))
        except Exception as e:
            self.fail(f"Model loading test raised an exception: {e}")


class TestGenerationPipeline(unittest.TestCase):
    """Test the generation pipeline."""

    @unittest.skipIf(not GENERATIVE_AI_MODULE_AVAILABLE, "Generative AI module not available")
    def test_pipeline_initialization(self):
        """Test that we can initialize the generation pipeline."""
        try:
            # Initialize the pipeline with default parameters
            pipeline = ConsolidatedGenerationPipeline()
            
            # Check that the pipeline has the expected attributes
            self.assertTrue(hasattr(pipeline, "model"))
            self.assertTrue(hasattr(pipeline, "tokenizer"))
        except Exception as e:
            self.fail(f"Pipeline initialization test raised an exception: {e}")

    @unittest.skipIf(not GENERATIVE_AI_MODULE_AVAILABLE, "Generative AI module not available")
    def test_dataset_processor(self):
        """Test that we can initialize the dataset processor."""
        try:
            # Initialize the dataset processor with default parameters
            processor = ConsolidatedDatasetProcessor()
            
            # Check that the processor has the expected attributes
            self.assertTrue(hasattr(processor, "tokenizer"))
        except Exception as e:
            self.fail(f"Dataset processor initialization test raised an exception: {e}")


class TestDeepSeekIntegration(unittest.TestCase):
    """Test DeepSeek model integration."""

    @unittest.skipIf(not GENERATIVE_AI_MODULE_AVAILABLE, "Generative AI module not available")
    def test_deepseek_trainer_initialization(self):
        """Test that we can initialize the DeepSeek trainer."""
        try:
            # Initialize the trainer with default parameters
            trainer = DeepSeekTrainer()
            
            # Check that the trainer has the expected attributes
            self.assertTrue(hasattr(trainer, "model_name"))
            self.assertTrue(hasattr(trainer, "device"))
        except Exception as e:
            self.fail(f"DeepSeek trainer initialization test raised an exception: {e}")


class TestStorageManager(unittest.TestCase):
    """Test the storage manager."""

    @unittest.skipIf(not GENERATIVE_AI_MODULE_AVAILABLE, "Generative AI module not available")
    def test_storage_manager_initialization(self):
        """Test that we can initialize the storage manager."""
        try:
            # Initialize the storage manager with default parameters
            storage_manager = StorageManager()
            
            # Check that the storage manager has the expected attributes
            self.assertTrue(hasattr(storage_manager, "storage_type"))
        except Exception as e:
            self.fail(f"Storage manager initialization test raised an exception: {e}")

    @unittest.skipIf(not GENERATIVE_AI_MODULE_AVAILABLE, "Generative AI module not available")
    def test_storage_status(self):
        """Test that we can get the storage status."""
        try:
            # Get the storage status
            status = StorageManager.get_storage_status()
            
            # Check that the status has the expected keys
            self.assertIn("local", status)
            self.assertIn("gdrive", status)
        except Exception as e:
            self.fail(f"Storage status test raised an exception: {e}")


if __name__ == "__main__":
    unittest.main()
