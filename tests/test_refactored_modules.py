#!/usr/bin/env python3
"""
Test script to verify the refactored modules work correctly
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_import_utilities():
    """Test the import_utilities module"""
    logger.info("Testing import_utilities module...")
    try:
        from src.generative_ai_module import import_utilities
        
        # Test path fixing
        project_root = import_utilities.fix_path()
        logger.info(f"Project root: {project_root}")
        
        # Test check_imports
        import_results = import_utilities.check_imports(["os", "sys", "datetime"])
        logger.info(f"Import check results: {import_results}")
        
        logger.info("✅ import_utilities module works correctly")
        return True
    except Exception as e:
        logger.error(f"❌ Error testing import_utilities: {e}")
        return False

def test_nlp_utils():
    """Test the nlp_utils module"""
    logger.info("Testing nlp_utils module...")
    try:
        from src.generative_ai_module import nlp_utils
        
        # Test tokenization
        test_text = "This is a test sentence for tokenization."
        tokens = nlp_utils.tokenize_text(test_text)
        logger.info(f"Tokenized text: {tokens[:5]}...")
        
        # Test spaCy availability
        is_available, version = nlp_utils.is_spacy_available()
        logger.info(f"spaCy available: {is_available}, version: {version}")
        
        # Test minimal tokenizer
        minimal_tokens = nlp_utils.minimal_tokenize(test_text)
        logger.info(f"Minimal tokenizer results: {minimal_tokens[:5]}...")
        
        logger.info("✅ nlp_utils module works correctly")
        return True
    except Exception as e:
        logger.error(f"❌ Error testing nlp_utils: {e}")
        return False

def test_evaluation_metrics():
    """Test the evaluation_metrics module without GPU requirements"""
    logger.info("Testing evaluation_metrics module...")
    try:
        from src.generative_ai_module import evaluation_metrics
        
        # Test basic functionality of EvaluationMetrics class
        metrics = evaluation_metrics.EvaluationMetrics(use_gpu=False)
        
        # Test evaluate_generation with simple input
        result = metrics.evaluate_generation(
            prompt="Write a test function",
            generated_text="def test_function():\n    return True",
            reference_text="def sample_test():\n    return True",
            dataset_name="test_dataset",
            save_results=False
        )
        
        logger.info(f"Evaluation result keys: {list(result.keys())}")
        
        # Test save_metrics function
        tmp_dir = "tmp_metrics"
        os.makedirs(tmp_dir, exist_ok=True)
        filepath = evaluation_metrics.save_metrics(
            metrics={"test_metric": 0.95},
            model_name="test_model",
            dataset_name="test_dataset"
        )
        logger.info(f"Saved metrics to {filepath}")
        
        logger.info("✅ evaluation_metrics module works correctly")
        return True
    except Exception as e:
        logger.error(f"❌ Error testing evaluation_metrics: {e}")
        return False

def test_deepseek_handler_compatibility():
    """Test just the basic imports of deepseek_handler without GPU"""
    logger.info("Testing deepseek_handler module compatibility...")
    try:
        # Just test import without instantiating GPU-dependent classes
        from src.generative_ai_module import deepseek_handler
        
        # Test helper functions which don't require GPU
        is_apple = deepseek_handler.is_apple_silicon()
        logger.info(f"Running on Apple Silicon: {is_apple}")
        
        # Test if create_mini_dataset function is available
        assert hasattr(deepseek_handler, "create_mini_dataset")
        logger.info("create_mini_dataset function is available")
        
        logger.info("✅ deepseek_handler module is compatible (non-GPU functions)")
        return True
    except Exception as e:
        logger.error(f"❌ Error testing deepseek_handler compatibility: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("Testing refactored modules...")
    
    results = {
        "import_utilities": test_import_utilities(),
        "nlp_utils": test_nlp_utils(),
        "evaluation_metrics": test_evaluation_metrics(),
        "deepseek_handler": test_deepseek_handler_compatibility()
    }
    
    # Print summary
    logger.info("\n--- Test Results Summary ---")
    all_passed = True
    for module, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{module}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 All modules passed basic tests!")
        return 0
    else:
        logger.error("\n❌ Some tests failed. See above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 