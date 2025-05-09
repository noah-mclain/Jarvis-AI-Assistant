#!/usr/bin/env python3
"""
Minimal test script that directly imports the problematic files.
This script properly handles relative imports by setting up the package structure.
"""

import os
import sys
import importlib

# Set sys.path to include the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

print("Testing direct imports...")

# Disable logging to avoid extra output
import logging
logging.disable(logging.CRITICAL)

# First create a mock for the utils module to avoid GoogleDrive issues
class MockUtils:
    @staticmethod
    def get_storage_path(*args, **kwargs):
        return "/tmp/mock_storage"
    
    @staticmethod
    def sync_to_gdrive(*args, **kwargs):
        pass
    
    @staticmethod
    def sync_from_gdrive(*args, **kwargs):
        pass
    
    @staticmethod
    def ensure_directory_exists(*args, **kwargs):
        return "/tmp/mock_dir"
    
    @staticmethod
    def is_paperspace_environment(*args, **kwargs):
        return False
    
    @staticmethod
    def setup_logging(*args, **kwargs):
        pass
    
    @staticmethod
    def sync_logs(*args, **kwargs):
        pass

# Create the module
sys.modules['src.generative_ai_module.utils'] = MockUtils()

# Test direct import from evaluation_metrics.py
print("\nTesting evaluation_metrics.py...")
try:
    # Import directly using the standard import mechanism
    import src.generative_ai_module.evaluation_metrics as evaluation_metrics
    
    # Test if EvaluationMetrics class exists
    print(f"EvaluationMetrics class exists: {hasattr(evaluation_metrics, 'EvaluationMetrics')}")
    print(f"save_metrics function exists: {hasattr(evaluation_metrics, 'save_metrics')}")
    print("✅ evaluation_metrics.py loaded successfully")
    
except Exception as e:
    print(f"❌ Error loading evaluation_metrics.py: {e}")
    import traceback
    traceback.print_exc()

# Test direct import from train_models.py
print("\nTesting train_models.py...")
try:
    # Import directly using the standard import mechanism
    import src.generative_ai_module.train_models as train_models
    
    # Test if calculate_metrics function exists
    print(f"calculate_metrics function exists: {hasattr(train_models, 'calculate_metrics')}")
    print("✅ train_models.py loaded successfully")
    
except Exception as e:
    print(f"❌ Error loading train_models.py: {e}")
    import traceback
    traceback.print_exc()

print("\nDirect imports test completed.") 