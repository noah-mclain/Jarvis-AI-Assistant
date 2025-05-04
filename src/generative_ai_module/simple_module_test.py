#!/usr/bin/env python3
"""
Simple test script for the Jarvis AI Assistant.
This tests that the EvaluationMetrics class can be imported and instantiated.
"""

import os
import sys

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    print(f"Added {parent_dir} to sys.path")

# Create a mock class for utils.py functions to prevent import errors
class MockUtilsModule:
    @staticmethod
    def get_storage_path(*args, **kwargs):
        return "/tmp/mock_storage"
    
    @staticmethod
    def sync_to_gdrive(*args, **kwargs):
        print("Mock: sync_to_gdrive called")
    
    @staticmethod
    def sync_from_gdrive(*args, **kwargs):
        print("Mock: sync_from_gdrive called")
    
    @staticmethod
    def ensure_directory_exists(*args, **kwargs):
        print("Mock: ensure_directory_exists called")
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

# Replace the utils module with our mock
sys.modules['src.generative_ai_module.utils'] = MockUtilsModule

# Try to import the EvaluationMetrics class
try:
    from src.generative_ai_module.evaluation_metrics import EvaluationMetrics
    print("✅ Successfully imported EvaluationMetrics class")
    
    # Try to instantiate the class
    metrics = EvaluationMetrics(metrics_dir="/tmp/test_metrics")
    print("✅ Successfully instantiated EvaluationMetrics")
    
    # Import from train_models.py
    from src.generative_ai_module.train_models import calculate_metrics
    print("✅ Successfully imported calculate_metrics function")
    
    print("\nAll imports successful!")
    sys.exit(0)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1) 