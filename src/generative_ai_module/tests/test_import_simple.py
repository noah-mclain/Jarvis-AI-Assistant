#!/usr/bin/env python3
"""
Simple test script to verify that the specific imports that were failing now work.
This script handles relative imports properly.
"""

import sys
import os

# Add the parent directory to the path to make the module importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, project_root)

# Create a mock for the utils module to avoid GoogleDrive issues
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

# Disable automatic initialization of Google Drive directories
import types
mock_gdrive_sync = types.SimpleNamespace()
mock_gdrive_sync.ensure_local_dirs = lambda: None
sys.modules['src.generative_ai_module.google_drive_storage'] = types.ModuleType('google_drive_storage')
sys.modules['src.generative_ai_module.google_drive_storage'].GoogleDriveSync = mock_gdrive_sync

def test_direct_imports():
    """Test importing directly from the specific modules"""
    try:
        # Import directly from evaluation_metrics.py
        print("Testing import from evaluation_metrics.py...")
        import src.generative_ai_module.evaluation_metrics
        from src.generative_ai_module.evaluation_metrics import EvaluationMetrics, save_metrics
        print("✅ Successfully imported from evaluation_metrics.py")
        
        # Import directly from train_models.py
        print("\nTesting import from train_models.py...")
        import src.generative_ai_module.train_models
        from src.generative_ai_module.train_models import calculate_metrics
        print("✅ Successfully imported from train_models.py")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n🔍 Testing direct imports...")
    success = test_direct_imports()
    
    if success:
        print("\n✨ Direct imports are working correctly!")
        sys.exit(0)
    else:
        print("\n❌ Import issues remain. Please check the error messages above.")
        sys.exit(1) 