#!/usr/bin/env python3
"""
Isolated test for importing modules by mocking dependencies completely
"""

import os
import sys
import types

# Add the parent directory to the path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

def create_mocks():
    """Create mock modules for dependencies to avoid initialization issues"""
    
    # Create mock utils module with all required functions
    mock_utils = types.ModuleType('utils')
    mock_utils.get_storage_path = lambda *args, **kwargs: "/tmp/mock_storage"
    mock_utils.sync_to_gdrive = lambda *args, **kwargs: None
    mock_utils.sync_from_gdrive = lambda *args, **kwargs: None
    mock_utils.ensure_directory_exists = lambda *args, **kwargs: "/tmp/mock_dir"
    mock_utils.is_paperspace_environment = lambda *args, **kwargs: False
    mock_utils.setup_logging = lambda *args, **kwargs: None
    mock_utils.sync_logs = lambda *args, **kwargs: None
    mock_utils.save_log_file = lambda *args, **kwargs: None
    mock_utils.is_zipfile = lambda file_path: False
    mock_utils.process_zip = lambda zip_path, extract_to: True
    sys.modules['src.generative_ai_module.utils'] = mock_utils
    sys.modules['utils'] = mock_utils
    
    # Create mock google_drive_storage module
    mock_gdrive = types.ModuleType('google_drive_storage')
    
    # Add required attributes to GoogleDriveSync
    class MockGoogleDriveSync:
        LOCAL_BASE = "/tmp/jarvis"
        LOCAL_MODELS_DIR = os.path.join(LOCAL_BASE, "models")
        LOCAL_DATASETS_DIR = os.path.join(LOCAL_BASE, "datasets")
        LOCAL_METRICS_DIR = os.path.join(LOCAL_BASE, "metrics")
        LOCAL_EVALS_DIR = os.path.join(LOCAL_BASE, "evals")
        LOCAL_LOGS_DIR = os.path.join(LOCAL_BASE, "logs")
        LOCAL_CHECKPOINTS_DIR = os.path.join(LOCAL_BASE, "checkpoints")
        LOCAL_VISUALIZATIONS_DIR = os.path.join(LOCAL_BASE, "visualizations")
        
        @classmethod
        def ensure_local_dirs(cls):
            pass
            
        @classmethod
        def sync_to_gdrive(cls, *args, **kwargs):
            pass
            
        @classmethod
        def sync_from_gdrive(cls, *args, **kwargs):
            pass
    
    mock_gdrive.GoogleDriveSync = MockGoogleDriveSync
    sys.modules['src.generative_ai_module.google_drive_storage'] = mock_gdrive
    
    # Mock manage_storage module
    mock_storage = types.ModuleType('manage_storage')
    mock_storage.sync_everything_to_gdrive = lambda *args, **kwargs: None
    mock_storage.clear_local_storage = lambda *args, **kwargs: None
    mock_storage.show_storage_status = lambda *args, **kwargs: None
    sys.modules['src.generative_ai_module.manage_storage'] = mock_storage
    
    # Create directories
    for directory in ["/tmp/models", "/tmp/datasets", "/tmp/metrics", "/tmp/logs", "/tmp/checkpoints"]:
        os.makedirs(directory, exist_ok=True)
    
    # Return mocked modules for reference
    return {
        'utils': mock_utils,
        'google_drive_storage': mock_gdrive,
        'manage_storage': mock_storage
    }

def import_isolated_modules():
    """Try to import modules after mocking dependencies"""
    print("Attempting to import modules with mocked dependencies...")
    
    # First try to import modules without going through __init__.py
    modules_to_test = [
        "src.generative_ai_module.evaluation_metrics",
        "src.generative_ai_module.train_models",
        # Add other modules to test as needed
    ]
    
    successful_imports = []
    
    for module_name in modules_to_test:
        try:
            # Import module and add to local namespace
            module = __import__(module_name, fromlist=['*'])
            module_short_name = module_name.split('.')[-1]
            globals()[module_short_name] = module
            
            print(f"✅ Successfully imported {module_name}")
            successful_imports.append(module_name)
            
            # Test for specific attributes/classes in the module
            if module_short_name == "evaluation_metrics":
                if hasattr(module, "EvaluationMetrics"):
                    print(f"  - EvaluationMetrics class found in {module_short_name}")
                if hasattr(module, "save_metrics"):
                    print(f"  - save_metrics function found in {module_short_name}")
                    
            elif module_short_name == "train_models":
                if hasattr(module, "calculate_metrics"):
                    print(f"  - calculate_metrics function found in {module_short_name}")
                    
        except Exception as e:
            print(f"❌ Failed to import {module_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Now test importing specific functions/classes directly
    if "src.generative_ai_module.evaluation_metrics" in successful_imports:
        try:
            from src.generative_ai_module.evaluation_metrics import EvaluationMetrics, save_metrics
            print("✅ Successfully imported EvaluationMetrics and save_metrics directly")
            
            # Try to instantiate EvaluationMetrics
            metrics = EvaluationMetrics(metrics_dir="/tmp/metrics")
            print("✅ Successfully instantiated EvaluationMetrics")
        except Exception as e:
            print(f"❌ Failed to import or instantiate from evaluation_metrics: {e}")
    
    if "src.generative_ai_module.train_models" in successful_imports:
        try:
            from src.generative_ai_module.train_models import calculate_metrics
            print("✅ Successfully imported calculate_metrics directly")
        except Exception as e:
            print(f"❌ Failed to import calculate_metrics: {e}")

if __name__ == "__main__":
    print("\n🔧 Setting up mock environment...")
    mocks = create_mocks()
    
    print("\n🧪 Running isolated import tests...")
    import_isolated_modules()
    
    print("\n✅ Test completed!") 