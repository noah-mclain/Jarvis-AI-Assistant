#!/usr/bin/env python3
"""
Wrapper script that mocks dependencies and runs import tests
"""

import os
import sys
from pathlib import Path

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

# Set up mocks for all dependencies to prevent initialization code from running
def setup_mocks():
    import types
    
    # Mock the utils module
    mock_utils = types.ModuleType('utils')
    mock_utils.get_storage_path = lambda *args, **kwargs: "/tmp/mock_dir"
    mock_utils.sync_to_gdrive = lambda *args, **kwargs: None
    mock_utils.sync_from_gdrive = lambda *args, **kwargs: None
    mock_utils.ensure_directory_exists = lambda *args, **kwargs: "/tmp/mock_dir"
    mock_utils.is_paperspace_environment = lambda *args, **kwargs: False
    mock_utils.setup_logging = lambda *args, **kwargs: None
    mock_utils.sync_logs = lambda *args, **kwargs: None
    mock_utils.save_log_file = lambda *args, **kwargs: None
    sys.modules['src.generative_ai_module.utils'] = mock_utils
    
    # Mock google_drive_storage module
    mock_gdrive = types.ModuleType('google_drive_storage')
    mock_gdrive_sync = types.SimpleNamespace()
    mock_gdrive_sync.ensure_local_dirs = lambda: None
    mock_gdrive_sync.sync_to_gdrive = lambda *args, **kwargs: None
    mock_gdrive_sync.sync_from_gdrive = lambda *args, **kwargs: None
    mock_gdrive.GoogleDriveSync = mock_gdrive_sync
    sys.modules['src.generative_ai_module.google_drive_storage'] = mock_gdrive
    
    # Mock manage_storage module
    mock_storage = types.ModuleType('manage_storage')
    mock_storage.sync_everything_to_gdrive = lambda *args, **kwargs: None
    mock_storage.clear_local_storage = lambda *args, **kwargs: None
    mock_storage.show_storage_status = lambda *args, **kwargs: None
    sys.modules['src.generative_ai_module.manage_storage'] = mock_storage
    
    # Create directories to avoid errors
    for directory in ["models", "datasets", "metrics", "logs", "checkpoints"]:
        os.makedirs(os.path.join("/tmp", directory), exist_ok=True)

# Set up mocks
setup_mocks()

# Now test importing the modules directly
def run_test():
    print("Testing imports with mocks...")
    
    try:
        # Test importing from evaluation_metrics.py
        from src.generative_ai_module.evaluation_metrics import EvaluationMetrics, save_metrics
        print("✅ Successfully imported from evaluation_metrics.py")
        
        # Test instantiating EvaluationMetrics
        metrics = EvaluationMetrics(metrics_dir="/tmp/metrics")
        print("✅ Successfully instantiated EvaluationMetrics")
        
        # Test importing from train_models.py
        from src.generative_ai_module.train_models import calculate_metrics
        print("✅ Successfully imported calculate_metrics from train_models.py")
        
        print("\n✨ All imports successful!")
        return 0
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(run_test()) 