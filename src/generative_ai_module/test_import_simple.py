#!/usr/bin/env python3
"""
Simple test script to verify that the specific imports that were failing now work.
"""

import sys
import os

# Add the parent directory to the path to make the module importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

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