#!/usr/bin/env python3
"""
Test script to verify that imports are working correctly.
"""

import sys
import os

# Add the parent directory to the path to make the module importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

def test_imports():
    """Test the imports that were previously failing"""
    try:
        # Test importing the EvaluationMetrics class
        from src.generative_ai_module.evaluation_metrics import EvaluationMetrics, save_metrics
        print("✅ Successfully imported EvaluationMetrics and save_metrics")
        
        # Test importing calculate_metrics
        from src.generative_ai_module.train_models import calculate_metrics
        print("✅ Successfully imported calculate_metrics")
        
        # Test importing from __init__.py
        from src.generative_ai_module import (
            EvaluationMetrics, 
            save_metrics, 
            calculate_metrics
        )
        print("✅ Successfully imported all functions via __init__.py")
        
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n🔍 Testing fixed imports...")
    success = test_imports()
    if success:
        print("\n✨ All imports are working correctly!")
        sys.exit(0)
    else:
        print("\n❌ Import issues remain. Please check the error messages above.")
        sys.exit(1) 