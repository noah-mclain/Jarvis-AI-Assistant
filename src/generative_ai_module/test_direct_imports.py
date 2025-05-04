#!/usr/bin/env python3
"""
Minimal test script that directly imports the problematic files.
This script avoids importing from __init__.py to test if the files themselves are correct.
"""

import os
import sys

# Set sys.path to include the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

print("Testing direct imports...")

# Disable logging to avoid extra output
import logging
logging.disable(logging.CRITICAL)

# Test direct import from evaluation_metrics.py
print("\nTesting evaluation_metrics.py...")
try:
    # Load the file directly using importlib to avoid __init__.py
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "evaluation_metrics", 
        os.path.join(project_root, "src/generative_ai_module/evaluation_metrics.py")
    )
    evaluation_metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation_metrics)
    
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
    # Load the file directly using importlib to avoid __init__.py
    spec = importlib.util.spec_from_file_location(
        "train_models", 
        os.path.join(project_root, "src/generative_ai_module/train_models.py")
    )
    train_models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_models)
    
    # Test if calculate_metrics function exists
    print(f"calculate_metrics function exists: {hasattr(train_models, 'calculate_metrics')}")
    print("✅ train_models.py loaded successfully")
    
except Exception as e:
    print(f"❌ Error loading train_models.py: {e}")
    import traceback
    traceback.print_exc()

print("\nDirect imports test completed.") 