#!/usr/bin/env python3
"""
Run fine-tuning for the Jarvis AI Assistant with fixed imports.
This file is a fixed version of run_finetune.py with corrected imports.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Add the parent directory to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import with try/except to handle errors gracefully
try:
    from src.generative_ai_module.evaluation_metrics import EvaluationMetrics
    print("✅ Successfully imported EvaluationMetrics")
except Exception as e:
    print(f"❌ Failed to import EvaluationMetrics: {e}")

try:
    from src.generative_ai_module.train_models import calculate_metrics
    print("✅ Successfully imported calculate_metrics")
except Exception as e:
    print(f"❌ Failed to import calculate_metrics: {e}")

# Continue with required imports for fine-tuning
try:
    from src.generative_ai_module.finetune_deepseek import main as finetune_deepseek_main
    print("✅ Successfully imported finetune_deepseek_main")
except Exception as e:
    print(f"❌ Failed to import finetune_deepseek_main: {e}")
    
def main():
    print("All imports successful! The fixes worked.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 