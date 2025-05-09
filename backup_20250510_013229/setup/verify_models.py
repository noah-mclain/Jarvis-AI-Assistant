#!/usr/bin/env python3
"""
Verify that models were saved correctly.
"""

import os
import sys
import torch

def verify_models(model_type):
    """
    Verify that models were saved correctly.
    
    Args:
        model_type (str): Type of model to verify ('code', 'text', 'cnn-text', or 'custom-model')
    """
    # Define expected model directories based on model type
    model_dirs = {
        'code': '/notebooks/Jarvis_AI_Assistant/models/deepseek-coder-6.7b-finetuned',
        'text': '/notebooks/Jarvis_AI_Assistant/models/flan-ul2-finetuned',
        'cnn-text': '/notebooks/Jarvis_AI_Assistant/models/cnn-flan-ul2-finetuned',
        'custom-model': '/notebooks/Jarvis_AI_Assistant/models/custom-encoder-decoder'
    }

    if model_type in model_dirs:
        model_dir = model_dirs[model_type]
        if os.path.exists(model_dir):
            print(f'✓ Model directory {model_dir} exists')
            required_files = ['model.pt'] if model_type != 'code' else ['config.json', 'adapter_config.json']
            missing = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
            if missing:
                print(f'❌ WARNING: Missing files: {missing}')
            else:
                print(f'✓ All required files present')
        else:
            print(f'❌ WARNING: Directory {model_dir} missing')
    else:
        print(f'❌ Unknown model type: {model_type}')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print('✓ CUDA cache cleared')

if __name__ == "__main__":
    # Get model type from command line arguments
    if len(sys.argv) >= 2:
        model_type = sys.argv[1]
    else:
        print("Please specify a model type: code, text, cnn-text, or custom-model")
        sys.exit(1)
    
    verify_models(model_type)
