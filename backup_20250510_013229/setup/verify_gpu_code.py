#!/usr/bin/env python3
"""
Verify GPU availability and specifications for code model training.
"""

import sys
import torch

def verify_gpu_for_code_model():
    """
    Verify that a suitable GPU is available for code model training.
    """
    if not torch.cuda.is_available():
        print('❌ ERROR: CUDA is not available. Cannot proceed with GPU training.')
        sys.exit(1)
    else:
        device_name = torch.cuda.get_device_name(0)
        device_capability = torch.cuda.get_device_capability(0)
        free_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f'✓ Using GPU: {device_name} with CUDA capability {device_capability[0]}.{device_capability[1]}')
        print(f'✓ Available GPU memory: {free_memory:.2f} GiB')

        # Verify minimum memory requirements - DeepSeek models need more memory
        if free_memory < 12:
            print(f'❌ ERROR: Not enough GPU memory. Need at least 12 GiB for DeepSeek model, but only {free_memory:.2f} GiB available.')
            sys.exit(1)

        # Clear CUDA cache
        torch.cuda.empty_cache()
        print('✓ CUDA cache cleared')
        
        return True

if __name__ == "__main__":
    verify_gpu_for_code_model()
