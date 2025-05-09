#!/usr/bin/env python3
"""
Clear CUDA cache to free up GPU memory.
"""

def clear_cuda_cache():
    """
    Clear CUDA cache if PyTorch and CUDA are available.
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("✓ CUDA cache cleared")
        else:
            print("✓ No CUDA device available, skipping cache clearing")
    except ImportError:
        print("⚠️ PyTorch not installed, skipping cache clearing")
        print("⚠️ Make sure PyTorch is installed with: pip install torch")

if __name__ == "__main__":
    clear_cuda_cache()
