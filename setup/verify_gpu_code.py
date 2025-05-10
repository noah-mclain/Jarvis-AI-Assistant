#!/usr/bin/env python3
"""
Verify GPU availability for code model training.
"""

import sys

def verify_gpu():
    """
    Verify GPU availability for code model training.
    
    Returns:
        bool: True if GPU is available and has sufficient memory
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("❌ CUDA is not available. GPU training is not possible.")
            return False
        
        # Get GPU information
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        
        print(f"Found {device_count} GPU(s)")
        print(f"Using GPU: {device_name}")
        
        # Check memory
        memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
        memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # Convert to GB
        
        print(f"GPU memory allocated: {memory_allocated:.2f} GB")
        print(f"GPU memory reserved: {memory_reserved:.2f} GB")
        
        # Try a simple tensor operation on GPU
        try:
            x = torch.rand(1000, 1000).cuda()
            y = torch.rand(1000, 1000).cuda()
            z = torch.matmul(x, y)
            del x, y, z
            torch.cuda.empty_cache()
            print("✅ GPU tensor operations successful")
        except Exception as e:
            print(f"❌ GPU tensor operations failed: {e}")
            return False
        
        print("✅ GPU verification successful")
        return True
    
    except ImportError:
        print("❌ PyTorch is not installed. Cannot verify GPU.")
        return False
    except Exception as e:
        print(f"❌ Error verifying GPU: {e}")
        return False

if __name__ == "__main__":
    success = verify_gpu()
    sys.exit(0 if success else 1)
