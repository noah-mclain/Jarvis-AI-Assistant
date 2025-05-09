#!/usr/bin/env python3
"""
Apple Silicon GPU Setup Script

This script helps install the correct version of PyTorch with 
Metal Performance Shaders (MPS) support for Apple Silicon (M1/M2/M3) Macs.

Run this script to ensure your installation can use the built-in GPU.
"""

import os
import sys
import subprocess
import platform

def run_command(cmd):
    """Run a shell command and return its output"""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    return result.stdout.strip()

def is_apple_silicon():
    """Check if running on Apple Silicon"""
    return (
        platform.system() == "Darwin" and 
        (platform.machine() == "arm64" or "M1" in platform.processor() or "M2" in platform.processor() or "M3" in platform.processor())
    )

def check_torch_installation():
    """Check if the correct version of PyTorch is installed"""
    try:
        import torch
        
        torch_version = torch.__version__
        print(f"PyTorch version: {torch_version}")
        
        # Check MPS availability
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
            is_mps_available = torch.backends.mps.is_available()
            is_mps_built = torch.backends.mps.is_built()
            print(f"MPS available: {is_mps_available}")
            print(f"MPS built: {is_mps_built}")
            
            if is_mps_available and is_mps_built:
                print("✅ PyTorch is correctly set up for Apple Silicon GPU!")
                test_mps_device(torch)
                return True
            else:
                print("❌ PyTorch does not have proper MPS support")
                return False
        else:
            print("❌ PyTorch version does not support MPS (Metal Performance Shaders)")
            return False
    except ImportError:
        print("❌ PyTorch is not installed")
        return False

def test_mps_device(torch):
    """Test if we can create tensors on the MPS device"""
    try:
        print("\nTesting MPS device with a simple tensor operation:")
        device = torch.device("mps")
        x = torch.ones(5, device=device)
        y = x + 1
        print(f"Tensor created on MPS device: {y}")
        print("✅ Successfully performed tensor operations on MPS device!\n")
    except Exception as e:
        print(f"❌ Error testing MPS device: {e}")
        
def install_torch_mps():
    """Install the correct version of PyTorch with MPS support"""
    print("Installing PyTorch with MPS support for Apple Silicon...")
    
    # Command for installing PyTorch with MPS support
    cmd = [
        sys.executable, "-m", "pip", "install", 
        "--upgrade", 
        "torch", 
        "torchvision", 
        "torchaudio"
    ]
    
    output = run_command(cmd)
    if output is not None:
        print("PyTorch installation complete. Checking installation...")
        
        # Check if installation was successful
        return check_torch_installation()
    else:
        return False

def main():
    print(f"Python version: {platform.python_version()}")
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    
    if not is_apple_silicon():
        print("This script is intended for Apple Silicon Macs (M1/M2/M3).")
        print("Your computer does not appear to be using Apple Silicon.")
        return
    
    print("Detected Apple Silicon Mac!")
    
    if check_torch_installation():
        # Already correctly installed
        print("Your PyTorch installation is ready to use the Apple Silicon GPU via MPS.")
        
        # Check for huggingface transformers
        try:
            import transformers
            print(f"transformers version: {transformers.__version__}")
            print("✅ transformers library is installed")
        except ImportError:
            print("⚠️ transformers library is not installed")
            print("Installing transformers...")
            run_command([sys.executable, "-m", "pip", "install", "transformers"])
        
        # Check for accelerate
        try:
            import accelerate
            print(f"accelerate version: {accelerate.__version__}")
            print("✅ accelerate library is installed")
        except ImportError:
            print("⚠️ accelerate library is not installed")
            print("Installing accelerate...")
            run_command([sys.executable, "-m", "pip", "install", "accelerate"])
            
    else:
        print("\nWould you like to install PyTorch with MPS support? (y/n)")
        response = input("> ").strip().lower()
        
        if response == 'y':
            success = install_torch_mps()
            if success:
                print("\n✅ PyTorch with MPS support installed successfully!")
                print("\nYou can now run your fine-tuning scripts with GPU support:")
                print("  python run_finetune.py")
            else:
                print("\n❌ Installation failed. Please try manually:")
                print("  pip install --upgrade torch torchvision torchaudio")
        else:
            print("Installation skipped.")

if __name__ == "__main__":
    main() 