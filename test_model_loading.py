#!/usr/bin/env python3
"""
Test script for loading the DeepSeek Coder model with optimized memory settings.
This script tests the improved memory management in the CodeGenerator class.
"""

import os
import sys
import torch
import gc
import argparse
from pathlib import Path

# Add the parent directory to the path to make the module importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "./")))

def setup_logging():
    """Set up basic logging configuration"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def print_gpu_info():
    """Print GPU information"""
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print(f"Current GPU memory usage: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
        print(f"Max GPU memory usage: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
    else:
        print("No GPU available")

def main():
    """Main function to test model loading"""
    logger = setup_logging()
    
    parser = argparse.ArgumentParser(description="Test DeepSeek Coder model loading")
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--use_8bit", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--force_gpu", action="store_true", help="Force GPU usage")
    args = parser.parse_args()
    
    # Force GPU usage if requested
    if args.force_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    # Print initial GPU info
    print("\n=== Initial GPU State ===")
    print_gpu_info()
    
    # Force garbage collection and clear CUDA cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Set memory fraction to avoid OOM
        torch.cuda.set_per_process_memory_fraction(0.9)
    
    # Print GPU info after cleanup
    print("\n=== GPU State After Cleanup ===")
    print_gpu_info()
    
    # Import CodeGenerator here to avoid early GPU allocation
    try:
        from src.generative_ai_module.code_generator import CodeGenerator
        
        # Create a code generator instance
        print("\n=== Creating CodeGenerator ===")
        logger.info(f"Creating CodeGenerator with 4-bit: {args.use_4bit}, 8-bit: {args.use_8bit}")
        
        code_gen = CodeGenerator(
            use_deepseek=True,
            load_in_8bit=args.use_8bit,
            load_in_4bit=args.use_4bit,
            force_gpu=args.force_gpu
        )
        
        # Print GPU info after model loading
        print("\n=== GPU State After Model Loading ===")
        print_gpu_info()
        
        # Test a simple code generation
        print("\n=== Testing Code Generation ===")
        prompt = "Write a function to calculate the factorial of a number"
        generated_code = code_gen.generate_code(prompt, length=100)
        
        print("\nGenerated Code:")
        print("-" * 40)
        print(generated_code)
        print("-" * 40)
        
        # Print final GPU info
        print("\n=== Final GPU State ===")
        print_gpu_info()
        
        print("\n✅ Test completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
