#!/usr/bin/env python3
"""
Optimize memory usage for training on A6000 GPU

This script applies aggressive memory optimizations to allow training
large models on GPUs with limited memory.
"""

import os
import sys
import logging
import torch
import gc
import psutil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("memory_optimizer")

def print_memory_stats():
    """Print current memory usage statistics"""
    # GPU memory
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        gpu_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        gpu_max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
        
        logger.info(f"GPU Memory: {gpu_allocated:.2f} GB allocated, {gpu_reserved:.2f} GB reserved")
        logger.info(f"GPU Max Memory Allocated: {gpu_max_allocated:.2f} GB")
        
        # Per-device statistics
        for i in range(torch.cuda.device_count()):
            gpu_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            gpu_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            logger.info(f"GPU {i}: {gpu_allocated:.2f} GB allocated, {gpu_reserved:.2f} GB reserved")
    
    # CPU memory
    process = psutil.Process(os.getpid())
    cpu_memory = process.memory_info().rss / (1024 ** 3)
    total_memory = psutil.virtual_memory().total / (1024 ** 3)
    available_memory = psutil.virtual_memory().available / (1024 ** 3)
    
    logger.info(f"CPU Memory: {cpu_memory:.2f} GB used by process")
    logger.info(f"System Memory: {available_memory:.2f} GB available out of {total_memory:.2f} GB total")

def clear_gpu_memory():
    """Clear GPU memory cache and run garbage collection"""
    if torch.cuda.is_available():
        # Empty CUDA cache
        torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        logger.info("Cleared GPU memory cache and ran garbage collection")
        print_memory_stats()

def optimize_torch_settings():
    """Apply optimal PyTorch settings for memory efficiency"""
    # Set environment variables for optimal memory usage
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,garbage_collection_threshold:0.6"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Better error messages
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid deadlocks
    
    # Disable gradient synchronization for DataParallel
    os.environ["NCCL_P2P_DISABLE"] = "1"
    
    # Set PyTorch to use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Enable TF32 precision on Ampere GPUs (A6000)
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("Enabled TF32 precision for Ampere GPUs")
    
    logger.info("Applied optimal PyTorch settings for memory efficiency")

def patch_transformers_for_memory_efficiency():
    """Patch transformers library for better memory efficiency"""
    try:
        from transformers import modeling_utils
        
        # Store original method
        original_from_pretrained = modeling_utils.PreTrainedModel.from_pretrained
        
        # Create patched method with memory-efficient defaults
        def patched_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
            # Set memory-efficient defaults if not explicitly provided
            if 'low_cpu_mem_usage' not in kwargs:
                kwargs['low_cpu_mem_usage'] = True
            
            if 'torch_dtype' not in kwargs and torch.cuda.is_available():
                # Use BFloat16 on Ampere+ GPUs, Float16 otherwise
                if torch.cuda.get_device_capability()[0] >= 8:
                    kwargs['torch_dtype'] = torch.bfloat16
                else:
                    kwargs['torch_dtype'] = torch.float16
            
            # Call original method with updated kwargs
            return original_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs)
        
        # Apply the patch
        modeling_utils.PreTrainedModel.from_pretrained = classmethod(patched_from_pretrained)
        logger.info("Patched transformers.PreTrainedModel.from_pretrained for memory efficiency")
        return True
    except Exception as e:
        logger.error(f"Failed to patch transformers: {e}")
        return False

def optimize_for_a6000():
    """Apply specific optimizations for A6000 48GB GPU"""
    # Check if we're running on an A6000
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        if "A6000" in device_name:
            logger.info(f"Detected {device_name} GPU - applying specific optimizations")
            
            # Set optimal CUDA settings for A6000
            torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of available memory
            
            # Enable tensor cores
            torch.set_float32_matmul_precision('high')
            
            # Set optimal batch size and sequence length
            return {
                "batch_size": 1,
                "gradient_accumulation_steps": 64,  # Increased from 32 to 64
                "max_length": 256,  # Reduced from 512 to 256
                "use_8bit_quantization": True,
                "use_4bit_quantization": True,
                "use_cpu_offloading": True,
                "use_checkpoint_activation": True,
                "use_flash_attention": False,  # Disable as it's not compatible with all models
                "use_memory_efficient_attention": True
            }
        else:
            logger.info(f"Running on {device_name} - using default optimizations")
    
    # Default settings for other GPUs
    return {
        "batch_size": 1,
        "gradient_accumulation_steps": 32,
        "max_length": 512,
        "use_8bit_quantization": True,
        "use_4bit_quantization": False,
        "use_cpu_offloading": False,
        "use_checkpoint_activation": True,
        "use_flash_attention": False,
        "use_memory_efficient_attention": True
    }

def patch_model_forward(model):
    """Patch model's forward method to handle OOM errors gracefully"""
    if not hasattr(model, 'forward'):
        logger.warning("Model doesn't have a forward method to patch")
        return False
    
    # Store original forward method
    original_forward = model.forward
    
    # Create patched forward method
    def patched_forward(*args, **kwargs):
        try:
            # Try original forward pass
            return original_forward(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Clear memory and try again with smaller input
                logger.warning("OOM in forward pass, attempting recovery...")
                clear_gpu_memory()
                
                # Try to reduce input size if possible
                if 'input_ids' in kwargs and isinstance(kwargs['input_ids'], torch.Tensor):
                    original_shape = kwargs['input_ids'].shape
                    if original_shape[1] > 128:  # If sequence length > 128
                        # Truncate to 128 tokens
                        logger.info(f"Truncating input from {original_shape[1]} to 128 tokens")
                        kwargs['input_ids'] = kwargs['input_ids'][:, :128]
                        
                        # Also truncate attention_mask if present
                        if 'attention_mask' in kwargs and isinstance(kwargs['attention_mask'], torch.Tensor):
                            kwargs['attention_mask'] = kwargs['attention_mask'][:, :128]
                        
                        # Try again with truncated input
                        return original_forward(*args, **kwargs)
                
                # If we can't recover, re-raise the error
                raise
            else:
                # For other errors, just re-raise
                raise
    
    # Apply the patch
    model.forward = patched_forward
    logger.info("Patched model's forward method to handle OOM errors")
    return True

def apply_memory_optimizations():
    """Apply all memory optimizations"""
    logger.info("Applying memory optimizations...")
    
    # Print initial memory stats
    print_memory_stats()
    
    # Clear GPU memory
    clear_gpu_memory()
    
    # Apply PyTorch optimizations
    optimize_torch_settings()
    
    # Patch transformers
    patch_transformers_for_memory_efficiency()
    
    # Get optimal settings for current GPU
    settings = optimize_for_a6000()
    
    # Print final memory stats
    print_memory_stats()
    
    logger.info(f"Memory optimizations complete. Recommended settings: {settings}")
    return settings

if __name__ == "__main__":
    settings = apply_memory_optimizations()
    
    # Print settings as environment variables for shell scripts
    print("\n# Add these to your training script:")
    for key, value in settings.items():
        print(f"export JARVIS_OPT_{key.upper()}={value}")
