#!/usr/bin/env python3
"""
Clear CUDA Cache for Jarvis AI Assistant

This script clears the CUDA cache to free up GPU memory before training.
"""

import os
import sys
import logging
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def clear_cuda_cache():
    """Clear CUDA cache to free up GPU memory"""
    logger.info("Clearing CUDA cache")
    
    # Force garbage collection
    gc.collect()
    
    try:
        import torch
        
        if torch.cuda.is_available():
            # Get initial GPU memory usage
            initial_memory = torch.cuda.memory_allocated()
            initial_memory_gb = initial_memory / (1024 ** 3)
            logger.info(f"Initial GPU memory usage: {initial_memory_gb:.2f} GB")
            
            # Empty CUDA cache
            torch.cuda.empty_cache()
            
            # Force garbage collection again
            gc.collect()
            
            # Get final GPU memory usage
            final_memory = torch.cuda.memory_allocated()
            final_memory_gb = final_memory / (1024 ** 3)
            logger.info(f"Final GPU memory usage: {final_memory_gb:.2f} GB")
            
            # Calculate memory freed
            memory_freed = initial_memory - final_memory
            memory_freed_gb = memory_freed / (1024 ** 3)
            logger.info(f"Memory freed: {memory_freed_gb:.2f} GB")
            
            # Get maximum GPU memory
            try:
                max_memory = torch.cuda.get_device_properties(0).total_memory
                max_memory_gb = max_memory / (1024 ** 3)
                logger.info(f"Maximum GPU memory: {max_memory_gb:.2f} GB")
                
                # Calculate available memory
                available_memory = max_memory - final_memory
                available_memory_gb = available_memory / (1024 ** 3)
                logger.info(f"Available GPU memory: {available_memory_gb:.2f} GB")
            except Exception as e:
                logger.warning(f"Could not get maximum GPU memory: {e}")
        else:
            logger.warning("CUDA not available. Skipping CUDA cache clearing.")
    except ImportError:
        logger.warning("PyTorch not installed. Skipping CUDA cache clearing.")

if __name__ == "__main__":
    clear_cuda_cache()
