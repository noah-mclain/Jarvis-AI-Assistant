#!/usr/bin/env python3
"""
Fix for TensorBoard callback issues in transformers.

This script provides a custom callback handler that safely handles missing TensorBoard
and a custom trainer that uses this handler.
"""

import os
import sys
import logging
from typing import List, Optional, Union, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def fix_tensorboard_callback():
    """
    Fix TensorBoard callback issues by patching the Trainer class.
    
    This function:
    1. Creates a custom callback handler that safely handles missing TensorBoard
    2. Patches the Trainer class to use this handler
    """
    try:
        import transformers
        from transformers import Trainer, TrainingArguments
        from transformers.trainer_callback import CallbackHandler, TrainerCallback
        
        logger.info("Applying TensorBoard callback fix")
        
        # Create a custom callback handler that safely handles missing TensorBoard
        class TensorboardSafeCallbackHandler(CallbackHandler):
            """Custom callback handler that safely handles missing TensorBoard"""
            
            def __init__(self, callbacks=None, model=None, tokenizer=None, optimizer=None, lr_scheduler=None):
                """Initialize the callback handler with safe TensorBoard handling"""
                # Filter out TensorBoardCallback if tensorboard is not installed
                if callbacks:
                    filtered_callbacks = []
                    for callback in callbacks:
                        if isinstance(callback, type):
                            # This is a callback class, not an instance
                            callback_name = callback.__name__
                            if callback_name == "TensorBoardCallback":
                                try:
                                    # Try to import tensorboard
                                    import tensorboard
                                    # If successful, keep the callback
                                    filtered_callbacks.append(callback)
                                except ImportError:
                                    logger.warning("TensorBoard not installed, skipping TensorBoardCallback")
                            else:
                                # Keep other callback classes
                                filtered_callbacks.append(callback)
                        else:
                            # This is a callback instance
                            callback_name = callback.__class__.__name__
                            if callback_name == "TensorBoardCallback":
                                try:
                                    # Try to import tensorboard
                                    import tensorboard
                                    # If successful, keep the callback
                                    filtered_callbacks.append(callback)
                                except ImportError:
                                    logger.warning("TensorBoard not installed, skipping TensorBoardCallback")
                            else:
                                # Keep other callback instances
                                filtered_callbacks.append(callback)
                    
                    # Replace callbacks with filtered list
                    callbacks = filtered_callbacks
                
                # Call parent constructor with filtered callbacks
                super().__init__(callbacks, model, tokenizer, optimizer, lr_scheduler)
        
        # Store the original Trainer.__init__ method
        original_trainer_init = Trainer.__init__
        
        # Define a patched __init__ method for Trainer
        def patched_trainer_init(self, *args, **kwargs):
            """Patched __init__ method for Trainer that uses TensorboardSafeCallbackHandler"""
            # Call the original __init__ method
            original_trainer_init(self, *args, **kwargs)
            
            # Replace the callback handler with our custom one
            self.callback_handler = TensorboardSafeCallbackHandler(
                callbacks=self.callback_handler.callbacks,
                model=self.model,
                tokenizer=self.tokenizer,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler
            )
            logger.info("Using TensorboardSafeCallbackHandler to handle missing TensorBoard")
        
        # Apply the patch
        Trainer.__init__ = patched_trainer_init
        logger.info("✅ Successfully patched Trainer.__init__ to use TensorboardSafeCallbackHandler")
        
        # Create a custom trainer that uses our callback handler
        class TensorboardSafeTrainer(Trainer):
            """Custom trainer that safely handles missing TensorBoard"""
            
            def __init__(self, *args, **kwargs):
                """Initialize with custom callback handler"""
                # Call the original __init__ method
                super().__init__(*args, **kwargs)
                
                # Replace the callback handler with our custom one
                self.callback_handler = TensorboardSafeCallbackHandler(
                    callbacks=self.callback_handler.callbacks,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    optimizer=self.optimizer,
                    lr_scheduler=self.lr_scheduler
                )
                logger.info("Using TensorboardSafeCallbackHandler in TensorboardSafeTrainer")
        
        # Make the custom trainer available
        transformers.TensorboardSafeTrainer = TensorboardSafeTrainer
        logger.info("✅ Successfully added TensorboardSafeTrainer to transformers")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to fix TensorBoard callback: {e}")
        return False

def install_tensorboard():
    """
    Install TensorBoard if it's not already installed.
    """
    try:
        import tensorboard
        logger.info(f"TensorBoard is already installed (version: {tensorboard.__version__})")
        return True
    except ImportError:
        logger.info("TensorBoard is not installed. Attempting to install...")
        
        try:
            import subprocess
            # Install tensorboard
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorboard", "--no-deps"])
            
            # Verify installation
            import tensorboard
            logger.info(f"✅ Successfully installed TensorBoard (version: {tensorboard.__version__})")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to install TensorBoard: {e}")
            return False

if __name__ == "__main__":
    # Try to install TensorBoard first
    install_success = install_tensorboard()
    
    # Apply the fix regardless of whether TensorBoard was installed
    fix_success = fix_tensorboard_callback()
    
    # Exit with appropriate code
    sys.exit(0 if fix_success else 1)
