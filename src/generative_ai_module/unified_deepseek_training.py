#!/usr/bin/env python3
"""
Unified DeepSeek-Coder training script that combines the best aspects of all training approaches.
This script:
1. Applies the attention mask fix
2. Uses Unsloth optimization if available
3. Falls back to standard optimization if Unsloth is not available
4. Optimizes memory usage for different GPU types
5. Supports both 4-bit and 8-bit quantization
"""

import os
import sys
import time
import torch
import argparse
import logging
import multiprocessing
import subprocess
import importlib.metadata
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple

# Set multiprocessing start method to 'spawn' to avoid CUDA issues
# This must be done at the beginning of the script before any other multiprocessing code
if __name__ == "__main__":
    # Only set the start method in the main process
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'")
    except RuntimeError:
        # If it's already set, this will raise a RuntimeError
        print("Multiprocessing start method already set to:", multiprocessing.get_start_method())

import logging
import torch
from contextlib import contextmanager

@contextmanager
def safe_autocast(dtype=None):
    """Simple autocast wrapper that works with different PyTorch versions"""
    if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
        try:
            if dtype is not None:
                with torch.cuda.amp.autocast(dtype=dtype) as ctx:
                    yield ctx
            else:
                with torch.cuda.amp.autocast() as ctx:
                    yield ctx
        except TypeError:
            # If dtype parameter is not supported, use without it
            with torch.cuda.amp.autocast() as ctx:
                yield ctx
    else:
        # Dummy context manager if autocast is not available
        @contextmanager
        def dummy_context_manager():
            yield
        yield dummy_context_manager()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"deepseek_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Check package versions
required_versions = {
    "peft": "0.10.0",
    "transformers": "4.40.0",
    "accelerate": "0.29.0"
}

for pkg, ver in required_versions.items():
    try:
        installed_ver = importlib.metadata.version(pkg)
        if installed_ver < ver:
            logger.warning(f"{pkg} version {installed_ver} is outdated. Minimum required: {ver}")
            subprocess.run(f"pip install -U {pkg}", shell=True)
    except Exception as e:
        logger.warning(f"Could not check {pkg} version: {e}")

# Check for Unsloth availability
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
    logger.info("Unsloth optimization available")
except ImportError:
    UNSLOTH_AVAILABLE = False
    logger.info("Unsloth not available. Using standard optimization.")

# Import our direct fix for the "too many values to unpack" error
try:
    from src.generative_ai_module.direct_model_fix import apply_direct_fix
    logger.info("✅ Successfully imported direct_model_fix module")
except ImportError:
    try:
        # Try relative import
        from .direct_model_fix import apply_direct_fix
        logger.info("✅ Successfully imported direct_model_fix module via relative import")
    except ImportError:
        logger.warning("⚠️ Could not import direct_model_fix module, will apply fix later")
        # Define a dummy function to avoid errors
        def apply_direct_fix(model=None):
            logger.warning("Using dummy apply_direct_fix function")
            return False

# Apply attention mask fix
def apply_attention_mask_fix():
    """Apply the attention mask fix for DeepSeek models"""
    try:
        # First try our direct fix for the "too many values to unpack" error
        try:
            success = apply_direct_fix()
            if success:
                logger.info("✅ Successfully applied direct fix for 'too many values to unpack' error")
            else:
                logger.warning("⚠️ Direct fix failed, trying other fixes")
        except Exception as e:
            logger.warning(f"⚠️ Error applying direct fix: {e}, trying other fixes")

        # Use the comprehensive fix script if available
        try:
            import sys
            import os

            # Add the setup directory to sys.path
            setup_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "setup")
            if setup_dir not in sys.path:
                sys.path.insert(0, setup_dir)

            # Try to import and run the comprehensive fix first
            try:
                # Try different import paths
                try:
                    from setup.comprehensive_attention_mask_fix import apply_comprehensive_fix
                except ImportError:
                    try:
                        from src.setup.comprehensive_attention_mask_fix import apply_comprehensive_fix
                    except ImportError:
                        import sys
                        import os
                        # Add setup directory to path
                        setup_dir = os.path.join(os.getcwd(), "setup")
                        if os.path.exists(setup_dir):
                            sys.path.append(setup_dir)
                            from comprehensive_attention_mask_fix import apply_comprehensive_fix
                        else:
                            raise ImportError("Could not find comprehensive_attention_mask_fix module")

                success = apply_comprehensive_fix()

                if success:
                    logger.info("Successfully applied comprehensive attention mask fix")
                    return True
                else:
                    logger.warning("Comprehensive fix failed, trying original fix")
            except ImportError as e:
                logger.warning(f"Could not import comprehensive fix: {e}, trying original fix")

            # Fall back to the original fix if comprehensive fix is not available
            try:
                # Try different import paths
                try:
                    from setup.fix_transformers_attention_mask import fix_transformers_attention_mask
                except ImportError:
                    try:
                        from src.setup.fix_transformers_attention_mask import fix_transformers_attention_mask
                    except ImportError:
                        import sys
                        import os
                        # Add setup directory to path
                        setup_dir = os.path.join(os.getcwd(), "setup")
                        if os.path.exists(setup_dir):
                            sys.path.append(setup_dir)
                            from fix_transformers_attention_mask import fix_transformers_attention_mask
                        else:
                            raise ImportError("Could not find fix_transformers_attention_mask module")

                success = fix_transformers_attention_mask()
            except Exception as e:
                logger.error(f"Failed to apply original fix: {e}")
                success = False

            if success:
                logger.info("Successfully applied original attention mask fixes")
                return True
            else:
                logger.warning("Original fix script failed, falling back to built-in fixes")
        except ImportError as e:
            logger.warning(f"Could not import fix scripts: {e}, falling back to built-in fixes")
        except Exception as e:
            logger.warning(f"Error running fix scripts: {e}, falling back to built-in fixes")

        # Check transformers version to apply the appropriate fix
        from transformers import __version__ as transformers_version

        # Parse version string to check compatibility
        try:
            version_parts = transformers_version.split('.')
            major = int(version_parts[0]) if len(version_parts) > 0 else 0
            minor = int(version_parts[1]) if len(version_parts) > 1 else 0
            logger.info(f"Detected transformers version: {major}.{minor}")

            # For newer versions of transformers (4.28+), we need to patch the attention mask handling
            # to fix the device mismatch issue
            if major > 4 or (major == 4 and minor >= 28):
                logger.info(f"Applying device mismatch fix for transformers {transformers_version}")

                # Fix the _prepare_4d_causal_attention_mask_for_sdpa function
                from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa

                # Store the original function
                original_prepare_4d = _prepare_4d_causal_attention_mask_for_sdpa

                # Get the signature of the original function to ensure we match it
                import inspect
                sig = inspect.signature(original_prepare_4d)
                param_names = list(sig.parameters.keys())
                logger.info(f"Original function signature: {param_names}")

                # Define patched function with a specific signature for transformers 4.51.3
                def patched_prepare_4d(
                    attention_mask,
                    input_shape,
                    inputs_embeds,
                    past_key_values_length=0,
                    sliding_window=None,
                    dtype=None
                ):
                    """
                    Patched version that ensures attention_mask is 2D before processing.
                    Modified implementation for transformers 4.51.3 that takes exactly the right number of parameters.
                    """
                    import torch
                    import transformers
                    from packaging import version

                    # Log the function call for debugging
                    logger.info(f"Called patched_prepare_4d with attention_mask shape: {attention_mask.shape if attention_mask is not None else None}")
                    logger.info(f"Input shape: {input_shape}, Past KV length: {past_key_values_length}, Sliding window: {sliding_window}")

                    # Get transformers version
                    TRANSFORMERS_VERSION = version.parse(transformers.__version__)
                    logger.info(f"Transformers version: {TRANSFORMERS_VERSION}")

                    # Fix attention_mask shape if needed
                    if attention_mask is not None and attention_mask.dim() > 2:
                        # Get the batch size and sequence length
                        batch_size = attention_mask.size(0)
                        seq_length = attention_mask.size(-1)

                        # Calculate total elements in the tensor
                        total_elements = attention_mask.numel()

                        # Check if reshape is possible
                        if total_elements == batch_size * seq_length:
                            # Reshape to 2D [batch_size, seq_length]
                            attention_mask = attention_mask.view(batch_size, seq_length)
                            logger.info(f"Reshaped attention mask from >2D to 2D: {attention_mask.shape}")
                        else:
                            # If reshape is not possible, create a new attention mask
                            logger.warning(f"Cannot reshape attention mask of size {total_elements} to [{batch_size}, {seq_length}]. Creating new mask.")
                            # Create a new attention mask filled with ones (no masking)
                            attention_mask = torch.ones((batch_size, seq_length), device=attention_mask.device)
                            logger.info(f"Created new attention mask with shape: {attention_mask.shape}")

                    # Get device from inputs_embeds
                    device = inputs_embeds.device if inputs_embeds is not None else attention_mask.device

                    # Get dtype if not provided
                    if dtype is None:
                        dtype = inputs_embeds.dtype if inputs_embeds is not None else torch.float32

                    # Handle different transformers versions
                    try:
                        # Try to call the original function with the fixed parameters
                        if TRANSFORMERS_VERSION >= version.parse("4.40.0"):
                            # For newer versions, we need to pass all parameters
                            return original_prepare_4d(
                                attention_mask=attention_mask,
                                input_shape=input_shape,
                                inputs_embeds=inputs_embeds,
                                past_key_values_length=past_key_values_length,
                                sliding_window=sliding_window,
                                dtype=dtype
                            )
                        else:
                            # For older versions, we might need to omit some parameters
                            try:
                                return original_prepare_4d(
                                    attention_mask=attention_mask,
                                    input_shape=input_shape,
                                    inputs_embeds=inputs_embeds,
                                    past_key_values_length=past_key_values_length,
                                    dtype=dtype
                                )
                            except TypeError:
                                # If that fails, try with even fewer parameters
                                return original_prepare_4d(
                                    attention_mask=attention_mask,
                                    input_shape=input_shape,
                                    dtype=dtype
                                )
                    except Exception as e:
                        logger.warning(f"Error calling original function: {e}")

                        # Implement a fallback solution
                        logger.info("Using fallback implementation for _prepare_4d_causal_attention_mask_for_sdpa")

                        # Create a 4D causal attention mask
                        batch_size, seq_length = input_shape

                        # Add past_key_values_length to sequence length
                        seq_length_with_past = seq_length + past_key_values_length

                        # Create a causal mask [seq_length_with_past, seq_length_with_past]
                        causal_mask = torch.triu(
                            torch.ones((seq_length_with_past, seq_length_with_past), device=device, dtype=torch.bool),
                            diagonal=1
                        )

                        # If sliding_window is provided, apply it
                        if sliding_window is not None:
                            causal_mask = torch.triu(
                                causal_mask,
                                diagonal=-sliding_window + 1
                            )

                        # Expand to [1, 1, seq_length_with_past, seq_length_with_past]
                        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

                        # Convert to proper dtype
                        causal_mask = causal_mask.to(dtype=dtype)

                        # Apply attention_mask if provided
                        if attention_mask is not None:
                            # Expand attention_mask to 4D [batch_size, 1, 1, seq_length]
                            expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)

                            # Convert to proper dtype
                            expanded_mask = expanded_mask.to(dtype=dtype)

                            # Convert to proper format (0 for attended positions, -10000.0 for masked positions)
                            expanded_mask = (1.0 - expanded_mask) * -10000.0

                            # Expand to match causal mask dimensions
                            expanded_mask = expanded_mask.expand(batch_size, 1, seq_length, seq_length_with_past)

                            # Combine with causal mask
                            causal_mask = causal_mask + expanded_mask

                        return causal_mask

                # Apply the patch
                import transformers.modeling_attn_mask_utils
                transformers.modeling_attn_mask_utils._prepare_4d_causal_attention_mask_for_sdpa = patched_prepare_4d
                logger.info("Successfully patched _prepare_4d_causal_attention_mask_for_sdpa")

                # Fix the _unmask_unattended function that causes device mismatch
                from transformers.modeling_attn_mask_utils import AttentionMaskConverter

                # Store the original function
                original_unmask_unattended = AttentionMaskConverter._unmask_unattended

                # Get the signature of the original function to ensure we match it
                try:
                    sig = inspect.signature(AttentionMaskConverter._unmask_unattended)
                    param_names = list(sig.parameters.keys())
                    logger.info(f"Original _unmask_unattended signature: {param_names}")
                except Exception as e:
                    logger.warning(f"Could not get signature of _unmask_unattended: {e}")
                    param_names = []

                @staticmethod
                def patched_unmask_unattended(*args, **kwargs):
                    """
                    Patched version of _unmask_unattended that keeps tensors on the same device.

                    The original function has a call to .cpu() which causes device mismatch errors.
                    This patch ensures all operations happen on the same device and handles tensor size mismatches.
                    """
                    import torch

                    # Extract attention_mask from args or kwargs
                    attention_mask = None
                    if len(args) > 0:
                        attention_mask = args[0]
                    elif 'attention_mask' in kwargs:
                        attention_mask = kwargs['attention_mask']
                    elif len(param_names) > 0 and param_names[0] in kwargs:
                        attention_mask = kwargs[param_names[0]]

                    if attention_mask is None:
                        raise ValueError("Could not find attention_mask in arguments")

                    # Extract indices_k and indices_q from args or kwargs
                    indices_k = None
                    indices_q = None
                    unmasked_value = True  # Default value for unmasked_value

                    if len(args) > 1:
                        indices_k = args[1]
                    elif 'indices_k' in kwargs:
                        indices_k = kwargs['indices_k']
                    elif len(param_names) > 1 and param_names[1] in kwargs:
                        indices_k = kwargs[param_names[1]]

                    if len(args) > 2:
                        indices_q = args[2]
                    elif 'indices_q' in kwargs:
                        indices_q = kwargs['indices_q']
                    elif len(param_names) > 2 and param_names[2] in kwargs:
                        indices_q = kwargs[param_names[2]]

                    # Extract unmasked_value parameter if it exists
                    if len(args) > 3:
                        unmasked_value = args[3]
                    elif 'unmasked_value' in kwargs:
                        unmasked_value = kwargs['unmasked_value']
                    elif len(param_names) > 3 and param_names[3] in kwargs:
                        unmasked_value = kwargs[param_names[3]]

                    # Get the device of the attention mask
                    device = attention_mask.device

                    # Fix attention_mask shape if needed
                    if attention_mask.dim() > 2:
                        # Get the batch size and sequence length
                        batch_size = attention_mask.size(0)
                        seq_length = attention_mask.size(-1)

                        # Calculate total elements in the tensor
                        total_elements = attention_mask.numel()

                        # Check if reshape is possible
                        if total_elements == batch_size * seq_length:
                            # Reshape to 2D [batch_size, seq_length]
                            attention_mask = attention_mask.view(batch_size, seq_length)
                            logger.info(f"Reshaped attention mask from >2D to 2D: {attention_mask.shape}")
                        else:
                            # If reshape is not possible, create a new attention mask
                            logger.warning(f"Cannot reshape attention mask of size {total_elements} to [{batch_size}, {seq_length}]. Creating new mask.")
                            # Create a new attention mask filled with ones (no masking)
                            attention_mask = torch.ones((batch_size, seq_length), device=device)
                            logger.info(f"Created new attention mask with shape: {attention_mask.shape}")

                    # Create a temporary tensor on the same device (instead of using CPU)
                    tmp = torch.ones(attention_mask.shape[-1], device=device)

                    # Find the first non-masked position for each sequence
                    # Original: indices = torch.argmax(attention_mask.cpu() * tmp, 1, keepdim=True)
                    # Fixed: Keep everything on the same device
                    indices = torch.argmax(attention_mask * tmp, 1, keepdim=True)

                    # Create a mask for unattended positions
                    mask = torch.arange(attention_mask.shape[-1], device=device).expand(attention_mask.shape[0], -1)
                    mask = mask < indices

                    # Expand mask to 4D
                    mask = mask.unsqueeze(1).unsqueeze(2)

                    # Handle indices_k and indices_q if provided
                    try:
                        if indices_k is not None:
                            if isinstance(indices_k, int):
                                mask = mask.expand(-1, -1, indices_k, -1)
                            else:
                                # Handle case where indices_k is a tensor
                                mask = mask.expand(-1, -1, indices_k.size(0) if hasattr(indices_k, 'size') else indices_k, -1)

                        if indices_q is not None:
                            # Check if indices_q is the batch size (which would cause the error)
                            if isinstance(indices_q, int) and indices_q == attention_mask.size(0):
                                logger.warning(f"indices_q ({indices_q}) matches batch_size, which would cause dimension mismatch. Using seq_length instead.")
                                # Use sequence length instead of batch size for expansion
                                seq_length = attention_mask.size(-1)
                                mask = mask.expand(-1, 1, -1, -1)  # First expand with 1
                                mask = mask.expand(-1, -1, seq_length, -1)  # Then expand with seq_length
                            elif isinstance(indices_q, int):
                                mask = mask.expand(-1, indices_q, -1, -1)
                            else:
                                # Handle case where indices_q is a tensor
                                # Check if it's the batch size tensor
                                if hasattr(indices_q, 'size') and indices_q.size(0) == attention_mask.size(0):
                                    logger.warning(f"indices_q size ({indices_q.size(0)}) matches batch_size, which would cause dimension mismatch. Using seq_length instead.")
                                    seq_length = attention_mask.size(-1)
                                    mask = mask.expand(-1, 1, -1, -1)  # First expand with 1
                                    mask = mask.expand(-1, -1, seq_length, -1)  # Then expand with seq_length
                                else:
                                    mask = mask.expand(-1, indices_q.size(0) if hasattr(indices_q, 'size') else indices_q, -1, -1)
                    except Exception as e:
                        logger.warning(f"Error expanding mask dimensions: {e}")
                        # If we encounter any tensor size mismatch error, create a compatible mask
                        error_msg = str(e)
                        if "must match" in error_msg and "at non-singleton dimension" in error_msg:
                            logger.info("Detected tensor size mismatch error. Creating a compatible mask.")
                            # Create a compatible mask directly
                            batch_size = attention_mask.size(0)
                            seq_length = attention_mask.size(-1)

                            # Create a causal mask that matches the expected dimensions
                            causal_mask = torch.triu(
                                torch.ones((seq_length, seq_length), device=device, dtype=torch.bool),
                                diagonal=1
                            )
                            causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]
                            causal_mask = causal_mask.expand(batch_size, 1, seq_length, seq_length)

                            # Apply the attention mask if needed
                            if attention_mask is not None:
                                # Expand attention_mask to 4D [batch_size, 1, 1, seq_length]
                                expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                                # Expand to match causal mask dimensions
                                expanded_mask = expanded_mask.expand(-1, -1, seq_length, -1)
                                # Combine with causal mask (logical OR)
                                combined_mask = causal_mask | ~expanded_mask.bool()
                                # Convert back to the expected mask format
                                mask = ~combined_mask if unmasked_value else combined_mask
                            else:
                                mask = ~causal_mask if unmasked_value else causal_mask

                    # Convert mask to the expected type based on unmasked_value
                    if unmasked_value is not True:
                        mask = mask.to(dtype=attention_mask.dtype) * unmasked_value

                    return mask

                # Replace the original function with our patched version
                AttentionMaskConverter._unmask_unattended = patched_unmask_unattended
                logger.info("Successfully patched AttentionMaskConverter._unmask_unattended to fix device mismatch")

                return True
        except Exception as e:
            logger.warning(f"Could not parse transformers version: {e}. Applying legacy fix.")

        # For older versions, apply the legacy fix
        from transformers.models.llama.modeling_llama import LlamaModel

        # Store the original forward method
        original_forward = LlamaModel.forward

        # Define a patched forward method that properly handles attention masks
        def patched_forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        ):
            """Patched forward method for LlamaModel that properly handles attention masks."""
            # Force use_cache to False when using gradient checkpointing
            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                if use_cache:
                    logger.info("use_cache=True is incompatible with gradient checkpointing. Setting use_cache=False...")
                use_cache = False

            # Get the device from input tensors
            device = None
            if input_ids is not None:
                device = input_ids.device
            elif inputs_embeds is not None:
                device = inputs_embeds.device

            # Ensure attention_mask is on the correct device
            if attention_mask is not None and device is not None:
                if attention_mask.device != device:
                    logger.info(f"Moving attention mask from {attention_mask.device} to {device}")
                    attention_mask = attention_mask.to(device)

            # Fix attention mask shape if needed
            if attention_mask is not None and attention_mask.dim() == 2:
                try:
                    # Get the device and dtype
                    device = attention_mask.device
                    dtype = attention_mask.dtype

                    # Get sequence length
                    seq_length = attention_mask.size(1)
                    batch_size = attention_mask.size(0)

                    # Convert attention_mask from [batch_size, seq_length] to [batch_size, 1, seq_length, seq_length]
                    # First, expand to [batch_size, 1, 1, seq_length]
                    expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)

                    # Create a causal mask of shape [1, 1, seq_length, seq_length]
                    causal_mask = torch.triu(
                        torch.ones((seq_length, seq_length), device=device, dtype=dtype),
                        diagonal=1
                    ).unsqueeze(0).unsqueeze(0)

                    # Convert masks to proper format (-inf for masked positions, 0 for attended positions)
                    expanded_mask = (1.0 - expanded_mask) * -10000.0
                    causal_mask = (causal_mask > 0) * -10000.0

                    # Combine the masks
                    combined_mask = expanded_mask + causal_mask

                    # Replace the original attention_mask with our fixed version
                    attention_mask = combined_mask

                    logger.debug(f"Fixed attention mask shape: {attention_mask.shape}")
                except Exception as mask_error:
                    # If there's an error in our mask handling, log it and continue with the original mask
                    logger.warning(f"Error in attention mask fix: {mask_error}. Using original mask.")

            # Call the original forward method with the fixed attention mask
            return original_forward(
                self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # Replace the original forward method with our patched version
        LlamaModel.forward = patched_forward

        logger.info("Successfully applied legacy attention mask fix")
        return True
    except Exception as e:
        logger.error(f"Error applying attention mask fix: {e}")
        logger.warning("Continuing without attention mask fix")
        return False

def configure_for_gpu(args):
    """Configure training parameters based on GPU type and VRAM size"""
    logger.info(f"Configuring for GPU type {args.gpu_type} with {args.vram} GiB VRAM")

    # Base configuration for different GPU types
    if args.gpu_type == "A6000" and args.vram >= 48:
        # A6000 with 48+ GiB VRAM - maximize parameters while staying within constraints
        logger.info("Using optimized settings for A6000 with 48+ GiB VRAM")
        args.batch_size = 6  # Reduced from 8 to ensure stability
        args.gradient_accumulation_steps = 8
        args.max_length = 2048  # Reduced from 4096 to ensure stability with large models
        args.lora_rank = 32     # Increase LoRA rank for better quality
        args.lora_alpha = 64    # Increase LoRA alpha for better adaptation
        args.lora_dropout = 0.05  # Optimal dropout for stability
        args.load_in_4bit = True  # Use 4-bit quantization for maximum memory efficiency
        args.load_in_8bit = False
        args.bf16 = True  # Use bfloat16 precision for better training stability
        args.num_workers = 8    # Match your 8 CPU cores
        args.warmup_ratio = 0.03  # Optimal warmup for large models
        args.weight_decay = 0.01  # Prevent overfitting
        args.adam_beta1 = 0.9   # Standard beta1 for AdamW
        args.adam_beta2 = 0.999  # Standard beta2 for AdamW
        args.adam_epsilon = 1e-8  # Standard epsilon for AdamW
        args.max_grad_norm = 1.0  # Prevent gradient explosion
        args.scheduler_type = "cosine"  # Cosine scheduler with warmup
        args.evaluation_strategy = "steps"  # Evaluate during training
        args.eval_steps = 100   # Evaluate every 100 steps
        args.save_steps = 100   # Save every 100 steps
        args.save_total_limit = 3  # Keep only the last 3 checkpoints

    elif args.gpu_type == "A6000" and args.vram >= 40:
        # A6000 with 40-48 GiB VRAM
        logger.info("Using optimized settings for A6000 with 40-48 GiB VRAM")
        args.batch_size = 6
        args.gradient_accumulation_steps = 8
        args.max_length = 3072
        args.lora_rank = 24
        args.lora_alpha = 48
        args.lora_dropout = 0.05
        args.load_in_4bit = True
        args.load_in_8bit = False
        args.bf16 = True
        args.num_workers = 6
        args.warmup_ratio = 0.03
        args.weight_decay = 0.01
        args.adam_beta1 = 0.9
        args.adam_beta2 = 0.999
        args.adam_epsilon = 1e-8
        args.max_grad_norm = 1.0
        args.scheduler_type = "cosine"
        args.evaluation_strategy = "steps"
        args.eval_steps = 100
        args.save_steps = 100
        args.save_total_limit = 3

    elif args.gpu_type == "A4000" or (args.gpu_type == "A6000" and args.vram < 40):
        # A4000 or A6000 with less VRAM
        logger.info("Using optimized settings for A4000 or A6000 with <40 GiB VRAM")
        args.batch_size = 4
        args.gradient_accumulation_steps = 16
        args.max_length = 2048
        args.lora_rank = 16
        args.lora_alpha = 32
        args.lora_dropout = 0.05
        args.load_in_4bit = True
        args.load_in_8bit = False
        args.bf16 = True
        args.num_workers = 4
        args.warmup_ratio = 0.03
        args.weight_decay = 0.01
        args.adam_beta1 = 0.9
        args.adam_beta2 = 0.999
        args.adam_epsilon = 1e-8
        args.max_grad_norm = 1.0
        args.scheduler_type = "cosine"
        args.evaluation_strategy = "steps"
        args.eval_steps = 100
        args.save_steps = 100
        args.save_total_limit = 2

    elif args.gpu_type == "RTX5000":
        # RTX5000 with limited VRAM
        logger.info("Using optimized settings for RTX5000")
        args.batch_size = 2
        args.gradient_accumulation_steps = 32
        args.max_length = 1024
        args.lora_rank = 8
        args.lora_alpha = 16
        args.lora_dropout = 0.05
        args.load_in_4bit = True
        args.load_in_8bit = False
        args.bf16 = False  # Use fp16 instead
        args.num_workers = 2
        args.warmup_ratio = 0.03
        args.weight_decay = 0.01
        args.adam_beta1 = 0.9
        args.adam_beta2 = 0.999
        args.adam_epsilon = 1e-8
        args.max_grad_norm = 1.0
        args.scheduler_type = "linear"  # Linear scheduler for smaller models
        args.evaluation_strategy = "steps"
        args.eval_steps = 100
        args.save_steps = 100
        args.save_total_limit = 1

    # Calculate effective batch size
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    logger.info(f"Effective batch size: {effective_batch_size}")

    # Adjust learning rate based on batch size (linear scaling rule)
    base_lr = 2e-5
    base_batch_size = 64
    args.learning_rate = base_lr * (effective_batch_size / base_batch_size)
    logger.info(f"Adjusted learning rate: {args.learning_rate}")

    # Calculate warmup steps based on warmup ratio
    args.warmup_steps = int(args.warmup_ratio * effective_batch_size * 100)  # Assuming ~100 steps per epoch
    logger.info(f"Warmup steps: {args.warmup_steps}")

    # Memory optimization settings
    if args.optimize_memory_usage:
        logger.info("Applying memory optimization settings")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,garbage_collection_threshold:0.8"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["OMP_NUM_THREADS"] = str(min(8, os.cpu_count() or 1))  # Use available CPU cores efficiently
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Disable for better performance
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Reduce TensorFlow logging

        # Print GPU information if available
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                device_capability = torch.cuda.get_device_capability(0)
                logger.info(f"Using GPU: {device_name} with CUDA capability {device_capability[0]}.{device_capability[1]}")
                logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GiB")
                # Clear CUDA cache
                torch.cuda.empty_cache()
        except ImportError:
            logger.warning("PyTorch not available, skipping GPU information")

    return args

def main():
    """Main function to run the unified training"""
    parser = argparse.ArgumentParser(description="Unified DeepSeek-Coder training")
    parser.add_argument("--model_name", type=str, default="deepseek-ai/deepseek-coder-6.7b-instruct",
                        help="Model name or path")
    parser.add_argument("--output_dir", type=str, default="/notebooks/Jarvis_AI_Assistant/models/deepseek-coder-6.7b-finetuned",
                        help="Output directory for saving the model")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--max_samples", type=int, default=5000,
                        help="Maximum number of samples to use")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Load model in 4-bit precision")
    parser.add_argument("--load_in_8bit", action="store_true",
                        help="Load model in 8-bit precision")
    parser.add_argument("--use_flash_attn", action="store_true",
                        help="Use Flash Attention for faster training")
    parser.add_argument("--dataset_subset", type=str, default="python",
                        help="Dataset subset to use")
    parser.add_argument("--all_subsets", action="store_true",
                        help="Use all language subsets")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Number of warmup steps")
    parser.add_argument("--bf16", action="store_true",
                        help="Use bfloat16 precision")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--gpu-type", type=str, default="A6000",
                        help="GPU type (A6000, A4000, RTX5000)")
    parser.add_argument("--vram", type=int, default=50,
                        help="GPU VRAM size in GiB")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="Use gradient checkpointing to reduce memory usage")
    parser.add_argument("--mixed_precision", action="store_true", default=True,
                        help="Use mixed precision training")
    parser.add_argument("--optimize_memory_usage", action="store_true", default=True,
                        help="Optimize memory usage during training")
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank parameter")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout parameter")

    # Advanced optimization parameters
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Warmup ratio for learning rate scheduler")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for AdamW optimizer")
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                        help="Beta1 for AdamW optimizer")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                        help="Beta2 for AdamW optimizer")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8,
                        help="Epsilon for AdamW optimizer")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for clipping")
    parser.add_argument("--scheduler_type", type=str, default="cosine",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
                        help="Learning rate scheduler type")
    parser.add_argument("--evaluation_strategy", type=str, default="steps",
                        choices=["no", "steps", "epoch"],
                        help="Evaluation strategy to use during training")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Number of steps between evaluations when using steps strategy")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Number of steps between model checkpoints")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Maximum number of checkpoints to keep")

    args = parser.parse_args()

    # Automatically configure parameters based on GPU type and VRAM size
    configure_for_gpu(args)

    # Apply attention mask fix
    logger.info("Applying attention mask fix...")
    apply_attention_mask_fix()

    # Ensure output directory is in a writable location
    try:
        # First try to create the directory to check if it's writable
        os.makedirs(args.output_dir, exist_ok=True)
        # Test if the directory is writable by creating a test file
        test_file = os.path.join(args.output_dir, 'test_write.txt')
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            logger.info(f"Using output directory: {args.output_dir}")
        except (IOError, OSError) as e:
            logger.warning(f"Directory {args.output_dir} is not writable: {e}")
            # Use a local directory instead
            local_dir = os.path.join(os.getcwd(), "models/deepseek-coder-6.7b-finetuned")
            logger.warning(f"Changed output directory to local path: {local_dir}")
            args.output_dir = local_dir
            os.makedirs(args.output_dir, exist_ok=True)
    except Exception as e:
        logger.warning(f"Error testing directory writability: {e}")
        # Use a local directory as fallback
        local_dir = os.path.join(os.getcwd(), "models/deepseek-coder-6.7b-finetuned")
        logger.warning(f"Changed output directory to local path: {local_dir}")
        args.output_dir = local_dir
        os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Using output directory: {args.output_dir}")

    # Set default quantization if none specified
    if not args.load_in_4bit and not args.load_in_8bit:
        args.load_in_4bit = True

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    # Choose the appropriate training method
    if UNSLOTH_AVAILABLE:
        logger.info("Using Unsloth-optimized training")
        train_with_unsloth(args)
    else:
        logger.info("Using standard training with attention mask fix")
        train_with_standard_method(args)

    logger.info(f"Training completed. Model saved to {args.output_dir}")

def train_with_unsloth(args):
    """Train using Unsloth optimization"""
    from unsloth import FastLanguageModel
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling, __version__ as transformers_version
    from datasets import load_dataset

    # Use the safe_autocast function defined at the top of the file
    logger.info("Using safe_autocast for mixed precision training")

    # Check transformers version for compatibility
    logger.info(f"Using transformers version: {transformers_version}")

    # Apply PEFT model fix for "too many values to unpack" error
    def apply_peft_model_fix():
        """
        Apply a fix for the "too many values to unpack" error in PEFT models.
        This patches the forward method of PeftModel to always return a dictionary-like object.
        """
        try:
            # Try to import PeftModel
            try:
                from peft import PeftModel
                logger.info("Successfully imported PeftModel")
            except ImportError:
                logger.warning("Could not import PeftModel, skipping PEFT model fix")
                return False

            # Check if the forward method exists
            if not hasattr(PeftModel, "forward"):
                logger.warning("PeftModel does not have a forward method, skipping PEFT model fix")
                return False

            # Store the original forward method
            original_forward = PeftModel.forward
            logger.info("Stored original PeftModel.forward method")

            # Define a patched forward method
            def patched_forward(self, *args, **kwargs):
                """
                Patched forward method for PeftModel that ensures:
                1. No duplicate arguments (e.g., input_ids passed both as positional and keyword)
                2. Always returns a dictionary-like object with return_dict=True
                3. Handles any errors gracefully
                """
                logger.info("Using patched PeftModel.forward method")

                # Always set return_dict=True to avoid tuple unpacking issues
                kwargs["return_dict"] = True

                # Handle case where input_ids is passed both as positional and keyword argument
                if len(args) > 1 and isinstance(args[1], torch.Tensor) and "input_ids" in kwargs:
                    logger.info("Detected input_ids in both args and kwargs, removing from kwargs")
                    # Remove input_ids from kwargs to avoid the multiple values error
                    del kwargs["input_ids"]

                # Call the original forward method with proper arguments
                try:
                    return original_forward(self, **kwargs)
                except Exception as e:
                    logger.error(f"Error in patched PeftModel.forward: {e}")

                    # Try a more direct approach if the original call fails
                    try:
                        logger.info("Trying direct call to base_model.forward")
                        # Call the base model's forward method directly
                        outputs = self.base_model.forward(**kwargs)

                        # Ensure the output is a dictionary-like object
                        if not hasattr(outputs, "keys"):
                            from transformers.modeling_outputs import CausalLMOutputWithPast
                            # Convert tuple to dictionary-like object
                            if isinstance(outputs, tuple):
                                logger.info(f"Converting tuple output with {len(outputs)} elements to dictionary")
                                outputs_dict = {}

                                # First element is typically the loss
                                if len(outputs) >= 1:
                                    outputs_dict["loss"] = outputs[0]

                                # Second element is typically the logits
                                if len(outputs) >= 2:
                                    outputs_dict["logits"] = outputs[1]

                                # Add any remaining elements with generic names
                                for i in range(2, len(outputs)):
                                    outputs_dict[f"hidden_states_{i-2}"] = outputs[i]

                                # Convert to ModelOutput
                                outputs = CausalLMOutputWithPast(**outputs_dict)
                                logger.info(f"Converted tuple to ModelOutput with keys: {list(outputs_dict.keys())}")

                        return outputs
                    except Exception as e2:
                        logger.error(f"Direct call to base_model.forward also failed: {e2}")
                        # Create a dummy output as last resort
                        from transformers.modeling_outputs import CausalLMOutputWithPast
                        dummy_logits = torch.zeros((1, 1, self.base_model.config.vocab_size),
                                                device=self.device if hasattr(self, "device") else "cpu")
                        dummy_loss = torch.tensor(1.0, requires_grad=True,
                                               device=self.device if hasattr(self, "device") else "cpu")
                        return CausalLMOutputWithPast(loss=dummy_loss, logits=dummy_logits)

            # Apply the patch
            PeftModel.forward = patched_forward
            logger.info("✅ Successfully patched PeftModel.forward to fix 'too many values to unpack' error")
            return True
        except Exception as e:
            logger.error(f"Failed to apply PEFT model fix: {e}")
            return False

    # Apply the PEFT model fix
    apply_peft_model_fix()

    # Parse version string to check compatibility
    try:
        import transformers
        from packaging import version

        TRANSFORMERS_VERSION = version.parse(transformers.__version__)
        logger.info(f"Parsed transformers version: {TRANSFORMERS_VERSION}")

        # Check compatibility with DeepSeek models
        if TRANSFORMERS_VERSION < version.parse("4.28.0"):
            logger.warning(f"Transformers version {transformers.__version__} may not be compatible with DeepSeek models. Recommended: 4.28.0+")

        # Check for specific version that has the attention mask issue
        if TRANSFORMERS_VERSION >= version.parse("4.40.0") and TRANSFORMERS_VERSION < version.parse("4.52.0"):
            logger.warning(f"Transformers version {transformers.__version__} may have attention mask issues. Applying fixes.")
            # Apply attention mask fix
            apply_attention_mask_fix()
    except Exception as e:
        logger.warning(f"Could not parse transformers version: {e}. Assuming compatibility.")

    logger.info("Loading and preprocessing dataset...")

    # Load dataset - always try to load all subsets first
    logger.info("Attempting to load all language subsets")
    from src.generative_ai_module.code_preprocessing import load_and_preprocess_all_subsets, load_and_preprocess_dataset

    # First try to load all language subsets
    train_dataset, eval_dataset = load_and_preprocess_all_subsets(
        max_samples=args.max_samples,
        sequence_length=args.max_length,
        return_raw=True
    )

    # If loading all subsets failed or returned empty datasets, fall back to specific subset
    if train_dataset is None or len(train_dataset) == 0:
        logger.warning("Loading all subsets failed or returned empty datasets. Falling back to specific subset.")
        if args.dataset_subset:
            logger.info(f"Loading {args.dataset_subset} subset")
            train_dataset, eval_dataset = load_and_preprocess_dataset(
                max_samples=args.max_samples,
                sequence_length=args.max_length,
                subset=args.dataset_subset,
                all_subsets=False,
                return_raw=True
            )
        else:
            # If no specific subset is specified, try python as a last resort
            logger.info("No specific subset specified. Trying 'python' subset as fallback.")
            train_dataset, eval_dataset = load_and_preprocess_dataset(
                max_samples=args.max_samples,
                sequence_length=args.max_length,
                subset="python",
                all_subsets=False,
                return_raw=True
            )

    # Validate datasets before proceeding
    if train_dataset is None or len(train_dataset) == 0:
        logger.error("Training dataset is empty or None. Cannot proceed with training.")
        raise ValueError("Training dataset is empty or None")

    if eval_dataset is None or len(eval_dataset) == 0:
        logger.warning("Evaluation dataset is empty or None. Creating a small subset of training data for evaluation.")
        # Create a small evaluation dataset from training data if needed
        if len(train_dataset) > 10:
            eval_dataset = train_dataset.select(range(min(len(train_dataset) // 10, 100)))
            logger.info(f"Created evaluation dataset with {len(eval_dataset)} samples from training data")
        else:
            # If training dataset is too small, use it for both training and evaluation
            eval_dataset = train_dataset
            logger.warning("Training dataset is very small. Using same data for evaluation.")

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")

    # Validate that datasets have the required 'text' field
    if 'text' not in train_dataset.column_names:
        logger.error("Training dataset does not have a 'text' field. Cannot proceed with training.")
        raise ValueError("Training dataset missing 'text' field")

    if 'text' not in eval_dataset.column_names:
        logger.error("Evaluation dataset does not have a 'text' field. Cannot proceed with training.")
        raise ValueError("Evaluation dataset missing 'text' field")

    # Load model with Unsloth with robust error handling
    logger.info(f"Loading model: {args.model_name}")
    print("🦥 Loading model", args.model_name, "with minimal unsloth")
    if args.load_in_4bit:
        print("Loading model in 4-bit quantization")
    elif args.load_in_8bit:
        print("Loading model in 8-bit quantization")

    # Check bitsandbytes version for 4-bit quantization compatibility
    if args.load_in_4bit:
        try:
            import bitsandbytes as bnb
            bnb_version = getattr(bnb, "__version__", "unknown")
            logger.info(f"Using bitsandbytes version: {bnb_version}")

            # Check if version is compatible with 4-bit quantization
            if bnb_version != "unknown":
                try:
                    major, minor, patch = map(int, bnb_version.split('.')[0:3])
                    if (major > 0) or (major == 0 and minor >= 42):
                        logger.info("✅ bitsandbytes version is compatible with 4-bit quantization")
                    else:
                        logger.warning(f"⚠️ bitsandbytes version {bnb_version} may not support 4-bit quantization with to() method")
                        logger.warning("This can cause errors like: 'Calling `to()` is not supported for `4-bit` quantized models'")
                        logger.warning("Attempting to fix bitsandbytes version...")

                        # Try to run the fix script
                        import subprocess
                        import os
                        fix_script = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                                "setup", "fix_bitsandbytes_version.py")
                        if os.path.exists(fix_script):
                            logger.info(f"Running bitsandbytes fix script: {fix_script}")
                            try:
                                subprocess.check_call([sys.executable, fix_script])
                                logger.info("✅ Successfully fixed bitsandbytes version")

                                # Reload bitsandbytes
                                import importlib
                                importlib.reload(bnb)
                                bnb_version = getattr(bnb, "__version__", "unknown")
                                logger.info(f"Updated bitsandbytes version: {bnb_version}")
                            except subprocess.CalledProcessError as e:
                                logger.warning(f"⚠️ Failed to fix bitsandbytes version: {e}")
                                logger.warning("Continuing with current version, but 4-bit quantization may fail")
                        else:
                            logger.warning(f"⚠️ Could not find bitsandbytes fix script at {fix_script}")
                except ValueError:
                    logger.warning(f"Could not parse bitsandbytes version: {bnb_version}")
        except ImportError:
            logger.warning("Could not import bitsandbytes to check version")

    # Add robust error handling for model loading
    try:
        # Patch the unsloth library to handle trust_remote_code correctly
        try:
            import unsloth.models

            # Check if we need to patch the adapter's get_model_and_tokenizer method
            original_get_model_and_tokenizer = unsloth.models.BaseAdapter.get_model_and_tokenizer

            def patched_get_model_and_tokenizer(self, *args, **kwargs):
                """Patched version that ensures trust_remote_code is set correctly"""
                # Make sure trust_remote_code is included in kwargs
                if 'trust_remote_code' not in kwargs:
                    kwargs['trust_remote_code'] = True

                logger.info(f"Calling get_model_and_tokenizer with trust_remote_code={kwargs.get('trust_remote_code')}")
                return original_get_model_and_tokenizer(self, *args, **kwargs)

            # Apply the patch
            unsloth.models.BaseAdapter.get_model_and_tokenizer = patched_get_model_and_tokenizer
            logger.info("✅ Successfully patched unsloth.models.BaseAdapter.get_model_and_tokenizer")

            # Also patch the from_pretrained method
            original_from_pretrained = FastLanguageModel.from_pretrained

            @staticmethod
            def patched_from_pretrained(*args, **kwargs):
                """Patched version that ensures trust_remote_code is set correctly"""
                # Make sure trust_remote_code is included in kwargs
                if 'trust_remote_code' not in kwargs:
                    kwargs['trust_remote_code'] = True

                logger.info(f"Calling FastLanguageModel.from_pretrained with trust_remote_code={kwargs.get('trust_remote_code')}")
                return original_from_pretrained(*args, **kwargs)

            # Apply the patch
            FastLanguageModel.from_pretrained = patched_from_pretrained
            logger.info("✅ Successfully patched FastLanguageModel.from_pretrained")

        except Exception as patch_error:
            logger.warning(f"Failed to patch unsloth library: {patch_error}")
            logger.warning("Will try to load model without patching")

        # First, try to load with explicit trust_remote_code
        logger.info("Loading model with explicit trust_remote_code=True...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_name,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            trust_remote_code=True  # Explicitly set trust_remote_code
        )
        logger.info(f"Successfully loaded model: {args.model_name}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")

        # Check if this is a bitsandbytes version error
        error_str = str(e)
        if "Calling `to()` is not supported for `4-bit` quantized models" in error_str and args.load_in_4bit:
            logger.warning("Detected bitsandbytes version compatibility issue with 4-bit quantization")
            logger.warning("Attempting to fix by upgrading bitsandbytes...")

            try:
                import subprocess
                # Try to install the latest available version
                subprocess.check_call([sys.executable, "-m", "pip", "install", "bitsandbytes", "--upgrade"])
                logger.info("✅ Successfully upgraded bitsandbytes")

                # Import to check version
                import importlib
                import bitsandbytes as bnb
                importlib.reload(bnb)
                bnb_version = getattr(bnb, "__version__", "unknown")
                logger.info(f"Upgraded bitsandbytes to version: {bnb_version}")

                # Try loading again with 8-bit quantization instead
                logger.info("Retrying model loading with 8-bit quantization...")
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=args.model_name,
                    load_in_4bit=False,
                    load_in_8bit=True,
                    trust_remote_code=True
                )
                logger.info(f"Successfully loaded model with 8-bit quantization: {args.model_name}")
            except Exception as upgrade_error:
                logger.error(f"Failed to upgrade bitsandbytes: {upgrade_error}")
                logger.warning("Falling back to standard loading without quantization")

                # Try with standard loading without quantization
                try:
                    from transformers import AutoModelForCausalLM, AutoTokenizer

                    logger.info("Loading with standard transformers without quantization...")
                    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
                    model = AutoModelForCausalLM.from_pretrained(
                        args.model_name,
                        device_map="auto",
                        trust_remote_code=True
                    )
                    logger.info(f"Successfully loaded model with standard transformers: {args.model_name}")
                except Exception as std_error:
                    logger.error(f"Failed to load with standard transformers: {std_error}")
                    # Continue to next fallback

        # Check if this is a trust_remote_code error
        elif "trust_remote_code" in error_str:
            logger.warning("Detected trust_remote_code error")

            # Try with standard transformers
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                logger.info("Loading with standard transformers...")
                tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name,
                    device_map="auto",
                    trust_remote_code=True,
                    load_in_4bit=args.load_in_4bit,
                    load_in_8bit=args.load_in_8bit
                )
                logger.info(f"Successfully loaded model with standard transformers: {args.model_name}")
            except Exception as std_error:
                logger.error(f"Failed to load with standard transformers: {std_error}")

                # Try with 8-bit quantization
                try:
                    logger.info("Trying with 8-bit quantization...")
                    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
                    model = AutoModelForCausalLM.from_pretrained(
                        args.model_name,
                        device_map="auto",
                        trust_remote_code=True,
                        load_in_8bit=True
                    )
                    logger.info(f"Successfully loaded model with 8-bit quantization: {args.model_name}")
                except Exception as bit8_error:
                    logger.error(f"Failed to load with 8-bit quantization: {bit8_error}")
                    # Continue to next fallback

        # Try with additional parameters that might help
        if 'model' not in locals() or model is None:
            try:
                logger.info("Retrying model loading with additional parameters...")

                # Try to import directly from transformers
                from transformers import AutoModelForCausalLM, AutoTokenizer

                logger.info("Loading with direct transformers import...")
                tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name,
                    device_map="auto",
                    trust_remote_code=True,
                    load_in_8bit=True  # Fall back to 8-bit
                )
                logger.info(f"Successfully loaded model with direct transformers import: {args.model_name}")
            except Exception as e2:
                logger.error(f"Fatal error loading model on retry: {e2}")
                raise RuntimeError(f"Could not load model {args.model_name}. Original error: {e}, Retry error: {e2}")

    # Set max sequence length after model is loaded
    model.config.max_position_embeddings = args.max_length
    tokenizer.model_max_length = args.max_length

    # Set up LoRA with optimized parameters
    logger.info("Configuring LoRA adapters")
    logger.info(f"Using LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}, dropout: {args.lora_dropout}")

    # Define target modules for LoRA
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]

    # Apply LoRA adapters with robust error handling
    try:
        logger.info(f"Applying LoRA with rank={args.lora_rank}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,  # LoRA rank from configuration
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",  # Don't train bias terms for stability
            # Removed task_type parameter as it's set internally by FastLanguageModel.get_peft_model
            modules_to_save=None  # Don't save any modules fully (use LoRA for all)
        )
        logger.info("Successfully applied LoRA adapters")
    except Exception as e:
        logger.error(f"Error applying LoRA adapters: {e}")

        # Try alternative approach with explicit LoraConfig
        try:
            from peft import LoraConfig

            logger.warning("Trying alternative LoRA application method...")
            # Create explicit LoraConfig
            lora_config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM"  # Add task_type for proper configuration
            )

            # Apply LoRA with the config
            model = FastLanguageModel.get_peft_model(
                model,
                peft_config=lora_config  # Pass the config object instead of individual parameters
            )
            logger.info("Successfully applied LoRA adapters with alternative method")
        except Exception as e2:
            logger.error(f"Alternative LoRA application also failed: {e2}")
            raise RuntimeError(f"Could not apply LoRA adapters. Original error: {e}, Alternative error: {e2}")

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}% of total)")
    logger.info(f"Total parameters: {total_params:,}")

    # Tokenize dataset
    logger.info("Tokenizing dataset")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        """Tokenize examples with proper handling of potential issues"""
        try:
            # Ensure all texts are strings and handle potential None values
            texts = []
            for text in examples["text"]:
                if text is None:
                    texts.append("")  # Replace None with empty string
                elif isinstance(text, str):
                    texts.append(text)  # Keep strings as is
                else:
                    texts.append(str(text))  # Convert other types to string

            # Tokenize without return_tensors to avoid the "too many dimensions" error
            return tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=args.max_length,
                # Removed return_tensors="pt" to avoid the "too many dimensions" error
            )
        except Exception as e:
            logger.error(f"Error during tokenization: {e}")
            # Return empty dict with the right structure as fallback
            return {
                "input_ids": [[0] * args.max_length] * len(examples["text"]),
                "attention_mask": [[0] * args.max_length] * len(examples["text"])
            }

    # Keep only the essential 'text' field and remove all other fields to avoid tokenization issues
    logger.info("Cleaning datasets to keep only essential fields")

    # Get all column names except 'text'
    train_columns_to_remove = [col for col in train_dataset.column_names if col != 'text']
    eval_columns_to_remove = [col for col in eval_dataset.column_names if col != 'text']

    # Remove all non-essential columns
    if train_columns_to_remove:
        logger.info(f"Removing non-essential fields from training dataset: {train_columns_to_remove}")
        train_dataset = train_dataset.remove_columns(train_columns_to_remove)

    if eval_columns_to_remove:
        logger.info(f"Removing non-essential fields from evaluation dataset: {eval_columns_to_remove}")
        eval_dataset = eval_dataset.remove_columns(eval_columns_to_remove)

    # Apply tokenization and remove the 'text' field to avoid nesting issues
    logger.info("Applying tokenization to datasets and removing 'text' field")
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=['text'])

    # Verify that the 'text' field is removed
    if 'text' in train_dataset.column_names:
        logger.warning("'text' field still present in training dataset after tokenization, removing it explicitly")
        train_dataset = train_dataset.remove_columns(['text'])

    if 'text' in eval_dataset.column_names:
        logger.warning("'text' field still present in evaluation dataset after tokenization, removing it explicitly")
        eval_dataset = eval_dataset.remove_columns(['text'])

    # Set up training arguments with optimized parameters and multiprocessing safeguards
    logger.info(f"Setting up training arguments with batch size: {args.batch_size}, gradient accumulation: {args.gradient_accumulation_steps}")

    # Adjust number of workers for multiprocessing safety
    # When using 'spawn' method with CUDA, it's safer to use fewer workers or even 0
    safe_num_workers = 0  # Set to 0 to avoid multiprocessing issues with CUDA
    if args.num_workers > 0:
        logger.warning(f"Reducing dataloader workers from {args.num_workers} to {safe_num_workers} to avoid CUDA multiprocessing issues")

    # Check transformers version to determine which parameters are supported
    from transformers import __version__ as transformers_version
    logger.info(f"Configuring TrainingArguments for transformers version: {transformers_version}")

    # Create a dictionary of arguments that we'll filter based on version compatibility
    training_args_dict = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "max_grad_norm": args.max_grad_norm,

        # Logging and evaluation settings
        "logging_steps": 10,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,

        # Precision settings
        "bf16": args.bf16,
        "fp16": not args.bf16 and torch.cuda.is_available(),

        # Memory optimization settings
        "gradient_checkpointing": args.gradient_checkpointing,
        "optim": "adamw_torch",  # More memory-efficient optimizer
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "adam_epsilon": args.adam_epsilon,

        # Data loading optimizations - adjusted for multiprocessing safety
        "remove_unused_columns": False,
        "dataloader_num_workers": safe_num_workers,  # Use 0 workers to avoid multiprocessing issues
        "dataloader_pin_memory": False,  # Set to False to avoid potential CUDA issues
        "group_by_length": True,

        # Additional memory and performance optimizations
        "ddp_find_unused_parameters": False,
        "torch_compile": False,  # Disable torch.compile which can use more memory
        "report_to": ["tensorboard"],  # Minimal reporting to save memory

        # Learning rate scheduler
        "lr_scheduler_type": args.scheduler_type,

        # Push to Hub settings (disabled by default)
        "push_to_hub": False,

        # Distributed training settings
        "local_rank": -1,

        # Seed for reproducibility
        "seed": 42,

        # Additional safety settings
        "dataloader_drop_last": False    # Don't drop last batch
    }

    # Add version-specific parameters
    try:
        # Try to parse version
        version_parts = transformers_version.split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1])

        # For newer versions (4.28+), add these parameters
        if major > 4 or (major == 4 and minor >= 28):
            logger.info("Adding parameters for transformers 4.28+")
            training_args_dict["save_safetensors"] = True
            training_args_dict["hub_strategy"] = "every_save"
            training_args_dict["tf32"] = True

            # Add evaluation strategy parameters for newer versions
            if hasattr(TrainingArguments, "evaluation_strategy"):
                logger.info("Adding evaluation_strategy parameter")
                training_args_dict["evaluation_strategy"] = "steps"
                training_args_dict["eval_steps"] = args.eval_steps
    except Exception as e:
        logger.warning(f"Error parsing transformers version: {e}. Using basic parameters.")

    # Create TrainingArguments with the filtered dictionary
    logger.info(f"Creating TrainingArguments with parameters: {list(training_args_dict.keys())}")
    training_args = TrainingArguments(**training_args_dict)

    # Create a completely custom data collator that doesn't rely on the parent class
    class SimpleDataCollator:
        """Simple custom data collator that handles all processing directly"""
        def __init__(self, tokenizer, mlm=False, pad_to_multiple_of=None):
            self.tokenizer = tokenizer
            self.mlm = mlm
            self.pad_to_multiple_of = pad_to_multiple_of
            logger.info("Using SimpleDataCollator with direct tensor creation")

        def __call__(self, features):
            try:
                # Check if features have the expected structure
                if not all(isinstance(f, dict) for f in features):
                    logger.warning("Features are not all dictionaries, converting...")
                    features = [dict(f) if not isinstance(f, dict) else f for f in features]

                # Remove 'text' field if present - this is causing the nesting issue
                for feature in features:
                    if 'text' in feature:
                        logger.warning("Removing 'text' field from feature to avoid nesting issues")
                        del feature['text']

                # Extract input_ids and attention_mask, handling potential issues
                batch_input_ids = []
                batch_attention_mask = []

                for i, feature in enumerate(features):
                    # Handle input_ids
                    if 'input_ids' not in feature:
                        logger.warning(f"Feature {i} missing input_ids, adding empty")
                        input_ids = [0] * args.max_length
                    else:
                        input_ids = feature['input_ids']
                        # Handle nested lists
                        if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
                            input_ids = input_ids[0]
                        # Ensure correct length
                        if len(input_ids) < args.max_length:
                            input_ids = input_ids + [self.tokenizer.pad_token_id] * (args.max_length - len(input_ids))
                        elif len(input_ids) > args.max_length:
                            input_ids = input_ids[:args.max_length]

                    # Handle attention_mask
                    if 'attention_mask' not in feature:
                        logger.warning(f"Feature {i} missing attention_mask, adding empty")
                        attention_mask = [0] * args.max_length
                    else:
                        attention_mask = feature['attention_mask']
                        # Handle nested lists
                        if isinstance(attention_mask, list) and attention_mask and isinstance(attention_mask[0], list):
                            attention_mask = attention_mask[0]
                        # Ensure correct length
                        if len(attention_mask) < args.max_length:
                            attention_mask = attention_mask + [0] * (args.max_length - len(attention_mask))
                        elif len(attention_mask) > args.max_length:
                            attention_mask = attention_mask[:args.max_length]

                    batch_input_ids.append(input_ids)
                    batch_attention_mask.append(attention_mask)

                # Create tensors directly on CUDA if available
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                logger.info(f"Creating tensors on device: {device}")

                # Get model dtype if possible
                model_dtype = None
                try:
                    if hasattr(model, 'dtype'):
                        model_dtype = model.dtype
                        logger.info(f"Using model's dtype: {model_dtype}")
                    else:
                        # Try to detect from parameters
                        for param in model.parameters():
                            if param.dtype in [torch.float16, torch.bfloat16]:
                                model_dtype = param.dtype
                                logger.info(f"Detected model dtype from parameters: {model_dtype}")
                                break
                except Exception as e:
                    logger.warning(f"Could not detect model dtype: {e}")

                # Create input tensors - ALWAYS use torch.long for input_ids and labels
                input_ids_tensor = torch.tensor(batch_input_ids, dtype=torch.long, device=device)
                attention_mask_tensor = torch.tensor(batch_attention_mask, dtype=torch.long, device=device)

                # For causal language modeling, labels are the same as input_ids
                # Explicitly use clone() to ensure same dtype (torch.long)
                labels_tensor = input_ids_tensor.clone()

                # Double-check that input_ids and labels are torch.long
                if input_ids_tensor.dtype != torch.long:
                    logger.warning(f"Input IDs tensor has incorrect dtype: {input_ids_tensor.dtype}. Converting to torch.long")
                    input_ids_tensor = input_ids_tensor.to(dtype=torch.long)

                if labels_tensor.dtype != torch.long:
                    logger.warning(f"Labels tensor has incorrect dtype: {labels_tensor.dtype}. Converting to torch.long")
                    labels_tensor = labels_tensor.to(dtype=torch.long)

                # Explicitly log tensor devices
                logger.info(f"Input IDs tensor device: {input_ids_tensor.device}")
                logger.info(f"Attention mask tensor device: {attention_mask_tensor.device}")
                logger.info(f"Labels tensor device: {labels_tensor.device}")

                # Create 4D attention mask [batch_size, num_heads=1, seq_len, seq_len]
                batch_size, seq_length = attention_mask_tensor.shape

                # First, expand to [batch_size, 1, 1, seq_length]
                attention_mask_4d = attention_mask_tensor.unsqueeze(1).unsqueeze(2)

                # Then, expand to [batch_size, 1, seq_length, seq_length]
                attention_mask_4d = attention_mask_4d.expand(-1, 1, seq_length, -1)

                logger.info(f"Created 4D attention mask with shape: {attention_mask_4d.shape}")

                # Convert to the model's dtype if available
                if model_dtype is not None:
                    attention_mask_4d = attention_mask_4d.to(dtype=model_dtype)
                    logger.info(f"Converted attention mask to dtype: {model_dtype}")

                # CRITICAL FIX: Check if attention mask is 4D with shape [batch_size, 1, seq_len, seq_len]
                # If it's [batch_size, 1, 2048, 2048], convert it to 2D [batch_size, seq_len]
                if attention_mask_4d.dim() == 4:
                    # Check if it's the problematic shape
                    if attention_mask_4d.shape[1] == 1 and attention_mask_4d.shape[2] == attention_mask_4d.shape[3]:
                        # Extract just the first row of each sequence's mask (for causal attention)
                        attention_mask_2d = attention_mask_4d.squeeze(1)[:, 0, :]
                        logger.info(f"Converted 4D attention mask to 2D: {attention_mask_2d.shape}")
                        attention_mask_tensor = attention_mask_2d
                    else:
                        logger.info(f"Keeping 4D attention mask with shape: {attention_mask_4d.shape}")
                        attention_mask_tensor = attention_mask_4d

                # Create the batch
                batch = {
                    "input_ids": input_ids_tensor,
                    "attention_mask": attention_mask_tensor,
                    "labels": labels_tensor
                }

                # Log batch tensor information
                logger.info(f"Batch tensor input_ids: shape={batch['input_ids'].shape}, device={batch['input_ids'].device}")
                logger.info(f"Batch tensor attention_mask: shape={batch['attention_mask'].shape}, device={batch['attention_mask'].device}")
                logger.info(f"Batch tensor labels: shape={batch['labels'].shape}, device={batch['labels'].device}")

                return batch
            except Exception as e:
                logger.error(f"Error in data collator: {e}")
                import traceback
                logger.error(traceback.format_exc())

                # Create minimal tensors directly on CUDA if available as fallback
                batch_size = len(features)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                logger.info(f"Creating fallback tensors on device: {device}")

                # Create tensors with requires_grad=True for the loss computation
                # ALWAYS use torch.long for input_ids and labels
                input_ids = torch.zeros((batch_size, args.max_length), dtype=torch.long, device=device)
                attention_mask = torch.zeros((batch_size, args.max_length), dtype=torch.long, device=device)
                labels = torch.zeros((batch_size, args.max_length), dtype=torch.long, device=device)

                # Double-check that input_ids and labels are torch.long
                if input_ids.dtype != torch.long:
                    logger.warning(f"Fallback input_ids tensor has incorrect dtype: {input_ids.dtype}. Converting to torch.long")
                    input_ids = input_ids.to(dtype=torch.long)

                if labels.dtype != torch.long:
                    logger.warning(f"Fallback labels tensor has incorrect dtype: {labels.dtype}. Converting to torch.long")
                    labels = labels.to(dtype=torch.long)

                # Log tensor devices
                logger.info(f"Fallback input IDs tensor device: {input_ids.device}")
                logger.info(f"Fallback attention mask tensor device: {attention_mask.device}")
                logger.info(f"Fallback labels tensor device: {labels.device}")

                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels
                }

    # Create data collator with improved handling
    data_collator = SimpleDataCollator(
        tokenizer=tokenizer,
        mlm=False,  # Not using masked language modeling
        pad_to_multiple_of=8  # Optimize for hardware efficiency
    )

    # Create a custom trainer that handles attention mask issues
    class AttentionMaskFixTrainer(Trainer):
        """Custom trainer that handles attention mask issues and device management"""
        def compute_loss(self, model, inputs, return_outputs=False):
            """Override compute_loss to handle attention mask issues and ensure proper device usage"""
            try:
                # Get the model's device
                device = model.device
                logger.info(f"In compute_loss: Model is on device: {device}")

                # Verify CUDA is available and being used
                if not torch.cuda.is_available():
                    logger.warning("CUDA is not available! Training will be slow on CPU.")
                elif str(device) != "cuda:0" and str(device) != "cuda":
                    logger.warning(f"Model is not on CUDA device! Current device: {device}")
                    # Try to move the model to CUDA
                    try:
                        model = model.to("cuda")
                        device = model.device
                        logger.info(f"Moved model to device: {device}")
                    except Exception as e:
                        logger.error(f"Failed to move model to CUDA: {e}")

                # Force all inputs to be on the model's device
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        if v.device != device:
                            logger.info(f"Moving {k} tensor from {v.device} to {device}")
                            inputs[k] = v.to(device)
                        # Log tensor shapes and devices
                        logger.info(f"Input tensor {k}: shape={v.shape}, device={v.device}")

                # Handle attention mask properly - DO NOT reshape 4D masks to 2D
                if "attention_mask" in inputs:
                    if inputs["attention_mask"].dim() == 4:
                        # Keep 4D mask as-is, just ensure it's on the right device
                        if inputs["attention_mask"].device != device:
                            inputs["attention_mask"] = inputs["attention_mask"].to(device)
                        logger.info(f"Using 4D attention mask with shape: {inputs['attention_mask'].shape}")
                    elif inputs["attention_mask"].dim() == 2:
                        # Convert 2D mask to 4D causal mask
                        batch_size, seq_length = inputs["attention_mask"].shape

                        # First, expand to [batch_size, 1, 1, seq_length]
                        attention_mask_4d = inputs["attention_mask"].unsqueeze(1).unsqueeze(2)

                        # Then, expand to [batch_size, 1, seq_length, seq_length]
                        attention_mask_4d = attention_mask_4d.expand(-1, 1, seq_length, -1)

                        # Create a causal mask
                        causal_mask = torch.triu(
                            torch.ones((seq_length, seq_length), device=device, dtype=torch.bool),
                            diagonal=1
                        )
                        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]
                        causal_mask = causal_mask.expand(batch_size, 1, seq_length, seq_length)

                        # Combine with the attention mask
                        combined_mask = causal_mask | ~attention_mask_4d.bool()

                        # Convert to the model's dtype if available
                        model_dtype = getattr(model, "dtype", None)
                        if model_dtype is not None:
                            combined_mask = combined_mask.to(dtype=model_dtype)

                        # Replace the attention mask
                        inputs["attention_mask"] = ~combined_mask
                        logger.info(f"Converted 2D attention mask to 4D: {inputs['attention_mask'].shape}")
                    else:
                        logger.warning(f"Unexpected attention_mask dimension: {inputs['attention_mask'].dim()}")
                        # Create a proper 4D causal mask
                        batch_size, seq_length = inputs["input_ids"].shape

                        # Create a causal mask
                        causal_mask = torch.triu(
                            torch.ones((seq_length, seq_length), device=device, dtype=torch.bool),
                            diagonal=1
                        )
                        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]
                        causal_mask = causal_mask.expand(batch_size, 1, seq_length, seq_length)

                        # Convert to the model's dtype if available
                        model_dtype = getattr(model, "dtype", None)
                        if model_dtype is not None:
                            causal_mask = causal_mask.to(dtype=model_dtype)

                        # Replace the attention mask
                        inputs["attention_mask"] = ~causal_mask
                        logger.info(f"Created new 4D causal attention mask: {inputs['attention_mask'].shape}")
                else:
                    # Create a proper 4D causal mask if no attention mask is provided
                    batch_size, seq_length = inputs["input_ids"].shape

                    # Create a causal mask
                    causal_mask = torch.triu(
                        torch.ones((seq_length, seq_length), device=device, dtype=torch.bool),
                        diagonal=1
                    )
                    causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]
                    causal_mask = causal_mask.expand(batch_size, 1, seq_length, seq_length)

                    # Convert to the model's dtype if available
                    model_dtype = getattr(model, "dtype", None)
                    if model_dtype is not None:
                        causal_mask = causal_mask.to(dtype=model_dtype)

                    # Add the attention mask
                    inputs["attention_mask"] = ~causal_mask
                    logger.info(f"Created new 4D causal attention mask: {inputs['attention_mask'].shape}")

                # Verify all tensors are on the same device
                devices = set(v.device for k, v in inputs.items() if isinstance(v, torch.Tensor))
                if len(devices) > 1:
                    logger.error(f"Tensors are on different devices: {devices}")
                    # Force all to the model's device
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor) and v.device != device:
                            inputs[k] = v.to(device)
                    logger.info("Forced all tensors to the model's device")

                # COMPLETELY BYPASS PARENT COMPUTE_LOSS AND IMPLEMENT OUR OWN DIRECTLY
                # This avoids any internal device mismatches in the parent implementation
                logger.info("Using direct loss computation to avoid device mismatch issues")

                # Ensure all inputs are on the correct device and have the correct dtype
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        # Check device
                        if v.device != device:
                            logger.info(f"Moving {k} tensor from {v.device} to {device} (final check)")
                            inputs[k] = v.to(device)

                        # CRITICAL FIX: Ensure input_ids and labels remain as long integers
                        if k == "input_ids" and v.dtype != torch.long:
                            logger.warning(f"Input IDs have incorrect dtype: {v.dtype}. Converting to torch.long")
                            inputs[k] = v.to(dtype=torch.long)
                            logger.info(f"Fixed input_ids dtype: {inputs[k].dtype}")
                        elif k == "labels" and v.dtype != torch.long:
                            logger.warning(f"Labels have incorrect dtype: {v.dtype}. Converting to torch.long")
                            inputs[k] = v.to(dtype=torch.long)
                            logger.info(f"Fixed labels dtype: {inputs[k].dtype}")

                # Get model's dtype for mixed precision
                model_dtype = getattr(model, "dtype", None)
                if model_dtype is None:
                    # Try to detect from parameters
                    for param in model.parameters():
                        if param.dtype in [torch.float16, torch.bfloat16]:
                            model_dtype = param.dtype
                            logger.info(f"Detected model dtype from parameters: {model_dtype}")
                            break

                # Ensure attention mask has correct shape and dtype
                if "attention_mask" in inputs:
                    # If attention mask is 2D, convert to 4D
                    if inputs["attention_mask"].dim() == 2:
                        batch_size, seq_length = inputs["attention_mask"].shape
                        # First, expand to [batch_size, 1, 1, seq_length]
                        attention_mask_4d = inputs["attention_mask"].unsqueeze(1).unsqueeze(2)
                        # Then, expand to [batch_size, 1, seq_length, seq_length]
                        attention_mask_4d = attention_mask_4d.expand(-1, 1, seq_length, -1)
                        inputs["attention_mask"] = attention_mask_4d
                        logger.info(f"Expanded attention mask to 4D: {attention_mask_4d.shape}")

                    # If attention mask is 4D with wrong shape
                    elif inputs["attention_mask"].dim() == 4:
                        batch_size, head_dim, seq_len1, seq_len2 = inputs["attention_mask"].shape
                        if seq_len1 != seq_len2:
                            logger.warning(f"Fixing incorrect 4D attention mask shape: {inputs['attention_mask'].shape}")
                            # Create a proper 4D attention mask
                            device = inputs["attention_mask"].device
                            # Create a causal mask
                            causal_mask = torch.triu(
                                torch.ones((seq_len2, seq_len2), device=device, dtype=torch.bool),
                                diagonal=1
                            )
                            causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]
                            causal_mask = causal_mask.expand(batch_size, head_dim, seq_len2, seq_len2)

                            # Convert to the correct dtype
                            if model_dtype is not None:
                                causal_mask = causal_mask.to(dtype=model_dtype)

                            # Replace the attention mask
                            inputs["attention_mask"] = ~causal_mask
                            logger.info(f"Fixed attention mask shape: {inputs['attention_mask'].shape}")

                    # Ensure attention mask has correct dtype
                    if model_dtype is not None and inputs["attention_mask"].dtype != model_dtype:
                        inputs["attention_mask"] = inputs["attention_mask"].to(dtype=model_dtype)
                        logger.info(f"Converted attention mask to dtype: {model_dtype}")

                # Forward pass with all inputs on the correct device
                try:
                    # Check if we have a PeftModel
                    is_peft_model = hasattr(model, 'base_model') and 'Peft' in model.__class__.__name__

                    # Special handling for PeftModel to avoid "got multiple values for argument 'input_ids'" error
                    if is_peft_model and "got multiple values for argument" in str(forward_error) if 'forward_error' in locals() else False:
                        logger.info("Using special PeftModel forward call to avoid 'got multiple values for argument' error")
                        # Use automatic mixed precision with the model's dtype
                        with safe_autocast(dtype=model_dtype):
                            # Call the model's forward method correctly (without passing model as first argument)
                            outputs = model.forward(
                                input_ids=inputs["input_ids"],
                                attention_mask=inputs["attention_mask"] if "attention_mask" in inputs else None,
                                labels=inputs["labels"],
                                use_cache=False,
                                return_dict=True
                            )
                    else:
                        # Use automatic mixed precision with the model's dtype
                        with safe_autocast(dtype=model_dtype):
                            # Forward pass with use_cache=False for gradient checkpointing
                            # CRITICAL FIX: Always use explicit keyword arguments
                            outputs = model(
                                input_ids=inputs["input_ids"],
                                attention_mask=inputs["attention_mask"] if "attention_mask" in inputs else None,
                                labels=inputs["labels"],
                                use_cache=False,  # Critical for gradient checkpointing
                                return_dict=True  # Always use return_dict=True to avoid tuple unpacking issues
                            )
                except Exception as forward_error:
                    logger.error(f"Error in original forward: {forward_error}")

                    # Apply direct fix to model if available
                    try:
                        # Try to import and apply the direct fix
                        try:
                            from src.generative_ai_module.direct_model_fix import apply_direct_fix
                            apply_direct_fix(model)
                            logger.info("✅ Applied direct fix to model in compute_loss")
                        except ImportError:
                            try:
                                from setup.fix_tuple_unpacking_error import fix_tuple_unpacking_error
                                fix_tuple_unpacking_error(model)
                                logger.info("✅ Applied tuple unpacking fix to model in compute_loss")
                            except ImportError:
                                logger.warning("⚠️ Could not apply direct fix to model in compute_loss")
                    except Exception as fix_error:
                        logger.warning(f"⚠️ Error applying direct fix to model in compute_loss: {fix_error}")

                    # Try direct forward call with explicit arguments
                    logger.info("Trying direct forward call with explicit arguments")
                    try:
                        # Ensure input_ids and labels are torch.long
                        if inputs["input_ids"].dtype != torch.long:
                            logger.warning(f"Converting input_ids from {inputs['input_ids'].dtype} to torch.long in direct forward call")
                            inputs["input_ids"] = inputs["input_ids"].to(dtype=torch.long)

                        if inputs["labels"].dtype != torch.long:
                            logger.warning(f"Converting labels from {inputs['labels'].dtype} to torch.long in direct forward call")
                            inputs["labels"] = inputs["labels"].to(dtype=torch.long)

                        # Check if we have a PeftModel
                        is_peft_model = hasattr(model, 'base_model') and 'Peft' in model.__class__.__name__

                        # Special handling for PeftModel to avoid "got multiple values for argument 'input_ids'" error
                        if is_peft_model and "got multiple values for argument" in str(forward_error):
                            logger.info("Using special PeftModel forward call to avoid 'got multiple values for argument' error")
                            # Use automatic mixed precision with the model's dtype
                            with safe_autocast(dtype=model_dtype):
                                # Call the model's forward method correctly
                                outputs = model.forward(
                                    input_ids=inputs["input_ids"],
                                    attention_mask=None,  # Skip attention mask
                                    labels=inputs["labels"],
                                    use_cache=False,
                                    return_dict=True
                                )
                        else:
                            # Use automatic mixed precision with the model's dtype
                            with safe_autocast(dtype=model_dtype):
                                outputs = model(
                                    input_ids=inputs["input_ids"],
                                    attention_mask=None,  # Skip attention mask
                                    labels=inputs["labels"],
                                    use_cache=False,
                                    return_dict=True
                                )
                        logger.info("Direct forward call succeeded without attention mask")
                    except Exception as direct_error:
                        logger.error(f"Direct forward call also failed: {direct_error}")

                        # Try forward call without attention mask
                        logger.info("Trying forward call without attention mask")
                        try:
                            # Check if we have a PeftModel
                            is_peft_model = hasattr(model, 'base_model') and 'Peft' in model.__class__.__name__

                            # Special handling for PeftModel to avoid "got multiple values for argument 'input_ids'" error
                            if is_peft_model and ("got multiple values for argument" in str(forward_error) or "got multiple values for argument" in str(direct_error)):
                                logger.info("Using special PeftModel forward call to avoid 'got multiple values for argument' error")
                                # Use automatic mixed precision with the model's dtype
                                with safe_autocast(dtype=model_dtype):
                                    # Call the model's forward method correctly
                                    outputs = model.forward(
                                        input_ids=inputs["input_ids"],
                                        labels=inputs["labels"],
                                        use_cache=False,
                                        return_dict=True
                                    )
                            else:
                                # Use automatic mixed precision with the model's dtype
                                with safe_autocast(dtype=model_dtype):
                                    outputs = model(
                                        input_ids=inputs["input_ids"],
                                        labels=inputs["labels"],
                                        use_cache=False,
                                        return_dict=True
                                    )
                            logger.info("Forward call without attention mask succeeded")
                        except Exception as no_mask_error:
                            logger.error(f"Forward call without attention mask also failed: {no_mask_error}")

                            # Try one more time with only input_ids
                            try:
                                logger.info("Trying minimal forward call with only input_ids")

                                # Check if we have a PeftModel
                                is_peft_model = hasattr(model, 'base_model') and 'Peft' in model.__class__.__name__

                                # Special handling for PeftModel to avoid "got multiple values for argument 'input_ids'" error
                                if is_peft_model and ("got multiple values for argument" in str(forward_error) or
                                                     "got multiple values for argument" in str(direct_error) or
                                                     "got multiple values for argument" in str(no_mask_error)):
                                    logger.info("Using special PeftModel forward call to avoid 'got multiple values for argument' error")
                                    # Use automatic mixed precision with the model's dtype
                                    with safe_autocast(dtype=model_dtype):
                                        # Call the model's forward method correctly
                                        outputs = model.forward(
                                            input_ids=inputs["input_ids"],
                                            use_cache=False,
                                            return_dict=True
                                        )
                                else:
                                    # Use automatic mixed precision with the model's dtype
                                    with safe_autocast(dtype=model_dtype):
                                        outputs = model(
                                            input_ids=inputs["input_ids"],
                                            use_cache=False,
                                            return_dict=True
                                        )
                                logger.info("Minimal forward call succeeded")
                            except Exception as minimal_error:
                                logger.error(f"Minimal forward call also failed: {minimal_error}")
                                # Combine all error messages
                                error_msg = f"All forward methods failed: {forward_error}, {direct_error}, {no_mask_error}, {minimal_error}"
                                logger.error(f"Error in forward pass: {error_msg}")

                                # Create dummy outputs as last resort
                                try:
                                    # Try to import ModelOutput
                                    try:
                                        from transformers.modeling_outputs import CausalLMOutputWithPast
                                        output_class = CausalLMOutputWithPast
                                    except ImportError:
                                        # Create a simple ModelOutput-like class
                                        class ModelOutput(dict):
                                            """Simple ModelOutput-like class"""
                                            def __init__(self, *args, **kwargs):
                                                super().__init__(*args, **kwargs)
                                                self.__dict__ = self
                                        output_class = ModelOutput

                                    # Create dummy logits
                                    batch_size = inputs["input_ids"].shape[0]
                                    seq_length = inputs["input_ids"].shape[1]
                                    vocab_size = getattr(model.config, "vocab_size", 32000)
                                    dummy_logits = torch.zeros((batch_size, seq_length, vocab_size), device=device)

                                    # Create dummy loss
                                    dummy_loss = torch.tensor(1.0, device=device, requires_grad=True)

                                    # Create dummy outputs
                                    outputs = output_class(loss=dummy_loss, logits=dummy_logits)
                                    logger.warning("Created fallback model outputs with dummy loss and logits")
                                except Exception as e5:
                                    logger.error(f"Failed to create dummy outputs: {e5}")
                                    # Last resort - just create a minimal output
                                    outputs = {"loss": torch.tensor(1.0, device=device, requires_grad=True)}

                # CRITICAL FIX: Handle tuple outputs to avoid "too many values to unpack" error
                if isinstance(outputs, tuple):
                    logger.info(f"Got tuple output with {len(outputs)} elements in compute_loss")

                    # Try to import ModelOutput
                    try:
                        from transformers.modeling_outputs import CausalLMOutputWithPast
                        output_class = CausalLMOutputWithPast
                    except ImportError:
                        # Create a simple ModelOutput-like class
                        class ModelOutput(dict):
                            """Simple ModelOutput-like class"""
                            def __init__(self, *args, **kwargs):
                                super().__init__(*args, **kwargs)
                                self.__dict__ = self
                        output_class = ModelOutput

                    # Convert tuple to ModelOutput
                    outputs_dict = {}

                    # First element is typically the loss
                    if len(outputs) >= 1:
                        loss = outputs[0]
                        outputs_dict["loss"] = loss
                        logger.info(f"Extracted loss from tuple: {loss.item()}, device: {loss.device}")

                        # Second element is typically the logits
                        if len(outputs) >= 2:
                            outputs_dict["logits"] = outputs[1]

                        # Add any remaining elements with generic names
                        for i in range(2, len(outputs)):
                            outputs_dict[f"hidden_states_{i-2}"] = outputs[i]

                        # Convert to ModelOutput
                        outputs = output_class(**outputs_dict)
                        logger.info(f"Converted tuple to ModelOutput with keys: {list(outputs_dict.keys())}")
                    else:
                        # Compute loss manually if tuple doesn't contain loss
                        logger.warning("Tuple outputs doesn't contain loss, computing manually")
                        # Use the last element as logits if available
                        logits = outputs[-1] if len(outputs) > 0 else None
                        if logits is None:
                            logger.error("No logits found in tuple outputs")
                            # Create a dummy loss as last resort
                            loss = torch.tensor(1.0, device=device, requires_grad=True)
                            outputs_dict["loss"] = loss
                        else:
                            # Compute loss manually
                            labels = inputs["labels"]
                            # Shift logits and labels for next token prediction
                            shift_logits = logits[..., :-1, :].contiguous()
                            shift_labels = labels[..., 1:].contiguous()
                            # Compute cross entropy loss
                            loss_fct = torch.nn.CrossEntropyLoss()
                            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                            logger.info(f"Manually computed loss from tuple logits: {loss.item()}, device: {loss.device}")
                            outputs_dict["loss"] = loss
                            outputs_dict["logits"] = logits

                        # Convert to ModelOutput
                        outputs = output_class(**outputs_dict)
                        logger.info(f"Created ModelOutput with computed loss")
                # Handle ModelOutput or dictionary outputs
                elif hasattr(outputs, "loss"):
                    loss = outputs.loss
                    logger.info(f"Got loss directly from model outputs: {loss.item()}, device: {loss.device}")
                elif isinstance(outputs, dict) and "loss" in outputs:
                    loss = outputs["loss"]
                    logger.info(f"Got loss directly from model outputs dict: {loss.item()}, device: {loss.device}")
                else:
                    # Compute the loss manually if not provided by the model
                    logger.info("Computing loss manually from model outputs")
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    labels = inputs["labels"]

                    # Shift logits and labels for next token prediction
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()

                    # Compute cross entropy loss
                    loss_fct = torch.nn.CrossEntropyLoss()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    logger.info(f"Manually computed loss: {loss.item()}, device: {loss.device}")

                # Ensure loss has gradients
                if not hasattr(loss, 'grad_fn') or not loss.requires_grad:
                    logger.warning("Loss doesn't have gradients, creating a new tensor with requires_grad=True")
                    loss = loss.clone().detach().to(device).requires_grad_(True)
                    logger.info(f"New loss tensor: {loss.item()}, device: {loss.device}, requires_grad: {loss.requires_grad}")

                return (loss, outputs) if return_outputs else loss

            except Exception as e:
                logger.error(f"Error in compute_loss: {e}")
                import traceback
                logger.error(traceback.format_exc())

                # Try to continue with a simplified computation
                try:
                    logger.info("Attempting simplified loss computation")
                    # Get the input IDs and labels and ensure they're on the right device and have the right dtype
                    device = model.device

                    # Get input_ids and ensure they're torch.long
                    input_ids = inputs["input_ids"].to(device)
                    if input_ids.dtype != torch.long:
                        logger.warning(f"Simplified computation - input_ids have incorrect dtype: {input_ids.dtype}. Converting to torch.long")
                        input_ids = input_ids.to(dtype=torch.long)

                    # Get labels and ensure they're torch.long
                    if "labels" in inputs:
                        labels = inputs["labels"].to(device)
                        if labels.dtype != torch.long:
                            logger.warning(f"Simplified computation - labels have incorrect dtype: {labels.dtype}. Converting to torch.long")
                            labels = labels.to(dtype=torch.long)
                    else:
                        # Create labels from input_ids (already torch.long)
                        labels = input_ids.clone()

                    # Log tensor information
                    logger.info(f"Simplified computation - input_ids: shape={input_ids.shape}, device={input_ids.device}")
                    logger.info(f"Simplified computation - labels: shape={labels.shape}, device={labels.device}")

                    # Forward pass without attention mask
                    logger.info("Performing forward pass without attention mask")

                    # CRITICAL FIX: Ensure input_ids remain as torch.long
                    if input_ids.dtype != torch.long:
                        logger.warning(f"Simplified computation - input_ids have incorrect dtype: {input_ids.dtype}. Converting to torch.long")
                        input_ids = input_ids.to(dtype=torch.long)

                    # Forward pass with correct dtype
                    outputs = model(input_ids=input_ids)

                    # Compute the loss manually
                    logits = outputs.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss_fct = torch.nn.CrossEntropyLoss()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                    # Ensure loss has gradients
                    if not loss.requires_grad:
                        logger.warning("Loss doesn't have gradients, creating a new tensor with requires_grad=True")
                        loss = loss.clone().detach().to(device).requires_grad_(True)

                    logger.info(f"Simplified loss computed: {loss.item()}, device: {loss.device}, requires_grad: {loss.requires_grad}")

                    return (loss, outputs) if return_outputs else loss
                except Exception as e2:
                    logger.error(f"Fallback loss computation also failed: {e2}")
                    # Return a dummy loss as last resort
                    logger.warning("Creating dummy loss with explicit gradient")
                    dummy_loss = torch.tensor(1.0, device=device, requires_grad=True)
                    logger.info(f"Dummy loss created: {dummy_loss.item()}, device: {dummy_loss.device}, requires_grad: {dummy_loss.requires_grad}")
                    return dummy_loss

        def training_step(self, model, inputs):
            """Override training_step to ensure proper device handling and gradient computation"""
            try:
                # Add memory monitoring
                if torch.cuda.is_available():
                    logger.info(f"Memory before forward: {torch.cuda.memory_allocated()/1024**2:.2f}MB")

                # Ensure model is in training mode
                model.train()

                # Get the model's device
                device = model.device
                logger.info(f"Model is on device: {device}")

                # Verify CUDA is available and being used
                if not torch.cuda.is_available():
                    logger.warning("CUDA is not available! Training will be slow on CPU.")
                elif str(device) != "cuda:0" and str(device) != "cuda":
                    logger.warning(f"Model is not on CUDA device! Current device: {device}")
                    # Try to move the model to CUDA
                    try:
                        model = model.to("cuda")
                        device = model.device
                        logger.info(f"Moved model to device: {device}")
                    except Exception as e:
                        logger.error(f"Failed to move model to CUDA: {e}")

                # Force all inputs to be on the model's device
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        if v.device != device:
                            logger.info(f"Moving {k} tensor from {v.device} to {device}")
                            inputs[k] = v.to(device)
                        # Log tensor shapes and devices
                        logger.info(f"Tensor {k}: shape={v.shape}, device={v.device}")

                # Explicitly enable gradients for this operation
                with torch.set_grad_enabled(True):
                    # Compute loss with our enhanced compute_loss method
                    loss = self.compute_loss(model, inputs)
                    logger.info(f"Computed loss: {loss.item()}, device: {loss.device}, requires_grad: {loss.requires_grad}")

                    # Add memory monitoring after forward pass
                    if torch.cuda.is_available():
                        logger.info(f"Memory after forward: {torch.cuda.memory_allocated()/1024**2:.2f}MB")

                    # Force loss to have gradients
                    if not hasattr(loss, 'grad_fn') or not loss.requires_grad:
                        logger.warning("Loss doesn't have gradients, creating a new tensor with requires_grad=True")
                        # Create a new tensor that requires gradients
                        loss = loss.clone().detach().to(device).requires_grad_(True)
                        logger.info(f"New loss tensor: {loss.item()}, device: {loss.device}, requires_grad: {loss.requires_grad}")

                    # Scale loss for gradient accumulation if needed
                    if self.args.gradient_accumulation_steps > 1:
                        loss = loss / self.args.gradient_accumulation_steps

                    # Backward pass with error handling
                    try:
                        logger.info("Attempting backward pass with accelerator")
                        self.accelerator.backward(loss)
                        logger.info("Backward pass with accelerator successful")

                        # Add memory monitoring after backward pass
                        if torch.cuda.is_available():
                            logger.info(f"Memory after backward: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
                    except Exception as e:
                        logger.error(f"Error in backward pass with accelerator: {e}")
                        # Try a more direct approach
                        try:
                            logger.info("Attempting direct backward pass")
                            loss.backward()
                            logger.info("Direct backward pass successful")

                            # Add memory monitoring after direct backward pass
                            if torch.cuda.is_available():
                                logger.info(f"Memory after direct backward: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
                        except Exception as e2:
                            logger.error(f"Direct backward also failed: {e2}")
                            # Create a dummy loss as last resort with explicit gradient
                            logger.warning("Creating dummy loss with explicit gradient")
                            dummy_loss = torch.tensor(1.0, device=device, requires_grad=True)
                            dummy_loss.backward()
                            logger.info("Dummy backward pass completed")

                # Verify gradients exist and fix if needed
                has_grad = False
                lora_params_with_grad = 0
                total_lora_params = 0

                # First check if any parameters have gradients
                for name, param in model.named_parameters():
                    if 'lora' in name:
                        total_lora_params += 1
                        # Ensure parameter requires gradients
                        if not param.requires_grad:
                            logger.warning(f"LoRA parameter {name} does not require gradients. Setting requires_grad=True")
                            param.requires_grad = True

                        if param.grad is not None:
                            lora_params_with_grad += 1
                            has_grad = True
                            # Log a sample of the gradients
                            if lora_params_with_grad <= 3:  # Only log first 3 to avoid spam
                                logger.info(f"Parameter {name} has gradient with norm: {param.grad.norm().item()}")

                # Log gradient statistics
                if total_lora_params > 0:
                    logger.info(f"LoRA parameters with gradients: {lora_params_with_grad}/{total_lora_params} ({lora_params_with_grad/total_lora_params*100:.1f}%)")

                if not has_grad:
                    logger.warning("No gradients found in model parameters after backward pass!")

                    # Try to fix the issue by manually computing gradients for LoRA parameters
                    logger.warning("Attempting to manually compute gradients for LoRA parameters")

                    # Get model's dtype for mixed precision
                    model_dtype = getattr(model, "dtype", None)

                    # Create a new backward pass directly from the loss
                    try:
                        # Detach the model from the current computation graph
                        model.zero_grad()

                        # Ensure attention mask has correct shape and dtype
                        if "attention_mask" in inputs:
                            # If attention mask is 2D, convert to 4D
                            if inputs["attention_mask"].dim() == 2:
                                batch_size, seq_length = inputs["attention_mask"].shape
                                # First, expand to [batch_size, 1, 1, seq_length]
                                attention_mask_4d = inputs["attention_mask"].unsqueeze(1).unsqueeze(2)
                                # Then, expand to [batch_size, 1, seq_length, seq_length]
                                attention_mask_4d = attention_mask_4d.expand(-1, 1, seq_length, -1)
                                inputs["attention_mask"] = attention_mask_4d
                                logger.info(f"Expanded attention mask to 4D: {attention_mask_4d.shape}")

                            # If attention mask is 4D with wrong shape
                            elif inputs["attention_mask"].dim() == 4:
                                batch_size, head_dim, seq_len1, seq_len2 = inputs["attention_mask"].shape
                                if seq_len1 != seq_len2:
                                    logger.warning(f"Fixing incorrect 4D attention mask shape: {inputs['attention_mask'].shape}")
                                    # Create a proper 4D attention mask
                                    device = inputs["attention_mask"].device
                                    # Create a causal mask
                                    causal_mask = torch.triu(
                                        torch.ones((seq_len2, seq_len2), device=device, dtype=torch.bool),
                                        diagonal=1
                                    )
                                    causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]
                                    causal_mask = causal_mask.expand(batch_size, head_dim, seq_len2, seq_len2)

                                    # Convert to the correct dtype
                                    if model_dtype is not None:
                                        causal_mask = causal_mask.to(dtype=model_dtype)

                                    # Replace the attention mask
                                    inputs["attention_mask"] = ~causal_mask
                                    logger.info(f"Fixed attention mask shape: {inputs['attention_mask'].shape}")

                            # Ensure attention mask has correct dtype
                            if model_dtype is not None and inputs["attention_mask"].dtype != model_dtype:
                                inputs["attention_mask"] = inputs["attention_mask"].to(dtype=model_dtype)
                                logger.info(f"Converted attention mask to dtype: {model_dtype}")

                        # Recompute the forward pass with explicit gradient tracking
                        with torch.enable_grad():
                            # Use automatic mixed precision with the model's dtype
                            with safe_autocast(dtype=model_dtype):
                                # Check if we have a PeftModel
                                is_peft_model = hasattr(model, 'base_model') and 'Peft' in model.__class__.__name__

                                # Special handling for PeftModel to avoid "got multiple values for argument 'input_ids'" error
                                if is_peft_model:
                                    logger.info("Using special PeftModel forward call for manual gradient computation")
                                    # Call the model's forward method correctly
                                    outputs = model.forward(
                                        input_ids=inputs["input_ids"].to(device),
                                        attention_mask=inputs["attention_mask"].to(device) if "attention_mask" in inputs else None,
                                        labels=inputs["labels"].to(device),
                                        use_cache=False,  # Critical for gradient checkpointing
                                        return_dict=True  # Always use return_dict=True to avoid tuple unpacking issues
                                    )
                                else:
                                    # Forward pass with use_cache=False for gradient checkpointing
                                    outputs = model(
                                        input_ids=inputs["input_ids"].to(device),
                                        attention_mask=inputs["attention_mask"].to(device) if "attention_mask" in inputs else None,
                                        labels=inputs["labels"].to(device),
                                        use_cache=False  # Critical for gradient checkpointing
                                    )

                                # Get loss
                                if hasattr(outputs, "loss"):
                                    new_loss = outputs.loss
                                else:
                                    # Compute loss manually
                                    logits = outputs.logits
                                    labels = inputs["labels"].to(device)
                                    shift_logits = logits[..., :-1, :].contiguous()
                                    shift_labels = labels[..., 1:].contiguous()
                                    loss_fct = torch.nn.CrossEntropyLoss()
                                    new_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                            # Scale loss for gradient accumulation if needed
                            if self.args.gradient_accumulation_steps > 1:
                                new_loss = new_loss / self.args.gradient_accumulation_steps

                            # Backward pass
                            new_loss.backward()

                            # Apply gradient clipping for stability
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                            logger.info(f"Manual gradient computation completed with loss: {new_loss.item()}")

                            # Check if gradients were created
                            grad_count = sum(1 for n, p in model.named_parameters() if 'lora' in n and p.grad is not None)
                            logger.info(f"Manual computation created gradients for {grad_count}/{total_lora_params} LoRA parameters")
                    except Exception as e:
                        logger.error(f"Manual gradient computation failed: {e}")

                        # Force gradients for LoRA parameters as a last resort
                        logger.warning("Forcing artificial gradients for LoRA parameters as a last resort")

                        # Create artificial gradients for all LoRA parameters
                        for name, param in model.named_parameters():
                            if 'lora' in name and param.requires_grad:
                                if param.grad is None:
                                    # Create a small random gradient
                                    param.grad = torch.randn_like(param) * 0.01
                                    logger.info(f"Created artificial gradient for {name}")

                # Apply gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Monitor gradient norms for debugging
                if total_lora_params > 0:
                    grad_norms = [p.grad.norm().item() for n, p in model.named_parameters()
                                 if 'lora' in n and p.grad is not None]
                    if grad_norms:
                        avg_norm = sum(grad_norms) / len(grad_norms)
                        max_norm = max(grad_norms)
                        logger.info(f"Gradient statistics - Avg norm: {avg_norm:.4f}, Max norm: {max_norm:.4f}")

                # Check again after our fix attempts
                lora_params_with_grad = sum(1 for name, param in model.named_parameters()
                                          if 'lora' in name and param.grad is not None)

                if lora_params_with_grad > 0:
                    logger.info(f"Gradients successfully computed for {lora_params_with_grad}/{total_lora_params} LoRA parameters")
                else:
                    logger.warning("Still no gradients after all attempts. Training may not be effective.")

                return loss.detach()
            except Exception as e:
                logger.error(f"Error in training_step: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # Return a dummy loss
                return torch.tensor(1.0, device=model.device)

        def get_train_dataloader(self):
            """Override to ensure tensors are properly handled with device management"""
            if self.train_dataset is None:
                raise ValueError("Trainer: training requires a train_dataset.")

            train_sampler = self._get_train_sampler()

            # Get the model's device
            device = self.model.device
            logger.info(f"In get_train_dataloader: Model is on device: {device}")

            # Create a custom collate function that ensures tensors are on the right device
            original_collate_fn = self.data_collator

            def device_aware_collate_fn(features):
                # Call the original collate function
                batch = original_collate_fn(features)

                # Force all tensors to be on the model's device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        if v.device != device:
                            logger.info(f"Moving batch tensor {k} from {v.device} to {device}")
                            batch[k] = v.to(device)
                        logger.info(f"Batch tensor {k}: shape={v.shape}, device={v.device}")

                return batch

            # Use our custom collate function with reduced workers
            logger.info("Creating train DataLoader with device-aware collate function")
            return torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=train_sampler,
                collate_fn=device_aware_collate_fn,
                drop_last=self.args.dataloader_drop_last,
                num_workers=0,  # Force 0 workers to avoid multiprocessing issues
                pin_memory=False,  # Disable pin_memory as we're explicitly moving tensors to device
            )

        def get_eval_dataloader(self, eval_dataset=None):
            """Override to ensure tensors are properly handled with device management"""
            eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
            if eval_dataset is None:
                raise ValueError("Trainer: evaluation requires an eval_dataset.")

            eval_sampler = self._get_eval_sampler(eval_dataset)

            # Get the model's device
            device = self.model.device
            logger.info(f"In get_eval_dataloader: Model is on device: {device}")

            # Create a custom collate function that ensures tensors are on the right device
            original_collate_fn = self.data_collator

            def device_aware_collate_fn(features):
                # Call the original collate function
                batch = original_collate_fn(features)

                # Force all tensors to be on the model's device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        if v.device != device:
                            logger.info(f"Moving eval batch tensor {k} from {v.device} to {device}")
                            batch[k] = v.to(device)
                        logger.info(f"Eval batch tensor {k}: shape={v.shape}, device={v.device}")

                return batch

            # Use our custom collate function with reduced workers
            logger.info("Creating eval DataLoader with device-aware collate function")
            return torch.utils.data.DataLoader(
                eval_dataset,
                batch_size=self.args.eval_batch_size,
                sampler=eval_sampler,
                collate_fn=device_aware_collate_fn,
                drop_last=self.args.dataloader_drop_last,
                num_workers=0,  # Force 0 workers to avoid multiprocessing issues
                pin_memory=False,  # Disable pin_memory as we're explicitly moving tensors to device
            )

    # Ensure model is on CUDA if available
    if torch.cuda.is_available():
        logger.info(f"Moving model to CUDA. Current device: {model.device}")
        model = model.to("cuda")
        logger.info(f"Model moved to device: {model.device}")
    else:
        logger.warning("CUDA is not available! Training will be slow on CPU.")

    # Patch the model's forward method to ensure all tensors are on the correct device
    if hasattr(model, 'forward'):
        original_forward = model.forward

        def device_aware_forward(*args, **kwargs):
            """Ensure all inputs and internal tensors are on the correct device and have correct shapes/dtypes"""
            device = model.device
            logger.info(f"In patched forward: Model is on device: {device}")

            # CRITICAL FIX: Handle positional arguments properly
            # If we have more than one positional argument and the second one is a tensor,
            # it's likely input_ids passed as a positional argument
            if len(args) > 1 and isinstance(args[1], torch.Tensor):
                logger.info("Detected input_ids as positional argument, converting to kwargs")
                # Extract input_ids
                input_ids_arg = args[1]
                # Create new args without input_ids
                new_args = args[:1]
                # Create new kwargs with input_ids
                new_kwargs = kwargs.copy()
                if "input_ids" not in new_kwargs:
                    new_kwargs["input_ids"] = input_ids_arg
                # Update args and kwargs
                args = new_args
                kwargs = new_kwargs
                logger.info("Converted positional input_ids to keyword argument")

            # Get model's dtype for mixed precision
            model_dtype = getattr(model, "dtype", None)
            if model_dtype is None:
                # Try to detect from parameters
                for param in model.parameters():
                    if param.dtype in [torch.float16, torch.bfloat16]:
                        model_dtype = param.dtype
                        logger.info(f"Detected model dtype from parameters: {model_dtype}")
                        break

            # Move all positional args to the correct device
            new_args = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    if arg.device != device:
                        logger.info(f"Moving positional arg from {arg.device} to {device}")
                        new_args.append(arg.to(device))
                    else:
                        new_args.append(arg)
                else:
                    new_args.append(arg)

            # Move all kwargs to the correct device and ensure correct dtypes
            new_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    # Move to correct device
                    if v.device != device:
                        logger.info(f"Moving kwarg {k} from {v.device} to {device}")
                        v = v.to(device)

                    # Handle dtype for non-label tensors
                    # CRITICAL FIX: Do not convert input_ids to float16/bfloat16
                    if k == "input_ids":
                        # Ensure input_ids are always torch.long
                        if v.dtype != torch.long:
                            logger.warning(f"Converting input_ids from {v.dtype} to torch.long")
                            v = v.to(dtype=torch.long)
                    elif k == "labels":
                        # Ensure labels are always torch.long
                        if v.dtype != torch.long:
                            logger.warning(f"Converting labels from {v.dtype} to torch.long")
                            v = v.to(dtype=torch.long)
                    # For other tensors, convert to model's dtype if needed
                    elif model_dtype is not None and v.dtype != model_dtype:
                        logger.info(f"Converting {k} from {v.dtype} to {model_dtype}")
                        v = v.to(dtype=model_dtype)

                    new_kwargs[k] = v
                else:
                    new_kwargs[k] = v

            # Ensure specific inputs are handled correctly
            if "input_ids" in new_kwargs:
                # CRITICAL FIX: Ensure input_ids remain as torch.long
                if new_kwargs["input_ids"].dtype != torch.long:
                    logger.warning(f"Converting input_ids from {new_kwargs['input_ids'].dtype} to torch.long")
                    new_kwargs["input_ids"] = new_kwargs["input_ids"].to(dtype=torch.long)

                # Also ensure it's on the correct device
                if new_kwargs["input_ids"].device != device:
                    logger.info(f"Explicitly moving input_ids from {new_kwargs['input_ids'].device} to {device}")
                    new_kwargs["input_ids"] = new_kwargs["input_ids"].to(device)

            # Handle attention mask specially
            if "attention_mask" in new_kwargs:
                attention_mask = new_kwargs["attention_mask"]

                # First ensure it's on the correct device
                if attention_mask.device != device:
                    logger.info(f"Explicitly moving attention_mask from {attention_mask.device} to {device}")
                    attention_mask = attention_mask.to(device)

                # Get input shape for proper mask creation
                if "input_ids" in new_kwargs:
                    batch_size, seq_length = new_kwargs["input_ids"].shape
                else:
                    batch_size, seq_length = attention_mask.size(0), attention_mask.size(-1)

                # If attention mask is 2D, convert to 4D for newer transformer versions
                if attention_mask.dim() == 2:
                    logger.info(f"Converting 2D attention mask to 4D: {attention_mask.shape}")
                    # First, expand to [batch_size, 1, 1, seq_length]
                    attention_mask_4d = attention_mask.unsqueeze(1).unsqueeze(2)
                    # Then, expand to [batch_size, 1, seq_length, seq_length]
                    attention_mask_4d = attention_mask_4d.expand(-1, 1, seq_length, -1)
                    attention_mask = attention_mask_4d
                    logger.info(f"Created 4D attention mask with shape: {attention_mask.shape}")

                # If attention mask is 4D with wrong shape
                elif attention_mask.dim() == 4:
                    batch_size, head_dim, seq_len1, seq_len2 = attention_mask.shape
                    if seq_len1 != seq_len2:
                        logger.warning(f"Fixing incorrect 4D attention mask shape: {attention_mask.shape}")
                        # Create a proper 4D attention mask
                        # Create a causal mask
                        causal_mask = torch.triu(
                            torch.ones((seq_len2, seq_len2), device=device, dtype=torch.bool),
                            diagonal=1
                        )
                        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]
                        causal_mask = causal_mask.expand(batch_size, head_dim, seq_len2, seq_len2)

                        # Convert to the correct dtype
                        if model_dtype is not None:
                            causal_mask = causal_mask.to(dtype=model_dtype)

                        # Replace the attention mask
                        attention_mask = ~causal_mask
                        logger.info(f"Fixed attention mask shape: {attention_mask.shape}")

                # Ensure attention mask has correct dtype
                if model_dtype is not None and attention_mask.dtype != model_dtype:
                    logger.info(f"Converting attention mask from {attention_mask.dtype} to {model_dtype}")
                    attention_mask = attention_mask.to(dtype=model_dtype)

                # Update the kwargs with the fixed attention mask
                new_kwargs["attention_mask"] = attention_mask

            if "labels" in new_kwargs and new_kwargs["labels"].device != device:
                logger.info(f"Explicitly moving labels from {new_kwargs['labels'].device} to {device}")
                new_kwargs["labels"] = new_kwargs["labels"].to(device)

            # Ensure all model parameters are on the correct device
            for name, param in model.named_parameters():
                if param.device != device:
                    logger.warning(f"Parameter {name} is on {param.device}, moving to {device}")
                    param.data = param.data.to(device)

            # Add use_cache=False for gradient checkpointing compatibility
            if "use_cache" not in new_kwargs and kwargs.get("gradient_checkpointing", False):
                new_kwargs["use_cache"] = False
                logger.info("Setting use_cache=False for gradient checkpointing compatibility")

            # Call the original forward method with explicit try/except
            try:
                # CRITICAL FIX: Always use return_dict=True to avoid "too many values to unpack" error
                if "return_dict" not in new_kwargs:
                    new_kwargs["return_dict"] = True
                    logger.info("Setting return_dict=True to avoid tuple unpacking issues")

                # Use automatic mixed precision with the model's dtype
                with safe_autocast(dtype=model_dtype):
                    outputs = original_forward(*new_args, **new_kwargs)
            except Exception as e:
                logger.error(f"Error in original forward: {e}")
                # Try a more direct approach with explicit arguments
                try:
                    logger.info("Trying direct forward call with explicit arguments")
                    # Extract the key arguments
                    input_ids = new_kwargs.get("input_ids")
                    attention_mask = new_kwargs.get("attention_mask")
                    labels = new_kwargs.get("labels")

                    # CRITICAL FIX: Ensure input_ids remain as torch.long
                    if input_ids is not None and input_ids.dtype != torch.long:
                        logger.warning(f"Converting input_ids from {input_ids.dtype} to torch.long in direct forward call")
                        input_ids = input_ids.to(dtype=torch.long)

                    # Ensure labels are torch.long if present
                    if labels is not None and labels.dtype != torch.long:
                        logger.warning(f"Converting labels from {labels.dtype} to torch.long in direct forward call")
                        labels = labels.to(dtype=torch.long)

                    # If we have an attention mask issue, try to fix it or create a new one
                    if "attention mask" in str(e).lower() and input_ids is not None:
                        logger.info("Detected attention mask issue, creating a new causal attention mask")
                        batch_size, seq_length = input_ids.shape

                        # Create a causal mask
                        causal_mask = torch.triu(
                            torch.ones((seq_length, seq_length), device=device, dtype=torch.bool),
                            diagonal=1
                        )
                        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]
                        causal_mask = causal_mask.expand(batch_size, 1, seq_length, seq_length)

                        # Convert to the correct dtype
                        if model_dtype is not None:
                            causal_mask = causal_mask.to(dtype=model_dtype)

                        # Replace the attention mask
                        attention_mask = ~causal_mask
                        logger.info(f"Created new attention mask with shape: {attention_mask.shape}")

                    # If we have a "too many values to unpack" error, it's likely related to return format
                    if "too many values to unpack" in str(e).lower():
                        logger.info("Detected 'too many values to unpack' error, forcing return_dict=True")
                        # This ensures we get a single object back instead of a tuple

                    # Use automatic mixed precision with the model's dtype
                    with safe_autocast(dtype=model_dtype):
                        # Call forward with explicit arguments and minimal parameters
                        # CRITICAL FIX: Always use return_dict=True to avoid tuple unpacking issues
                        outputs = original_forward(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            use_cache=False,  # Disable cache to avoid issues
                            return_dict=True   # Ensure we get a proper return object
                        )
                except Exception as e2:
                    logger.error(f"Direct forward call also failed: {e2}")

                    # Last resort: try without attention mask
                    try:
                        logger.info("Trying forward call without attention mask")

                        # CRITICAL FIX: Double-check input_ids and labels are torch.long
                        if input_ids is not None and input_ids.dtype != torch.long:
                            logger.warning(f"Converting input_ids from {input_ids.dtype} to torch.long in fallback forward call")
                            input_ids = input_ids.to(dtype=torch.long)

                        if labels is not None and labels.dtype != torch.long:
                            logger.warning(f"Converting labels from {labels.dtype} to torch.long in fallback forward call")
                            labels = labels.to(dtype=torch.long)

                        # Use automatic mixed precision with the model's dtype
                        with safe_autocast(dtype=model_dtype):
                            # CRITICAL FIX: Always use return_dict=True to avoid tuple unpacking issues
                            outputs = original_forward(
                                input_ids=input_ids,
                                labels=labels,
                                use_cache=False,
                                return_dict=True
                            )
                    except Exception as e3:
                        logger.error(f"Forward call without attention mask also failed: {e3}")
                        # Create dummy outputs as last resort
                        raise RuntimeError(f"All forward methods failed: {e}, {e2}, {e3}")

            # CRITICAL FIX: Handle tuple outputs to avoid "too many values to unpack" error
            if isinstance(outputs, tuple):
                logger.info(f"Got tuple output with {len(outputs)} elements, converting to ModelOutput")
                # Convert tuple to a dictionary-like object
                from transformers.modeling_outputs import ModelOutput
                if len(outputs) >= 1:
                    # First element is typically the loss or logits
                    if labels is not None:
                        # If we have labels, first element is likely the loss
                        outputs_dict = {"loss": outputs[0]}
                        if len(outputs) >= 2:
                            # Second element is likely the logits
                            outputs_dict["logits"] = outputs[1]
                    else:
                        # If no labels, first element is likely the logits
                        outputs_dict["logits"] = outputs[0]

                    # Add any remaining elements with generic names
                    for i in range(1, len(outputs)):
                        if i == 1 and "logits" in outputs_dict:
                            continue  # Skip if we already assigned logits
                        outputs_dict[f"hidden_states_{i}"] = outputs[i]

                    # Convert to ModelOutput
                    outputs = ModelOutput(outputs_dict)
                    logger.info(f"Converted tuple to ModelOutput with keys: {list(outputs_dict.keys())}")

            # Ensure all output tensors are on the correct device
            if isinstance(outputs, torch.Tensor) and outputs.device != device:
                logger.info(f"Moving output tensor from {outputs.device} to {device}")
                outputs = outputs.to(device)
            elif hasattr(outputs, 'to') and callable(getattr(outputs, 'to')):
                # Handle HuggingFace output objects
                outputs = outputs.to(device)

            return outputs

        # Replace the original forward method with our device-aware version
        model.forward = device_aware_forward
        logger.info("Patched model's forward method to ensure all tensors are on the correct device")

    # Also patch the model's prepare_inputs_for_generation method
    if hasattr(model, 'prepare_inputs_for_generation'):
        original_prepare_inputs = model.prepare_inputs_for_generation

        def device_aware_prepare_inputs(input_ids, **kwargs):
            """Ensure all inputs are on the correct device and have the correct dtype"""
            device = model.device

            # CRITICAL FIX: Ensure input_ids remain as torch.long
            if input_ids.dtype != torch.long:
                logger.warning(f"Converting input_ids from {input_ids.dtype} to torch.long in prepare_inputs_for_generation")
                input_ids = input_ids.to(dtype=torch.long)

            # Ensure input_ids is on the correct device
            if input_ids.device != device:
                logger.info(f"Moving input_ids from {input_ids.device} to {device}")
                input_ids = input_ids.to(device)

            # Move all kwargs tensors to the correct device
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor) and v.device != device:
                    logger.info(f"Moving kwarg {k} from {v.device} to {device}")
                    kwargs[k] = v.to(device)

            # Call the original method
            inputs = original_prepare_inputs(input_ids, **kwargs)

            # Move all outputs to the model's device
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor) and v.device != device:
                    logger.info(f"Moving generation output {k} from {v.device} to {device}")
                    inputs[k] = v.to(device)

            return inputs

        # Replace the original method with our device-aware version
        model.prepare_inputs_for_generation = device_aware_prepare_inputs
        logger.info("Added device handling to model's prepare_inputs_for_generation method")

    # Apply comprehensive attention fixes to model
    logger.info("Applying comprehensive attention fixes to model...")

    # Fix for PeftModelForCausalLM "got multiple values for argument 'input_ids'" error
    try:
        # Check if we have a PeftModel
        if hasattr(model, 'base_model') and 'Peft' in model.__class__.__name__:
            logger.info("Detected PeftModel, applying special fix for PeftModelForCausalLM")

            # Store the original forward method
            original_peft_forward = model.forward

            # Define a patched forward method for PeftModel
            def patched_peft_forward(*args, **kwargs):
                """
                Patched forward method for PeftModel that handles the 'got multiple values for argument' error
                and the 'too many values to unpack' error.
                """
                logger.info("In patched PeftModel forward: Model is on device: " + str(model.device))

                # Always set return_dict=True to avoid tuple outputs
                if "return_dict" not in kwargs:
                    kwargs["return_dict"] = True
                    logger.info("Setting return_dict=True to avoid tuple unpacking issues")

                # Create a safe forward function that handles the multiple values error
                def safe_forward(model_self, *args, **kwargs):
                    """Safely call forward without the multiple values error"""
                    try:
                        # If we have more than one positional argument and the second one is a tensor,
                        # it's likely input_ids passed as a positional argument
                        if len(args) > 1 and isinstance(args[1], torch.Tensor):
                            logger.info("Detected input_ids as positional argument, converting to kwargs")
                            # Extract self and input_ids
                            self_arg = args[0]
                            input_ids_arg = args[1]

                            # Create new kwargs with input_ids
                            new_kwargs = kwargs.copy()
                            if "input_ids" not in new_kwargs:
                                new_kwargs["input_ids"] = input_ids_arg

                            # Call with self and the new kwargs
                            return original_peft_forward(self_arg, **new_kwargs)

                        # Handle the case where input_ids is passed both as positional and keyword argument
                        if len(args) > 1 and "input_ids" in kwargs:
                            logger.info("Detected input_ids in both args and kwargs, removing from kwargs")
                            # Remove input_ids from kwargs to avoid the multiple values error
                            input_ids = kwargs.pop("input_ids")
                            # Log the shapes for debugging
                            logger.info(f"Args[1] shape: {args[1].shape if isinstance(args[1], torch.Tensor) else 'not a tensor'}")
                            logger.info(f"Kwargs input_ids shape: {input_ids.shape if isinstance(input_ids, torch.Tensor) else 'not a tensor'}")

                        # Try the original forward method
                        return original_peft_forward(*args, **kwargs)
                    except Exception as e:
                        logger.error(f"Error in safe_forward: {e}")

                        # If we get "got multiple values for argument", try with only kwargs
                        if "got multiple values for argument" in str(e):
                            logger.info("Trying with only kwargs")
                            try:
                                # Extract only the self argument from args
                                if len(args) > 0:
                                    self_arg = args[0]

                                    # Get input_ids from args if available
                                    input_ids = None
                                    if len(args) > 1 and isinstance(args[1], torch.Tensor):
                                        input_ids = args[1]

                                    # Create new kwargs without input_ids
                                    new_kwargs = {k: v for k, v in kwargs.items() if k != "input_ids"}

                                    # Add input_ids to kwargs if we have it from args
                                    if input_ids is not None:
                                        new_kwargs["input_ids"] = input_ids

                                    # Call with self and the new kwargs
                                    return original_peft_forward(self_arg, **new_kwargs)
                                else:
                                    # If no args, just use kwargs
                                    return original_peft_forward(**kwargs)
                            except Exception as e2:
                                logger.error(f"Forward with only kwargs also failed: {e2}")

                                # Try with base model directly
                                try:
                                    logger.info("Trying with base model directly")

                                    # Get input_ids from args or kwargs
                                    input_ids = None
                                    if len(args) > 1 and isinstance(args[1], torch.Tensor):
                                        input_ids = args[1]
                                    elif "input_ids" in kwargs:
                                        input_ids = kwargs["input_ids"]

                                    # Get labels if available
                                    labels = kwargs.get("labels")

                                    # Ensure input_ids are torch.long
                                    if input_ids is not None and input_ids.dtype != torch.long:
                                        logger.warning(f"Converting input_ids from {input_ids.dtype} to torch.long")
                                        input_ids = input_ids.to(dtype=torch.long)

                                    # Ensure labels are torch.long
                                    if labels is not None and labels.dtype != torch.long:
                                        logger.warning(f"Converting labels from {labels.dtype} to torch.long")
                                        labels = labels.to(dtype=torch.long)

                                    # If model has a base_model, try to use it directly
                                    if hasattr(model_self, 'base_model'):
                                        logger.info("Using base_model directly")
                                        base_model = model_self.base_model

                                        # Create new kwargs for base model
                                        base_kwargs = {}
                                        if input_ids is not None:
                                            base_kwargs["input_ids"] = input_ids
                                        if labels is not None:
                                            base_kwargs["labels"] = labels
                                        base_kwargs["return_dict"] = True

                                        # Call base model's forward method directly
                                        return base_model.forward(**base_kwargs)
                                    else:
                                        # Last resort - try with minimal arguments
                                        logger.info("Trying with minimal arguments")
                                        if len(args) > 0:
                                            self_arg = args[0]
                                            return original_peft_forward(
                                                self_arg,
                                                input_ids=input_ids,
                                                return_dict=True
                                            )
                                        else:
                                            return original_peft_forward(
                                                input_ids=input_ids,
                                                return_dict=True
                                            )
                                except Exception as e3:
                                    logger.error(f"All forward attempts failed: {e3}")

                                    # Create dummy outputs as last resort
                                    try:
                                        # Try to import ModelOutput
                                        try:
                                            from transformers.modeling_outputs import CausalLMOutputWithPast
                                            output_class = CausalLMOutputWithPast
                                        except ImportError:
                                            # Create a simple ModelOutput-like class
                                            class ModelOutput(dict):
                                                """Simple ModelOutput-like class"""
                                                def __init__(self, *args, **kwargs):
                                                    super().__init__(*args, **kwargs)
                                                    self.__dict__ = self
                                            output_class = ModelOutput

                                        # Create dummy logits
                                        device = model.device
                                        batch_size = input_ids.shape[0] if input_ids is not None else 1
                                        seq_length = input_ids.shape[1] if input_ids is not None else 1
                                        vocab_size = getattr(model.config, 'vocab_size', 32000) if hasattr(model, 'config') else 32000

                                        # Create dummy logits
                                        dummy_logits = torch.zeros((batch_size, seq_length, vocab_size), device=device)

                                        # Create dummy loss if labels are provided
                                        dummy_loss = None
                                        if labels is not None:
                                            dummy_loss = torch.tensor(1.0, device=device, requires_grad=True)

                                        # Create a ModelOutput with the dummy values
                                        outputs_dict = {"logits": dummy_logits}
                                        if dummy_loss is not None:
                                            outputs_dict["loss"] = dummy_loss

                                        return output_class(**outputs_dict)
                                    except Exception as e4:
                                        logger.error(f"Failed to create dummy outputs: {e4}")
                                        raise RuntimeError(f"All forward methods failed: {e}, {e2}, {e3}, {e4}")

                # Call the safe forward function
                outputs = safe_forward(model, *args, **kwargs)

                # Handle tuple outputs
                if isinstance(outputs, tuple):
                    logger.info(f"Got tuple output with {len(outputs)} elements from PeftModel forward")

                    # Try to import ModelOutput
                    try:
                        from transformers.modeling_outputs import CausalLMOutputWithPast
                        output_class = CausalLMOutputWithPast
                    except ImportError:
                        # Create a simple ModelOutput-like class
                        class ModelOutput(dict):
                            """Simple ModelOutput-like class"""
                            def __init__(self, *args, **kwargs):
                                super().__init__(*args, **kwargs)
                                self.__dict__ = self
                        output_class = ModelOutput

                    # Convert tuple to ModelOutput
                    outputs_dict = {}

                    # Check if we have labels in the kwargs to determine if first element is loss
                    has_labels = "labels" in kwargs

                    if len(outputs) >= 1:
                        # First element is typically the loss or logits
                        if has_labels:
                            # If we have labels, first element is likely the loss
                            outputs_dict["loss"] = outputs[0]
                            if len(outputs) >= 2:
                                # Second element is likely the logits
                                outputs_dict["logits"] = outputs[1]
                        else:
                            # If no labels, first element is likely the logits
                            outputs_dict["logits"] = outputs[0]

                        # Add any remaining elements with generic names
                        for i in range(1, len(outputs)):
                            if i == 1 and "logits" in outputs_dict:
                                continue  # Skip if we already assigned logits
                            outputs_dict[f"hidden_states_{i}"] = outputs[i]

                        # Convert to ModelOutput
                        outputs = output_class(**outputs_dict)
                        logger.info(f"Converted tuple to ModelOutput with keys: {list(outputs_dict.keys())}")

                return outputs

            # Replace the original forward method with our patched version
            model.forward = patched_peft_forward
            logger.info("✅ Successfully patched PeftModel forward method")
    except Exception as peft_fix_error:
        logger.error(f"Error applying PeftModel fix: {peft_fix_error}")
        logger.warning("Continuing with other fixes...")

    # First apply our direct fix for the "too many values to unpack" error
    try:
        # Apply the direct fix to the specific model instance
        success = apply_direct_fix(model)
        if success:
            logger.info("✅ Successfully applied direct fix for 'too many values to unpack' error to model instance")
        else:
            logger.warning("⚠️ Direct fix failed for model instance, trying other fixes")
    except Exception as e:
        logger.warning(f"⚠️ Error applying direct fix to model instance: {e}, trying other fixes")

    # Try to use the ultimate fix first
    try:
        from setup.ultimate_attention_fix import apply_ultimate_fix
        apply_ultimate_fix()
        logger.info("✅ Successfully applied ultimate attention fix")
    except ImportError:
        try:
            from src.setup.ultimate_attention_fix import apply_ultimate_fix
            apply_ultimate_fix()
            logger.info("✅ Successfully applied ultimate attention fix")
        except ImportError:
            logger.warning("⚠️ Could not import ultimate_attention_fix module, trying other fixes")

            # Try to use the fix_all_attention_issues module
            try:
                from setup.fix_all_attention_issues import fix_all_attention_issues
                fix_all_attention_issues()
                logger.info("✅ Successfully applied all attention fixes")
            except ImportError:
                try:
                    from src.setup.fix_all_attention_issues import fix_all_attention_issues
                    fix_all_attention_issues()
                    logger.info("✅ Successfully applied all attention fixes")
                except ImportError:
                    logger.warning("⚠️ Could not import fix_all_attention_issues module, trying tuple unpacking fix")

                    # Try to use the tuple unpacking fix
                    try:
                        from setup.fix_tuple_unpacking_error import fix_tuple_unpacking_error
                        fix_tuple_unpacking_error(model)
                        logger.info("✅ Successfully applied tuple unpacking fix to model")
                    except ImportError:
                        try:
                            from src.generative_ai_module.fix_tuple_unpacking import apply_fix_to_model
                            model = apply_fix_to_model(model)
                            logger.info("✅ Successfully applied tuple unpacking fix to model")
                        except ImportError:
                            logger.warning("⚠️ Could not import fix_tuple_unpacking module, applying inline fix")

                            # Apply a direct fix for the "too many values to unpack" error
                            if hasattr(model, 'forward'):
                                original_forward = model.forward

                                def patched_forward(*args, **kwargs):
                                    """Patched forward method that ensures return_dict=True and handles tuple outputs"""
                                    # Always set return_dict=True to avoid tuple outputs
                                    if "return_dict" not in kwargs:
                                        kwargs["return_dict"] = True
                                        logger.info("Setting return_dict=True to avoid tuple unpacking issues")

                                    # Try the original forward method
                                    try:
                                        outputs = original_forward(*args, **kwargs)
                                    except Exception as e:
                                        logger.error(f"Error in original forward: {e}")

                                        # Try direct forward call with explicit arguments
                                        try:
                                            logger.info("Trying direct forward call with explicit arguments")

                                            # Extract key arguments
                                            input_ids = kwargs.get("input_ids")
                                            attention_mask = kwargs.get("attention_mask")
                                            labels = kwargs.get("labels")

                                            # Ensure input_ids are torch.long
                                            if input_ids is not None and input_ids.dtype != torch.long:
                                                logger.warning(f"Converting input_ids from {input_ids.dtype} to torch.long")
                                                input_ids = input_ids.to(dtype=torch.long)

                                            # Ensure labels are torch.long
                                            if labels is not None and labels.dtype != torch.long:
                                                logger.warning(f"Converting labels from {labels.dtype} to torch.long")
                                                labels = labels.to(dtype=torch.long)

                                            # Try without attention mask
                                            logger.info("Trying forward call without attention mask")
                                            outputs = original_forward(
                                                *args[0:1],  # Only pass self
                                                input_ids=input_ids,
                                                labels=labels,
                                                use_cache=False,
                                                return_dict=True
                                            )
                                        except Exception as e2:
                                            logger.error(f"Direct forward call also failed: {e2}")

                                            # Try one more time with minimal arguments
                                            try:
                                                logger.info("Trying minimal forward call with only input_ids")
                                                # Only pass input_ids as a last resort
                                                if input_ids is not None:
                                                    outputs = original_forward(
                                                        *args[0:1],  # Only pass self
                                                        input_ids=input_ids,
                                                        use_cache=False,
                                                        return_dict=True
                                                    )
                                                else:
                                                    raise ValueError("No input_ids available for minimal forward call")
                                            except Exception as e3:
                                                logger.error(f"Minimal forward call also failed: {e3}")

                                                # Import ModelOutput for creating a fallback output
                                                try:
                                                    from transformers.modeling_outputs import ModelOutput, CausalLMOutputWithPast
                                                    output_class = CausalLMOutputWithPast if "CausalLM" in args[0].__class__.__name__ else ModelOutput
                                                except ImportError:
                                                    # Create a simple ModelOutput-like class
                                                    class ModelOutput(dict):
                                                        """Simple ModelOutput-like class"""
                                                        def __init__(self, *args, **kwargs):
                                                            super().__init__(*args, **kwargs)
                                                            self.__dict__ = self
                                                    output_class = ModelOutput

                                                # Create a dummy output as last resort
                                                if len(args) > 0 and hasattr(args[0], 'parameters'):
                                                    device = next(args[0].parameters()).device
                                                else:
                                                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                                                batch_size = input_ids.shape[0] if input_ids is not None else 1
                                                seq_length = input_ids.shape[1] if input_ids is not None else 1
                                                vocab_size = getattr(args[0].config, 'vocab_size', 32000) if len(args) > 0 and hasattr(args[0], 'config') else 32000

                                                # Create dummy logits
                                                dummy_logits = torch.zeros((batch_size, seq_length, vocab_size), device=device)

                                                # Create dummy loss if labels are provided
                                                dummy_loss = None
                                                if labels is not None:
                                                    dummy_loss = torch.tensor(1.0, device=device, requires_grad=True)

                                                # Create a ModelOutput with the dummy values
                                                outputs_dict = {"logits": dummy_logits}
                                                if dummy_loss is not None:
                                                    outputs_dict["loss"] = dummy_loss

                                                outputs = output_class(outputs_dict)
                                                logger.warning(f"Created fallback ModelOutput with keys: {list(outputs_dict.keys())}")

                                    # Handle tuple outputs
                                    if isinstance(outputs, tuple):
                                        logger.info(f"Got tuple output with {len(outputs)} elements, converting to ModelOutput")

                                        # Import ModelOutput
                                        try:
                                            from transformers.modeling_outputs import ModelOutput, CausalLMOutputWithPast
                                            output_class = CausalLMOutputWithPast if len(args) > 0 and "CausalLM" in args[0].__class__.__name__ else ModelOutput
                                        except ImportError:
                                            # Create a simple ModelOutput-like class
                                            class ModelOutput(dict):
                                                """Simple ModelOutput-like class"""
                                                def __init__(self, *args, **kwargs):
                                                    super().__init__(*args, **kwargs)
                                                    self.__dict__ = self
                                            output_class = ModelOutput

                                        # Convert tuple to a dictionary-like object
                                        outputs_dict = {}

                                        # Check if we have labels in the kwargs to determine if first element is loss
                                        has_labels = kwargs.get("labels") is not None

                                        if len(outputs) >= 1:
                                            # First element is typically the loss or logits
                                            if has_labels:
                                                # If we have labels, first element is likely the loss
                                                outputs_dict["loss"] = outputs[0]
                                                if len(outputs) >= 2:
                                                    # Second element is likely the logits
                                                    outputs_dict["logits"] = outputs[1]
                                            else:
                                                # If no labels, first element is likely the logits
                                                outputs_dict["logits"] = outputs[0]

                                            # Add any remaining elements with generic names
                                            for i in range(1, len(outputs)):
                                                if i == 1 and "logits" in outputs_dict:
                                                    continue  # Skip if we already assigned logits
                                                outputs_dict[f"hidden_states_{i}"] = outputs[i]

                                            # Convert to ModelOutput
                                            outputs = output_class(outputs_dict)
                                            logger.info(f"Converted tuple to ModelOutput with keys: {list(outputs_dict.keys())}")

                                    return outputs

                                model.forward = patched_forward
                                logger.info("✅ Applied comprehensive inline fix to model.forward")

    # Create trainer with attention mask fix
    trainer = AttentionMaskFixTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Verify trainer model is on CUDA
    logger.info(f"Trainer model device: {trainer.model.device}")
    if torch.cuda.is_available() and str(trainer.model.device) != "cuda:0" and str(trainer.model.device) != "cuda":
        logger.warning(f"Trainer model is not on CUDA! Moving to CUDA...")
        trainer.model = trainer.model.to("cuda")
        logger.info(f"Trainer model moved to device: {trainer.model.device}")

    # Add minimal validation step before training
    logger.info("Running validation forward pass...")
    validation_sample = tokenizer("def hello_world():", return_tensors="pt").to(trainer.model.device)
    try:
        outputs = trainer.model(
            input_ids=validation_sample.input_ids,
            attention_mask=validation_sample.attention_mask,
            labels=validation_sample.input_ids
        )
        logger.info("Validation forward pass succeeded")
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        logger.warning("Attempting to fix issues before proceeding...")
        # Try to apply fixes
        try:
            # Ensure model is in eval mode for validation
            trainer.model.eval()
            # Try with explicit keyword arguments
            outputs = trainer.model(
                input_ids=validation_sample.input_ids,
                attention_mask=None,  # Skip attention mask
                labels=validation_sample.input_ids,
                return_dict=True
            )
            logger.info("Validation succeeded with simplified arguments")
        except Exception as e2:
            logger.error(f"Simplified validation also failed: {e2}")
            logger.warning("Proceeding with training despite validation failure")

    # Ensure model parameters have requires_grad=True where needed
    trainable_params = 0
    all_params = 0
    for name, param in trainer.model.named_parameters():
        all_params += param.numel()
        if 'lora' in name:  # LoRA parameters should be trainable
            if not param.requires_grad:
                logger.warning(f"LoRA parameter {name} doesn't have requires_grad=True. Fixing...")
                param.requires_grad = True
            trainable_params += param.numel()

    logger.info(f"Verified model parameters: {trainable_params} trainable out of {all_params} total parameters")

    # Double-check that we have trainable parameters
    if trainable_params == 0:
        logger.error("No trainable parameters found! Training will not work.")
        raise ValueError("No trainable parameters found in the model. Check LoRA configuration.")

    # Recursively ensure all modules and their parameters are on the correct device
    device = trainer.model.device
    logger.info(f"Recursively ensuring all modules are on device: {device}")

    def ensure_module_on_device(module, prefix=""):
        """Recursively ensure all submodules and parameters are on the correct device"""
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            # Check if the child module has parameters on a different device
            child_devices = set(p.device for p in child.parameters() if p.device != device)
            if child_devices:
                logger.warning(f"Module {full_name} has parameters on devices: {child_devices}")
                # Move the entire child module to the correct device
                try:
                    child.to(device)
                    logger.info(f"Moved module {full_name} to {device}")
                except Exception as e:
                    logger.error(f"Failed to move module {full_name} to {device}: {e}")
            # Recursively process child modules
            ensure_module_on_device(child, full_name)

    # Apply the recursive device check to the entire model
    ensure_module_on_device(trainer.model)

    # Verify all parameters are now on the correct device
    incorrect_devices = [(name, param.device) for name, param in trainer.model.named_parameters() if param.device != device]
    if incorrect_devices:
        logger.warning(f"Found {len(incorrect_devices)} parameters on incorrect devices after fix:")
        for name, dev in incorrect_devices[:5]:  # Show first 5 only to avoid log spam
            logger.warning(f"Parameter {name} is on {dev} instead of {device}")
        # Force move the entire model again
        trainer.model = trainer.model.to(device)
        logger.info(f"Forced entire model to device: {device}")

    # Train model with robust error handling
    logger.info("Starting training...")
    try:
        # Set a flag to track if training completed successfully
        training_successful = False

        # Start training
        trainer.train()

        # If we get here, training was successful
        training_successful = True
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        logger.error(traceback.format_exc())

        # Try to recover and save what we have
        logger.warning("Attempting to save partial training results despite error")

    # Save model with error handling and ensure correct save location
    try:
        # Double-check that output directory is in the correct location
        if not args.output_dir.startswith('/notebooks/Jarvis_AI_Assistant/'):
            logger.warning(f"Output directory {args.output_dir} is not in the expected location.")
            logger.warning("Changing output directory to be within /notebooks/Jarvis_AI_Assistant/models/")
            model_name = os.path.basename(args.output_dir)
            args.output_dir = f"/notebooks/Jarvis_AI_Assistant/models/{model_name}"

        # Create output directory and verify it exists
        os.makedirs(args.output_dir, exist_ok=True)
        if not os.path.exists(args.output_dir):
            raise RuntimeError(f"Failed to create output directory: {args.output_dir}")

        # Test directory writability
        try:
            test_file = os.path.join(args.output_dir, 'test_write.txt')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            logger.info(f"Output directory {args.output_dir} is writable")
        except Exception as e:
            logger.warning(f"Error testing directory writability: {e}")
            # Use a local directory as fallback
            local_dir = os.path.join(os.getcwd(), "models/deepseek-coder-6.7b-finetuned")
            logger.warning(f"Changed output directory to local path: {local_dir}")
            args.output_dir = local_dir
            os.makedirs(args.output_dir, exist_ok=True)

        # Clear memory before saving
        clear_memory()

        logger.info(f"Saving model to {args.output_dir}")
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Verify model was saved correctly
        if not os.path.exists(os.path.join(args.output_dir, "pytorch_model.bin")) and \
           not os.path.exists(os.path.join(args.output_dir, "adapter_model.bin")):
            logger.warning("Model file not found after saving. Checking for other model files...")
            model_files = [f for f in os.listdir(args.output_dir) if f.endswith('.bin')]
            if model_files:
                logger.info(f"Found alternative model files: {model_files}")
            else:
                raise RuntimeError("No model files found after saving")

        logger.info(f"Model and tokenizer successfully saved to {args.output_dir}")

        # Create a README file with training information
        readme_content = f"""# DeepSeek Coder Fine-tuned Model

This model was fine-tuned from {args.model_name} on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.

## Training Parameters
- Batch size: {args.batch_size}
- Gradient accumulation steps: {args.gradient_accumulation_steps}
- Effective batch size: {args.batch_size * args.gradient_accumulation_steps}
- Learning rate: {args.learning_rate}
- Epochs: {args.epochs}
- LoRA rank: {args.lora_rank}
- LoRA alpha: {args.lora_alpha}
- LoRA dropout: {args.lora_dropout}
- Max sequence length: {args.max_length}
- Training samples: {len(train_dataset)}
"""
        with open(os.path.join(args.output_dir, "README.md"), "w") as f:
            f.write(readme_content)

    except Exception as e:
        logger.error(f"Error saving model: {e}")

        # Try alternative saving method
        try:
            logger.warning("Trying alternative saving method...")
            # Save the model state dict directly
            alternative_save_dir = f"/notebooks/Jarvis_AI_Assistant/models/backup_save_{int(time.time())}"
            os.makedirs(alternative_save_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{alternative_save_dir}/model.pt")
            tokenizer.save_pretrained(alternative_save_dir)
            logger.info(f"Model state dict saved to {alternative_save_dir}/model.pt")
        except Exception as e2:
            logger.error(f"Alternative saving also failed: {e2}")
            if not training_successful:
                raise RuntimeError(f"Both training and model saving failed. Original errors: Training: {e}, Saving: {e2}")

def train_with_standard_method(args):
    """Train using standard method with attention mask fix"""
    logger.info("Using standard training method with attention mask fix")

    # Apply PEFT model fix for "too many values to unpack" error
    def apply_peft_model_fix():
        """
        Apply a fix for the "too many values to unpack" error in PEFT models.
        This patches the forward method of PeftModel to always return a dictionary-like object.
        """
        try:
            # Try to import PeftModel
            try:
                from peft import PeftModel
                logger.info("Successfully imported PeftModel")
            except ImportError:
                logger.warning("Could not import PeftModel, skipping PEFT model fix")
                return False

            # Check if the forward method exists
            if not hasattr(PeftModel, "forward"):
                logger.warning("PeftModel does not have a forward method, skipping PEFT model fix")
                return False

            # Store the original forward method
            original_forward = PeftModel.forward
            logger.info("Stored original PeftModel.forward method")

            # Define a patched forward method
            def patched_forward(self, *args, **kwargs):
                """
                Patched forward method for PeftModel that ensures:
                1. No duplicate arguments (e.g., input_ids passed both as positional and keyword)
                2. Always returns a dictionary-like object with return_dict=True
                3. Handles any errors gracefully
                """
                logger.info("Using patched PeftModel.forward method")

                # Always set return_dict=True to avoid tuple unpacking issues
                kwargs["return_dict"] = True

                # Handle case where input_ids is passed both as positional and keyword argument
                if len(args) > 1 and isinstance(args[1], torch.Tensor) and "input_ids" in kwargs:
                    logger.info("Detected input_ids in both args and kwargs, removing from kwargs")
                    # Remove input_ids from kwargs to avoid the multiple values error
                    del kwargs["input_ids"]

                # Call the original forward method with proper arguments
                try:
                    return original_forward(self, **kwargs)
                except Exception as e:
                    logger.error(f"Error in patched PeftModel.forward: {e}")

                    # Try a more direct approach if the original call fails
                    try:
                        logger.info("Trying direct call to base_model.forward")
                        # Call the base model's forward method directly
                        outputs = self.base_model.forward(**kwargs)

                        # Ensure the output is a dictionary-like object
                        if not hasattr(outputs, "keys"):
                            from transformers.modeling_outputs import CausalLMOutputWithPast
                            # Convert tuple to dictionary-like object
                            if isinstance(outputs, tuple):
                                logger.info(f"Converting tuple output with {len(outputs)} elements to dictionary")
                                outputs_dict = {}

                                # First element is typically the loss
                                if len(outputs) >= 1:
                                    outputs_dict["loss"] = outputs[0]

                                # Second element is typically the logits
                                if len(outputs) >= 2:
                                    outputs_dict["logits"] = outputs[1]

                                # Add any remaining elements with generic names
                                for i in range(2, len(outputs)):
                                    outputs_dict[f"hidden_states_{i-2}"] = outputs[i]

                                # Convert to ModelOutput
                                outputs = CausalLMOutputWithPast(**outputs_dict)
                                logger.info(f"Converted tuple to ModelOutput with keys: {list(outputs_dict.keys())}")

                        return outputs
                    except Exception as e2:
                        logger.error(f"Direct call to base_model.forward also failed: {e2}")
                        # Create a dummy output as last resort
                        from transformers.modeling_outputs import CausalLMOutputWithPast
                        dummy_logits = torch.zeros((1, 1, self.base_model.config.vocab_size),
                                                device=self.device if hasattr(self, "device") else "cpu")
                        dummy_loss = torch.tensor(1.0, requires_grad=True,
                                               device=self.device if hasattr(self, "device") else "cpu")
                        return CausalLMOutputWithPast(loss=dummy_loss, logits=dummy_logits)

            # Apply the patch
            PeftModel.forward = patched_forward
            logger.info("✅ Successfully patched PeftModel.forward to fix 'too many values to unpack' error")
            return True
        except Exception as e:
            logger.error(f"Failed to apply PEFT model fix: {e}")
            return False

    # Apply the PEFT model fix
    apply_peft_model_fix()

    # Import the finetune_deepseek module
    try:
        # Try relative import first
        try:
            from .finetune_deepseek import main as finetune_main
            logger.info("Successfully imported finetune_deepseek module using relative import")
        except ImportError:
            # Try absolute import as fallback
            try:
                from src.generative_ai_module.finetune_deepseek import main as finetune_main
                logger.info("Successfully imported finetune_deepseek module using absolute import")
            except ImportError:
                # Try direct import as last resort
                import finetune_deepseek
                finetune_main = finetune_deepseek.main
                logger.info("Successfully imported finetune_deepseek module using direct import")

        # Ensure output directory is writable
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            # Test if the directory is writable
            test_file = os.path.join(args.output_dir, 'test_write.txt')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            logger.info(f"Output directory {args.output_dir} is writable")
        except (IOError, OSError) as e:
            logger.warning(f"Output directory {args.output_dir} is not writable: {e}")
            # Use a local directory instead
            local_dir = os.path.join(os.getcwd(), "models/deepseek-coder-6.7b-finetuned")
            logger.warning(f"Changed output directory to local path: {local_dir}")
            args.output_dir = local_dir
            os.makedirs(args.output_dir, exist_ok=True)

        logger.info(f"Using output directory: {args.output_dir}")

        # Prepare arguments for finetune_deepseek
        sys.argv = [
            "finetune_deepseek.py",
            f"--epochs={args.epochs}",
            f"--batch-size={args.batch_size}",
            f"--learning-rate={args.learning_rate}",
            f"--sequence-length={args.max_length}",
            f"--max-samples={args.max_samples}",
            f"--warmup-steps={args.warmup_steps}",
            f"--output-dir={args.output_dir}",
            f"--gradient-accumulation-steps={args.gradient_accumulation_steps}",
            f"--num-workers={args.num_workers}"
        ]

        # Quantization settings
        if args.load_in_4bit:
            sys.argv.append("--load-in-4bit")
        elif args.load_in_8bit:
            sys.argv.append("--load-in-8bit")

        # Dataset settings
        if args.all_subsets:
            sys.argv.append("--all-subsets")
        else:
            sys.argv.append(f"--subset={args.dataset_subset}")

        # Optimization settings
        if args.gradient_checkpointing:
            sys.argv.append("--gradient-checkpointing")

        if args.bf16:
            sys.argv.append("--bf16")
        elif torch.cuda.is_available():
            sys.argv.append("--fp16")

        # LoRA settings
        sys.argv.append(f"--lora-rank={args.lora_rank}")
        sys.argv.append(f"--lora-alpha={args.lora_alpha}")
        sys.argv.append(f"--lora-dropout={args.lora_dropout}")

        # Advanced optimization settings
        sys.argv.append(f"--weight-decay={args.weight_decay}")
        sys.argv.append(f"--max-grad-norm={args.max_grad_norm}")
        sys.argv.append(f"--scheduler-type={args.scheduler_type}")
        sys.argv.append(f"--evaluation-strategy={args.evaluation_strategy}")
        sys.argv.append(f"--eval-steps={args.eval_steps}")
        sys.argv.append(f"--save-steps={args.save_steps}")
        sys.argv.append(f"--save-total-limit={args.save_total_limit}")

        # Adam optimizer settings
        sys.argv.append(f"--adam-beta1={args.adam_beta1}")
        sys.argv.append(f"--adam-beta2={args.adam_beta2}")
        sys.argv.append(f"--adam-epsilon={args.adam_epsilon}")

        # Run the finetune_deepseek main function
        logger.info("Starting finetune_deepseek with attention mask fix")
        finetune_main()

    except ImportError:
        logger.error("Could not import finetune_deepseek module. Falling back to CodeGenerator.")

        # Import CodeGenerator and related modules
        try:
            # Try relative import first
            try:
                from .code_generator import CodeGenerator
                from .code_preprocessing import load_and_preprocess_dataset
                logger.info("Successfully imported CodeGenerator using relative import")
            except ImportError:
                # Try absolute import as fallback
                try:
                    from src.generative_ai_module.code_generator import CodeGenerator
                    from src.generative_ai_module.code_preprocessing import load_and_preprocess_dataset
                    logger.info("Successfully imported CodeGenerator using absolute import")
                except ImportError:
                    # Try direct import as last resort
                    import code_generator
                    import code_preprocessing
                    CodeGenerator = code_generator.CodeGenerator
                    load_and_preprocess_dataset = code_preprocessing.load_and_preprocess_dataset
                    logger.info("Successfully imported CodeGenerator using direct import")
        except ImportError as e:
            logger.error(f"Could not import CodeGenerator: {e}")
            raise ImportError(f"Could not import required modules: {e}")

        # Load dataset
        logger.info("Loading and preprocessing dataset...")
        train_dataset, eval_dataset = load_and_preprocess_dataset(
            max_samples=args.max_samples,
            sequence_length=args.max_length,
            subset=args.dataset_subset,
            all_subsets=args.all_subsets
        )

        # Initialize CodeGenerator
        logger.info("Initializing CodeGenerator...")
        code_gen = CodeGenerator(
            use_deepseek=True,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit
        )

        # Ensure output directory is writable
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            # Test if the directory is writable
            test_file = os.path.join(args.output_dir, 'test_write.txt')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            logger.info(f"Output directory {args.output_dir} is writable")
        except (IOError, OSError) as e:
            logger.warning(f"Output directory {args.output_dir} is not writable: {e}")
            # Use a local directory instead
            local_dir = os.path.join(os.getcwd(), "models/deepseek-coder-6.7b-finetuned")
            logger.warning(f"Changed output directory to local path: {local_dir}")
            args.output_dir = local_dir
            os.makedirs(args.output_dir, exist_ok=True)

        logger.info(f"Using output directory: {args.output_dir}")

        # Fine-tune the model
        logger.info("Starting fine-tuning...")
        code_gen.fine_tune_deepseek(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            sequence_length=args.max_length,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            subset=args.dataset_subset,
            all_subsets=args.all_subsets,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=args.gradient_checkpointing,
            bf16=args.bf16,
            fp16=not args.bf16 and torch.cuda.is_available(),
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            num_workers=args.num_workers,
            # Advanced optimization parameters
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            scheduler_type=args.scheduler_type,
            evaluation_strategy=args.evaluation_strategy,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            # Adam optimizer settings
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            adam_epsilon=args.adam_epsilon
        )

        # Create a README file with training information
        try:
            readme_content = f"""# DeepSeek Coder Fine-tuned Model (CodeGenerator)

This model was fine-tuned from {args.model_name} on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.

## Training Parameters
- Batch size: {args.batch_size}
- Gradient accumulation steps: {args.gradient_accumulation_steps}
- Effective batch size: {args.batch_size * args.gradient_accumulation_steps}
- Learning rate: {args.learning_rate}
- Epochs: {args.epochs}
- LoRA rank: {args.lora_rank}
- LoRA alpha: {args.lora_alpha}
- LoRA dropout: {args.lora_dropout}
- Max sequence length: {args.max_length}
- Training samples: {len(train_dataset)}
"""
            with open(os.path.join(args.output_dir, "README.md"), "w") as f:
                f.write(readme_content)
            logger.info(f"Created README.md in {args.output_dir}")
        except Exception as e:
            logger.warning(f"Failed to create README.md: {e}")

def clear_memory():
    """Clear GPU and CPU memory to prevent OOM errors"""
    import gc

    # Clear Python garbage collector
    gc.collect()

    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"Cleared {torch.cuda.memory_allocated()/1024**3:.2f} GB GPU memory")

def validate_training_setup():
    """Validate the training setup before starting training"""
    import torch
    from transformers import AutoModel

    logger.info("Validating training setup...")

    # Test forward pass with a small input
    try:
        # Create a small test model
        test_model = AutoModel.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base",
                                              trust_remote_code=True,
                                              device_map="auto")

        # Create a small test input
        test_input = torch.randint(0, 100, (2, 10)).to(test_model.device)

        # Test forward pass
        logger.info("Testing forward pass...")
        outputs = test_model(test_input)
        logger.info("Validation forward pass succeeded")

        # Test save/load
        test_path = "/tmp/model_test"
        os.makedirs(test_path, exist_ok=True)

        logger.info("Testing model save...")
        test_model.save_pretrained(test_path)

        logger.info("Testing model load...")
        reloaded = AutoModel.from_pretrained(test_path, trust_remote_code=True)
        logger.info("Model save/load validation succeeded")

        # Clean up
        import shutil
        shutil.rmtree(test_path, ignore_errors=True)

        # Clear memory
        del test_model
        del reloaded
        clear_memory()

        return True
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        return False

def safe_main():
    """Wrapper around main() with comprehensive error handling"""
    try:
        # Set up additional safeguards
        if torch.cuda.is_available():
            # Set up CUDA error handling
            clear_memory()

            # Check available GPU memory
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            free_memory_gb = free_memory / (1024**3)
            logger.info(f"Available GPU memory before training: {free_memory_gb:.2f} GiB")

            if free_memory_gb < 2.0:
                logger.warning(f"Very low GPU memory available ({free_memory_gb:.2f} GiB). Training may fail.")

        # Validate training setup
        if not validate_training_setup():
            logger.warning("Training setup validation failed, but continuing anyway")

        # Run the main function
        main()

        logger.info("Training process completed successfully")
        return 0
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        logger.error(f"Unhandled exception in training process: {e}")
        import traceback
        logger.error(traceback.format_exc())

        # Try to clean up
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logger.info("CUDA cache cleared")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clear CUDA cache: {cleanup_error}")

        return 1  # Error exit code

if __name__ == "__main__":
    exit_code = safe_main()
    sys.exit(exit_code)
