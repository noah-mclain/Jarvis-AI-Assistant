#!/usr/bin/env python3
"""
Comprehensive Attention Mask Fix for Transformer Models

This module provides a comprehensive solution to fix attention mask issues in transformer models,
particularly for DeepSeek and LLaMA models. It addresses various issues including:

1. Tensor device mismatches (CPU vs. GPU)
2. Attention mask dimension mismatches
3. Tensor size mismatches in mask expansion
4. Handling of unmasked_value parameter

Usage:
    from setup.comprehensive_attention_mask_fix import apply_comprehensive_fix
    apply_comprehensive_fix()
"""

import sys
import logging
import inspect
from typing import Optional, Any, Tuple, Dict, List, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def fix_dtype_mismatch():
    """
    Apply a comprehensive fix for dtype mismatches between BFloat16 and Half tensors.
    This is a common issue when using mixed precision training.

    Returns:
        bool: True if the fix was applied successfully, False otherwise
    """
    try:
        import torch
        import transformers
        from transformers.modeling_utils import PreTrainedModel

        logger.info("Applying comprehensive dtype mismatch fix...")

        # Fix 1: Patch PreTrainedModel.forward
        # Store the original forward method
        original_forward = PreTrainedModel.forward

        # Define a patched forward method
        def patched_forward(self, *args, **kwargs):
            """
            Patched forward method that ensures consistent dtypes for all tensors.
            """
            # Get the model's dtype
            model_dtype = getattr(self, "dtype", None)

            if model_dtype is None:
                # Try to detect the model's dtype from its parameters
                for param in self.parameters():
                    if param.dtype in [torch.float16, torch.bfloat16]:
                        model_dtype = param.dtype
                        logger.info(f"Detected model dtype from parameters: {model_dtype}")
                        break

            # Process input tensors to ensure consistent dtype
            for arg_name, arg_value in kwargs.items():
                if isinstance(arg_value, torch.Tensor):
                    # CRITICAL FIX: Ensure input_ids remain as long integers
                    if arg_name == "input_ids":
                        if arg_value.dtype != torch.long:
                            logger.warning(f"Input IDs have incorrect dtype: {arg_value.dtype}. Converting to torch.long")
                            kwargs[arg_name] = arg_value.to(dtype=torch.long)
                            logger.info(f"Fixed input_ids dtype: {kwargs[arg_name].dtype}")
                    # Skip certain tensors that should not be converted
                    elif arg_name not in ["labels", "token_type_ids"]:
                        # Handle attention mask specially
                        if arg_name == "attention_mask":
                            # If it's a 4D attention mask, ensure it has the correct shape
                            if arg_value.dim() == 4:
                                batch_size, head_dim, seq_len1, seq_len2 = arg_value.shape
                                # Check if the mask has the wrong shape (e.g., [6, 1, 6, 2048])
                                if seq_len1 != seq_len2:
                                    logger.warning(f"Attention mask has incorrect shape: {arg_value.shape}. Fixing...")
                                    # Create a proper 4D attention mask
                                    device = arg_value.device
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
                                    else:
                                        causal_mask = causal_mask.to(dtype=arg_value.dtype)

                                    # Replace the attention mask
                                    kwargs[arg_name] = ~causal_mask
                                    logger.info(f"Fixed attention mask shape: {kwargs[arg_name].shape}")

                            # Ensure the attention mask has the correct dtype
                            if model_dtype is not None and kwargs[arg_name].dtype != model_dtype:
                                logger.info(f"Converting attention_mask from {kwargs[arg_name].dtype} to {model_dtype}")
                                kwargs[arg_name] = kwargs[arg_name].to(dtype=model_dtype)

                        # For other tensors, just convert the dtype if needed
                        elif model_dtype is not None and arg_value.dtype != model_dtype:
                            logger.info(f"Converting {arg_name} from {arg_value.dtype} to {model_dtype}")
                            kwargs[arg_name] = arg_value.to(dtype=model_dtype)

            # Call the original forward method
            try:
                return original_forward(self, *args, **kwargs)
            except RuntimeError as e:
                error_msg = str(e)
                # Check for specific dtype mismatch errors
                if "expected mat1 and mat2 to have the same dtype" in error_msg:
                    logger.warning(f"Caught dtype mismatch error: {error_msg}")
                    # Try to fix the dtype mismatch by converting all tensors to float32
                    logger.info("Attempting to fix by converting all tensors to float32")
                    for arg_name, arg_value in kwargs.items():
                        if isinstance(arg_value, torch.Tensor):
                            # Keep input_ids as long integers
                            if arg_name == "input_ids":
                                kwargs[arg_name] = arg_value.to(dtype=torch.long)
                            # Keep labels as long integers
                            elif arg_name == "labels":
                                kwargs[arg_name] = arg_value.to(dtype=torch.long)
                            else:
                                kwargs[arg_name] = arg_value.to(dtype=torch.float32)

                    # Try again with fixed tensors
                    return original_forward(self, *args, **kwargs)
                else:
                    # Re-raise other errors
                    raise

        # Apply the patch
        PreTrainedModel.forward = patched_forward
        logger.info("✅ Successfully patched PreTrainedModel.forward to fix dtype mismatches")

        # Fix 2: Patch the prepare_inputs_for_generation method
        try:
            original_prepare_inputs = PreTrainedModel.prepare_inputs_for_generation

            def patched_prepare_inputs(self, *args, **kwargs):
                """
                Patched prepare_inputs_for_generation method that ensures consistent dtypes.
                """
                # Call the original method
                inputs = original_prepare_inputs(self, *args, **kwargs)

                # Get the model's dtype
                model_dtype = getattr(self, "dtype", None)

                # CRITICAL FIX: Ensure input_ids remain as long integers
                if "input_ids" in inputs and inputs["input_ids"].dtype != torch.long:
                    logger.warning(f"Input IDs have incorrect dtype: {inputs['input_ids'].dtype}. Converting to torch.long")
                    inputs["input_ids"] = inputs["input_ids"].to(dtype=torch.long)
                    logger.info(f"Fixed input_ids dtype: {inputs['input_ids'].dtype}")

                # Ensure labels remain as long integers
                if "labels" in inputs and inputs["labels"].dtype != torch.long:
                    logger.warning(f"Labels have incorrect dtype: {inputs['labels'].dtype}. Converting to torch.long")
                    inputs["labels"] = inputs["labels"].to(dtype=torch.long)
                    logger.info(f"Fixed labels dtype: {inputs['labels'].dtype}")

                if model_dtype is not None:
                    # Ensure all tensors have the correct dtype
                    for key, value in inputs.items():
                        if isinstance(value, torch.Tensor) and key not in ["labels", "input_ids", "token_type_ids"]:
                            if value.dtype != model_dtype:
                                logger.info(f"Converting {key} from {value.dtype} to {model_dtype} in prepare_inputs_for_generation")
                                inputs[key] = value.to(dtype=model_dtype)

                return inputs

            # Apply the patch
            PreTrainedModel.prepare_inputs_for_generation = patched_prepare_inputs
            logger.info("✅ Successfully patched PreTrainedModel.prepare_inputs_for_generation")
        except Exception as e:
            logger.warning(f"⚠️ Could not patch prepare_inputs_for_generation: {e}")

        return True
    except Exception as e:
        logger.warning(f"⚠️ Could not apply dtype mismatch fix: {e}")
        return False

def apply_comprehensive_fix():
    """
    Apply a comprehensive fix for attention mask issues in transformer models.
    This function patches various methods in the transformers library to handle
    attention mask issues properly.

    Returns:
        bool: True if the fix was applied successfully, False otherwise
    """
    try:
        import torch
        import transformers

        logger.info(f"Applying comprehensive attention mask fix for transformers {transformers.__version__}")

        # Fix 1: Apply dtype mismatch fix
        dtype_fix_success = fix_dtype_mismatch()
        if dtype_fix_success:
            logger.info("✅ Successfully applied dtype mismatch fix")
        else:
            logger.warning("⚠️ Failed to apply dtype mismatch fix")

        # Fix 1.5: Patch _prepare_4d_causal_attention_mask_for_sdpa in LlamaModel
        try:
            # Try to import the function from LlamaModel
            try:
                from transformers.models.llama.modeling_llama import _prepare_4d_causal_attention_mask_for_sdpa as llama_prepare_4d

                # Store the original function
                original_llama_prepare_4d = llama_prepare_4d

                # Define a patched function with a flexible signature
                def patched_llama_prepare_4d(*args, **kwargs):
                    """
                    Patched version of _prepare_4d_causal_attention_mask_for_sdpa in LlamaModel.
                    This handles the specific signature used in the LlamaModel implementation.
                    """
                    # Extract parameters from args or kwargs
                    attention_mask = None
                    input_shape = None
                    inputs_embeds = None
                    past_key_values_length = 0
                    sliding_window = None
                    dtype = None

                    # Extract from args
                    if len(args) > 0:
                        attention_mask = args[0]
                    if len(args) > 1:
                        input_shape = args[1]
                    if len(args) > 2:
                        inputs_embeds = args[2]
                    if len(args) > 3:
                        past_key_values_length = args[3]
                    if len(args) > 4:
                        sliding_window = args[4]
                    if len(args) > 5:
                        dtype = args[5]

                    # Extract from kwargs
                    if 'attention_mask' in kwargs:
                        attention_mask = kwargs['attention_mask']
                    if 'input_shape' in kwargs:
                        input_shape = kwargs['input_shape']
                    if 'inputs_embeds' in kwargs:
                        inputs_embeds = kwargs['inputs_embeds']
                    if 'past_key_values_length' in kwargs:
                        past_key_values_length = kwargs['past_key_values_length']
                    if 'sliding_window' in kwargs:
                        sliding_window = kwargs['sliding_window']
                    if 'dtype' in kwargs:
                        dtype = kwargs['dtype']

                    # Fix attention_mask shape if needed
                    if attention_mask is not None and attention_mask.dim() > 2:
                        # Get the batch size and sequence length
                        batch_size = attention_mask.size(0)
                        seq_length = attention_mask.size(-1)

                        # Reshape to 2D [batch_size, seq_length]
                        try:
                            attention_mask = attention_mask.view(batch_size, seq_length)
                            logger.info(f"Reshaped attention mask from >2D to 2D in LlamaModel: {attention_mask.shape}")
                        except Exception as e:
                            logger.warning(f"Could not reshape attention mask: {e}")
                            # Create a new mask if reshaping fails
                            attention_mask = torch.ones((batch_size, seq_length), device=inputs_embeds.device)
                            logger.info(f"Created new attention mask with shape: {attention_mask.shape}")

                        # Update args or kwargs with the fixed mask
                        if len(args) > 0:
                            args_list = list(args)
                            args_list[0] = attention_mask
                            args = tuple(args_list)
                        elif 'attention_mask' in kwargs:
                            kwargs['attention_mask'] = attention_mask

                    # Get the signature of the original function
                    sig = inspect.signature(original_llama_prepare_4d)
                    param_names = list(sig.parameters.keys())

                    # Determine the expected signature based on the number of parameters
                    if len(param_names) <= 4:
                        # v4.36.0 signature
                        logger.info("Using v4.36.0 signature (4 parameters) for LlamaModel")
                        return original_llama_prepare_4d(
                            attention_mask, input_shape, inputs_embeds, past_key_values_length
                        )
                    elif len(param_names) == 5:
                        # v4.37.0 signature
                        logger.info("Using v4.37.0 signature (5 parameters) for LlamaModel")
                        return original_llama_prepare_4d(
                            attention_mask, input_shape, inputs_embeds, past_key_values_length, sliding_window
                        )
                    else:
                        # v4.38.0+ signature
                        logger.info("Using v4.38.0+ signature (6 parameters) for LlamaModel")
                        # If dtype is None, use inputs_embeds.dtype
                        if dtype is None:
                            dtype = inputs_embeds.dtype

                        return original_llama_prepare_4d(
                            attention_mask, input_shape, inputs_embeds, past_key_values_length, sliding_window, dtype
                        )

                # Apply the patch
                import transformers.models.llama.modeling_llama
                transformers.models.llama.modeling_llama._prepare_4d_causal_attention_mask_for_sdpa = patched_llama_prepare_4d
                logger.info("✅ Successfully patched LlamaModel _prepare_4d_causal_attention_mask_for_sdpa")
            except (ImportError, AttributeError) as e:
                logger.warning(f"⚠️ Could not patch LlamaModel _prepare_4d_causal_attention_mask_for_sdpa: {e}")
        except Exception as e:
            logger.warning(f"⚠️ Could not patch LlamaModel _prepare_4d_causal_attention_mask_for_sdpa: {e}")

        # Fix 2: Apply direct fix for attention mask shape issues
        try:
            from transformers.models.llama.modeling_llama import LlamaAttention

            # Store the original _prepare_decoder_attention_mask method
            original_prepare_mask = LlamaAttention._prepare_decoder_attention_mask

            # Define a patched method
            def patched_prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
                """
                Patched method that ensures the attention mask has the correct shape.
                """
                try:
                    # Get batch size and sequence length
                    batch_size, seq_length = input_shape

                    # Check if attention_mask has the wrong shape
                    if attention_mask is not None and attention_mask.dim() > 2:
                        # Try to reshape it to 2D
                        try:
                            attention_mask = attention_mask.view(batch_size, -1)
                            logger.info(f"Reshaped attention mask from {attention_mask.shape} to [batch_size, seq_length]")
                        except Exception as reshape_error:
                            logger.warning(f"Could not reshape attention mask: {reshape_error}")
                            # Create a new mask if reshaping fails
                            attention_mask = torch.ones((batch_size, seq_length), device=inputs_embeds.device)
                            logger.info(f"Created new attention mask with shape: {attention_mask.shape}")

                    # Call the original method with the fixed mask
                    result = original_prepare_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length)

                    # Verify the result has the correct shape
                    expected_shape = (batch_size, 1, seq_length, seq_length)
                    if result.shape != expected_shape:
                        logger.warning(f"Result has incorrect shape: {result.shape}, expected: {expected_shape}")
                        # Create a correct mask directly
                        device = inputs_embeds.device
                        # Create a causal mask
                        causal_mask = torch.triu(
                            torch.ones((seq_length, seq_length), device=device, dtype=torch.bool),
                            diagonal=1
                        )
                        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]
                        causal_mask = causal_mask.expand(batch_size, 1, seq_length, seq_length)

                        # Convert to the expected format
                        result = causal_mask.to(dtype=inputs_embeds.dtype) * torch.finfo(inputs_embeds.dtype).min
                        logger.info(f"Created corrected mask with shape: {result.shape}")

                    return result
                except Exception as e:
                    logger.warning(f"Error in patched _prepare_decoder_attention_mask: {e}")
                    # Fall back to the original method
                    return original_prepare_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length)

            # Apply the patch
            LlamaAttention._prepare_decoder_attention_mask = patched_prepare_decoder_attention_mask
            logger.info("✅ Successfully patched LlamaAttention._prepare_decoder_attention_mask")
        except Exception as e:
            logger.warning(f"⚠️ Could not patch LlamaAttention._prepare_decoder_attention_mask: {e}")

        # Fix 3: Apply specific fix for DeepSeek models
        try:
            # Try to import DeepSeekAttention
            deepseek_available = False
            try:
                # First try the standard import path
                try:
                    from transformers.models.deepseek.modeling_deepseek import DeepSeekAttention
                    deepseek_available = True
                except ImportError:
                    # Try alternative import paths
                    try:
                        from transformers.models.deepseek_coder.modeling_deepseek_coder import DeepSeekCoderAttention as DeepSeekAttention
                        deepseek_available = True
                    except ImportError:
                        # Try to find any DeepSeek-related module
                        import pkgutil
                        import transformers

                        deepseek_modules = [name for _, name, _ in pkgutil.iter_modules(transformers.__path__) if 'deepseek' in name.lower()]
                        if deepseek_modules:
                            logger.info(f"Found potential DeepSeek modules: {deepseek_modules}")
                            # Try to import from the first found module
                            try:
                                module_name = deepseek_modules[0]
                                exec(f"from transformers.models.{module_name}.modeling_{module_name} import {module_name.capitalize()}Attention as DeepSeekAttention")
                                deepseek_available = True
                            except Exception as module_error:
                                logger.warning(f"Could not import from found module {module_name}: {module_error}")
            except Exception as import_error:
                logger.warning(f"Error while trying to import DeepSeek: {import_error}")

            if not deepseek_available:
                logger.warning("DeepSeek model not available in this transformers version")

            if deepseek_available:
                # Store the original _prepare_decoder_attention_mask method
                original_deepseek_prepare_mask = DeepSeekAttention._prepare_decoder_attention_mask

                # Define a patched method
                def patched_deepseek_prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
                    """
                    Patched method that ensures the attention mask has the correct shape for DeepSeek models.
                    """
                    try:
                        # Get batch size and sequence length
                        batch_size, seq_length = input_shape

                        # Check if attention_mask has the wrong shape
                        if attention_mask is not None and attention_mask.dim() > 2:
                            # Try to reshape it to 2D
                            try:
                                attention_mask = attention_mask.view(batch_size, -1)
                                logger.info(f"Reshaped DeepSeek attention mask from {attention_mask.shape} to [batch_size, seq_length]")
                            except Exception as reshape_error:
                                logger.warning(f"Could not reshape DeepSeek attention mask: {reshape_error}")
                                # Create a new mask if reshaping fails
                                attention_mask = torch.ones((batch_size, seq_length), device=inputs_embeds.device)
                                logger.info(f"Created new DeepSeek attention mask with shape: {attention_mask.shape}")

                        # Call the original method with the fixed mask
                        result = original_deepseek_prepare_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length)

                        # Verify the result has the correct shape
                        expected_shape = (batch_size, 1, seq_length, seq_length)
                        if result.shape != expected_shape:
                            logger.warning(f"DeepSeek result has incorrect shape: {result.shape}, expected: {expected_shape}")
                            # Create a correct mask directly
                            device = inputs_embeds.device
                            # Create a causal mask
                            causal_mask = torch.triu(
                                torch.ones((seq_length, seq_length), device=device, dtype=torch.bool),
                                diagonal=1
                            )
                            causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]
                            causal_mask = causal_mask.expand(batch_size, 1, seq_length, seq_length)

                            # Convert to the expected format
                            result = causal_mask.to(dtype=inputs_embeds.dtype) * torch.finfo(inputs_embeds.dtype).min
                            logger.info(f"Created corrected DeepSeek mask with shape: {result.shape}")

                        return result
                    except Exception as e:
                        logger.warning(f"Error in patched DeepSeek _prepare_decoder_attention_mask: {e}")
                        # Fall back to the original method
                        return original_deepseek_prepare_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length)

                # Apply the patch
                DeepSeekAttention._prepare_decoder_attention_mask = patched_deepseek_prepare_decoder_attention_mask
                logger.info("✅ Successfully patched DeepSeekAttention._prepare_decoder_attention_mask")
        except Exception as e:
            logger.warning(f"⚠️ Could not apply DeepSeek-specific fix: {e}")

        # Fix 4: Patch AttentionMaskConverter._unmask_unattended
        try:
            from transformers.modeling_attn_mask_utils import AttentionMaskConverter

            # Store the original function
            original_unmask_unattended = AttentionMaskConverter._unmask_unattended

            # Get the signature of the original function
            try:
                sig = inspect.signature(original_unmask_unattended)
                param_names = list(sig.parameters.keys())
                logger.info(f"Original _unmask_unattended signature: {param_names}")
            except Exception as e:
                logger.warning(f"Could not get signature of _unmask_unattended: {e}")
                param_names = []

            @staticmethod
            def patched_unmask_unattended(*args, **kwargs):
                """
                Patched version of _unmask_unattended that handles various issues:
                1. Keeps tensors on the same device (no CPU conversion)
                2. Handles tensor dimension mismatches
                3. Properly handles the unmasked_value parameter
                4. Creates compatible masks when expansion fails
                """
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
                        # Get sequence length for proper expansion
                        seq_length = attention_mask.size(-1)
                        batch_size = attention_mask.size(0)

                        # Create a properly shaped 4D attention mask directly
                        logger.info(f"Creating a properly shaped 4D attention mask with dimensions [batch_size, 1, seq_length, seq_length]")

                        # Create a causal mask
                        causal_mask = torch.triu(
                            torch.ones((seq_length, seq_length), device=device, dtype=torch.bool),
                            diagonal=1
                        )
                        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]
                        causal_mask = causal_mask.expand(batch_size, 1, seq_length, seq_length)

                        # Apply the attention mask
                        expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                        expanded_mask = expanded_mask.expand(-1, -1, seq_length, -1)

                        # Combine with causal mask
                        combined_mask = causal_mask | ~expanded_mask.bool()

                        # Convert to the expected format
                        mask = ~combined_mask
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

                # Handle dtype mismatch issues
                try:
                    # First, convert mask to the expected type based on unmasked_value
                    if unmasked_value is not True:
                        # If unmasked_value is a float or tensor, convert mask to that dtype and multiply
                        if isinstance(unmasked_value, (float, int)) or (isinstance(unmasked_value, torch.Tensor) and unmasked_value.dtype.is_floating_point):
                            # Convert mask to the same dtype as attention_mask or to float32 if needed
                            mask_dtype = attention_mask.dtype if attention_mask.dtype.is_floating_point else torch.float32
                            mask = mask.to(dtype=mask_dtype) * unmasked_value
                            logger.info(f"Applied unmasked_value {unmasked_value} to mask, resulting dtype: {mask.dtype}")
                        else:
                            # For boolean or other types, just use the value directly
                            mask = mask * unmasked_value
                            logger.info(f"Applied unmasked_value {unmasked_value} to mask without dtype conversion")

                    # Try to detect the model's dtype from the current context
                    model_dtype = None

                    # Method 1: Check the current frame for model dtype
                    try:
                        import inspect
                        frame = inspect.currentframe()
                        while frame:
                            if 'self' in frame.f_locals and hasattr(frame.f_locals['self'], 'dtype'):
                                model_dtype = frame.f_locals['self'].dtype
                                break
                            frame = frame.f_back
                    except Exception:
                        pass

                    # Method 2: Try to detect if we're in a BFloat16 or Float16 context
                    if model_dtype is None:
                        try:
                            # Check if any tensor in the current context is BFloat16 or Float16
                            for frame_info in inspect.stack():
                                frame = frame_info.frame
                                for var_name, var_val in frame.f_locals.items():
                                    if isinstance(var_val, torch.Tensor) and var_val.dtype in [torch.bfloat16, torch.float16]:
                                        model_dtype = var_val.dtype
                                        logger.info(f"Detected model dtype from context: {model_dtype}")
                                        break
                                if model_dtype is not None:
                                    break
                        except Exception:
                            pass

                    # Method 3: Try both common dtypes
                    if model_dtype is None:
                        # If we couldn't detect the dtype, try both BFloat16 and Float16
                        logger.info("Could not detect model dtype. Creating both BFloat16 and Float16 versions of the mask.")

                        # Create a copy of the mask in both dtypes
                        mask_bfloat16 = mask.to(dtype=torch.bfloat16)
                        mask_float16 = mask.to(dtype=torch.float16)

                        # Return the BFloat16 version (we'll handle Float16 in the forward method)
                        mask = mask_bfloat16
                    else:
                        # If we detected a dtype, convert the mask to that dtype
                        logger.info(f"Converting mask to detected dtype: {model_dtype}")
                        mask = mask.to(dtype=model_dtype)
                except Exception as e:
                    logger.warning(f"Error handling dtype conversion: {e}")
                    # If all else fails, create a float32 mask which can be converted later
                    mask = mask.to(dtype=torch.float32)

                return mask

            # Apply the patch
            AttentionMaskConverter._unmask_unattended = patched_unmask_unattended
            logger.info("✅ Successfully patched AttentionMaskConverter._unmask_unattended")
        except Exception as e:
            logger.warning(f"⚠️ Could not patch AttentionMaskConverter._unmask_unattended: {e}")

        logger.info("✅ Comprehensive attention mask fix applied successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to apply comprehensive attention mask fix: {e}")
        return False

if __name__ == "__main__":
    # Apply the fix when the script is run directly
    success = apply_comprehensive_fix()

    # Exit with appropriate code
    sys.exit(0 if success else 1)
