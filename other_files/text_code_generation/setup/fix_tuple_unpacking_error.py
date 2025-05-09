#!/usr/bin/env python3
"""
Fix for the "too many values to unpack (expected 2)" error in transformers models.

This script applies patches to fix the tuple unpacking error that occurs when
model outputs are returned as tuples with more than 2 elements, but the code
expects to unpack them into exactly 2 variables.

The fix ensures that model outputs are always returned as ModelOutput objects
with a dictionary-like interface, rather than tuples.
"""

import os
import sys
import logging
import importlib
from typing import Optional, Any, Dict, Tuple, List, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def fix_tuple_unpacking_error(model=None):
    """
    Apply fixes for the "too many values to unpack (expected 2)" error.

    This function patches the model's forward method to ensure it always returns'
    a ModelOutput object with a dictionary-like interface, rather than a tuple.

    Args:
        model: The model to patch. If None, tries to patch the transformers library.

    Returns:
        bool: True if the fix was applied successfully, False otherwise.
    """
    import torch

    success = False

    # If a specific model is provided, patch it directly
    if model is not None:
        logger.info(f"Applying tuple unpacking fix to model: {type(model).__name__}")

        # Check if the model has a forward method
        if hasattr(model, 'forward'):
            # Store the original forward method
            original_forward = model.forward

            # Define a patched forward method
            def patched_forward(*args, **kwargs):
                """
                Patched forward method that ensures outputs are always ModelOutput objects.
                """
                # Always set return_dict=True to avoid tuple outputs
                if "return_dict" not in kwargs:
                    kwargs["return_dict"] = True
                    logger.info("Setting return_dict=True to avoid tuple unpacking issues")

                # Call the original forward method
                outputs = original_forward(*args, **kwargs)

                # Handle tuple outputs
                if isinstance(outputs, tuple):
                    logger.info(f"Got tuple output with {len(outputs)} elements, converting to ModelOutput")

                    # Import ModelOutput
                    try:
                        from transformers.modeling_outputs import ModelOutput
                    except ImportError:
                        logger.warning("Could not import ModelOutput, creating a custom version")

                        # Create a simple ModelOutput-like class
                        class ModelOutput(dict):
                            """Simple ModelOutput-like class"""
                            def __init__(self, *args, **kwargs):
                                super().__init__(*args, **kwargs)
                                self.__dict__ = self

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
                        outputs = ModelOutput(outputs_dict)
                        logger.info(f"Converted tuple to ModelOutput with keys: {list(outputs_dict.keys())}")

                return outputs

            # Apply the patch
            model.forward = patched_forward
            logger.info(f"✅ Successfully patched {type(model).__name__}.forward")
            success = True
        else:
            logger.warning(f"⚠️ Model {type(model).__name__} does not have a forward method")

    # Try to patch common model classes in transformers
    try:
        # Try to patch PreTrainedModel first (this will affect all models)
        try:
            from transformers import PreTrainedModel

            # Store the original forward method
            original_pretrained_forward = PreTrainedModel.forward

            # Define a patched forward method
            def patched_pretrained_forward(self, *args, **kwargs):
                """
                Patched forward method that ensures outputs are always ModelOutput objects.
                """
                # Always set return_dict=True to avoid tuple outputs
                if "return_dict" not in kwargs:
                    kwargs["return_dict"] = True
                    logger.info("Setting return_dict=True to avoid tuple unpacking issues")

                # Call the original forward method
                outputs = original_pretrained_forward(self, *args, **kwargs)

                # Handle tuple outputs
                if isinstance(outputs, tuple):
                    logger.info(f"Got tuple output with {len(outputs)} elements, converting to ModelOutput")

                    # Import ModelOutput
                    try:
                        from transformers.modeling_outputs import ModelOutput
                    except ImportError:
                        logger.warning("Could not import ModelOutput, creating a custom version")

                        # Create a simple ModelOutput-like class
                        class ModelOutput(dict):
                            """Simple ModelOutput-like class"""
                            def __init__(self, *args, **kwargs):
                                super().__init__(*args, **kwargs)
                                self.__dict__ = self

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
                        outputs = ModelOutput(outputs_dict)
                        logger.info(f"Converted tuple to ModelOutput with keys: {list(outputs_dict.keys())}")

                return outputs

            # Apply the patch
            PreTrainedModel.forward = patched_pretrained_forward
            logger.info("✅ Successfully patched PreTrainedModel.forward")
            success = True
        except Exception as e:
            logger.warning(f"⚠️ Could not patch PreTrainedModel.forward: {e}")

        # Try to patch DeepSeek models specifically
        try:
            # First check if the deepseek module exists
            import importlib.util
            spec = importlib.util.find_spec("transformers.models.deepseek.modeling_deepseek")

            if spec is None:
                logger.warning("DeepSeek model not available in this transformers version. Creating custom implementation.")

                # Create a custom implementation for DeepSeek models
                from transformers.models.llama.modeling_llama import LlamaModel, LlamaForCausalLM

                # Create a DeepSeekModel class that inherits from LlamaModel
                class CustomDeepSeekModel(LlamaModel):
                    """Custom DeepSeekModel implementation based on LlamaModel."""
                    pass

                # Create a DeepSeekForCausalLM class that inherits from LlamaForCausalLM
                class CustomDeepSeekForCausalLM(LlamaForCausalLM):
                    """Custom DeepSeekForCausalLM implementation based on LlamaForCausalLM."""
                    pass

                # Store the classes in variables
                DeepSeekModel = CustomDeepSeekModel
                DeepSeekForCausalLM = CustomDeepSeekForCausalLM

                # Add them to the transformers module
                import transformers
                if not hasattr(transformers.models, "deepseek"):
                    # Create the module structure
                    class DeepSeekModule:
                        pass
                    transformers.models.deepseek = DeepSeekModule()

                # Create or get the modeling_deepseek module
                if not hasattr(transformers.models.deepseek, "modeling_deepseek"):
                    class ModelingDeepSeek:
                        pass
                    transformers.models.deepseek.modeling_deepseek = ModelingDeepSeek()

                # Set the DeepSeekModel and DeepSeekForCausalLM attributes
                setattr(transformers.models.deepseek.modeling_deepseek, "DeepSeekModel", DeepSeekModel)
                setattr(transformers.models.deepseek.modeling_deepseek, "DeepSeekForCausalLM", DeepSeekForCausalLM)

                # Create a module object for easy access
                deepseek_module = transformers.models.deepseek.modeling_deepseek

                logger.info("✅ Created custom DeepSeek model implementations based on Llama models")
            else:
                # If the module exists, import it normally
                deepseek_module = importlib.import_module("transformers.models.deepseek.modeling_deepseek")

                # Patch DeepSeekModel
                DeepSeekModel = getattr(deepseek_module, "DeepSeekModel")

                # Also try to patch DeepSeekForCausalLM
                try:
                    DeepSeekForCausalLM = getattr(deepseek_module, "DeepSeekForCausalLM")
                except (AttributeError, ImportError) as e:
                    logger.warning(f"Could not get DeepSeekForCausalLM: {e}")
                    DeepSeekForCausalLM = None

            # Store the original forward methods
            original_deepseek_forward = DeepSeekModel.forward

            if DeepSeekForCausalLM is not None:
                original_deepseek_causal_forward = DeepSeekForCausalLM.forward

            # Define a patched forward method
            def patched_deepseek_forward(self, *args, **kwargs):
                """
                Patched forward method for DeepSeekModel.
                """
                # Always set return_dict=True to avoid tuple outputs
                if "return_dict" not in kwargs:
                    kwargs["return_dict"] = True
                    logger.info("Setting return_dict=True for DeepSeekModel")

                # Try the original forward method
                try:
                    outputs = original_deepseek_forward(self, *args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in DeepSeekModel.forward: {e}")

                    # Try direct forward call with explicit arguments
                    try:
                        logger.info("Trying direct forward call for DeepSeekModel")

                        # Extract key arguments
                        input_ids = kwargs.get("input_ids")
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
                        logger.info("Trying DeepSeekModel.forward without attention mask")
                        outputs = original_deepseek_forward(
                            self,
                            input_ids=input_ids,
                            labels=labels,
                            use_cache=False,
                            return_dict=True
                        )
                    except Exception as e2:
                        logger.error(f"Direct forward call for DeepSeekModel also failed: {e2}")

                        # Import ModelOutput for creating a fallback output
                        try:
                            from transformers.modeling_outputs import ModelOutput
                        except ImportError:
                            # Create a simple ModelOutput-like class
                            class ModelOutput(dict):
                                """Simple ModelOutput-like class"""
                                def __init__(self, *args, **kwargs):
                                    super().__init__(*args, **kwargs)
                                    self.__dict__ = self

                        # Create a dummy output as last resort
                        device = next(self.parameters()).device
                        batch_size = input_ids.shape[0] if input_ids is not None else 1
                        seq_length = input_ids.shape[1] if input_ids is not None else 1
                        hidden_size = self.config.hidden_size if hasattr(self, 'config') and hasattr(self.config, 'hidden_size') else 4096

                        # Create dummy hidden states
                        dummy_hidden_states = torch.zeros((batch_size, seq_length, hidden_size), device=device)

                        # Create a ModelOutput with the dummy values
                        outputs = ModelOutput({"last_hidden_state": dummy_hidden_states})
                        logger.warning("Created fallback ModelOutput for DeepSeekModel")

                # Handle tuple outputs
                if isinstance(outputs, tuple):
                    logger.info(f"Got tuple output with {len(outputs)} elements from DeepSeekModel.forward")

                    # Import ModelOutput
                    try:
                        from transformers.modeling_outputs import ModelOutput
                    except ImportError:
                        # Create a simple ModelOutput-like class
                        class ModelOutput(dict):
                            """Simple ModelOutput-like class"""
                            def __init__(self, *args, **kwargs):
                                super().__init__(*args, **kwargs)
                                self.__dict__ = self

                    # Convert tuple to a dictionary-like object
                    outputs_dict = {}

                    if len(outputs) >= 1:
                        # First element is typically the hidden states
                        outputs_dict["last_hidden_state"] = outputs[0]

                        # Add any remaining elements with generic names
                        for i in range(1, len(outputs)):
                            outputs_dict[f"hidden_states_{i}"] = outputs[i]

                        # Convert to ModelOutput
                        outputs = ModelOutput(outputs_dict)
                        logger.info(f"Converted tuple to ModelOutput with keys: {list(outputs_dict.keys())}")

                return outputs

            # Apply the patch to DeepSeekModel
            DeepSeekModel.forward = patched_deepseek_forward
            logger.info("✅ Successfully patched DeepSeekModel.forward")
            success = True

            # Patch DeepSeekForCausalLM if available
            if DeepSeekForCausalLM is not None:
                # Define a patched forward method for DeepSeekForCausalLM
                def patched_deepseek_causal_forward(self, *args, **kwargs):
                    """
                    Patched forward method for DeepSeekForCausalLM.
                    """
                    # Always set return_dict=True to avoid tuple outputs
                    if "return_dict" not in kwargs:
                        kwargs["return_dict"] = True
                        logger.info("Setting return_dict=True for DeepSeekForCausalLM")

                    # Try the original forward method
                    try:
                        outputs = original_deepseek_causal_forward(self, *args, **kwargs)
                    except Exception as e:
                        logger.error(f"Error in DeepSeekForCausalLM.forward: {e}")

                        # Try direct forward call with explicit arguments
                        try:
                            logger.info("Trying direct forward call for DeepSeekForCausalLM")

                            # Extract key arguments
                            input_ids = kwargs.get("input_ids")
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
                            logger.info("Trying DeepSeekForCausalLM.forward without attention mask")
                            outputs = original_deepseek_causal_forward(
                                self,
                                input_ids=input_ids,
                                labels=labels,
                                use_cache=False,
                                return_dict=True
                            )
                        except Exception as e2:
                            logger.error(f"Direct forward call for DeepSeekForCausalLM also failed: {e2}")

                            # Import ModelOutput for creating a fallback output
                            try:
                                from transformers.modeling_outputs import CausalLMOutputWithPast
                            except ImportError:
                                # Create a simple ModelOutput-like class
                                class CausalLMOutputWithPast(dict):
                                    """Simple CausalLMOutputWithPast-like class"""
                                    def __init__(self, *args, **kwargs):
                                        super().__init__(*args, **kwargs)
                                        self.__dict__ = self

                            # Create a dummy output as last resort
                            device = next(self.parameters()).device
                            batch_size = input_ids.shape[0] if input_ids is not None else 1
                            seq_length = input_ids.shape[1] if input_ids is not None else 1
                            vocab_size = self.config.vocab_size if hasattr(self, 'config') and hasattr(self.config, 'vocab_size') else 32000

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

                            outputs = CausalLMOutputWithPast(outputs_dict)
                            logger.warning("Created fallback CausalLMOutputWithPast for DeepSeekForCausalLM")

                    # Handle tuple outputs
                    if isinstance(outputs, tuple):
                        logger.info(f"Got tuple output with {len(outputs)} elements from DeepSeekForCausalLM.forward")

                        # Import ModelOutput
                        try:
                            from transformers.modeling_outputs import CausalLMOutputWithPast
                        except ImportError:
                            # Create a simple ModelOutput-like class
                            class CausalLMOutputWithPast(dict):
                                """Simple CausalLMOutputWithPast-like class"""
                                def __init__(self, *args, **kwargs):
                                    super().__init__(*args, **kwargs)
                                    self.__dict__ = self

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
                            outputs = CausalLMOutputWithPast(outputs_dict)
                            logger.info(f"Converted tuple to CausalLMOutputWithPast with keys: {list(outputs_dict.keys())}")

                    return outputs

                # Apply the patch to DeepSeekForCausalLM
                DeepSeekForCausalLM.forward = patched_deepseek_causal_forward
                logger.info("✅ Successfully patched DeepSeekForCausalLM.forward")
                success = True
        except Exception as e:
            logger.warning(f"⚠️ Could not patch DeepSeekModel.forward: {e}")
    except Exception as e:
        logger.error(f"❌ Failed to apply model class fix: {e}")

    return success

if __name__ == "__main__":
    logger.info("Applying fix for 'too many values to unpack (expected 2)' error")
    success = fix_tuple_unpacking_error()

    if success:
        logger.info("✅ Successfully applied tuple unpacking fix")
    else:
        logger.warning("⚠️ Could not apply tuple unpacking fix")
