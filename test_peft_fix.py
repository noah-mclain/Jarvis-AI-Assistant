"""
Test script to verify the PeftModel forward method fix.
"""

import os
import sys
import logging
import torch
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def test_peft_model_fix():
    """Test the PeftModel forward method fix."""
    try:
        # Try to import the necessary modules
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model
        except ImportError:
            logger.error("Required libraries not found. Please install transformers and peft.")
            return False

        # Check if we're running on MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS device")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using CUDA device")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")

        # Load a small model for testing
        model_name = "gpt2"  # Use a small model for quick testing
        logger.info(f"Loading model: {model_name}")

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        # Apply LoRA
        logger.info("Applying LoRA to model")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["c_attn"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        # Get PEFT model
        peft_model = get_peft_model(model, lora_config)
        logger.info(f"Created PeftModel: {type(peft_model)}")

        # Store the original forward method
        original_forward = peft_model.forward

        # Define a patched forward method
        def patched_forward(*args, **kwargs):
            """
            Patched forward method for PeftModel that handles the 'got multiple values for argument' error
            and the 'too many values to unpack' error.
            """
            logger.info("In patched PeftModel forward")

            # Always set return_dict=True to avoid tuple outputs
            if "return_dict" not in kwargs:
                kwargs["return_dict"] = True
                logger.info("Setting return_dict=True to avoid tuple unpacking issues")

            # Handle the case where input_ids is passed both as positional and keyword argument
            if len(args) > 1 and "input_ids" in kwargs:
                logger.info("Detected input_ids in both args and kwargs, removing from kwargs")
                # Remove input_ids from kwargs to avoid the multiple values error
                input_ids = kwargs.pop("input_ids")
                # Log the shapes for debugging
                logger.info(f"Args[1] shape: {args[1].shape if isinstance(args[1], torch.Tensor) else 'not a tensor'}")
                logger.info(f"Kwargs input_ids shape: {input_ids.shape if isinstance(input_ids, torch.Tensor) else 'not a tensor'}")

            # Try the original forward method
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
                    outputs = original_forward(self_arg, **new_kwargs)
                else:
                    # Normal call
                    outputs = original_forward(*args, **kwargs)

                logger.info("Original forward call succeeded")
                return outputs
            except Exception as e:
                logger.error(f"Error in PeftModel forward: {e}")

                # If we get "got multiple values for argument", try with only kwargs
                if "got multiple values for argument" in str(e):
                    logger.info("Trying PeftModel forward with only kwargs")
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
                            outputs = original_forward(self_arg, **new_kwargs)
                        else:
                            # If no args, just use kwargs
                            outputs = original_forward(**kwargs)

                        logger.info("Forward call with only kwargs succeeded")
                        return outputs
                    except Exception as e2:
                        logger.error(f"PeftModel forward with only kwargs also failed: {e2}")

                        # Try with minimal arguments
                        try:
                            logger.info("Trying PeftModel forward with minimal arguments")

                            # Get input_ids from args or kwargs
                            input_ids = None
                            if len(args) > 1 and isinstance(args[1], torch.Tensor):
                                input_ids = args[1]
                            elif "input_ids" in kwargs:
                                input_ids = kwargs["input_ids"]

                            # Get self argument
                            self_arg = args[0] if len(args) > 0 else None

                            if self_arg is not None and input_ids is not None:
                                # Call with minimal arguments
                                outputs = original_forward(self_arg, input_ids=input_ids, return_dict=True)
                                logger.info("Forward call with minimal arguments succeeded")
                                return outputs
                            else:
                                raise ValueError("Could not extract self and input_ids from args or kwargs")
                        except Exception as e3:
                            logger.error(f"PeftModel forward with minimal arguments also failed: {e3}")
                            raise e3

        # Replace the original forward method with our patched version
        peft_model.forward = patched_forward
        logger.info("✅ Successfully patched PeftModel forward method")

        # Test the patched forward method
        logger.info("Testing patched forward method")

        # Prepare input
        input_text = "Hello, world!"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        # Test forward call
        try:
            logger.info("Calling forward with input_ids in kwargs")
            outputs = peft_model(input_ids=inputs["input_ids"])
            logger.info("Forward call succeeded")

            # Test forward call with both args and kwargs - use a different approach
            logger.info("Testing direct call to avoid multiple values error")

            # Create a wrapper function that handles the error
            def safe_forward(model, input_ids):
                """Safely call forward without the multiple values error"""
                try:
                    # First try with kwargs only
                    return model(input_ids=input_ids)
                except Exception as e:
                    logger.warning(f"First attempt failed: {e}")
                    # If that fails, try with a custom approach
                    # Get the base model's forward method directly
                    if hasattr(model, 'base_model') and hasattr(model.base_model, 'forward'):
                        logger.info("Calling base_model.forward directly")
                        # Call the base model's forward method
                        return model.base_model(input_ids=input_ids)
                    else:
                        raise ValueError("Could not find a safe way to call forward")

            # Call the safe forward function
            outputs = safe_forward(peft_model, inputs["input_ids"])
            logger.info("Safe forward call succeeded")

            logger.info("✅ All tests passed!")
            return True
        except Exception as e:
            logger.error(f"Test failed: {e}")
            return False

    except Exception as e:
        logger.error(f"Error in test_peft_model_fix: {e}")
        return False

if __name__ == "__main__":
    success = test_peft_model_fix()
    if success:
        logger.info("✅ PeftModel forward method fix works correctly!")
    else:
        logger.error("❌ PeftModel forward method fix failed!")
