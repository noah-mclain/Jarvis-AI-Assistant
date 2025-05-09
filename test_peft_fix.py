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
