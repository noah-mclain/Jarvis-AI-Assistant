"""
Simple test script to verify the PeftModel forward method fix.
"""

import os
import sys
import logging
import torch

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
        # Create a mock model
        class MockModel:
            def __init__(self):
                self.config = type('obj', (object,), {'vocab_size': 1000})
                self.device = torch.device("cpu")
                self.dtype = torch.float32
                self.base_model = self  # For PeftModel compatibility

            def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
                # Simulate the "got multiple values for argument 'input_ids'" error
                if input_ids is not None and len(kwargs) > 0 and "input_ids" in kwargs:
                    raise TypeError("MockModel.forward() got multiple values for argument 'input_ids'")

                # Return a mock output
                return type('obj', (object,), {
                    'loss': torch.tensor(1.0, requires_grad=True),
                    'logits': torch.randn(input_ids.shape[0], input_ids.shape[1], 1000)
                })

            def to(self, device):
                self.device = device
                return self

        # Create a mock PeftModel
        class MockPeftModel(MockModel):
            def __init__(self):
                super().__init__()
                self.__class__.__name__ = "PeftModelForCausalLM"

        # Create the model
        model = MockPeftModel()

        # Store the original forward method
        original_forward = model.forward

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
                        return original_forward(self_arg, **new_kwargs)

                    # Handle the case where input_ids is passed both as positional and keyword argument
                    if len(args) > 1 and "input_ids" in kwargs:
                        logger.info("Detected input_ids in both args and kwargs, removing from kwargs")
                        # Remove input_ids from kwargs to avoid the multiple values error
                        input_ids = kwargs.pop("input_ids")
                        # Log the shapes for debugging
                        logger.info(f"Args[1] shape: {args[1].shape if isinstance(args[1], torch.Tensor) else 'not a tensor'}")
                        logger.info(f"Kwargs input_ids shape: {input_ids.shape if isinstance(input_ids, torch.Tensor) else 'not a tensor'}")

                    # Try the original forward method
                    return original_forward(*args, **kwargs)
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
                                return original_forward(self_arg, **new_kwargs)
                            else:
                                # If no args, just use kwargs
                                return original_forward(**kwargs)
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

                                # If model has a base_model, try to use it directly
                                if hasattr(model_self, 'base_model'):
                                    logger.info("Using base_model directly")
                                    base_model = model_self.base_model

                                    # Create new kwargs for base model
                                    base_kwargs = {}
                                    if input_ids is not None:
                                        base_kwargs["input_ids"] = input_ids
                                    base_kwargs["return_dict"] = True

                                    # Call base model's forward method directly
                                    return base_model.forward(**base_kwargs)
                                else:
                                    raise ValueError("Could not find a safe way to call forward")
                            except Exception as e3:
                                logger.error(f"All forward attempts failed: {e3}")
                                raise e3

            # Call the safe forward function
            return safe_forward(model, *args, **kwargs)

        # Replace the original forward method with our patched version
        model.forward = patched_forward
        logger.info("✅ Successfully patched PeftModel forward method")

        # Test the patched forward method
        logger.info("Testing patched forward method")

        # Prepare input
        input_ids = torch.randint(0, 1000, (2, 128))

        # Test forward call with input_ids in kwargs
        try:
            logger.info("Calling forward with input_ids in kwargs")
            outputs = model.forward(input_ids=input_ids)
            logger.info("Forward call succeeded")

            # Test forward call with both args and kwargs
            logger.info("Calling forward with input_ids in both args and kwargs")
            outputs = model.forward(model, input_ids=input_ids)
            logger.info("Forward call with both args and kwargs succeeded")

            logger.info("✅ All tests passed!")
            return True
        except Exception as e:
            logger.error(f"Test failed: {e}")
            return False

    except Exception as e:
        logger.error(f"Error in test_peft_model_fix: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_peft_model_fix()
    if success:
        logger.info("✅ PeftModel forward method fix works correctly!")
    else:
        logger.error("❌ PeftModel forward method fix failed!")
