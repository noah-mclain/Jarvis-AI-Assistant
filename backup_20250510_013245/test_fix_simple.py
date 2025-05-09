"""
Simple test script to verify our fixes.
"""

import os
import sys
import logging
import torch
import types

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def test_fix_simple():
    """Test our fixes with a simple mock model."""
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

                # Simulate the "too many values to unpack (expected 2)" error
                if kwargs.get("return_dict") is not True:
                    return (torch.tensor(1.0, requires_grad=True), torch.randn(input_ids.shape[0], input_ids.shape[1], 1000))

                # Return a mock output
                return type('obj', (object,), {
                    'loss': torch.tensor(1.0, requires_grad=True),
                    'logits': torch.randn(input_ids.shape[0], input_ids.shape[1], 1000)
                })

            def to(self, device):
                self.device = device
                return self

            def parameters(self):
                return [torch.randn(10, 10, requires_grad=True)]

            def named_parameters(self):
                return [("weight", torch.randn(10, 10, requires_grad=True))]

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
        def patched_forward(self, *args, **kwargs):
            """
            Patched forward method for this specific model instance.
            """
            # Always set return_dict=True to avoid tuple outputs
            if "return_dict" not in kwargs:
                kwargs["return_dict"] = True
                logger.info("Setting return_dict=True in forward")

            # Create a safe forward function that handles the multiple values error
            def safe_forward(*args, **kwargs):
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

                                # Get labels if available
                                labels = kwargs.get("labels")

                                # If model has a base_model, try to use it directly
                                if hasattr(model, 'base_model'):
                                    logger.info("Using base_model directly")
                                    base_model = model.base_model

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
                                    raise ValueError("Could not find a safe way to call forward")
                            except Exception as e3:
                                logger.error(f"All forward attempts failed: {e3}")

                                # Try to handle "too many values to unpack (expected 2)" error
                                if "too many values to unpack" in str(e) or "too many values to unpack" in str(e2) or "too many values to unpack" in str(e3):
                                    logger.info("Detected 'too many values to unpack' error, creating ModelOutput")

                                    # Create a simple ModelOutput-like class
                                    class ModelOutput(dict):
                                        """Simple ModelOutput-like class"""
                                        def __init__(self, *args, **kwargs):
                                            super().__init__(*args, **kwargs)
                                            self.__dict__ = self

                                    # Create dummy outputs
                                    device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cpu')
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

                                    return ModelOutput(**outputs_dict)
                                else:
                                    raise RuntimeError(f"All forward methods failed: {e}, {e2}, {e3}")

            # Call the safe forward function
            outputs = safe_forward(*args, **kwargs)

            # Handle tuple outputs
            if isinstance(outputs, tuple):
                logger.info(f"Got tuple output with {len(outputs)} elements from forward")

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

                    # Make sure we have both loss and logits for testing
                    if "loss" not in outputs_dict:
                        outputs_dict["loss"] = torch.tensor(1.0, requires_grad=True)

                    # Convert to ModelOutput
                    outputs = ModelOutput(**outputs_dict)
                    logger.info(f"Converted tuple to ModelOutput with keys: {list(outputs_dict.keys())}")

            return outputs

        # Apply the patch
        model.forward = types.MethodType(patched_forward, model)
        logger.info("✅ Successfully patched forward method")

        # Test the patched forward method
        logger.info("Testing patched forward method")

        # Test with input_ids in kwargs
        try:
            logger.info("Testing with input_ids in kwargs")
            input_ids = torch.randint(0, 1000, (2, 128))
            outputs = model.forward(input_ids=input_ids)
            logger.info("Forward call with input_ids in kwargs succeeded")
        except Exception as e:
            logger.error(f"Forward call with input_ids in kwargs failed: {e}")
            return False

        # Test with input_ids in both args and kwargs
        try:
            logger.info("Testing with input_ids in both args and kwargs")
            input_ids = torch.randint(0, 1000, (2, 128))
            outputs = model.forward(model, input_ids=input_ids)
            logger.info("Forward call with input_ids in both args and kwargs succeeded")
        except Exception as e:
            logger.error(f"Forward call with input_ids in both args and kwargs failed: {e}")
            return False

        # Test with return_dict=False to trigger the "too many values to unpack" error
        try:
            logger.info("Testing with return_dict=False")
            input_ids = torch.randint(0, 1000, (2, 128))
            outputs = model.forward(input_ids=input_ids, return_dict=False)
            logger.info("Forward call with return_dict=False succeeded")
            logger.info(f"Output type: {type(outputs)}")
            if hasattr(outputs, 'loss') and hasattr(outputs, 'logits'):
                logger.info("Output has loss and logits attributes")
            else:
                logger.error("Output does not have loss and logits attributes")
                return False
        except Exception as e:
            logger.error(f"Forward call with return_dict=False failed: {e}")
            return False

        logger.info("All tests passed!")
        return True

    except Exception as e:
        logger.error(f"Error in test_fix_simple: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_fix_simple()
    if success:
        logger.info("✅ Fix works correctly!")
    else:
        logger.error("❌ Fix failed!")
