"""
Test script to verify the integration of our fixes.
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

def test_fix_integration():
    """Test the integration of our fixes."""
    try:
        # Add the project root to the Python path
        project_root = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, project_root)
        
        # Import our fixes
        from src.generative_ai_module.direct_model_fix import apply_direct_fix
        
        logger.info("Successfully imported direct_model_fix")
        
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
        
        # Test the fix
        logger.info("Creating mock objects for testing")
        model = MockPeftModel()
        
        # Apply the fix to the model
        logger.info("Applying the fix to the model")
        success = apply_direct_fix(model)
        
        if not success:
            logger.error("Failed to apply direct fix")
            return False
        
        # Test the patched forward method
        logger.info("Testing the patched forward method")
        
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
        logger.error(f"Error in test_fix_integration: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_fix_integration()
    if success:
        logger.info("✅ Fix integration works correctly!")
    else:
        logger.error("❌ Fix integration failed!")
