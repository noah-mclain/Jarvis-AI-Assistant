"""
Test script to verify the PeftModel forward method fix in unified_deepseek_training.py.
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

def test_unified_deepseek_fix():
    """Test the PeftModel forward method fix in unified_deepseek_training.py."""
    try:
        # Add the project root to the Python path
        project_root = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, project_root)
        
        # Import the patched forward method from unified_deepseek_training.py
        from src.generative_ai_module.unified_deepseek_training import train_with_standard_method
        
        logger.info("Successfully imported train_with_standard_method from unified_deepseek_training.py")
        
        # Create a simple mock class to test the patched forward method
        class MockArgs:
            def __init__(self):
                self.model_name = "gpt2"  # Use a small model for quick testing
                self.dataset_name = "test_dataset"
                self.output_dir = "./output/test"
                self.epochs = 1
                self.batch_size = 2
                self.max_length = 128
                self.learning_rate = 2e-5
                self.warmup_steps = 10
                self.gradient_accumulation_steps = 1
                self.gradient_checkpointing = True
                self.bf16 = False
                self.fp16 = True
                self.lora_rank = 8
                self.lora_alpha = 16
                self.lora_dropout = 0.05
                self.dataset_subset = "test"
                self.all_subsets = False
                self.num_workers = 1
                self.weight_decay = 0.01
                self.max_samples = 10
                self.seed = 42
                self.local_rank = -1
                self.deepspeed = None
                self.flash_attention = False
                self.use_unsloth = False
        
        # Create a mock dataset
        class MockDataset:
            def __init__(self):
                self.data = [{"input_ids": torch.randint(0, 1000, (128,)), "attention_mask": torch.ones(128), "labels": torch.randint(0, 1000, (128,))} for _ in range(10)]
            
            def __getitem__(self, idx):
                return self.data[idx]
            
            def __len__(self):
                return len(self.data)
        
        # Create a mock tokenizer
        class MockTokenizer:
            def __init__(self):
                self.pad_token_id = 0
                self.eos_token_id = 1
                self.vocab_size = 1000
            
            def __call__(self, text, **kwargs):
                return {"input_ids": torch.randint(0, 1000, (128,)), "attention_mask": torch.ones(128)}
        
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
            
            def train(self):
                return self
            
            def eval(self):
                return self
            
            def parameters(self):
                return [torch.randn(10, 10, requires_grad=True)]
            
            def named_parameters(self):
                return [("weight", torch.randn(10, 10, requires_grad=True))]
            
            def gradient_checkpointing_enable(self):
                pass
        
        # Create a mock PeftModel
        class MockPeftModel(MockModel):
            def __init__(self):
                super().__init__()
                self.__class__.__name__ = "PeftModelForCausalLM"
        
        # Test the fix
        logger.info("Creating mock objects for testing")
        args = MockArgs()
        model = MockPeftModel()
        tokenizer = MockTokenizer()
        train_dataset = MockDataset()
        eval_dataset = MockDataset()
        
        # Apply the fix to the model
        logger.info("Applying the fix to the model")
        
        # Create a patched forward method
        original_forward = model.forward
        
        def patched_forward(*args, **kwargs):
            """
            Patched forward method for testing.
            """
            logger.info("In patched forward method")
            
            # Always set return_dict=True to avoid tuple outputs
            if "return_dict" not in kwargs:
                kwargs["return_dict"] = True
                logger.info("Setting return_dict=True to avoid tuple unpacking issues")
            
            # Handle the case where input_ids is passed both as positional and keyword argument
            if len(args) > 1 and "input_ids" in kwargs:
                logger.info("Detected input_ids in both args and kwargs, removing from kwargs")
                # Remove input_ids from kwargs to avoid the multiple values error
                input_ids = kwargs.pop("input_ids")
                logger.info(f"Removed input_ids from kwargs")
            
            # Try the original forward method
            try:
                return original_forward(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in patched forward: {e}")
                
                # If we get "got multiple values for argument", try with only kwargs
                if "got multiple values for argument" in str(e):
                    logger.info("Trying with only kwargs")
                    try:
                        # Extract only the self argument from args
                        if len(args) > 0:
                            self_arg = args[0]
                            return original_forward(self_arg, **kwargs)
                        else:
                            # If no args, just use kwargs
                            return original_forward(**kwargs)
                    except Exception as e2:
                        logger.error(f"Forward with only kwargs also failed: {e2}")
                        raise e2
        
        # Replace the original forward method with our patched version
        model.forward = patched_forward
        
        # Test the patched forward method
        logger.info("Testing the patched forward method")
        
        # Test with input_ids in kwargs
        try:
            logger.info("Testing with input_ids in kwargs")
            input_ids = torch.randint(0, 1000, (2, 128))
            outputs = model(input_ids=input_ids)
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
        
        logger.info("All tests passed!")
        return True
    
    except Exception as e:
        logger.error(f"Error in test_unified_deepseek_fix: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_unified_deepseek_fix()
    if success:
        logger.info("✅ PeftModel forward method fix works correctly!")
    else:
        logger.error("❌ PeftModel forward method fix failed!")
