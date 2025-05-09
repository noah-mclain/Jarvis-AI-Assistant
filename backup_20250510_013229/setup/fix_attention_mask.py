#!/usr/bin/env python3
"""
Fix for the transformers attention mask issue in DeepSeek-Coder models.

This script patches the LlamaModel.forward method to properly handle attention masks,
fixing the 'Attention mask should be of size (batch_size, 1, seq_length, seq_length)' error.
"""

import torch
import sys
import os
import inspect
from typing import Optional, Tuple, Union, List, Dict, Any

def debug_function_signature(func):
    """Print detailed information about a function's signature"""'
    sig = inspect.signature(func)
    print(f"Function: {func.__name__}")
    print(f"Signature: {sig}")
    print(f"Parameters:")
    for name, param in sig.parameters.items():
        print(f"  - {name}: {param.kind} (default: {param.default if param.default is not param.empty else 'required'})")
    print()

def patch_llama_model_forward():
    """
    Patch the LlamaModel.forward method to properly handle attention masks.
    This fixes the 'Attention mask should be of size (batch_size, 1, seq_length, seq_length)' error.
    """
    try:
        from transformers.models.llama.modeling_llama import LlamaModel
        
        # Store the original forward method
        original_forward = LlamaModel.forward
        
        # Debug the original forward method
        print("\n===== Original LlamaModel.forward Method =====")
        debug_function_signature(original_forward)
        
        # Define a patched forward method that properly handles attention masks
        def patched_forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ):
            """
            Patched forward method for LlamaModel that properly handles attention masks.
            """
            # Force use_cache to False when using gradient checkpointing
            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                if use_cache:
                    print("use_cache=True is incompatible with gradient checkpointing. Setting use_cache=False...")
                use_cache = False
            
            # Fix attention mask shape if needed
            if attention_mask is not None and attention_mask.dim() == 2:
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
                
                print(f"Fixed attention mask shape: {attention_mask.shape}")
            
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
        
        print("Successfully patched LlamaModel.forward method")
        return True
        
    except ImportError as e:
        print(f"Error importing transformers: {e}")
        return False
    except Exception as e:
        print(f"Error patching LlamaModel.forward: {e}")
        return False

def patch_attention_mask_in_dataset_collator():
    """
    Patch the data collator to ensure attention masks have the correct shape.
    """
    try:
        from transformers import DataCollatorForLanguageModeling
        
        # Store the original __call__ method
        original_call = DataCollatorForLanguageModeling.__call__
        
        # Define a patched __call__ method
        def patched_call(self, features, return_tensors=None):
            batch = original_call(self, features, return_tensors)
            
            # Fix attention mask shape if needed
            if "attention_mask" in batch and batch["attention_mask"].dim() == 2:
                print("Fixing attention mask shape in data collator...")
                
                # Get dimensions
                batch_size, seq_length = batch["attention_mask"].shape
                device = batch["attention_mask"].device
                
                # Reshape to 4D
                batch["attention_mask"] = batch["attention_mask"].unsqueeze(1).unsqueeze(2).expand(
                    batch_size, 1, seq_length, seq_length
                )
                
                print(f"Attention mask shape after fix: {batch['attention_mask'].shape}")
            
            return batch
        
        # Replace the original __call__ method with our patched version
        DataCollatorForLanguageModeling.__call__ = patched_call
        
        print("Successfully patched DataCollatorForLanguageModeling.__call__ method")
        return True
        
    except ImportError as e:
        print(f"Error importing transformers: {e}")
        return False
    except Exception as e:
        print(f"Error patching data collator: {e}")
        return False

def main():
    """Main function to fix the attention mask error"""
    print("=" * 50)
    print("FIXING ATTENTION MASK ERROR")
    print("=" * 50)
    
    # Patch the LlamaModel.forward method
    success1 = patch_llama_model_forward()
    
    # Patch the data collator
    success2 = patch_attention_mask_in_dataset_collator()
    
    if success1 and success2:
        print("\nSuccessfully applied all patches!")
        print("The attention mask error should now be fixed.")
    else:
        print("\nSome patches failed to apply. The error might still occur.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
