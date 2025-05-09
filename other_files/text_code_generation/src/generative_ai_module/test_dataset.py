#!/usr/bin/env python3
"""
Test for the TextDataset class and dataset handling

This script tests the TextDataset class from jarvis_unified.py 
to ensure it handles both callable tokenizers and pre-tokenized inputs correctly.
It also checks that tensors remain on CPU for DataLoader compatibility.
"""

import torch
import sys
import os
from pathlib import Path

# Add the parent directory to sys.path
current_dir = Path(__file__).parent
if str(current_dir.parent.parent) not in sys.path:
    sys.path.insert(0, str(current_dir.parent.parent))

from src.generative_ai_module.jarvis_unified import TextDataset

def test_dataset_with_dict_tokenizer():
    """Test TextDataset with pre-tokenized inputs (dictionary)"""
    # Create mock tokenized inputs - explicitly on CPU
    mock_tokenized = {
        'input_ids': torch.tensor([[1, 2, 3], [4, 5, 6]], device='cpu'),
        'attention_mask': torch.tensor([[1, 1, 1], [1, 1, 0]], device='cpu')
    }
    
    # Create dataset
    dataset = TextDataset(['text1', 'text2'], mock_tokenized)
    
    # Verify dataset length
    assert len(dataset) == 2, f"Expected dataset length 2, got {len(dataset)}"
    
    # Verify tensors are on CPU
    for key, tensor in dataset.encodings.items():
        assert tensor.device.type == 'cpu', f"Tensor {key} not on CPU: {tensor.device}"
    
    # Test __getitem__
    item = dataset[0]
    assert 'input_ids' in item, "input_ids missing from dataset item"
    assert 'attention_mask' in item, "attention_mask missing from dataset item"
    assert 'labels' in item, "labels missing from dataset item"
    
    # Ensure we compare on same device
    assert torch.equal(item['input_ids'], torch.tensor([1, 2, 3], device='cpu')), "Incorrect input_ids"
    assert torch.equal(item['labels'], torch.tensor([1, 2, 3], device='cpu')), "Labels should match input_ids"
    
    print("✅ Test with dict tokenizer passed!")

def test_dataset_with_callable_tokenizer():
    """Test TextDataset with callable tokenizer"""
    # Create mock tokenizer function that explicitly returns CPU tensors
    def mock_tokenizer(texts, truncation=None, padding=None, max_length=None, return_tensors=None):
        # Simple mock implementation with explicit CPU device
        return {
            'input_ids': torch.tensor([[10, 11, 12], [13, 14, 15]], device='cpu'),
            'attention_mask': torch.tensor([[1, 1, 1], [1, 1, 0]], device='cpu')
        }
    
    # Create dataset
    dataset = TextDataset(['text1', 'text2'], mock_tokenizer)
    
    # Verify dataset length
    assert len(dataset) == 2, f"Expected dataset length 2, got {len(dataset)}"
    
    # Verify tensors are on CPU
    for key, tensor in dataset.encodings.items():
        assert tensor.device.type == 'cpu', f"Tensor {key} not on CPU: {tensor.device}"
    
    # Test __getitem__
    item = dataset[0]
    assert 'input_ids' in item, "input_ids missing from dataset item"
    assert 'attention_mask' in item, "attention_mask missing from dataset item"
    assert 'labels' in item, "labels missing from dataset item"
    assert torch.equal(item['input_ids'], torch.tensor([10, 11, 12], device='cpu')), "Incorrect input_ids"
    
    print("✅ Test with callable tokenizer passed!")

def test_dataloader_compatibility():
    """Test compatibility with DataLoader"""
    from torch.utils.data import DataLoader
    
    # Create mock tokenized inputs - explicitly on CPU
    mock_tokenized = {
        'input_ids': torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], device='cpu'),
        'attention_mask': torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1]], device='cpu')
    }
    
    # Create dataset
    dataset = TextDataset(['text1', 'text2', 'text3', 'text4'], mock_tokenized)
    
    # Create DataLoader
    batch_size = 2
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Verify DataLoader works without errors
    batch = next(iter(dataloader))
    assert isinstance(batch, dict), "Batch should be a dictionary"
    assert 'input_ids' in batch, "input_ids missing from batch"
    assert 'attention_mask' in batch, "attention_mask missing from batch"
    assert 'labels' in batch, "labels missing from batch"
    assert batch['input_ids'].shape[0] == batch_size, f"Expected batch size {batch_size}, got {batch['input_ids'].shape[0]}"
    
    print("✅ Test with DataLoader passed!")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    
    print("🧪 Running TextDataset tests...")
    
    test_dataset_with_dict_tokenizer()
    test_dataset_with_callable_tokenizer()
    test_dataloader_compatibility()
    
    print("✅ All tests passed!") 