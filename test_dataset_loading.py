#!/usr/bin/env python
"""
Test script for memory-efficient HuggingFace dataset loading
"""
import os
import sys
from pathlib import Path

# Add parent directory to path so we can import the module
sys.path.insert(0, str(Path(__file__).parent))

def test_dataset_loading():
    """Test loading a HuggingFace dataset with memory efficiency improvements"""
    try:
        from src.generative_ai_module.unified_dataset_handler import UnifiedDatasetHandler
        from src.generative_ai_module.dataset_processor import DatasetProcessor
        from transformers import AutoTokenizer
        
        print("Initializing tokenizer and handlers...")
        
        # Initialize a small tokenizer
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        
        # Initialize dataset handler
        dataset_handler = UnifiedDatasetHandler(
            tokenizer=tokenizer,
            max_length=128,  # Small value for testing
            batch_size=4
        )
        
        # Test with a max_samples limit to prevent OOM
        max_samples = 500  # Small number for testing
        hf_dataset = "teknium/GPTeacher-General-Instruct"
        
        print(f"Loading HuggingFace dataset: {hf_dataset} with max_samples={max_samples}")
        dataset = dataset_handler.load_dataset(
            dataset_name=hf_dataset,
            split="train",
            max_samples=max_samples,
            use_cache=False  # Force fresh load to test the implementation
        )
        
        # Print dataset information
        print(f"Successfully loaded dataset with {len(dataset['train'])} samples")
        
        # Verify our dataset structure
        if 'text' in dataset['train'].column_names:
            print(f"Dataset has 'text' column with {len(dataset['train']['text'])} items")
            print(f"Sample text entry: {dataset['train']['text'][0][:100]}...")
            
        if 'input_ids' in dataset['train'].column_names:
            print(f"Dataset has 'input_ids' column with {len(dataset['train']['input_ids'])} items")
            
        # Test the DatasetProcessor directly
        print("\nTesting DatasetProcessor directly...")
        processor = DatasetProcessor()
        
        # Test with a very small sample to avoid memory issues
        processor_max_samples = 100
        print(f"Processing HuggingFace dataset with processor, max_samples={processor_max_samples}")
        
        dataset_output = processor.prepare_dataset(
            source=hf_dataset,
            max_samples=processor_max_samples,
            sequence_length=64,
            batch_size=4
        )
        
        if dataset_output and 'batches' in dataset_output:
            print(f"DatasetProcessor returned {len(dataset_output['batches'])} batches")
            
        print("\nAll tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"Error testing dataset loading: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run the test function
    success = test_dataset_loading()
    sys.exit(0 if success else 1) 