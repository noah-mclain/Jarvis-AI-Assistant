"""
Unified testing tools for generative_ai_module
Consolidates test_imports.py, test_preprocessed_data.py, and test_data_adapter.py
"""

import os
import sys
import torch
import argparse

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_imports():
    """Test that imports work correctly"""
    print("\n===== Testing Module Imports =====")
    
    # Try the direct imports
    try:
        from src.generative_ai_module.text_generator import TextGenerator
        print("✓ Successfully imported TextGenerator with absolute path")
    except ImportError as e:
        print(f"✗ Failed to import TextGenerator with absolute path: {e}")
        
    # Try relative imports
    try:
        from generative_ai_module.text_generator import TextGenerator
        print("✓ Successfully imported TextGenerator with relative path")
    except ImportError as e:
        print(f"✗ Failed to import TextGenerator with relative path: {e}")
    
    # Try intra-package imports
    try:
        import sys
        sys.path.append('..')  # Add one more level up to the path
        from src.generative_ai_module import text_generator
        print("✓ Successfully imported text_generator module")
    except ImportError as e:
        print(f"✗ Failed to import text_generator module: {e}")
    
    print("\nTesting non-standard imports...")
    
    # Try prompt_enhancer which uses utils
    try:
        from src.generative_ai_module.prompt_enhancer import PromptEnhancer
        print("✓ Successfully imported PromptEnhancer")
    except ImportError as e:
        print(f"✗ Failed to import PromptEnhancer: {e}")
    
    # Test importing the BasicTokenizer
    try:
        from generative_ai_module.basic_tokenizer import BasicTokenizer
        print("✓ Successfully imported BasicTokenizer")
    except ImportError as e:
        print(f"✗ Failed to import BasicTokenizer: {e}")
    
    # Try full chain of imports
    print("\nTesting import chain...")
    try:
        from src.generative_ai_module.unified_generation_pipeline import (
            load_preprocessed_data, 
            train_text_generator
        )
        print("✓ Successfully imported from unified_generation_pipeline")
    except ImportError as e:
        print(f"✗ Failed to import from unified_generation_pipeline: {e}")
        
    print("\nImport test completed")

def load_preprocessed_data(dataset_name="persona_chat"):
    """Load preprocessed data from disk"""
    # Get the current directory (where this file lives) and project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../.."))

    # Path to 'preprocessed_data' at the root
    preprocessed_dir = os.path.join(project_root, "preprocessed_data")
    preprocessed_path = os.path.join(preprocessed_dir, f"{dataset_name}_preprocessed.pt")

    print(f"Looking for preprocessed data at: {preprocessed_path}")

    if not os.path.exists(preprocessed_path):
        print(f"❌ Preprocessed data not found at {preprocessed_path}")
        return None

    print(f"✅ Found preprocessed data at {preprocessed_path}")
    try:
        data = torch.load(preprocessed_path)
        print("✅ Successfully loaded data")
        return data
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def inspect_data_structure(data):
    """Print details about the data structure"""
    if data is None:
        print("❌ No data to inspect")
        return

    print("\n=== Data Structure ===")

    # Check top-level keys
    print(f"Top-level keys: {', '.join(data.keys())}")

    # Check batches
    if 'batches' in data:
        batches = data['batches']
        print(f"Number of batches: {len(batches)}")

        if batches:
            batch_inspection(batches)
    else:
        print("❌ No batches found in data")

    # Check other structures like vocabulary if present
    if 'vocab_size' in data:
        print(f"\nVocabulary size: {data['vocab_size']}")

    if 'dataset_name' in data:
        print(f"Dataset name: {data['dataset_name']}")


def batch_inspection(batches):
    # Look at the first batch
    inputs, targets = batches[0]
    print(f"First batch input shape: {inputs.shape}")
    print(f"First batch target shape: {targets.shape}")
    print(f"Input dtype: {inputs.dtype}")
    print(f"Target dtype: {targets.dtype}")

    # Sample some values
    print("\nSample input values:")
    try:
        if inputs.dim() == 2:
            print(inputs[0, :10])  # First row, first 10 columns
        elif inputs.dim() == 3:
            print(inputs[0, 0, :10])  # First row, first seq, first 10 features
    except Exception as e:
        print(f"Error sampling input values: {e}")

    print("\nSample target values:")
    try:
        if targets.dim() == 1:
            print(targets[:10])  # First 10 values
        else:
            print(targets[0, :10] if targets.shape[1] >= 10 else targets[0])
    except Exception as e:
        print(f"Error sampling target values: {e}")

def adapt_data_for_model(data, verbose=True):
    """Adapt data format for the model"""
    if data is None or 'batches' not in data or not data['batches']:
        print("❌ No valid data to adapt")
        return data
    
    print("\n=== Adapting Data Format ===")
    
    adapted_batches = []
    
    for i, (inputs, targets) in enumerate(data['batches']):
        if verbose and i < 3:
            print(f"\nBatch {i}:")
            print(f"  Original input shape: {inputs.shape}")
            print(f"  Original target shape: {targets.shape}")
        
        # Ensure inputs are appropriate shape
        if inputs.dim() == 3 and inputs.shape[1] == 1:
            inputs = inputs.squeeze(1)
            if verbose and i < 3:
                print(f"  ➡️ Squeezed input to shape: {inputs.shape}")
        
        # Ensure targets are appropriate shape  
        if targets.dim() > 1:
            targets = targets.squeeze()
            if verbose and i < 3:
                print(f"  ➡️ Squeezed target to shape: {targets.shape}")
        
        adapted_batches.append((inputs, targets))
    
    adapted_data = data.copy()
    adapted_data['batches'] = adapted_batches
    print(f"✅ Adapted {len(adapted_batches)} batches")
    
    return adapted_data

def test_data_loading():
    """Test data loading and preprocessing"""
    print("\n===== Testing Data Loading and Adaptation =====")

    # 1. Test persona_chat data
    print("\n== Testing persona_chat dataset ==")
    if persona_data := load_preprocessed_data("persona_chat"):
        inspect_data_structure(persona_data)
        adapted_persona = adapt_data_for_model(persona_data)

    # 2. Test writing_prompts data
    print("\n== Testing writing_prompts dataset ==")
    if prompts_data := load_preprocessed_data("writing_prompts"):
        inspect_data_structure(prompts_data)
        adapted_prompts = adapt_data_for_model(prompts_data)

    print("\n=== Test Complete ===")

def test_preprocessing():
    """Test preprocessing and dataset functionality"""
    print("\n===== Testing Preprocessing Functionality =====")

    try:
        from generative_ai_module.dataset_processor import DatasetProcessor
        from generative_ai_module.text_generator import TextGenerator

        # Initialize processor and generator
        generator = TextGenerator()
        processor = DatasetProcessor(generator)

        print("✓ Successfully initialized DatasetProcessor and TextGenerator")

        # Test loading preprocessed data
        print("\nTesting preprocessed data loading:")
        try:
            data = processor.load_preprocessed_data("persona_chat")
            print("✓ Successfully loaded preprocessed data")
            print(f"  Dataset contains {len(data.get('batches', []))} batches")
        except Exception as e:
            print(f"✗ Failed to load preprocessed data: {e}")

        # Test sequence creation with sample text
        print("\nTesting sequence creation:")
        try:
            sample_text = """<PERSONA>
- I am a teacher
<DIALOGUE>
USER: What do you do for a living?
ASSISTANT: I teach mathematics at a high school."""

            cleaned_text = processor.clean_text(sample_text)
            sequences = processor.create_sequences(cleaned_text, sequence_length=50)

            print(f"✓ Successfully created {len(sequences)} sequences from sample text")

            # Create batches
            if sequences:
                batches = processor.create_batches(sequences, batch_size=16)
                print(f"✓ Successfully created {len(batches)} batches")

                if batches:
                    inputs, targets = batches[0]
                    print(f"  Batch input shape: {inputs.shape}")
                    print(f"  Batch target shape: {targets.shape}")
        except Exception as e:
            print(f"✗ Failed in sequence/batch creation: {e}")
            import traceback
            traceback.print_exc()

    except ImportError as e:
        print(f"✗ Failed to import preprocessing components: {e}")

    print("\n=== Preprocessing Test Complete ===")

def main():
    """Main function combining all tests"""
    parser = argparse.ArgumentParser(description="Unified testing tools for generative_ai_module")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--imports", action="store_true", help="Test imports")
    parser.add_argument("--data", action="store_true", help="Test data loading and adaptation")
    parser.add_argument("--preprocessing", action="store_true", help="Test preprocessing functionality")
    
    args = parser.parse_args()
    
    # If no specific tests are selected, run all tests
    run_all = args.all or not (args.imports or args.data or args.preprocessing)
    
    print("===== Unified Testing Tools =====")
    
    if args.imports or run_all:
        test_imports()
    
    if args.data or run_all:
        test_data_loading()
    
    if args.preprocessing or run_all:
        test_preprocessing()
        
    print("\n===== Testing Complete =====")

if __name__ == "__main__":
    main() 