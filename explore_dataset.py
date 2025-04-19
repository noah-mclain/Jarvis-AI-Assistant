"""
Dataset Explorer

This script explores the structure of the code_search_net dataset to understand its schema.
"""

from datasets import load_dataset
import json

def explore_dataset(subset="python"):
    print(f"Loading code_search_net dataset ({subset} subset) to examine its structure...")
    try:
        # Load dataset with trust_remote_code=True
        dataset = load_dataset("code_search_net", subset, trust_remote_code=True)
        
        # Print available splits
        print(f"Available splits: {dataset.keys()}")
        
        # Get a single example
        train_example = dataset["train"][0]
        
        # Print all keys in the example
        print("\nKeys in dataset example:")
        for key in train_example.keys():
            print(f"  - {key}")
        
        # Print a sample of each field (truncated for readability)
        print("\nSample values (truncated):")
        for key, value in train_example.items():
            if isinstance(value, str):
                # Truncate string values to 100 chars
                display_value = value[:100] + "..." if len(value) > 100 else value
            else:
                display_value = value
            print(f"  {key}: {display_value}")
        
        # Save full example to a file for inspection
        with open(f"sample_{subset}_example.json", "w") as f:
            json.dump(train_example, f, indent=2)
        print(f"\nFull example saved to sample_{subset}_example.json")
        
    except Exception as e:
        print(f"Error exploring dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    # Use command line argument or default to python
    subset = sys.argv[1] if len(sys.argv) > 1 else "python"
    explore_dataset(subset) 