"""Simple example of using LAION 2B dataset with streaming.

This script provides a minimal example of how to:
1. Stream data from the LAION 2B dataset
2. Process a few samples
3. Display basic information about the samples

"""

import os
import sys
from tqdm import tqdm
from huggingface_hub import login
from dotenv import load_dotenv


# Add the project root to the path so we can import from data.preprocessing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing.dataset import load_laion_dataset
from data.preprocessing.image_utils import download_image
load_dotenv()
login(token=os.getenv('token'))

print(os.getenv('laion2b'))

def main():
    # Load LAION-2B English dataset in streaming mode
    print("Loading LAION-2B English dataset...")
    dataset = load_laion_dataset(dataset_name="laion/laion2B-en-aesthetic", split="train", streaming=True)
    
    # Process a few samples
    num_samples = 5
    print(f"Processing {num_samples} samples...")
    
    for i, sample in enumerate(dataset.take(num_samples)):
        print(f"\nSample {i+1}:")
        print(f"  Caption: {sample['TEXT'][:100]}..." if len(sample['TEXT']) > 100 else f"  Caption: {sample['TEXT']}")
        print(f"  URL: {sample['URL']}")
        
        # Print metadata if available
        if 'WIDTH' in sample and 'HEIGHT' in sample:
            print(f"  Dimensions: {sample['WIDTH']} x {sample['HEIGHT']}")
        
        # Try to download the image (but don't display it)
        print("  Downloading image...", end="")
        image = download_image(sample['URL'])
        if image is not None:
            print(f" Success! Image size: {image.size}")
        else:
            print(" Failed to download image")
    
    print("  python scripts/laion_dataset_demo.py --num_samples 10 --visualize")
    print("\nFor batch processing and saving:")
    print("  python scripts/laion_batch_processor.py --output_dir ./processed_laion --num_batches 5")


if __name__ == "__main__":
    main()