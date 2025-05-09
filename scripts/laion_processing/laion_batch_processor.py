"""LAION 2B dataset batch processing and saving.

This script demonstrates how to:
1. Stream data from the LAION 2B dataset using Hugging Face datasets
2. Process data in batches for efficiency
3. Apply filtering based on metadata
4. Save processed samples to disk for later use
5. Resume processing from checkpoints
"""

import os
import sys
import torch
import argparse
import json
from tqdm import tqdm
import pickle
from huggingface_hub import login
from pathlib import Path

# Add the project root to the path so we can import from data.preprocessing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing.dataset import load_laion_dataset, ProcessedImageDataset
from data.preprocessing.image_utils import (
    download_image, 
    get_image_transforms, 
    filter_laion_samples,
    extract_clip_features
)
login(token=os.getenv('token'))



def process_batch(batch, transform, clip_model=None, clip_processor=None):
    """Process a batch of LAION samples.
    
    Args:
        batch (list): List of LAION samples
        transform: Image transformation pipeline
        clip_model: Optional CLIP model for feature extraction
        clip_processor: Optional CLIP processor for feature extraction
        
    Returns:
        list: List of processed samples
    """
    processed_samples = []
    
    for sample in batch:
        # Download image
        image = download_image(sample['URL'])
        if image is None:
            continue
        
        # Apply transformations
        try:
            transformed_image = transform(image)
        except Exception as e:
            print(f"Error transforming image: {e}")
            continue
        
        # Create processed sample
        processed_sample = {
            'id': sample['SAMPLE_ID'],
            'image': transformed_image,
            'caption': sample['TEXT'],
            'image_features': None,
            'text_features': None,
            'metadata': {
                'width': sample.get('WIDTH'),
                'height': sample.get('HEIGHT'),
                'url': sample['URL']
            }
        }
        
        # Extract CLIP features if model is available
        if clip_model is not None and clip_processor is not None:
            try:
                image_features, text_features = extract_clip_features(
                    clip_model, clip_processor, image, sample['TEXT'])
                processed_sample['image_features'] = image_features
                processed_sample['text_features'] = text_features
            except Exception as e:
                print(f"Error extracting CLIP features: {e}")
        
        processed_samples.append(processed_sample)
    
    return processed_samples


def save_checkpoint(processed_samples, output_dir, batch_idx):
    """Save processed samples to disk.
    
    Args:
        processed_samples (list): List of processed samples
        output_dir (str): Directory to save to
        batch_idx (int): Batch index for filename
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"batch_{batch_idx:05d}.pkl")
    
    with open(output_file, 'wb') as f:
        pickle.dump(processed_samples, f)
    
    # Save metadata separately (without the large tensors) for easy inspection
    metadata = []
    for sample in processed_samples:
        metadata.append({
            'id': sample['id'],
            'caption': sample['caption'],
            'metadata': sample['metadata']
        })
    
    metadata_file = os.path.join(output_dir, f"metadata_{batch_idx:05d}.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)


def load_checkpoint(output_dir):
    """Load checkpoint information to resume processing.
    
    Args:
        output_dir (str): Directory with saved batches
        
    Returns:
        int: Next batch index to process
    """
    if not os.path.exists(output_dir):
        return 0
    
    # Find the highest batch index
    batch_files = list(Path(output_dir).glob("batch_*.pkl"))
    if not batch_files:
        return 0
    
    # Extract batch indices from filenames
    indices = [int(f.stem.split('_')[1]) for f in batch_files]
    return max(indices) + 1


def load_clip_model():
    """Load CLIP model if available.
    
    Returns:
        tuple: (model, processor) or (None, None) if not available
    """
    try:
        import clip
        model, processor = clip.load("ViT-B/32")
        model.eval()
        return model, processor
    except ImportError:
        print("CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")
        return None, None


def main():
    parser = argparse.ArgumentParser(description="LAION dataset batch processing")
    parser.add_argument("--dataset", type=str, default="laion/laion2B-en", 
                        help="LAION dataset to use (default: laion/laion2B-en)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save processed batches")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for processing")
    parser.add_argument("--num_batches", type=int, default=10,
                        help="Number of batches to process")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Size to resize images to")
    parser.add_argument("--use_clip", action="store_true",
                        help="Extract CLIP features")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--min_width", type=int, default=256,
                        help="Minimum image width from metadata")
    parser.add_argument("--min_height", type=int, default=256,
                        help="Minimum image height from metadata")
    parser.add_argument("--max_aspect_ratio", type=float, default=2.0,
                        help="Maximum aspect ratio (width/height or height/width)")
    args = parser.parse_args()
    
    # Set up image transformations
    transform = get_image_transforms(image_size=args.image_size)
    
    # Load CLIP model if requested
    clip_model, clip_processor = None, None
    if args.use_clip:
        clip_model, clip_processor = load_clip_model()
    
    # Determine starting batch index
    start_batch = 0
    if args.resume:
        start_batch = load_checkpoint(args.output_dir)
        print(f"Resuming from batch {start_batch}")
    
    # Load LAION dataset in streaming mode
    print(f"Loading dataset: {args.dataset}")
    dataset = load_laion_dataset(dataset_name=args.dataset, split="train", streaming=True)
    
    # Process batches
    samples_processed = 0
    batch_idx = start_batch
    
    # Skip already processed batches
    if start_batch > 0:
        for _ in tqdm(range(start_batch * args.batch_size), desc="Skipping processed samples"):
            next(iter(dataset))
    
    # Process new batches
    with tqdm(total=args.num_batches, initial=0, desc="Processing batches") as pbar:
        while batch_idx < start_batch + args.num_batches:
            # Collect batch
            batch = []
            for _ in range(args.batch_size * 2):  # Get more samples to account for filtering
                try:
                    sample = next(iter(dataset))
                    batch.append(sample)
                    if len(batch) >= args.batch_size * 2:
                        break
                except StopIteration:
                    print("Reached end of dataset")
                    break
            
            if not batch:
                break
            
            # Filter samples based on metadata
            filtered_batch = filter_laion_samples(
                batch, 
                min_width=args.min_width,
                min_height=args.min_height,
                max_aspect_ratio=args.max_aspect_ratio
            )
            
            # Take only batch_size samples after filtering
            filtered_batch = filtered_batch[:args.batch_size]
            
            if not filtered_batch:
                continue
            
            # Process batch
            processed_batch = process_batch(
                filtered_batch, transform, clip_model, clip_processor)
            
            if processed_batch:
                # Save checkpoint
                save_checkpoint(processed_batch, args.output_dir, batch_idx)
                samples_processed += len(processed_batch)
            
            batch_idx += 1
            pbar.update(1)
            pbar.set_postfix({"samples": samples_processed})
    
    print(f"Finished processing {samples_processed} samples in {batch_idx - start_batch} batches")
    print(f"Processed data saved to {args.output_dir}")


def load_processed_dataset(output_dir):
    """Load all processed batches into a single dataset.
    
    Args:
        output_dir (str): Directory with saved batches
        
    Returns:
        ProcessedImageDataset: Dataset with all processed samples
    """
    all_samples = []
    batch_files = sorted(Path(output_dir).glob("batch_*.pkl"))
    
    for batch_file in tqdm(batch_files, desc="Loading batches"):
        with open(batch_file, 'rb') as f:
            batch_samples = pickle.load(f)
            all_samples.extend(batch_samples)
    
    return ProcessedImageDataset(all_samples)


if __name__ == "__main__":
    main()