"""Utility functions for image preprocessing with LAION datasets."""

import torch
import numpy as np
from PIL import Image
from io import BytesIO
import requests
from torchvision import transforms


def download_image(url, timeout=5):
    """Download an image from a URL.
    
    Args:
        url (str): The URL of the image to download
        timeout (int): Timeout in seconds for the request
        
    Returns:
        PIL.Image or None: The downloaded image, or None if download failed
    """
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code != 200:
            return None
        
        image = Image.open(BytesIO(response.content)).convert('RGB')
        return image
    except Exception as e:
        print(f"Error downloading image {url}: {e}")
        return None


def get_image_transforms(image_size=224, augment=False):
    """Get image transformation pipeline.
    
    Args:
        image_size (int): Size to resize images to
        augment (bool): Whether to apply data augmentation
        
    Returns:
        torchvision.transforms.Compose: The transformation pipeline
    """
    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def filter_laion_samples(samples, min_width=256, min_height=256, max_aspect_ratio=2.0):
    """Filter LAION samples based on metadata criteria.
    
    Args:
        samples (list): List of LAION samples
        min_width (int): Minimum image width
        min_height (int): Minimum image height
        max_aspect_ratio (float): Maximum aspect ratio (width/height or height/width)
        
    Returns:
        list: Filtered list of samples
    """
    filtered = []
    
    for sample in samples:
        # Skip if no metadata
        if 'WIDTH' not in sample or 'HEIGHT' not in sample:
            continue
            
        width = sample['WIDTH']
        height = sample['HEIGHT']
        
        # Skip small images
        if width < min_width or height < min_height:
            continue
            
        # Skip extreme aspect ratios
        aspect_ratio = max(width / height, height / width)
        if aspect_ratio > max_aspect_ratio:
            continue
            
        filtered.append(sample)
        
    return filtered


def extract_clip_features(model, processor, image, text):
    """Extract CLIP features for image and text.
    
    Args:
        model: CLIP model
        processor: CLIP processor/transform
        image: PIL image or tensor
        text (str): Text to encode
        
    Returns:
        tuple: (image_features, text_features) as numpy arrays
    """
    try:
        import clip
    except ImportError:
        print("CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")
        return None, None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with torch.no_grad():
        # Process image
        if isinstance(image, Image.Image):
            image_input = processor(image).unsqueeze(0).to(device)
        else:
            # Assume it's already a tensor
            image_input = image.unsqueeze(0).to(device) if image.dim() == 3 else image.to(device)
        
        # Process text
        text_tokens = clip.tokenize([text]).to(device)
        
        # Get features
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_tokens)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy(), text_features.cpu().numpy()
    

def create_image_batch_loader(dataset, batch_size=32, num_workers=4):
    """Create a batch loader for LAION dataset that handles downloading images.
    
    Args:
        dataset: Hugging Face dataset
        batch_size (int): Batch size
        num_workers (int): Number of workers for parallel processing
        
    Returns:
        function: A generator function that yields batches
    """
    def batch_loader():
        from torch.utils.data import DataLoader
        from functools import partial
        from multiprocessing import Pool
        
        # Create batches from the dataset
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            
            # Download images in parallel
            with Pool(num_workers) as p:
                images = p.map(download_image, [sample['URL'] for sample in batch])
                
            # Filter out failed downloads
            valid_samples = []
            for sample, image in zip(batch, images):
                if image is not None:
                    sample['image'] = image
                    valid_samples.append(sample)
                    
            if valid_samples:
                yield valid_samples
    
    return batch_loader