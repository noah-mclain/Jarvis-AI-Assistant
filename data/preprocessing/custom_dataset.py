"""Custom dataset handling for Stable Diffusion fine-tuning."""

import os
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class TextImagePairDataset(Dataset):
    """Dataset for text-image pairs used in Stable Diffusion fine-tuning."""
    def __init__(self, image_dir, prompt_file=None, image_size=512, concept_name=None):
        """
        Initialize the dataset.
        
        Args:
            image_dir: Directory containing images
            prompt_file: JSON file with image-to-prompt mapping or text file with one prompt per line
            image_size: Size to resize images to
            concept_name: Optional concept name to append to prompts (e.g., "a photo of [concept] dog")
        """
        self.image_dir = Path(image_dir)
        self.concept_name = concept_name
        
        # Set up image transforms
        self.transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        # Get all image files
        self.image_files = []
        for ext in ['.png', '.jpg', '.jpeg', '.webp']:
            self.image_files.extend(list(self.image_dir.glob(f'**/*{ext}')))
        
        print(f"Found {len(self.image_files)} images in {image_dir}")
        
        # Load prompts if provided
        self.prompts = {}
        if prompt_file:
            self._load_prompts(prompt_file)
        
    def _load_prompts(self, prompt_file):
        """Load prompts from file."""
        prompt_path = Path(prompt_file)
        
        if not prompt_path.exists():
            print(f"Warning: Prompt file {prompt_file} not found. Using filenames as prompts.")
            return
            
        if prompt_path.suffix.lower() == '.json':
            # Load JSON format (image_filename -> prompt)
            with open(prompt_path, 'r') as f:
                self.prompts = json.load(f)
                
            print(f"Loaded {len(self.prompts)} prompts from JSON file")
        else:
            # Load text file (one prompt per line, matched with images by index)
            with open(prompt_path, 'r') as f:
                prompt_lines = [line.strip() for line in f if line.strip()]
            
            # Match prompts with images by index (cycling if needed)
            for i, img_path in enumerate(self.image_files):
                prompt_idx = i % len(prompt_lines)
                self.prompts[img_path.name] = prompt_lines[prompt_idx]
                
            print(f"Matched {len(self.image_files)} images with {len(prompt_lines)} prompts")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # Load and transform image
        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = self.transforms(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a random image as fallback
            image_tensor = torch.randn(3, 512, 512)
        
        # Get prompt for this image
        if img_path.name in self.prompts:
            prompt = self.prompts[img_path.name]
        else:
            # Use filename as prompt if no mapping exists
            prompt = img_path.stem.replace('_', ' ')
        
        # Add concept name if provided
        if self.concept_name:
            prompt = f"a photo of {self.concept_name} {prompt}"
        
        return {
            "image": image_tensor,
            "prompt": prompt,
            "image_path": str(img_path)
        }


class DreamBoothDataset(Dataset):
    """Dataset specifically for DreamBooth-style fine-tuning with a unique identifier."""
    def __init__(self, image_dir, class_name, instance_prompt, class_prompt, image_size=512):
        """
        Initialize the DreamBooth dataset.
        
        Args:
            image_dir: Directory containing instance images
            class_name: Name of the class (e.g., "dog", "cat")
            instance_prompt: Prompt template for instance images (e.g., "a photo of sks dog")
            class_prompt: Prompt template for class images (e.g., "a photo of dog")
            image_size: Size to resize images to
        """
        self.image_dir = Path(image_dir)
        self.class_name = class_name
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt
        
        # Set up image transforms
        self.transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        # Get all image files
        self.image_files = []
        for ext in ['.png', '.jpg', '.jpeg', '.webp']:
            self.image_files.extend(list(self.image_dir.glob(f'**/*{ext}')))
        
        print(f"Found {len(self.image_files)} instance images in {image_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # Load and transform image
        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = self.transforms(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a random image as fallback
            image_tensor = torch.randn(3, 512, 512)
        
        # Use instance prompt for all images
        prompt = self.instance_prompt
        
        return {
            "image": image_tensor,
            "prompt": prompt,
            "class_prompt": self.class_prompt,
            "image_path": str(img_path)
        }