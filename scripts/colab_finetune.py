# -*- coding: utf-8 -*-
"""DreamBooth Fine-tuning in Google Colab

This script adapts the local fine-tuning implementation for Google Colab environment,
handling Google Drive mounting, dependencies installation, and model fine-tuning.
"""

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install required dependencies
!pip install -q diffusers==0.10.2 transformers ftfy accelerate bitsandbytes gradio natsort
!pip install -q torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

import os
import sys
import argparse
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from diffusers import StableDiffusionPipeline

def setup_environment():
    """Set up the Colab environment and verify GPU availability."""
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a GPU runtime. Please change runtime type in Colab.")
    print(f"GPU available: {torch.cuda.get_device_name(0)}")

def preprocess_images(input_dir, output_dir, image_size=512):
    """Preprocess training images for DreamBooth."""
    os.makedirs(output_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.ToPILImage()
    ])
    
    processed_count = 0
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            image_path = os.path.join(input_dir, filename)
            try:
                image = Image.open(image_path).convert('RGB')
                processed_image = transform(image)
                save_path = os.path.join(output_dir, f"processed_{filename}")
                processed_image.save(save_path, 'PNG')
                processed_count += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    return processed_count

def setup_dreambooth_args(args):
    """Configure DreamBooth training arguments."""
    instance_prompt = f"a photo of {args.identifier} {args.class_name}"
    class_prompt = f"a photo of {args.class_name}"
    
    print(f"\nDreamBooth configuration:")
    print(f"Instance prompt: {instance_prompt}")
    print(f"Class prompt: {class_prompt}")
    
    return {
        "pretrained_model_name_or_path": args.base_model,
        "instance_data_dir": args.image_dir,
        "output_dir": args.output_dir,
        "instance_prompt": instance_prompt,
        "class_prompt": class_prompt,
        "resolution": args.image_size,
        "train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "max_train_steps": args.max_steps,
        "mixed_precision": "fp16" if args.mixed_precision else "no"
    }

def main():
    parser = argparse.ArgumentParser(description="DreamBooth fine-tuning in Google Colab")
    parser.add_argument("--base_model", type=str, required=True,
                        help="Path to the base model checkpoint in Google Drive")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing training images in Google Drive")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the fine-tuned model in Google Drive")
    parser.add_argument("--class_name", type=str, required=True,
                        help="Class name for training (e.g., 'person')")
    parser.add_argument("--identifier", type=str, required=True,
                        help="Unique identifier for instance (e.g., 'Amr')")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Size of training images")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of steps for gradient accumulation")
    parser.add_argument("--learning_rate", type=float, default=2e-6,
                        help="Learning rate for training")
    parser.add_argument("--max_steps", type=int, default=800,
                        help="Maximum number of training steps")
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Enable mixed precision training")
    
    args = parser.parse_args()
        
    # Preprocess images
    print("\nPreprocessing training images...")
    processed_dir = os.path.join(os.path.dirname(args.image_dir), "processed_images")
    num_processed = preprocess_images(args.image_dir, processed_dir, args.image_size)
    print(f"Successfully processed {num_processed} images")
    
    # Update image directory to processed images
    args.image_dir = processed_dir
    
    # Setup and start training
    training_args = setup_dreambooth_args(args)
    
    # Initialize the pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        training_args["pretrained_model_name_or_path"],
        torch_dtype=torch.float16 if args.mixed_precision else torch.float32
    )
    
    # Start training
    print("\nStarting DreamBooth fine-tuning...")
    pipeline.train(**training_args)
    
    # Save the fine-tuned model
    pipeline.save_pretrained(args.output_dir)
    print(f"\nFine-tuning completed! Model saved to: {args.output_dir}")

if __name__ == "__main__":
    main()