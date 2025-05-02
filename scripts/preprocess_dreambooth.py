import os
import argparse
import shutil
from pathlib import Path
from PIL import Image
import json

def resize_and_center_crop(image, target_size):
    """Resize and center crop an image to target size while maintaining aspect ratio."""
    width, height = image.size
    aspect_ratio = width / height
    target_aspect = target_size[0] / target_size[1]

    if aspect_ratio > target_aspect:
        # Image is wider than target aspect ratio
        new_height = target_size[1]
        new_width = int(new_height * aspect_ratio)
    else:
        # Image is taller than target aspect ratio
        new_width = target_size[0]
        new_height = int(new_width / aspect_ratio)

    # Resize maintaining aspect ratio
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Center crop
    left = (new_width - target_size[0]) // 2
    top = (new_height - target_size[1]) // 2
    right = left + target_size[0]
    bottom = top + target_size[1]

    return image.crop((left, top, right, bottom))

def preprocess_images(input_dir, output_dir, image_size):
    """Preprocess images for DreamBooth training."""
    os.makedirs(output_dir, exist_ok=True)
    processed_count = 0

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            try:
                image_path = os.path.join(input_dir, filename)
                image = Image.open(image_path).convert('RGB')
                
                # Resize and center crop
                processed_image = resize_and_center_crop(image, (image_size, image_size))
                
                # Save processed image
                output_path = os.path.join(output_dir, f"{processed_count:04d}.jpg")
                processed_image.save(output_path, 'JPEG', quality=95)
                processed_count += 1
                
                print(f"Processed {filename} -> {os.path.basename(output_path)}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    return processed_count

def generate_prompts(class_name, identifier, num_images, output_file):
    """Generate prompts for DreamBooth training."""
    instance_prompt = f"a photo of {identifier} {class_name}"
    class_prompt = f"a photo of {class_name}"
    
    prompts = {
        "instance_prompt": instance_prompt,
        "class_prompt": class_prompt,
        "training_prompts": [instance_prompt] * num_images
    }
    
    with open(output_file, 'w') as f:
        json.dump(prompts, f, indent=2)
    
    return prompts

def main():
    parser = argparse.ArgumentParser(description="Preprocess images for DreamBooth training")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing source images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save processed images")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Target size for images (square)")
    parser.add_argument("--class_name", type=str, required=True,
                        help="Class name for training (e.g., 'person')")
    parser.add_argument("--identifier", type=str, required=True,
                        help="Unique identifier for instance (e.g., 'Amr')")
    
    args = parser.parse_args()
    
    # Process images
    print(f"\nProcessing images from {args.input_dir}...")
    num_processed = preprocess_images(args.input_dir, args.output_dir, args.image_size)
    print(f"Successfully processed {num_processed} images")
    
    # Generate prompts
    prompts_file = os.path.join(args.output_dir, "prompts.json")
    prompts = generate_prompts(args.class_name, args.identifier, num_processed, prompts_file)
    print(f"\nGenerated prompts:")
    print(f"Instance prompt: {prompts['instance_prompt']}")
    print(f"Class prompt: {prompts['class_prompt']}")
    print(f"Prompts saved to: {prompts_file}")

if __name__ == "__main__":
    main()