import torch
import sys
import os
from pathlib import Path
import argparse
from PIL import Image
import gc
import time
import re

# Add the parent directory to system path to find the modules
sys.path.append(str(Path(__file__).parent))

from src.utils import memory_efficient_loader
from src.pipelines import pipeline
from transformers import CLIPTokenizer

def sanitize_filename(text):
    # Replace spaces and illegal filename characters
    return re.sub(r'[^a-zA-Z0-9_-]', '_', text.strip())[:100]

def setup_gpu():
    """Configure GPU settings for optimal performance."""
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(1)
        device = "cuda:1"
        print(f"Using GPU 1: {torch.cuda.get_device_name(1)}")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"
    torch.backends.cudnn.benchmark = True

    if "cuda" in device:
        torch.cuda.empty_cache()
        gc.collect()
    
    return device

def load_models(device):
    """Load and prepare all necessary models."""
    # Fix tokenizer paths using Path for proper resolution
    data_dir = Path(__file__).parent / "data" / "tokenizer"
    print("Loading tokenizer...")
    tokenizer = CLIPTokenizer(
        str(data_dir / "vocab.json"),
        merges_file=str(data_dir / "merges.txt")
    )

    print("Loading model file...")
    model_file = str(Path(__file__).parent / "checkpoints" / "v1-5-pruned-emaonly.ckpt")

    print("Loading models with memory efficient loader...")
    models = memory_efficient_loader.preload_models_from_standard_weights(model_file, device)

    return tokenizer, models

def warmup_gpu(models, tokenizer, device):
    """Perform GPU warmup for more consistent first generation."""
    print("Warming up GPU...")
    warmup_prompt = "A simple test image"
    warmup_tokens = tokenizer.batch_encode_plus(
        [warmup_prompt], padding="max_length", max_length=77
    ).input_ids
    warmup_tokens = torch.tensor(warmup_tokens, dtype=torch.long, device=device)
    with torch.no_grad():
        models["clip"](warmup_tokens)
        torch.cuda.synchronize()

def generate_image(prompt, models, tokenizer, device, args):
    """Generate image based on the prompt and parameters."""
    print(f"\nGenerating image for prompt: {prompt}")
    print(f"Using parameters: Steps={args.steps}, CFG Scale={args.cfg_scale}, Seed={args.seed}")
    
    start_time = time.time()
    output_image = pipeline.generate(
        prompt=prompt,
        uncond_prompt=args.negative_prompt,
        input_image=None,
        strength=0.9,
        do_cfg=True,
        cfg_scale=args.cfg_scale,
        sampler_name=args.sampler,
        n_inference_steps=args.steps,
        seed=args.seed,
        models=models,
        device=device,
        idle_device="cpu",
        tokenizer=tokenizer,
    )
    end_time = time.time()
    print(f"Generation completed in {end_time - start_time:.2f} seconds")
    
    return output_image

def save_image(image_array, prompt, args):
    """Save the generated image with proper naming and organization."""
    output_image_pil = Image.fromarray(image_array)
    
    # Create filename with parameters
    param_str = f"_s{args.steps}_cfg{args.cfg_scale}_seed{args.seed}"
    filename = sanitize_filename(prompt) + param_str + ".png"
    
    try:
        # Try to save in the project's images directory
        images_dir = Path(__file__).parent / "images"
        images_dir.mkdir(exist_ok=True)
        output_path = images_dir / filename
        output_image_pil.save(output_path)
        print(f"\nImage saved as: {output_path}")
    except Exception as e:
        # Fallback: save in the same directory as the script
        fallback_path = Path(__file__).parent / filename
        output_image_pil.save(fallback_path)
        print(f"Warning: Couldn't save to images directory ({str(e)})")
        print(f"Image saved as: {fallback_path}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Text-to-Image Generation Script")
    parser.add_argument("--prompt", type=str, help="The prompt to generate an image from")
    parser.add_argument("--negative-prompt", type=str, default="", help="Negative prompt to guide generation")
    parser.add_argument("--steps", type=int, default=40, help="Number of inference steps (20-50)")
    parser.add_argument("--cfg-scale", type=float, default=8.0, help="Classifier-free guidance scale (1-20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"], help="Sampling method")
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Get prompt from arguments or input
    prompt = args.prompt if args.prompt else input("Enter your prompt: ").strip()
    if not prompt:
        print("Prompt is empty. Exiting.")
        return

    # Setup and initialization
    device = setup_gpu()
    tokenizer, models = load_models(device)
    warmup_gpu(models, tokenizer, device)

    # Generate and save image
    try:
        output_image = generate_image(prompt, models, tokenizer, device, args)
        save_image(output_image, prompt, args)
    except Exception as e:
        print(f"\nError during image generation: {str(e)}")
    finally:
        if "cuda" in device:
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    main()