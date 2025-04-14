import torch
import sys
import os
from pathlib import Path

# Add the parent directory to system path to find the modules
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import memory_efficient_loader
from src.pipelines import pipeline
from transformers import CLIPTokenizer
from PIL import Image
import gc
import time
import re

def sanitize_filename(text):
    # Replace spaces and illegal filename characters
    return re.sub(r'[^a-zA-Z0-9_-]', '_', text.strip())[:100]

def main():
    # Take prompt as input
    prompt = input("Enter your prompt: ").strip()
    if not prompt:
        print("Prompt is empty. Exiting.")
        return

    # Set CUDA device to GPU 1 (GTX 1650)
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(1)
        DEVICE = "cuda:1"
        print(f"Using GPU 1: {torch.cuda.get_device_name(1)}")
    else:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {DEVICE}")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"
    torch.backends.cudnn.benchmark = True

    if "cuda" in DEVICE:
        torch.cuda.empty_cache()
        gc.collect()

    print("Loading tokenizer...")
    tokenizer = CLIPTokenizer("./data/vocab.json", merges_file="./data/merges.txt")

    print("Loading model file...")
    model_file = "./data/v1-5-pruned-emaonly.ckpt"

    print("Loading models with memory efficient loader...")
    models = memory_efficient_loader.preload_models_from_standard_weights(model_file, DEVICE)

    uncond_prompt = ""
    do_cfg = True
    cfg_scale = 8

    input_image = None
    strength = 0.9

    sampler = "ddpm"
    num_inference_steps = 40
    seed = 42

    if "cuda" in DEVICE:
        torch.cuda.empty_cache()
        gc.collect()

    print("Warming up GPU...")
    warmup_prompt = "A simple test image"
    warmup_tokens = tokenizer.batch_encode_plus(
        [warmup_prompt], padding="max_length", max_length=77
    ).input_ids
    warmup_tokens = torch.tensor(warmup_tokens, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        models["clip"](warmup_tokens)
        torch.cuda.synchronize()

    print("Generating image...")
    start_time = time.time()
    output_image = pipeline.generate(
        prompt=prompt,
        uncond_prompt=uncond_prompt,
        input_image=input_image,
        strength=strength,
        do_cfg=do_cfg,
        cfg_scale=cfg_scale,
        sampler_name=sampler,
        n_inference_steps=num_inference_steps,
        seed=seed,
        models=models,
        device=DEVICE,
        idle_device="cpu",
        tokenizer=tokenizer,
    )
    end_time = time.time()
    print(f"Generation completed in {end_time - start_time:.2f} seconds")

    if "cuda" in DEVICE:
        torch.cuda.empty_cache()
        gc.collect()

    filename = sanitize_filename(prompt) + ".png"
    print(f"Saving output image as '{filename}'...")
    output_image_pil = Image.fromarray(output_image)
    output_image_pil.save(f"./images/{filename}")
    print(f"Done! Image saved as '{filename}'")

if __name__ == "__main__":
    main()
