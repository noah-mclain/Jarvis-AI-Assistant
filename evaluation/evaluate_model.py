import sys
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import argparse
from transformers import CLIPTokenizer, CLIPModel, CLIPProcessor

# Adding the parent directory to system path to find the modules
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import memory_efficient_loader
from src.pipelines import pipeline

def load_model(model_path, device):
    """Load the model from checkpoint."""
    print(f"Loading model from {model_path}...")
    models = memory_efficient_loader.preload_models_from_standard_weights(model_path, device)
    return models

def generate_images(prompts, models, tokenizer, device, cfg_scales=[7.5], seeds=[42], memory_efficient=True):
    """Generate images for a list of prompts with different cfg_scales and seeds."""
    results = {}
    
    # Set memory optimization if needed
    if memory_efficient and "cuda" in device:
        # Try to optimize CUDA memory allocation
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'memory_reserved'):
            print(f"CUDA memory reserved before generation: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
        print(f"Setting memory-efficient mode for generation")
        
        # Set environment variable for memory allocation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    for prompt in tqdm(prompts, desc="Processing prompts"):
        results[prompt] = {}
        
        for cfg_scale in cfg_scales:
            results[prompt][f"cfg_{cfg_scale}"] = {}
            
            for seed in seeds:
                print(f"Generating: '{prompt}' (CFG: {cfg_scale}, Seed: {seed})")
                start_time = time.time()
                
                # Clear cache before each generation
                if "cuda" in device:
                    torch.cuda.empty_cache()
                
                try:
                    output_image = pipeline.generate(
                        prompt=prompt,
                        uncond_prompt="",
                        input_image=None,
                        strength=0.9,
                        do_cfg=True,
                        cfg_scale=cfg_scale,
                        sampler_name="ddpm",
                        n_inference_steps=40,
                        seed=seed,
                        models=models,
                        device=device,
                        idle_device="cpu",
                        tokenizer=tokenizer,
                    )
                except torch.cuda.OutOfMemoryError:
                    print("CUDA out of memory! Trying with reduced parameters...")
                    # Clear cache and try with fewer steps
                    torch.cuda.empty_cache()
                    time.sleep(2)  # Give GPU a moment to recover
                    
                    output_image = pipeline.generate(
                        prompt=prompt,
                        uncond_prompt="",
                        input_image=None,
                        strength=0.9,
                        do_cfg=True,
                        cfg_scale=cfg_scale,
                        sampler_name="ddpm",
                        n_inference_steps=20,  # Reduced steps
                        seed=seed,
                        models=models,
                        device=device,
                        idle_device="cpu",
                        tokenizer=tokenizer,
                    )
                
                end_time = time.time()
                generation_time = end_time - start_time
                
                results[prompt][f"cfg_{cfg_scale}"][f"seed_{seed}"] = {
                    "image": output_image,
                    "generation_time": generation_time
                }
                
                print(f"Generation completed in {generation_time:.2f} seconds")
                
                # Clear CUDA cache after each generation
                if "cuda" in device:
                    torch.cuda.empty_cache()
                    time.sleep(1)  # Small delay to let memory settle
    
    return results

def calculate_clip_score(images, prompts, clip_model, clip_processor, device):
    """Calculate CLIP score for generated images."""
    clip_scores = {}
    
    for prompt in prompts:
        clip_scores[prompt] = {}
        
        for cfg_key, cfg_results in images[prompt].items():
            clip_scores[prompt][cfg_key] = {}
            
            for seed_key, seed_results in cfg_results.items():
                image = seed_results["image"]
                pil_image = Image.fromarray(image)
                
                # Process image and text with CLIP
                inputs = clip_processor(
                    text=[prompt],
                    images=pil_image,
                    return_tensors="pt",
                    padding=True
                ).to(device)
                
                with torch.no_grad():
                    outputs = clip_model(**inputs)
                    # Normalize features
                    image_features = outputs.image_embeds / outputs.image_embeds.norm(dim=1, keepdim=True)
                    text_features = outputs.text_embeds / outputs.text_embeds.norm(dim=1, keepdim=True)
                    # Calculate similarity
                    similarity = (100 * image_features @ text_features.T).item()
                
                clip_scores[prompt][cfg_key][seed_key] = similarity
    
    return clip_scores

def visualize_results(results, clip_scores, output_dir):
    """Visualize and save the generated images with their metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create a summary file
    with open(output_dir / "evaluation_summary.txt", "w") as f:
        f.write("# Stable Diffusion Evaluation Results\n\n")
        
        for prompt in results:
            f.write(f"## Prompt: \"{prompt}\"\n\n")
            
            for cfg_key in results[prompt]:
                cfg_value = float(cfg_key.split("_")[1])
                f.write(f"### CFG Scale: {cfg_value}\n\n")
                
                for seed_key in results[prompt][cfg_key]:
                    seed_value = int(seed_key.split("_")[1])
                    generation_time = results[prompt][cfg_key][seed_key]["generation_time"]
                    clip_score = clip_scores[prompt][cfg_key][seed_key]
                    
                    f.write(f"- Seed: {seed_value}\n")
                    f.write(f"  - Generation Time: {generation_time:.2f} seconds\n")
                    f.write(f"  - CLIP Score: {clip_score:.2f}\n\n")
    
    # Export results to CSV
    import csv
    with open(output_dir / "evaluation_results.csv", "w", newline="") as csvfile:
        fieldnames = ["prompt", "cfg_scale", "seed", "clip_score", "generation_time"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for prompt in results:
            for cfg_key in results[prompt]:
                cfg_value = float(cfg_key.split("_")[1])
                
                for seed_key in results[prompt][cfg_key]:
                    seed_value = int(seed_key.split("_")[1])
                    generation_time = results[prompt][cfg_key][seed_key]["generation_time"]
                    clip_score = clip_scores[prompt][cfg_key][seed_key]
                    
                    writer.writerow({
                        "prompt": prompt,
                        "cfg_scale": cfg_value,
                        "seed": seed_value,
                        "clip_score": clip_score,
                        "generation_time": generation_time
                    })
    
    # Create visualizations
    for prompt in results:
        prompt_dir = output_dir / prompt.replace(" ", "_")[:50]
        prompt_dir.mkdir(exist_ok=True)
        
        # Save individual images
        for cfg_key in results[prompt]:
            for seed_key in results[prompt][cfg_key]:
                image = results[prompt][cfg_key][seed_key]["image"]
                clip_score = clip_scores[prompt][cfg_key][seed_key]
                generation_time = results[prompt][cfg_key][seed_key]["generation_time"]
                
                cfg_value = float(cfg_key.split("_")[1])
                seed_value = int(seed_key.split("_")[1])
                
                # Create a figure with the image and metrics
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(image)
                ax.set_title(f"Prompt: {prompt}\nCFG: {cfg_value}, Seed: {seed_value}\nCLIP Score: {clip_score:.2f}, Time: {generation_time:.2f}s")
                ax.axis('off')
                
                # Save the figure
                fig.savefig(prompt_dir / f"cfg_{cfg_value}_seed_{seed_value}.png", bbox_inches='tight')
                plt.close(fig)
                
                # Also save the raw image
                Image.fromarray(image).save(prompt_dir / f"cfg_{cfg_value}_seed_{seed_value}_raw.png")
        
        # Create a comparison grid for different CFG values and seeds
        if len(results[prompt]) > 0 and len(list(results[prompt].values())[0]) > 0:
            cfg_keys = list(results[prompt].keys())
            seed_keys = list(results[prompt][cfg_keys[0]].keys())
            
            if len(cfg_keys) > 1 or len(seed_keys) > 1:
                fig, axes = plt.subplots(len(cfg_keys), len(seed_keys), figsize=(len(seed_keys)*5, len(cfg_keys)*5))
                
                if len(cfg_keys) == 1 and len(seed_keys) > 1:
                    axes = [axes]  # Make it 2D for consistent indexing
                elif len(cfg_keys) > 1 and len(seed_keys) == 1:
                    axes = [[ax] for ax in axes]  # Make it 2D for consistent indexing
                
                for i, cfg_key in enumerate(cfg_keys):
                    cfg_value = float(cfg_key.split("_")[1])
                    
                    for j, seed_key in enumerate(seed_keys):
                        seed_value = int(seed_key.split("_")[1])
                        image = results[prompt][cfg_key][seed_key]["image"]
                        clip_score = clip_scores[prompt][cfg_key][seed_key]
                        
                        axes[i][j].imshow(image)
                        axes[i][j].set_title(f"CFG: {cfg_value}, Seed: {seed_value}\nCLIP: {clip_score:.2f}")
                        axes[i][j].axis('off')
                
                plt.tight_layout()
                plt.savefig(prompt_dir / "comparison_grid.png")
                plt.close(fig)
    
    print(f"Evaluation results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Stable Diffusion model")
    parser.add_argument("--model_path", type=str, default="../checkpoints/v1-5-pruned-emaonly.ckpt", 
                        help="Path to the model checkpoint")
    parser.add_argument("--output_dir", type=str, default="../evaluation_results", 
                        help="Directory to save evaluation results")
    parser.add_argument("--prompts", type=str, nargs="+", 
                        default=["a photo of a japanese anime lady", "a beautiful sunset over mountains"],
                        help="Prompts to generate images for")
    parser.add_argument("--cfg_scales", type=float, nargs="+", default=[5.0, 7.5, 10.0],
                        help="CFG scales to test")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1024],
                        help="Seeds to test")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda:0, cuda:1, cpu)")
    parser.add_argument("--memory_efficient", action="store_true", 
                        help="Enable memory-efficient mode for CUDA")
    parser.add_argument("--inference_steps", type=int, default=40,
                        help="Number of inference steps (lower for less memory usage)")
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        if torch.cuda.device_count() > 1:
            device = "cuda:1"
            torch.cuda.set_device(1)
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        if "cuda" in device:
            torch.cuda.set_device(int(device.split(":")[-1]))
    
    print(f"Using device: {device}")
    
    # Display GPU info if using CUDA
    if "cuda" in device:
        gpu_id = int(device.split(":")[-1]) if ":" in device else 0
        print(f"GPU: {torch.cuda.get_device_name(gpu_id)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.2f} GB")
        print(f"Available memory: {torch.cuda.mem_get_info(gpu_id)[0] / 1024**3:.2f} GB")
    
    # Set up paths
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = Path(__file__).parent.parent / args.model_path.lstrip("./").lstrip("../")
    
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).parent.parent / args.output_dir.lstrip("./").lstrip("../")
    
    print(f"Model path: {model_path}")
    print(f"Output directory: {output_dir}")
    
    # Ensure model file exists
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load tokenizer
    tokenizer_dir = Path(__file__).parent.parent / "data" / "tokenizer"
    tokenizer = CLIPTokenizer(
        str(tokenizer_dir / "vocab.json"),
        merges_file=str(tokenizer_dir / "merges.txt")
    )
    
    # Load CLIP model for evaluation - move to CPU if memory is tight
    if args.memory_efficient and "cuda" in device:
        print("Loading CLIP model in memory-efficient mode...")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cpu")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    else:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Load model
    models = load_model(str(model_path), device)
    
    # Generate images
    results = generate_images(args.prompts, models, tokenizer, device, 
                             args.cfg_scales, args.seeds, args.memory_efficient)
    
    # Move CLIP model to device for scoring if it was on CPU
    if args.memory_efficient and "cuda" in device:
        clip_model = clip_model.to(device)
    
    # Calculate CLIP scores
    clip_scores = calculate_clip_score(results, args.prompts, clip_model, clip_processor, device)
    
    # Visualize and save results
    visualize_results(results, clip_scores, output_dir)

if __name__ == "__main__":
    main()