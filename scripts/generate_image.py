import torch
import sys
import os
import time
import gc
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Add the parent directory to system path to find the modules
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import memory_efficient_loader
from src.pipelines import pipeline
from transformers import CLIPTokenizer

def get_user_inputs():
    """Get user inputs from terminal."""
    # Get prompt from user
    prompt = input("Write the prompt: ").strip()
    while not prompt:
        print("Prompt cannot be empty. Please try again.")
        prompt = input("Write the prompt: ").strip()
    
    # Get input image path from user
    input_image = input("Write the image path: ").strip()
    while not input_image or not os.path.exists(input_image):
        print(f"Image path '{input_image}' does not exist. Please enter a valid path.")
        input_image = input("Write the image path: ").strip()
    
    # Create a class to mimic argparse namespace
    class Args:
        pass
    
    args = Args()
    args.prompt = prompt
    args.input_image = input_image
    args.model_path = str(Path(__file__).parent.parent / "checkpoints" / "v1-5-pruned-emaonly.ckpt")
    args.negative_prompt = ""
    args.strength = 0.8
    args.output_dir = "../outputs"
    args.cfg_scale = 7.0  # Reduced from 7.5 to save memory
    args.steps = 20  # Reduced from 50 to save memory
    args.seed = None
    args.device = None
    args.sampler = "ddpm"
    args.use_memory_efficient = True  # Enable memory efficient mode by default
    
    # Ask about low memory mode for GPUs with very limited VRAM
    low_memory = input("Enable low-memory mode for GPUs with limited VRAM? (y/n, default: n): ").lower().strip()
    if low_memory == 'y':
        print("Low-memory mode enabled. Using minimal settings for image generation.")
        args.steps = 15  # Minimum steps
        args.cfg_scale = 5.0  # Lower CFG scale
        args.strength = 0.6  # Lower strength preserves more of original image
    
    # Optionally ask for advanced parameters
    advanced = input("Do you want to set advanced parameters? (y/n): ").lower().strip() == 'y'
    if advanced:
        args.negative_prompt = input("Negative prompt (press Enter to skip): ").strip()
        
        strength = input("Strength (0.0-1.0, default 0.8): ").strip()
        if strength:
            try:
                args.strength = float(strength)
                args.strength = max(0.0, min(1.0, args.strength))  # Clamp between 0 and 1
            except ValueError:
                print("Invalid strength value, using default 0.8")
        
        cfg_scale = input("CFG Scale (default 7.5): ").strip()
        if cfg_scale:
            try:
                args.cfg_scale = float(cfg_scale)
            except ValueError:
                print("Invalid CFG scale value, using default 7.5")
        
        steps = input("Steps (default 30): ").strip()
        if steps:
            try:
                args.steps = int(steps)
            except ValueError:
                print("Invalid steps value, using default 30")
        
        seed = input("Seed (press Enter for random): ").strip()
        if seed:
            try:
                args.seed = int(seed)
            except ValueError:
                print("Invalid seed value, using random seed")
                
        memory_efficient = input("Enable memory-efficient mode? (y/n, default: y): ").lower().strip()
        if memory_efficient and memory_efficient == 'n':
            args.use_memory_efficient = False
            print("Memory-efficient mode disabled. This may cause out-of-memory errors on GPUs with limited VRAM.")
        else:
            print("Memory-efficient mode enabled. Parameters will be automatically adjusted based on available GPU memory.")
    
    return args

def setup_device(args):
    """Set up the device for generation."""
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
        
        # Set memory optimization settings
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        torch.backends.cudnn.benchmark = True
        
        # Enable memory efficient operations
        if torch.cuda.is_available():
            # Clear CUDA cache
            torch.cuda.empty_cache()
            gc.collect()
            
            # Enable TF32 precision if available (on NVIDIA Ampere GPUs)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    return device

def load_tokenizer():
    """Load the CLIP tokenizer."""
    tokenizer_dir = Path(__file__).parent.parent / "data" / "tokenizer"
    return CLIPTokenizer(
        str(tokenizer_dir / "vocab.json"),
        merges_file=str(tokenizer_dir / "merges.txt")
    )

def load_input_image(image_path, max_size=384):
    """Load and prepare input image for image-to-image generation.
    Automatically resizes large images to save memory.
    """
    if not image_path or not os.path.exists(image_path):
        return None
    
    try:
        print(f"Loading input image from {image_path}")
        image = Image.open(image_path).convert("RGB")
        
        # Check if image needs resizing to save memory
        width, height = image.size
        if width > max_size or height > max_size:
            print(f"Resizing image from {width}x{height} to max dimension {max_size} to save memory")
            # Calculate new dimensions while preserving aspect ratio
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            
            # Resize the image
            image = image.resize((new_width, new_height), Image.LANCZOS)
            print(f"Resized to {new_width}x{new_height}")
        
        return image
    except Exception as e:
        print(f"Error loading input image: {e}")
        return None

def optimize_for_inference(device):
    """Apply optimizations for inference to reduce memory usage."""
    if "cuda" in device:
        # Enable mixed precision for inference
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set PyTorch to optimize for inference
        torch.set_grad_enabled(False)
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        gc.collect()

def auto_adjust_parameters(device, args):
    """Automatically adjust generation parameters based on available GPU memory."""
    if "cuda" not in device or not args.use_memory_efficient:
        return args
    
    try:
        # Get available GPU memory
        gpu_id = int(device.split(":")[-1]) if ":" in device else 0
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3  # GB
        available_memory = torch.cuda.mem_get_info(gpu_id)[0] / 1024**3  # GB
        
        print(f"\nAuto-adjusting parameters based on available GPU memory:")
        print(f"Total GPU memory: {total_memory:.2f} GB")
        print(f"Available memory: {available_memory:.2f} GB")
        
        # Adjust steps based on available memory - more aggressive reductions
        if available_memory < 1.5:  # Less than 1.5GB available
            args.steps = min(args.steps, 15)
            print(f"Limited memory detected, reducing steps to {args.steps}")
        elif available_memory < 2.5:  # Less than 2.5GB available
            args.steps = min(args.steps, 20)
            print(f"Moderate memory detected, setting steps to {args.steps}")
        
        # For very limited memory, also reduce other parameters
        if available_memory < 1.0:  # Very limited memory
            if args.cfg_scale > 7.0:
                args.cfg_scale = 7.0
                print(f"Reduced CFG scale to {args.cfg_scale} to save memory")
            
            # Reduce strength to preserve more of the original image (less processing)
            if args.strength > 0.7:
                args.strength = 0.7
                print(f"Reduced strength to {args.strength} to save memory")
    
    except Exception as e:
        print(f"Error during auto parameter adjustment: {e}")
    
    return args

def main():
    args = get_user_inputs()
    device = setup_device(args)
    
    # Apply memory optimizations
    optimize_for_inference(device)
    
    # Auto-adjust parameters based on available GPU memory
    args = auto_adjust_parameters(device, args)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load models
    print(f"Loading model from {args.model_path}...")
    print("Loading model weights to CPU first...")
    models = memory_efficient_loader.preload_models_from_standard_weights(args.model_path, device)
    
    # Load tokenizer
    tokenizer = load_tokenizer()
    
    # Load input image (now a required argument)
    input_image = load_input_image(args.input_image)
    if input_image is None:
        print(f"Error: Could not load input image from {args.input_image}")
        return
    
    # Generate timestamp for unique filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Set generation mode to img2img since input image is required
    mode = "img2img"
    
    # Set initial parameters
    steps = args.steps
    cfg_scale = args.cfg_scale
    
    # Print generation parameters
    print(f"\nGeneration Parameters:")
    print(f"Mode: {mode}")
    print(f"Prompt: {args.prompt}")
    print(f"Negative Prompt: {args.negative_prompt}")
    if input_image:
        print(f"Input Image: {args.input_image}")
        print(f"Strength: {args.strength}")
    print(f"CFG Scale: {cfg_scale}")
    print(f"Steps: {steps}")
    print(f"Seed: {args.seed}")
    print(f"Sampler: {args.sampler}")
    
    # Generate image with automatic fallback for memory errors
    print("\nGenerating image...")
    start_time = time.time()
    
    # Clear CUDA cache before generation
    if "cuda" in device:
        torch.cuda.empty_cache()
        gc.collect()
    
    # Try with progressively lower resource settings until successful
    max_attempts = 3
    attempt = 0
    success = False
    
    while attempt < max_attempts and not success:
        try:
            attempt += 1
            
            if attempt > 1:
                print(f"\nRetrying with optimized parameters (attempt {attempt}/{max_attempts})...")
                # Reduce parameters to save memory
                if steps > 20:
                    steps = max(20, steps // 2)  # Reduce steps by half, minimum 20
                    print(f"Reduced steps to {steps}")
                
                # Clear memory again
                if "cuda" in device:
                    torch.cuda.empty_cache()
                    gc.collect()
                    time.sleep(1)  # Give GPU a moment to stabilize
            
            output_image = pipeline.generate(
                prompt=args.prompt,
                uncond_prompt=args.negative_prompt,
                input_image=input_image,
                strength=args.strength,
                do_cfg=True,
                cfg_scale=cfg_scale,
                sampler_name=args.sampler,
                n_inference_steps=steps,
                seed=args.seed,
                models=models,
                device=device,
                idle_device="cpu",
                tokenizer=tokenizer,
            )
            
            success = True
            
            # Convert to PIL Image
            output_pil = Image.fromarray(output_image)
            
            # Save the generated image
            seed_used = args.seed if args.seed is not None else "random"
            filename = f"{timestamp}_{mode}_seed{seed_used}_steps{steps}.png"
            output_path = output_dir / filename
            output_pil.save(output_path)
            
            print(f"\nGeneration completed in {time.time() - start_time:.2f} seconds")
            print(f"Image saved to: {output_path}")
            
            # Display the image if in a notebook environment
            try:
                from IPython.display import display
                display(output_pil)
            except ImportError:
                pass
                
        except torch.cuda.OutOfMemoryError as e:
            print(f"CUDA out of memory error: {e}")
            if attempt >= max_attempts:
                print("\nFailed after multiple attempts. Try manually reducing parameters:")
                print("1. Decrease the number of steps (e.g., 20-30)")
                print("2. Use a smaller input image")
                print("3. Try a different sampler")
                break
                
            # Clear memory after error
            if "cuda" in device:
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(2)  # Give GPU more time to recover
                
        except Exception as e:
            print(f"Error during image generation: {e}")
            break

if __name__ == "__main__":
    main()