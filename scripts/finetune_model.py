import torch
import sys
import os
import argparse
import time
import gc
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

# Add the parent directory to system path to find the modules
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import memory_efficient_loader
from src.pipelines import pipeline
from src.models.diffusion import Diffusion
from transformers import CLIPTokenizer
from data.preprocessing.dataset import ProcessedImageDataset, load_laion_dataset
from data.preprocessing.custom_dataset import TextImagePairDataset, DreamBoothDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Stable Diffusion model")
    parser.add_argument("--base_model", type=str, default="../checkpoints/v1-5-pruned-emaonly.ckpt",
                        help="Path to the base model checkpoint")
    parser.add_argument("--output_dir", type=str, default="../checkpoints/finetuned",
                        help="Directory to save fine-tuned model checkpoints")
    parser.add_argument("--dataset", type=str, choices=["laion", "custom"], default="custom",
                        help="Dataset to use for fine-tuning")
    parser.add_argument("--custom_dataset_path", type=str, default=None,
                        help="Path to custom dataset (if dataset=custom)")
    parser.add_argument("--prompt_file", type=str, default=None,
                        help="Path to file containing prompts for training (one per line)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of steps to accumulate gradients")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every X steps")
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Use mixed precision training")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda:0, cuda:1, cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--cfg_scale", type=float, default=7.5,
                        help="Classifier-free guidance scale for training")
    parser.add_argument("--train_text_encoder", action="store_true",
                        help="Whether to train the text encoder (CLIP) as well")
    parser.add_argument("--train_unet_only", action="store_true",
                        help="Train only the UNet/diffusion model")
    parser.add_argument("--concept_name", type=str, default=None,
                        help="Name of the concept to fine-tune on (e.g., 'cat', 'painting style')")
    parser.add_argument("--class_name", type=str, default=None,
                        help="Class name for DreamBooth (e.g., 'dog', 'cat')")
    parser.add_argument("--identifier", type=str, default="sks",
                        help="Unique identifier for DreamBooth (e.g., 'sks')")
    
    return parser.parse_args()

def setup_device(args):
    """Set up the device for training."""
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
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"
        torch.backends.cudnn.benchmark = True
    
    return device

def load_tokenizer():
    """Load the CLIP tokenizer."""
    tokenizer_dir = Path(__file__).parent.parent / "data" / "tokenizer"
    return CLIPTokenizer(
        str(tokenizer_dir / "vocab.json"),
        merges_file=str(tokenizer_dir / "merges.txt")
    )

def load_dataset(args):
    """Load the dataset for fine-tuning."""
    if args.dataset == "laion":
        print("Loading LAION dataset...")
        return load_laion_dataset(split="train", streaming=True)
    elif args.dataset == "custom" and args.custom_dataset_path:
        print(f"Loading custom dataset from {args.custom_dataset_path}...")
        
        # Check if we're doing DreamBooth-style fine-tuning
        if args.class_name and args.identifier:
            print(f"Setting up DreamBooth dataset with class '{args.class_name}' and identifier '{args.identifier}'")
            instance_prompt = f"a photo of {args.identifier} {args.class_name}"
            class_prompt = f"a photo of {args.class_name}"
            
            return DreamBoothDataset(
                image_dir=args.custom_dataset_path,
                class_name=args.class_name,
                instance_prompt=instance_prompt,
                class_prompt=class_prompt
            )
        else:
            # Regular custom dataset
            return TextImagePairDataset(
                image_dir=args.custom_dataset_path,
                prompt_file=args.prompt_file,
                concept_name=args.concept_name
            )
    else:
        raise ValueError("Must specify either LAION dataset or a custom dataset path")

def prepare_models_for_training(models, args, device):
    """Prepare models for training by setting gradients."""
    # Set which models to train
    for model_name, model in models.items():
        requires_grad = False
        
        if model_name == "diffusion":
            # Always train the diffusion model
            requires_grad = True
        elif model_name == "clip" and args.train_text_encoder:
            # Train text encoder if specified
            requires_grad = True
        elif not args.train_unet_only and model_name in ["encoder", "decoder"]:
            # Train VAE if not training UNet only
            requires_grad = True
            
        for param in model.parameters():
            param.requires_grad = requires_grad
            
        print(f"Model {model_name} requires_grad: {requires_grad}")
    
    return models

def save_checkpoint(models, optimizer, epoch, step, args):
    """Save a checkpoint of the model."""
    checkpoint_dir = Path(args.output_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    checkpoint_name = f"checkpoint-epoch{epoch}-step{step}.pt"
    checkpoint_path = checkpoint_dir / checkpoint_name
    
    # Create state dict for all models
    state_dict = {
        "clip": models["clip"].state_dict(),
        "encoder": models["encoder"].state_dict(),
        "decoder": models["decoder"].state_dict(),
        "diffusion": models["diffusion"].state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "args": vars(args)
    }
    
    print(f"Saving checkpoint to {checkpoint_path}...")
    torch.save(state_dict, checkpoint_path)
    
    # Also save as latest
    latest_path = checkpoint_dir / "latest.pt"
    torch.save(state_dict, latest_path)

def train_step(batch, models, tokenizer, optimizer, device, args, scaler=None):
    """Perform a single training step."""
    # Extract data from batch
    prompts = batch["prompt"]
    images = batch["image"].to(device)
    
    # Clear CUDA cache
    if "cuda" in device:
        torch.cuda.empty_cache()
        gc.collect()
    
    # Move models to device
    for model_name, model in models.items():
        model.to(device)
    
    # Process text with CLIP
    clip = models["clip"]
    
    # Convert prompts into tokens
    tokens = tokenizer.batch_encode_plus(
        prompts, padding="max_length", max_length=77
    ).input_ids
    tokens = torch.tensor(tokens, dtype=torch.long, device=device)
    
    # Get CLIP embeddings
    with autocast() if args.mixed_precision else torch.no_grad():
        context = clip(tokens)
    
    # Encode images to latents
    encoder = models["encoder"]
    with autocast() if args.mixed_precision else torch.no_grad():
        # Generate random noise for the encoder
        encoder_noise = torch.randn((images.shape[0], 4, 64, 64), device=device)
        # Encode images to latents
        latents = encoder(images, encoder_noise)
    
    # Set up diffusion model
    diffusion = models["diffusion"]
    
    # Choose a random timestep
    timesteps = torch.randint(
        0, 1000, (images.shape[0],), device=device
    ).long()
    
    # Add noise to latents
    noise = torch.randn_like(latents)
    noisy_latents = latents + noise * timesteps.view(-1, 1, 1, 1).float()
    
    # Get time embeddings
    time_embeddings = torch.stack([pipeline.get_time_embedding(t) for t in timesteps]).to(device)
    
    # Predict noise with diffusion model
    with autocast() if args.mixed_precision else torch.no_grad():
        predicted_noise = diffusion(noisy_latents, context, time_embeddings)
    
    # Calculate loss
    loss = torch.nn.functional.mse_loss(predicted_noise, noise)
    
    # Scale loss by gradient accumulation steps
    loss = loss / args.gradient_accumulation_steps
    
    # Backward pass with mixed precision if enabled
    if args.mixed_precision and scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()
    
    return loss.item() * args.gradient_accumulation_steps

def train(args, models, tokenizer, train_dataloader, device):
    """Train the model."""
    # Prepare models for training
    models = prepare_models_for_training(models, args, device)
    
    # Set up optimizer - only optimize parameters that require gradients
    optimizer_params = []
    for model in models.values():
        optimizer_params.extend([p for p in model.parameters() if p.requires_grad])
    
    optimizer = AdamW(optimizer_params, lr=args.learning_rate)
    
    # Set up mixed precision training if enabled
    scaler = GradScaler() if args.mixed_precision else None
    
    # Training loop
    global_step = 0
    total_loss = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        progress_bar = tqdm(train_dataloader, desc=f"Training epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            # Perform training step
            loss = train_step(batch, models, tokenizer, optimizer, device, args, scaler)
            total_loss += loss
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss})
            
            # Perform optimizer step after gradient accumulation
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.mixed_precision and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
                global_step += 1
                
                # Log average loss
                if global_step % 10 == 0:
                    avg_loss = total_loss / 10
                    print(f"Step {global_step}: Average loss = {avg_loss:.4f}")
                    total_loss = 0
                
                # Save checkpoint
                if global_step % args.save_steps == 0:
                    save_checkpoint(models, optimizer, epoch, global_step, args)
                    
                # Clear CUDA cache periodically
                if "cuda" in device and global_step % 50 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
        
        # Save checkpoint at the end of each epoch
        save_checkpoint(models, optimizer, epoch, global_step, args)
    
    print("Training complete!")
    return models

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Set up device
    device = setup_device(args)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer()
    
    # Set up model paths
    model_path = Path(args.base_model)
    if not model_path.is_absolute():
        model_path = Path(__file__).parent.parent / args.base_model.lstrip("./").lstrip("../")
    
    print(f"Loading base model from {model_path}...")
    
    # Load models with memory efficient loader
    models = memory_efficient_loader.preload_models_from_standard_weights(str(model_path), device)
    
    # Load dataset
    train_dataset = load_dataset(args)
    
    # Create data loader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    # Train the model
    print("Starting training...")
    trained_models = train(args, models, tokenizer, train_dataloader, device)
    
    # Save final model
    final_checkpoint_path = Path(args.output_dir) / "final_model.pt"
    print(f"Saving final model to {final_checkpoint_path}...")
    
    # Create state dict for all models
    state_dict = {
        "clip": trained_models["clip"].state_dict(),
        "encoder": trained_models["encoder"].state_dict(),
        "decoder": trained_models["decoder"].state_dict(),
        "diffusion": trained_models["diffusion"].state_dict(),
    }
    
    torch.save(state_dict, final_checkpoint_path)
    print(f"Model saved to {final_checkpoint_path}")

if __name__ == "__main__":
    main()