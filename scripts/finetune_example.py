import argparse
import os
import sys
from pathlib import Path

# Add the parent directory to system path to find the modules
sys.path.append(str(Path(__file__).parent.parent))

from scripts.finetune_model import main as finetune_main
from data.preprocessing.custom_dataset import TextImagePairDataset, DreamBoothDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Example script for fine-tuning Stable Diffusion")
    parser.add_argument("--mode", type=str, choices=["dreambooth", "textual_inversion", "custom"], default="custom",
                        help="Fine-tuning mode to use")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing training images")
    parser.add_argument("--prompt_file", type=str, default=None,
                        help="Optional file containing prompts for images")
    parser.add_argument("--concept_name", type=str, default=None,
                        help="Name of the concept to fine-tune (e.g., 'cat', 'painting style')")
    parser.add_argument("--class_name", type=str, default=None,
                        help="Class name for DreamBooth (e.g., 'person', 'photo')")
    parser.add_argument("--identifier", type=str, default="Amr",
                        help="Unique identifier for DreamBooth (e.g., 'Amr')")
    parser.add_argument("--output_dir", type=str, default="../checkpoints/finetuned",
                        help="Directory to save fine-tuned model")
    parser.add_argument("--learning_rate", type=float, default=2e-7,
                        help="Learning rate for fine-tuning")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Number of steps to accumulate gradients")
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Use mixed precision training")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="Enable gradient checkpointing to save memory")
    parser.add_argument("--use_8bit_adam", action="store_true", default=True,
                        help="Use 8-bit Adam optimizer to reduce memory usage")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Size of training images (smaller = less VRAM usage)")
    
    return parser.parse_args()

def setup_dreambooth_args(args):
    """Set up arguments for DreamBooth fine-tuning."""
    if not args.class_name:
        print("Error: --class_name is required for DreamBooth mode")
        sys.exit(1)
    
    # Create instance and class prompts
    instance_prompt = f"a photo of {args.identifier} {args.class_name}"
    class_prompt = f"a photo of {args.class_name}"
    
    print(f"DreamBooth setup:")
    print(f"  - Instance prompt: {instance_prompt}")
    print(f"  - Class prompt: {class_prompt}")
    
    # Set up finetune arguments
    finetune_args = [
        "--base_model", "../checkpoints/v1-5-pruned-emaonly.ckpt",
        "--output_dir", args.output_dir,
        "--dataset", "custom",
        "--custom_dataset_path", args.image_dir,
        "--batch_size", str(args.batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--learning_rate", str(args.learning_rate),
        "--epochs", str(args.epochs),
        "--save_steps", "100",
        "--train_unet_only",  # For DreamBooth, typically only train UNet
        "--concept_name", args.identifier
    ]
    
    if args.mixed_precision:
        finetune_args.append("--mixed_precision")
    
    return finetune_args

def setup_custom_args(args):
    """Set up arguments for custom fine-tuning."""
    finetune_args = [
        "--base_model", "../checkpoints/v1-5-pruned-emaonly.ckpt",
        "--output_dir", args.output_dir,
        "--dataset", "custom",
        "--custom_dataset_path", args.image_dir,
        "--batch_size", str(args.batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--learning_rate", str(args.learning_rate),
        "--epochs", str(args.epochs),
        "--save_steps", "100"
    ]
    
    if args.prompt_file:
        finetune_args.extend(["--prompt_file", args.prompt_file])
    
    if args.concept_name:
        finetune_args.extend(["--concept_name", args.concept_name])
    
    if args.mixed_precision:
        finetune_args.append("--mixed_precision")
    
    return finetune_args

def main():
    args = parse_args()
    
    print(f"Starting fine-tuning in {args.mode} mode")
    print(f"Image directory: {args.image_dir}")
    
    # Set up arguments based on mode
    if args.mode == "dreambooth":
        finetune_args = setup_dreambooth_args(args)
    elif args.mode == "textual_inversion":
        # Textual inversion is similar to custom but with specific settings
        args.concept_name = args.concept_name or "<concept>"
        finetune_args = setup_custom_args(args)
        finetune_args.extend(["--train_text_encoder"])  # Train text encoder for textual inversion
    else:  # custom mode
        finetune_args = setup_custom_args(args)
    
    # Convert arguments to the format expected by finetune_main
    sys.argv = [sys.argv[0]] + finetune_args
    
    # Run fine-tuning
    print("\nStarting fine-tuning with the following arguments:")
    print(" ".join(sys.argv))
    print("\n" + "-"*50 + "\n")
    
    finetune_main()

if __name__ == "__main__":
    main()