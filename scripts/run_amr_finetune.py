import os
import sys
import subprocess
from pathlib import Path

# Add the parent directory to system path to find the modules
sys.path.append(str(Path(__file__).parent.parent))

def main():
    # Define paths
    image_dir = "d:\\programming\\FullProjects\\Jarvis-AI-Assistant\\data\\dreambooth_dataset\\Person\\amr raw"
    output_dir = "d:\\programming\\FullProjects\\Jarvis-AI-Assistant\\checkpoints\\finetuned_amr"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up command arguments
    cmd = [
        sys.executable,
        "d:\\programming\\FullProjects\\Jarvis-AI-Assistant\\scripts\\finetune_example.py",
        "--mode", "dreambooth",
        "--image_dir", image_dir,
        "--class_name", "person",
        "--identifier", "Amr",
        "--output_dir", output_dir,
        "--learning_rate", "2e-7",
        "--epochs", "50",
        "--batch_size", "1",
        "--gradient_accumulation_steps", "8",
        "--mixed_precision",
        "--image_size", "256"
    ]
    
    print("\n" + "="*50)
    print("Starting DreamBooth fine-tuning for 'Amr' images")
    print("="*50)
    print(f"Image directory: {image_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Training parameters:")
    print(f"  - Class name: person")
    print(f"  - Identifier: Amr")
    print(f"  - Learning rate: 2e-7")
    print(f"  - Epochs: 50")
    print(f"  - Batch size: 1")
    print(f"  - Gradient accumulation steps: 8")
    print(f"  - Image size: 256")
    print(f"  - Mixed precision: enabled")
    print("="*50 + "\n")
    
    # Run the fine-tuning script
    print("Running fine-tuning command:")
    print(" ".join(cmd))
    print("\n" + "-"*50 + "\n")
    
    # Execute the command
    subprocess.run(cmd)
    
    print("\n" + "-"*50)
    print("Fine-tuning process completed!")
    print("The fine-tuned model is saved in:", output_dir)
    print("-"*50 + "\n")

if __name__ == "__main__":
    main()