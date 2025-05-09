#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Jarvis AI Assistant - Google Colab Command Helper
This script provides ready-to-use commands for running tasks on Google Colab.
"""

import argparse
import os
import sys
from pathlib import Path

def print_setup_commands():
    """Print commands for initial setup in Google Colab"""
    print("# Jarvis AI Assistant - Google Colab Setup Commands")
    print("# ================================================")
    print("")
    print("# Clone the repository")
    print("!git clone https://github.com/your-username/Jarvis-AI-Assistant.git")
    print("!cd Jarvis-AI-Assistant")
    print("")
    print("# Setup the environment with A100 optimizations")
    print("!bash colab_setup.sh")
    print("")
    print("# Mount Google Drive for persistent storage")
    print("from google.colab import drive")
    print("drive.mount('/content/drive')")
    print("")
    print("# Create directories in Google Drive for persistent storage")
    print("!mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant/models")
    print("!mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant/datasets")
    print("!mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant/checkpoints")
    print("")
    print("# Verify GPU availability")
    print("import torch")
    print("print(f'PyTorch version: {torch.__version__}')")
    print("print(f'CUDA available: {torch.cuda.is_available()}')")
    print("if torch.cuda.is_available():")
    print("    print(f'GPU: {torch.cuda.get_device_name(0)}')")
    print("    print(f'CUDA version: {torch.version.cuda}')")
    print("    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')")

def print_finetune_commands():
    """Print commands for fine-tuning models in Google Colab"""
    print("# Jarvis AI Assistant - Fine-Tuning Commands for Google Colab")
    print("# =========================================================")
    print("")
    print("# Option 1: Fine-tune with standard HuggingFace dataset")
    print("!python src/generative_ai_module/run_finetune.py \\")
    print("    --base-model deepseek-ai/deepseek-coder-6.7b-base \\")
    print("    --dataset databricks/databricks-dolly-15k \\")
    print("    --output-dir /content/drive/MyDrive/Jarvis_AI_Assistant/models/fine_tuned \\")
    print("    --load-in-4bit \\")
    print("    --epochs 1 \\")
    print("    --batch-size 4 \\")
    print("    --gradient-accumulation-steps 4 \\")
    print("    --use-unsloth")
    print("")
    print("# Option 2: Fine-tune with custom JSON dataset")
    print("!python src/generative_ai_module/run_finetune.py \\")
    print("    --base-model deepseek-ai/deepseek-coder-6.7b-base \\")
    print("    --json-file /content/drive/MyDrive/Jarvis_AI_Assistant/datasets/your_dataset.json \\")
    print("    --output-dir /content/drive/MyDrive/Jarvis_AI_Assistant/models/fine_tuned \\")
    print("    --load-in-4bit \\")
    print("    --epochs 1 \\")
    print("    --batch-size 4 \\")
    print("    --gradient-accumulation-steps 4 \\")
    print("    --use-unsloth")
    print("")
    print("# Option 3: Fine-tune custom JSON with higher LoRA rank")
    print("!python src/generative_ai_module/run_finetune.py \\")
    print("    --base-model deepseek-ai/deepseek-coder-6.7b-base \\")
    print("    --json-file /content/drive/MyDrive/Jarvis_AI_Assistant/datasets/your_dataset.json \\")
    print("    --output-dir /content/drive/MyDrive/Jarvis_AI_Assistant/models/fine_tuned \\")
    print("    --load-in-4bit \\")
    print("    --epochs 2 \\")
    print("    --batch-size 2 \\")
    print("    --gradient-accumulation-steps 8 \\")
    print("    --lora-r 128 \\")
    print("    --lora-alpha 256 \\")
    print("    --use-unsloth")

def print_inference_commands():
    """Print commands for running inference in Google Colab"""
    print("# Jarvis AI Assistant - Inference Commands for Google Colab")
    print("# ====================================================")
    print("")
    print("# Option 1: Run with HuggingFace model in interactive mode")
    print("!python src/generative_ai_module/run_jarvis.py \\")
    print("    --model deepseek-ai/deepseek-coder-6.7b-instruct \\")
    print("    --interactive \\")
    print("    --output /content/drive/MyDrive/Jarvis_AI_Assistant/chat_history.json")
    print("")
    print("# Option 2: Run with fine-tuned model in interactive mode")
    print("!python src/generative_ai_module/run_jarvis.py \\")
    print("    --model-path /content/drive/MyDrive/Jarvis_AI_Assistant/models/fine_tuned/jarvis-YYYY-MM-DD_HH-MM-SS \\")
    print("    --interactive \\")
    print("    --output /content/drive/MyDrive/Jarvis_AI_Assistant/chat_history.json")
    print("")
    print("# Option 3: Run with a single prompt (non-interactive)")
    print("!python src/generative_ai_module/run_jarvis.py \\")
    print("    --model deepseek-ai/deepseek-coder-6.7b-instruct \\")
    print("    --prompt \"Write a Python function to calculate Fibonacci numbers\" \\")
    print("    --output /content/drive/MyDrive/Jarvis_AI_Assistant/chat_history.json")
    print("")
    print("# Option 4: Run with previous chat history")
    print("!python src/generative_ai_module/run_jarvis.py \\")
    print("    --model deepseek-ai/deepseek-coder-6.7b-instruct \\")
    print("    --interactive \\")
    print("    --history /content/drive/MyDrive/Jarvis_AI_Assistant/chat_history.json \\")
    print("    --output /content/drive/MyDrive/Jarvis_AI_Assistant/chat_history.json")

def print_all_commands():
    """Print all commands"""
    print_setup_commands()
    print("\n" + "=" * 80 + "\n")
    print_finetune_commands()
    print("\n" + "=" * 80 + "\n")
    print_inference_commands()

def main():
    """Main function to parse arguments and show commands"""
    parser = argparse.ArgumentParser(description="Google Colab command templates for Jarvis AI Assistant")
    parser.add_argument("--type", choices=["setup", "finetune", "inference", "all"], 
                        default="all", help="Type of commands to display")
    
    args = parser.parse_args()
    
    if args.type == "setup":
        print_setup_commands()
    elif args.type == "finetune":
        print_finetune_commands()
    elif args.type == "inference":
        print_inference_commands()
    else:
        print_all_commands()

if __name__ == "__main__":
    main() 