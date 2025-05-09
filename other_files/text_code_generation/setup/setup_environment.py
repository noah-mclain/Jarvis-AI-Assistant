#!/usr/bin/env python3
"""
Setup environment and create directories for Jarvis AI Assistant.
"""

import os
import sys

def setup_paperspace_env():
    """
    Set up Paperspace environment.
    """
    # Force Paperspace environment detection
    os.environ["PAPERSPACE"] = "true"
    print("✅ Forced Paperspace environment detection")

    # Set up environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,garbage_collection_threshold:0.8"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    # Force certain operations on CPU to save GPU memory
    os.environ["FORCE_CPU_ONLY_FOR_INITIAL_LOAD"] = "1"
    os.environ["FORCE_CPU_ONLY_FOR_TOKENIZATION"] = "1"
    os.environ["FORCE_CPU_ONLY_FOR_DATASET_PROCESSING"] = "1"
    os.environ["TOKENIZERS_FORCE_CPU"] = "1"
    os.environ["HF_DATASETS_CPU_ONLY"] = "1"
    os.environ["JARVIS_FORCE_CPU_TOKENIZER"] = "1"

    # Set PyTorch to use deterministic algorithms for reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = "42"

    print("✅ Environment variables set for Paperspace")

def create_directories():
    """
    Create necessary directories for Jarvis AI Assistant.
    """
    # Define base directories
    base_dirs = [
        "notebooks/Jarvis_AI_Assistant",
        "notebooks/Jarvis_AI_Assistant/models",
        "notebooks/Jarvis_AI_Assistant/datasets",
        "notebooks/Jarvis_AI_Assistant/checkpoints",
        "notebooks/Jarvis_AI_Assistant/logs",
        "notebooks/Jarvis_AI_Assistant/preprocessed_data",
        "notebooks/Jarvis_AI_Assistant/visualizations",
        "notebooks/Jarvis_AI_Assistant/metrics"
    ]

    # Create directories
    for directory in base_dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

    # Create model-specific directories
    model_dirs = [
        "notebooks/Jarvis_AI_Assistant/models/deepseek-coder-6.7b-finetuned",
        "notebooks/Jarvis_AI_Assistant/models/flan-ul2-finetuned",
        "notebooks/Jarvis_AI_Assistant/models/cnn-flan-ul2-finetuned",
        "notebooks/Jarvis_AI_Assistant/models/custom-encoder-decoder"
    ]

    for directory in model_dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created model directory: {directory}")

    print("✅ All directories created successfully")

if __name__ == "__main__":
    # Force Paperspace environment detection
    os.environ["PAPERSPACE"] = "true"
    print("✅ Forced Paperspace environment detection")

    # Setup environment and create directories
    setup_paperspace_env()
    create_directories()
    print("✅ Environment setup complete")
