#!/usr/bin/env python3
"""
CPU-first loading patch for DeepSeek Coder model training.
This script forces the initial model loading to happen on CPU,
then transfers only necessary parts to GPU for training.
"""

import os
import sys
import torch
import gc
import argparse

# Force CPU for initial model loading
os.environ["FORCE_CPU_ONLY_FOR_INITIAL_LOAD"] = "1"

# Set PyTorch to use CPU as default device initially
if hasattr(torch, 'set_default_device'):
    torch.set_default_device('cpu')
torch.set_default_tensor_type('torch.FloatTensor')

# Clear any existing CUDA memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

# Set environment variables for optimal memory usage
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,garbage_collection_threshold:0.9"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/tmp/hf_cache"  # Use temporary directory for cache

def parse_args():
    """Parse command line arguments and pass them to the main training script"""
    parser = argparse.ArgumentParser(description="CPU-first loading for DeepSeek Coder training")
    
    # Add all the arguments from train_models.py
    parser.add_argument('--model_type', type=str, default='code', help='Type of model to train (text, code)')
    parser.add_argument('--dataset', type=str, default="codeparrot/github-code:0.7,code-search-net/code_search_net:0.3", 
                        help='Dataset(s) to use for training (comma-separated)')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to use')
    parser.add_argument('--dataset_subset', type=str, default="python,javascript", 
                        help='Subset of dataset to use (comma-separated)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1.5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=64, 
                        help='Number of gradient accumulation steps')
    parser.add_argument('--model_name_or_path', type=str, default="deepseek-ai/deepseek-coder-5.7b-instruct", 
                        help='Model name or path')
    parser.add_argument('--output_dir', type=str, default="./models", help='Output directory')
    parser.add_argument('--eval_metrics_dir', type=str, default="./metrics", help='Evaluation metrics directory')
    parser.add_argument('--save_strategy', type=str, default="epoch", help='Save strategy')
    parser.add_argument('--evaluation_strategy', type=str, default="steps", help='Evaluation strategy')
    parser.add_argument('--logging_steps', type=int, default=50, help='Logging steps')
    parser.add_argument('--eval_steps', type=int, default=500, help='Evaluation steps')
    parser.add_argument('--use_deepspeed', action='store_true', help='Use DeepSpeed')
    parser.add_argument('--use_8bit', action='store_true', help='Use 8-bit quantization')
    parser.add_argument('--use_4bit', action='store_true', help='Use 4-bit quantization')
    parser.add_argument('--use_qlora', action='store_true', help='Use QLoRA')
    parser.add_argument('--fp16', action='store_true', help='Use FP16')
    parser.add_argument('--bf16', action='store_true', help='Use BF16')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--use_flash_attn', action='store_true', help='Use Flash Attention')
    parser.add_argument('--use_flash_attention_2', action='store_true', help='Use Flash Attention 2')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Use gradient checkpointing')
    parser.add_argument('--sequence_packing', action='store_true', help='Use sequence packing')
    parser.add_argument('--optim', type=str, default="adamw_bnb_8bit", help='Optimizer')
    parser.add_argument('--save_steps', type=int, default=1000, help='Save steps')
    parser.add_argument('--use_unsloth', action='store_true', help='Use Unsloth')
    parser.add_argument('--fim_rate', type=float, default=0.6, help='FIM rate')
    parser.add_argument('--pad_token_id', type=int, default=50256, help='Pad token ID')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--visualize_metrics', action='store_true', help='Visualize metrics')
    parser.add_argument('--cache_dir', type=str, default=".cache", help='Cache directory')
    parser.add_argument('--force_gpu', action='store_true', help='Force GPU usage')
    parser.add_argument('--cpu', action='store_true', help='Use CPU')
    parser.add_argument('--use_cnn', action='store_true', help='Use CNN')
    parser.add_argument('--cnn_layers', type=int, default=2, help='Number of CNN layers')
    parser.add_argument('--cnn_kernel_sizes', type=str, default="3,5,7", 
                        help='CNN kernel sizes (comma-separated)')
    
    return parser.parse_args()

def main():
    """Main function to run the training with CPU-first loading"""
    # Parse arguments
    args = parse_args()
    
    # Set default values for DeepSeek Coder training
    if not args.model_type:
        args.model_type = 'code'
    if not args.model_name_or_path:
        args.model_name_or_path = "deepseek-ai/deepseek-coder-5.7b-instruct"
    
    # Force 4-bit quantization for memory efficiency
    args.use_4bit = True
    args.use_qlora = True
    args.force_gpu = True
    
    # Import the actual training module
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "./")))
    from src.generative_ai_module.train_models import main as train_main
    
    # Run the main function with our arguments
    train_main(args)

if __name__ == "__main__":
    main()
