#!/usr/bin/env python3
"""
Consolidated Verification Module

This module consolidates all verification functionality for:
- GPU availability and configuration
- Model loading and compatibility
- Package installation and versions
- Environment setup

This consolidates functionality from:
- verify_gpu_cnn_text.py
- verify_gpu_code.py
- verify_gpu_custom_model.py
- verify_gpu_text.py
- verify_models.py
- verify_packages.py
"""

import os
import sys
import logging
import importlib
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to the path so we can import from the root
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def verify_gpu():
    """
    Verify GPU availability and configuration.
    
    Returns:
        dict: GPU information
    """
    gpu_info = {
        "available": False,
        "name": None,
        "memory": None,
        "cuda_version": None,
        "device_count": 0
    }
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["device_count"] = torch.cuda.device_count()
            gpu_info["name"] = torch.cuda.get_device_name(0)
            gpu_info["cuda_version"] = torch.version.cuda
            
            # Get GPU memory
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_info["memory"] = f"{gpu_memory / (1024 ** 3):.2f} GB"
            except Exception as e:
                logger.warning(f"Could not get GPU memory: {e}")
            
            logger.info(f"GPU available: {gpu_info['name']}")
            logger.info(f"CUDA version: {gpu_info['cuda_version']}")
            logger.info(f"GPU memory: {gpu_info['memory']}")
        else:
            logger.warning("No GPU available. Using CPU.")
            
            # Check if MPS (Metal Performance Shaders) is available for Apple Silicon
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                gpu_info["available"] = True
                gpu_info["name"] = "Apple MPS (Metal Performance Shaders)"
                gpu_info["device_count"] = 1
                logger.info("Apple MPS (Metal Performance Shaders) is available.")
            else:
                logger.warning("Neither CUDA nor MPS is available. Using CPU only.")
    
    except ImportError:
        logger.error("PyTorch not installed. Cannot verify GPU.")
    
    return gpu_info

def verify_packages():
    """
    Verify package installation and versions.
    
    Returns:
        dict: Package information
    """
    package_info = {}
    
    # List of packages to verify
    packages = [
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "bitsandbytes",
        "unsloth",
        "datasets",
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "joblib",
        "spacy",
        "nltk",
        "tensorboard",
        "tqdm",
        "huggingface_hub"
    ]
    
    for package in packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "unknown")
            package_info[package] = {
                "installed": True,
                "version": version
            }
            logger.info(f"{package} version: {version}")
        except ImportError:
            package_info[package] = {
                "installed": False,
                "version": None
            }
            logger.warning(f"{package} not installed.")
    
    # Special check for transformers.utils
    try:
        import transformers.utils
        package_info["transformers.utils"] = {
            "installed": True,
            "version": "available"
        }
        logger.info("transformers.utils is available.")
    except ImportError:
        package_info["transformers.utils"] = {
            "installed": False,
            "version": None
        }
        logger.warning("transformers.utils is not available.")
    
    return package_info

def verify_models(model_type="all"):
    """
    Verify model loading and compatibility.
    
    Args:
        model_type (str): Type of model to verify (code, text, cnn-text, custom-model, all)
        
    Returns:
        dict: Model verification results
    """
    model_results = {}
    
    # Verify PyTorch and transformers installation
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        logger.error(f"Required packages not installed: {e}")
        return {"error": f"Required packages not installed: {e}"}
    
    # Verify GPU
    gpu_info = verify_gpu()
    model_results["gpu"] = gpu_info
    
    # Define model configurations for each type
    model_configs = {}
    
    if model_type in ["code", "all"]:
        model_configs["code"] = {
            "model_name": "deepseek-ai/deepseek-coder-6.7b-base",
            "tokenizer_name": "deepseek-ai/deepseek-coder-6.7b-base",
            "test_input": "def fibonacci(n):"
        }
    
    if model_type in ["text", "all"]:
        model_configs["text"] = {
            "model_name": "deepseek-ai/deepseek-llm-7b-base",
            "tokenizer_name": "deepseek-ai/deepseek-llm-7b-base",
            "test_input": "The capital of France is"
        }
    
    if model_type in ["cnn-text", "all"]:
        model_configs["cnn-text"] = {
            "model_name": "gpt2",  # Using a smaller model for testing
            "tokenizer_name": "gpt2",
            "test_input": "CNN is a type of"
        }
    
    if model_type in ["custom-model", "all"]:
        model_configs["custom-model"] = {
            "model_name": "t5-small",  # Using a smaller model for testing
            "tokenizer_name": "t5-small",
            "test_input": "translate English to German: The house is wonderful."
        }
    
    # Verify each model
    for model_key, config in model_configs.items():
        logger.info(f"Verifying {model_key} model...")
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                config["tokenizer_name"],
                trust_remote_code=True
            )
            
            # Tokenize test input
            inputs = tokenizer(config["test_input"], return_tensors="pt")
            
            # Load model with minimal configuration for verification
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if gpu_info["available"] else torch.float32,
            }
            
            # Add device_map if GPU is available
            if gpu_info["available"]:
                model_kwargs["device_map"] = "auto"
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                config["model_name"],
                **model_kwargs
            )
            
            # Generate a short sequence
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_length=20,
                    num_return_sequences=1
                )
            
            # Decode the outputs
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            model_results[model_key] = {
                "success": True,
                "generated_text": generated_text
            }
            
            logger.info(f"Successfully verified {model_key} model.")
            logger.info(f"Generated text: {generated_text}")
            
        except Exception as e:
            model_results[model_key] = {
                "success": False,
                "error": str(e)
            }
            
            logger.error(f"Failed to verify {model_key} model: {e}")
    
    return model_results

def verify_environment():
    """
    Verify environment setup.
    
    Returns:
        dict: Environment information
    """
    env_info = {
        "python_version": sys.version,
        "platform": sys.platform,
        "in_paperspace": os.path.exists("/notebooks") or os.path.exists("/storage"),
        "in_colab": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "Not set"),
        "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING", "Not set"),
        "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM", "Not set")
    }
    
    # Check if running in Google Colab
    try:
        import google.colab
        env_info["in_colab"] = True
    except ImportError:
        pass
    
    logger.info(f"Python version: {env_info['python_version']}")
    logger.info(f"Platform: {env_info['platform']}")
    logger.info(f"In Paperspace: {env_info['in_paperspace']}")
    logger.info(f"In Google Colab: {env_info['in_colab']}")
    
    return env_info

def run_verification(model_type="all"):
    """
    Run all verification checks.
    
    Args:
        model_type (str): Type of model to verify (code, text, cnn-text, custom-model, all)
        
    Returns:
        dict: Verification results
    """
    results = {
        "environment": verify_environment(),
        "packages": verify_packages(),
        "gpu": verify_gpu()
    }
    
    # Only verify models if requested
    if model_type != "none":
        results["models"] = verify_models(model_type)
    
    return results

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Verify GPU, packages, and models.")
    parser.add_argument("--model-type", type=str, default="all",
                        choices=["code", "text", "cnn-text", "custom-model", "all", "none"],
                        help="Type of model to verify")
    args = parser.parse_args()
    
    # Run verification
    results = run_verification(args.model_type)
    
    # Print summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    # Environment summary
    print("\nEnvironment:")
    print(f"  Python version: {results['environment']['python_version'].split()[0]}")
    print(f"  Platform: {results['environment']['platform']}")
    print(f"  In Paperspace: {results['environment']['in_paperspace']}")
    print(f"  In Google Colab: {results['environment']['in_colab']}")
    
    # GPU summary
    print("\nGPU:")
    if results["gpu"]["available"]:
        print(f"  Available: Yes")
        print(f"  Name: {results['gpu']['name']}")
        print(f"  Memory: {results['gpu']['memory']}")
        print(f"  CUDA version: {results['gpu']['cuda_version']}")
    else:
        print(f"  Available: No")
    
    # Packages summary
    print("\nPackages:")
    for package, info in results["packages"].items():
        if info["installed"]:
            print(f"  {package}: {info['version']}")
        else:
            print(f"  {package}: Not installed")
    
    # Models summary
    if "models" in results:
        print("\nModels:")
        for model_type, info in results["models"].items():
            if model_type != "gpu":
                if info.get("success", False):
                    print(f"  {model_type}: Verified successfully")
                else:
                    print(f"  {model_type}: Verification failed - {info.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 80)
