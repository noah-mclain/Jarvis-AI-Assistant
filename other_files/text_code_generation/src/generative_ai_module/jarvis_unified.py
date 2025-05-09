"""
Jarvis Unified AI Module

This module provides a unified interface for the Jarvis AI capabilities,
including dataset processing, model training, context-aware text generation,
and interactive sessions. Optimized for Google Colab with A100 GPU and Paperspace with RTX4000/5000 GPUs.
"""

import os
import sys
import json
import time
import glob
import logging
import datetime
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Set
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    default_data_collator,
    BitsAndBytesConfig
)

# =========== CRITICAL GPU ENFORCEMENT MODULE ============
# This section ensures that GPU is used for all operations from the moment of import
def _force_gpu_usage():
    """Force GPU usage for all PyTorch operations throughout the codebase"""
    try:
        # Set environment variables to ensure CUDA visibility
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # For deterministic operations

        # Basic logging configuration
        print("Forcing GPU usage for all Jarvis AI operations...")

        # Set global seeds for reproducibility
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)

        # Try multiple approaches to ensure GPU usage
        if torch.cuda.is_available():
            # Set CUDA device
            torch.cuda.set_device(0)

            # IMPORTANT: Don't force CUDA as default tensor type
            # This causes issues with DataLoader workers
            # Instead, only set the default device in PyTorch 2.0+
            if hasattr(torch, 'set_default_device'):
                torch.set_default_device('cuda')

            # Set all CUDA seeds
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)

            # Get GPU info for logging
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else "Unknown"

            # Enable cudnn benchmark for faster training
            if hasattr(torch.backends, 'cudnn'):
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True

                # For reproducibility (may be slower but more stable)
                torch.backends.cudnn.deterministic = True

            # Try to get GPU memory
            try:
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"GPU Memory: {gpu_memory_gb:.2f} GB")
            except:
                pass

            # DO NOT override tensor creation to always use CUDA
            # This causes issues with DataLoader workers
            # Instead, explicitly move tensors to GPU where needed in the code

            print(f"✅ GPU enforcement successful. Using CUDA device: {gpu_name} (CUDA {cuda_version})")
            return True

        # Try Apple Silicon MPS as fallback
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Set MPS as device
            device = torch.device("mps")

            # Clear MPS cache to free memory
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()

            print(f"✅ GPU enforcement successful. Using Apple Silicon MPS device")
            return True

        else:
            print("⚠️ No GPU available. Performance will be significantly slower on CPU.")
            return False

    except Exception as e:
        print(f"⚠️ Error enforcing GPU usage: {str(e)}")
        print("⚠️ Operations will fall back to CPU, but performance will be degraded.")
        return False

# Run GPU enforcement immediately on module import
_GPU_AVAILABLE = _force_gpu_usage()

# ======== CRITICAL PATCH TO FIX CUDA GENERATOR ISSUES ========
# This patch ensures that PyTorch's random_split and DataLoader will work correctly
# with CUDA by always using CPU generators for samplers
def _patch_torch_for_cuda_generator_compatibility():
    # First set global seed for all operations for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    random.seed(42)
    np.random.seed(42)

    # 1. CRITICAL FIX: Monkey patch torch.randperm to ensure it never uses CUDA generators
    original_randperm = torch.randperm
    def safe_randperm(n, *, dtype=None, layout=None, device=None, requires_grad=False, pin_memory=False, generator=None):
        # Always force CPU device for randperm with generators
        if generator is not None and hasattr(generator, 'device') and generator.device.type != 'cpu':
            # Create a CPU generator with identical seed
            seed = generator.initial_seed()
            generator = torch.Generator().manual_seed(seed)

        # Always explicitly use CPU device for randperm to avoid CUDA generator issues
        device_to_use = 'cpu'
        result = original_randperm(n, dtype=dtype, layout=layout, device=device_to_use,
                                  requires_grad=requires_grad, pin_memory=pin_memory,
                                  generator=generator)

        # Move result to requested device if needed
        if device is not None and device != 'cpu':
            result = result.to(device)

        return result

    # Apply the randperm patch
    torch.randperm = safe_randperm

    # 2. Patch random_split to always use CPU generators
    if hasattr(torch.utils.data, 'random_split'):
        original_random_split = torch.utils.data.random_split

        def patched_random_split(dataset, lengths, generator=None):
            """Patched version that ensures CPU generator is used"""
            # Always use CPU generator regardless of what was passed
            cpu_generator = torch.Generator().manual_seed(42 if generator is None else generator.initial_seed())
            return original_random_split(dataset, lengths, generator=cpu_generator)

        # Apply the patch
        torch.utils.data.random_split = patched_random_split

    # 3. Also patch DataLoader to ensure it uses CPU generators for samplers
    original_dataloader_init = torch.utils.data.DataLoader.__init__

    def patched_dataloader_init(self, *args, **kwargs):
        # If a generator is provided but is on CUDA, convert to CPU
        if 'generator' in kwargs and kwargs['generator'] is not None:
            if hasattr(kwargs['generator'], 'device') and str(kwargs['generator'].device) != 'cpu':
                # Create a CPU generator with the same seed
                seed = kwargs['generator'].initial_seed()
                kwargs['generator'] = torch.Generator().manual_seed(seed)

        # Call the original init
        original_dataloader_init(self, *args, **kwargs)

    # Apply the DataLoader patch
    torch.utils.data.DataLoader.__init__ = patched_dataloader_init

    # 4. Override torch.Generator behavior to always create CPU generators
    original_generator = torch.Generator
    def safe_generator(*args, **kwargs):
        # Force device to CPU regardless of what was requested
        if 'device' in kwargs and kwargs['device'] != 'cpu':
            print(f"WARNING: Requested Generator on {kwargs['device']} - forcing CPU generator instead")
            kwargs['device'] = 'cpu'
        return original_generator(*args, **kwargs)

    # Apply the Generator patch cautiously - only in specific contexts where it's needed
    # This is to avoid breaking other functionality that might rely on CUDA generators
    torch.utils.data.generator = safe_generator

    # Log the patch application
    print("Applied critical patch for CUDA generator compatibility")

# Apply the patch immediately
_patch_torch_for_cuda_generator_compatibility()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("jarvis_ai.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("jarvis_unified")

# Additional GPU optimizations for specific hardware
try:
    # Check for CUDA again (redundant but ensures logging)
    if torch.cuda.is_available():
        # Check for RTX5000 GPU
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {gpu_name}")

        # Get CUDA version for optimizations
        cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else "Unknown"
        logger.info(f"CUDA Version: {cuda_version}")

        # Get GPU memory
        if hasattr(torch.cuda, 'get_device_properties'):
            props = torch.cuda.get_device_properties(0)
            memory_gb = props.total_memory / (1024**3)
            logger.info(f"GPU Memory: {memory_gb:.2f} GB")

        # Check for RTX5000 in Paperspace
        if "RTX5000" in gpu_name or "RTX 5000" in gpu_name:
            logger.info("RTX 5000 GPU detected - applying optimized settings")

            # Force GPU visibility (redundant but ensuring it's set)
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"

            # Memory optimization
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
            os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

            # Performance optimizations
            torch.backends.cudnn.benchmark = True

            # Check for BF16 support (unlikely on RTX5000 but check anyway)
            bf16_support = torch.cuda.is_bf16_supported() if hasattr(torch.cuda, 'is_bf16_supported') else False
            logger.info(f"BF16 support: {bf16_support}")

            # Check for Unsloth optimizations
            try:
                import unsloth
                logger.info("Unsloth optimizations available")
            except ImportError:
                pass
    else:
        if _GPU_AVAILABLE:
            logger.info("Using Apple Silicon MPS device")
        else:
            logger.warning("No GPU detected - using CPU (performance will be significantly slower)")

except Exception as e:
    logger.warning(f"Error during additional GPU initialization: {str(e)}")

# Conditionally import FastLanguageModel from unsloth only when CUDA is available
unsloth_available = False
if torch.cuda.is_available():
    try:
        from unsloth import FastLanguageModel
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        import bitsandbytes as bnb
        unsloth_available = True
        logger.info("Unsloth optimizations available")
    except (ImportError, NotImplementedError):
        logger.warning("Unsloth not available or CUDA not found. Using standard transformers.")
        unsloth_available = False

class ConversationMemory:
    """
    Manages conversation history and context for the AI assistant.

    This class provides methods for adding conversation exchanges,
    formatting conversation context for prompts, and saving/loading
    conversation memory.
    """

    def __init__(self, max_exchanges: int = 10, memory_file: Optional[str] = None):
        """
        Initialize conversation memory.

        Args:
            max_exchanges: Maximum number of exchanges to remember
            memory_file: File to save/load memory from
        """
        self.exchanges: List[Tuple[str, str]] = []
        self.max_exchanges = max_exchanges
        self.memory_file = memory_file
        self.user_preferences: Dict[str, Any] = {}

        # Load memory if file exists
        if memory_file and os.path.exists(memory_file):
            self.load_memory()

    def add_exchange(self, user_input: str, ai_response: str) -> None:
        """
        Add a conversation exchange to memory.

        Args:
            user_input: User's input text
            ai_response: AI's response text
        """
        self.exchanges.append((user_input, ai_response))

        # Trim to max exchanges
        if len(self.exchanges) > self.max_exchanges:
            self.exchanges = self.exchanges[-self.max_exchanges:]

        # Auto-save if memory file is set
        if self.memory_file:
            self.save_memory()

    def get_context(self, max_exchanges: Optional[int] = None, format_style: str = "default") -> str:
        """
        Format conversation history as context for prompts.

        Args:
            max_exchanges: Maximum number of exchanges to include in context
            format_style: Formatting style for different models ('default', 'deepseek', etc.)

        Returns:
            Formatted conversation context
        """
        if max_exchanges is None:
            max_exchanges = self.max_exchanges

        exchanges = self.exchanges[-max_exchanges:] if max_exchanges > 0 else self.exchanges

        if format_style == "deepseek":
            return "".join(
                f"USER: {user_input}\nASSISTANT: {ai_response}\n\n"
                for user_input, ai_response in exchanges
            )
        else:
            return "".join(
                f"User: {user_input}\nJarvis: {ai_response}\n\n"
                for user_input, ai_response in exchanges
            )

    def save_memory(self) -> None:
        """Save conversation memory to file."""
        if not self.memory_file:
            return

        memory_data = {
            "exchanges": self.exchanges,
            "user_preferences": self.user_preferences,
            "timestamp": datetime.datetime.now().isoformat()
        }

        try:
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            with open(self.memory_file, 'w') as f:
                json.dump(memory_data, f, indent=2)
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Error saving memory: {e}")

    def load_memory(self) -> None:
        """Load conversation memory from file."""
        if not self.memory_file or not os.path.exists(self.memory_file):
            return

        try:
            with open(self.memory_file, 'r') as f:
                memory_data = json.load(f)

            self.exchanges = memory_data.get("exchanges", [])
            self.user_preferences = memory_data.get("user_preferences", {})
            logger.info(f"Loaded {len(self.exchanges)} conversation exchanges from memory")
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Error loading memory: {e}")

    def update_preferences(self, preferences: Dict[str, Any]) -> None:
        """
        Update user preferences.

        Args:
            preferences: Dictionary of preference key-value pairs
        """
        self.user_preferences.update(preferences)
        if self.memory_file:
            self.save_memory()

    def clear(self) -> None:
        """Clear conversation memory."""
        self.exchanges = []
        if self.memory_file:
            self.save_memory()


class TextDataset(Dataset):
    """Dataset for training language models on text data."""

    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        """
        Initialize dataset.

        Args:
            texts: List of text samples
            tokenizer: Tokenizer for encoding text or a dictionary of pre-tokenized inputs
            max_length: Maximum sequence length
        """
        # Force tensors to stay on CPU for DataLoader compatibility
        with torch.device('cpu'):
            if callable(tokenizer):
                # If tokenizer is a callable (like HuggingFace tokenizer)
                self.encodings = tokenizer(
                    texts,
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                    return_tensors="pt"
                )
            else:
                # If tokenizer is already a dictionary of tokenized inputs
                self.encodings = tokenizer

            # Explicitly ensure tensors are on CPU
            for key, val in self.encodings.items():
                if isinstance(val, torch.Tensor) and val.device.type != 'cpu':
                    self.encodings[key] = val.cpu()

    def __getitem__(self, idx):
        """Get encoded item by index."""
        item = {key: val[idx].clone() for key, val in self.encodings.items()}

        # Add labels if not present (for causal language modeling)
        if "labels" not in item and "input_ids" in item:
            item["labels"] = item["input_ids"].clone()

        return item

    def __len__(self):
        """Get dataset length."""
        for val in self.encodings.values():
            if isinstance(val, torch.Tensor):
                return len(val)
        return 0


class JarvisAI:
    """
    Unified Jarvis AI class that handles training, inference,
    and interaction across multiple datasets. Optimized for A100 and RTX4000/5000 GPUs.
    """

    # Available datasets, focused on The Pile, OpenAssistant, and GPTeacher
    AVAILABLE_DATASETS = ["pile", "openassistant", "gpteacher"]

    # HuggingFace dataset prefixes for identification
    HUGGINGFACE_PREFIXES = ["google/", "agie-ai/", "teknium/", "euclaise/"]

    def __init__(
        self,
        models_dir: str = "models",
        use_best_models: bool = True,
        device: Optional[str] = None,
        memory_file: Optional[str] = None,
        load_in_4bit: bool = True,  # GPU optimization
        use_unsloth: bool = True,   # GPU optimization
        max_new_tokens: int = 1024,
        gradient_accumulation_steps: int = 4  # Added for RTX GPUs
    ):
        """
        Initialize Jarvis AI.

        Args:
            models_dir: Directory for storing models
            use_best_models: Whether to use best models or final models
            device: Device to use (cpu or cuda)
            memory_file: File to save/load conversation memory
            load_in_4bit: Whether to load models in 4-bit precision (GPU optimization)
            use_unsloth: Whether to use Unsloth optimizations (GPU optimization)
            max_new_tokens: Maximum number of tokens to generate
            gradient_accumulation_steps: Number of gradient accumulation steps (higher for RTX GPUs)
        """
        self.models_dir = Path(models_dir)
        self.use_best_models = use_best_models
        self.models = {}
        self.tokenizers = {}
        self.memory = ConversationMemory(memory_file=memory_file)
        self.load_in_4bit = load_in_4bit and unsloth_available  # Only use 4-bit if Unsloth is available
        self.use_unsloth = use_unsloth and unsloth_available    # Only use Unsloth if available
        self.max_new_tokens = max_new_tokens
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Setup environment-specific paths
        if os.path.exists("/storage"):
            # Use Paperspace persistent storage if available
            default_storage = Path("/storage/Jarvis_AI_Assistant")
            self.models_dir = default_storage / "models" if models_dir == "models" else Path(models_dir)
            logger.info(f"Using Paperspace persistent storage: {self.models_dir}")

        # Determine device
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Configure PyTorch to use the correct device
        torch.device(self.device)

        # Set the default generator for PyTorch operations
        if self.device == "cuda":
            # Force CUDA to initialize to avoid potential issues later
            _ = torch.tensor([1.0], device="cuda")
            # Set global generator
            torch.default_generator = torch.Generator(device="cuda").manual_seed(42)

        # Set up precision based on device capabilities and hardware
        self.use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
        if torch.cuda.is_available() and ("RTX5000" in torch.cuda.get_device_name(0) or "RTX 5000" in torch.cuda.get_device_name(0)):
            # RTX GPUs work better with FP16
            self.use_bf16 = False
            logger.info("Using FP16 precision for RTX GPU")

        # Set batch size based on GPU type for default values in train_models
        if torch.cuda.is_available() and ("RTX5000" in torch.cuda.get_device_name(0) or "RTX 5000" in torch.cuda.get_device_name(0)):
            self.default_batch_size = 4  # RTX5000 has less memory
            self.default_seq_length = 1024
        elif torch.cuda.is_available() and ("RTX4000" in torch.cuda.get_device_name(0) or "RTX 4000" in torch.cuda.get_device_name(0)):
            self.default_batch_size = 2  # RTX4000 has even less memory
            self.default_seq_length = 512
        else:
            self.default_batch_size = 2  # Default conservative values
            self.default_seq_length = 512

        logger.info(f"Using device: {self.device}")
        logger.info(f"Available built-in datasets: {', '.join(self.AVAILABLE_DATASETS)}")
        logger.info(f"HuggingFace datasets supported: prefixes {', '.join(self.HUGGINGFACE_PREFIXES)} or with '/' in name")
        logger.info(f"4-bit quantization: {'Enabled' if self.load_in_4bit else 'Disabled'}")
        logger.info(f"Unsloth optimizations: {'Enabled' if self.use_unsloth else 'Disabled'}")
        logger.info(f"BF16 precision: {'Enabled' if self.use_bf16 else 'Disabled'}")
        logger.info(f"Default batch size: {self.default_batch_size}")
        logger.info(f"Default sequence length: {self.default_seq_length}")
        logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")

        # Create models directory
        os.makedirs(self.models_dir, exist_ok=True)

    def is_huggingface_dataset(self, dataset_name: str) -> bool:
        """
        Check if a dataset name is a HuggingFace dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            True if it's a HuggingFace dataset, False otherwise
        """
        # Check if it has a '/' in it, which is characteristic of HuggingFace dataset names
        if '/' in dataset_name:
            return True

        # Check if it starts with any of the known HuggingFace prefixes
        for prefix in self.HUGGINGFACE_PREFIXES:
            if dataset_name.startswith(prefix):
                return True

        return False

    def get_device_generator(self, seed=42):
        """
        Get a generator appropriate for the current device.

        Args:
            seed: Random seed to use

        Returns:
            A torch Generator for the appropriate device
        """
        # CRITICAL: ALWAYS return a CPU generator regardless of context
        # This is the only way to ensure compatibility with PyTorch's data operations
        # IMPORTANT: DataLoader, random_split, and other dataset operations REQUIRE CPU generators

        # Check if we already created a generator with this seed
        if not hasattr(self, '_cpu_generators'):
            self._cpu_generators = {}

        # Reuse the same generator if we've created it before to avoid
        # creating multiple generators with the same seed
        if seed in self._cpu_generators:
            return self._cpu_generators[seed]

        # Create and cache the generator
        generator = torch.Generator().manual_seed(seed)

        # Verify it's on CPU
        if hasattr(generator, 'device') and generator.device.type != 'cpu':
            print(f"Warning: Non-CPU generator created! Forcing CPU generator instead.")
            generator = torch.Generator().manual_seed(seed)

        # Cache it
        self._cpu_generators[seed] = generator

        return generator

    def load_model(self, dataset_or_path: str, is_path: bool = False) -> Tuple[Any, Any]:
        """
        Load model and tokenizer for a specific dataset or from a path.

        Args:
            dataset_or_path: Name of the dataset or path to model
            is_path: Whether dataset_or_path is a path to a model

        Returns:
            Tuple of (model, tokenizer)
        """
        # If already loaded, return cached model
        if not is_path and dataset_or_path in self.models and dataset_or_path in self.tokenizers:
            return self.models[dataset_or_path], self.tokenizers[dataset_or_path]

        # Determine model path or name
        use_fallback = False
        original_path = None

        if is_path:
            model_path = dataset_or_path
            logger.info(f"Loading model from path: {model_path}")
        else:
            # Check if model exists in models directory
            model_type = "best" if self.use_best_models else "final"
            model_path = self.models_dir / f"{dataset_or_path}_{model_type}"
            original_path = str(model_path)

            # Check if the model path exists and has a config.json file
            if not model_path.exists() or not (Path(model_path) / "config.json").exists():
                use_fallback = True
                # Get fallback model
                if dataset_or_path == "pile":
                    model_path = "gpt2"
                elif dataset_or_path == "openassistant":
                    model_path = "facebook/opt-350m"
                elif dataset_or_path == "gpteacher":
                    model_path = "EleutherAI/pythia-410m"
                else:
                    # Default to deepseek-coder for all other cases
                    model_path = "deepseek-ai/deepseek-coder-6.7b-instruct"

                logger.info(f"Model for {dataset_or_path} not found at {original_path}, using fallback model: {model_path}")

        try:
            # Configure quantization for optimization
            if self.load_in_4bit and "deepseek" in str(model_path):
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16 if self.use_bf16 else torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
            else:
                quantization_config = None

            # Load tokenizer based on model path
            if use_fallback:
                # When using a fallback model, always use its direct name
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            else:
                # For custom models, try to load tokenizer from the path
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                except (OSError, EnvironmentError) as e:
                    logger.warning(f"Failed to load tokenizer from {model_path}: {e}")
                    # Fall back to a default tokenizer if custom one can't be loaded
                    logger.info(f"Falling back to gpt2 tokenizer")
                    model_path = "gpt2"
                    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                    use_fallback = True

            # Ensure pad token exists
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load model with optimizations based on hardware
            if self.use_unsloth and "deepseek" in str(model_path) and unsloth_available:
                # Use Unsloth optimizations for DeepSeek models
                logger.info(f"Loading {model_path} with Unsloth optimizations")
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_path,
                    max_seq_length=2048,
                    quantization_config=quantization_config
                    # Don't set device_map or trust_remote_code here as they're already set by FastLanguageModel internally
                )
            else:
                # Standard model loading
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto" if self.device == "cuda" else None,
                    quantization_config=quantization_config,
                    torch_dtype=torch.bfloat16 if self.use_bf16 else torch.float16,
                    trust_remote_code=True
                )

            # Cache model if it's a named dataset
            if not is_path:
                self.models[dataset_or_path] = model
                self.tokenizers[dataset_or_path] = tokenizer

            return model, tokenizer

        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            logger.info("Falling back to GPT2 as a last resort")

            # Last resort fallback - always use GPT2
            tokenizer = AutoTokenizer.from_pretrained("gpt2", trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                "gpt2",
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )

            # Cache the fallback model
            if not is_path:
                self.models[dataset_or_path] = model
                self.tokenizers[dataset_or_path] = tokenizer

            return model, tokenizer

    def determine_best_dataset(self, prompt: str) -> str:
        """
        Determine the best dataset to use for a given prompt.

        Args:
            prompt: User's input prompt

        Returns:
            Name of the best dataset to use
        """
        # Simple keyword matching for dataset selection
        prompt = prompt.lower()

        # Check for code-related queries
        code_keywords = ["code", "function", "program", "script", "class", "method", "algorithm"]
        if any(keyword in prompt for keyword in code_keywords):
            return "pile"  # The Pile has more code examples

        # Check for conversation and instruction following
        conversation_keywords = ["explain", "how to", "help me", "what is", "can you"]
        if any(keyword in prompt for keyword in conversation_keywords):
            return "openassistant"  # OpenAssistant is better for conversational tasks

        # Check for teaching-related queries
        teaching_keywords = ["teach", "learn", "understand", "concept", "example", "tutorial"]
        if any(keyword in prompt for keyword in teaching_keywords):
            return "gpteacher"  # GPTeacher is designed for educational content

        # Default to OpenAssistant for general queries
        return "openassistant"

    def generate_response(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_length: int = None,
        dataset: Optional[str] = None,
        from_path: Optional[str] = None
    ) -> str:
        """
        Generate a response to a prompt.

        Args:
            prompt: User's input prompt
            temperature: Sampling temperature (higher = more random)
            max_length: Maximum length of the generated response
            dataset: Dataset to use (will determine automatically if None)
            from_path: Path to a model to use instead of a dataset

        Returns:
            Generated response
        """
        # Set default max_length if not provided
        if max_length is None:
            max_length = self.max_new_tokens

        # Determine dataset to use
        if from_path:
            # Use model from specified path
            model, tokenizer = self.load_model(from_path, is_path=True)
            model_type = "custom"
        elif dataset:
            # Use specified dataset
            model, tokenizer = self.load_model(dataset)
            model_type = dataset
        else:
            # Auto-determine best dataset
            dataset = self.determine_best_dataset(prompt)
            model, tokenizer = self.load_model(dataset)
            model_type = dataset

        logger.info(f"Generating response using {model_type} model")

        # Format prompt with conversation history if it's a deepseek model
        if from_path and "deepseek" in str(from_path):
            context = self.memory.get_context(format_style="deepseek")
            full_prompt = f"{context}USER: {prompt}\nASSISTANT:"
        elif isinstance(model_type, str) and "deepseek" in model_type:
            context = self.memory.get_context(format_style="deepseek")
            full_prompt = f"{context}USER: {prompt}\nASSISTANT:"
        else:
            # Standard format for other models
            context = self.memory.get_context()
            full_prompt = f"{context}User: {prompt}\nJarvis:"

        # Tokenize prompt
        inputs = tokenizer(full_prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Generate text with A100 optimizations
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                top_k=40,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the model's response (after the prompt)
        response = generated_text[len(full_prompt):].strip()

        # Save to conversation memory
        self.memory.add_exchange(prompt, response)

        return response

    def load_dataset(
        self,
        dataset_name: str,
        max_samples: int = 1000,
        validation_split: float = 0.1,
        test_split: float = 0.1
    ) -> Tuple[TextDataset, TextDataset, TextDataset]:
        """
        Load and preprocess a dataset.

        Args:
            dataset_name: Name of the dataset to load
            max_samples: Maximum number of samples to load
            validation_split: Fraction of data to use for validation
            test_split: Fraction of data to use for testing

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # CRITICAL - Ensure GPU usage before dataset operations
        # Force CUDA usage for all operations if available
        if torch.cuda.is_available():
            # Create a dummy tensor on CUDA to ensure it's initialized
            _ = torch.zeros(1, device="cuda")

            # But ensure all generators are on CPU for compatibility
            cpu_generator = self.get_device_generator(seed=42)

        # Check if it's a HuggingFace dataset
        if self.is_huggingface_dataset(dataset_name):
            try:
                from datasets import load_dataset, Dataset
                from .unified_dataset_handler import UnifiedDatasetHandler

                # Load appropriate tokenizer - use a generic one for HF datasets
                _, tokenizer = self.load_model("pile")  # Use pile tokenizer as a generic one

                # Initialize unified dataset handler
                dataset_handler = UnifiedDatasetHandler(
                    dataset_name=dataset_name,
                    cache_dir=self.models_dir
                )

                # Add tokenizer and parameters as properties
                dataset_handler.tokenizer = tokenizer
                dataset_handler.max_length = self.default_seq_length
                dataset_handler.batch_size = self.default_batch_size

                logger.info(f"Loading HuggingFace dataset: {dataset_name}")

                # Load dataset
                dataset_dict = dataset_handler.load_dataset(
                    dataset_name=dataset_name,
                    split="train",
                    max_samples=max_samples,
                    use_cache=True
                )

                # Get train dataset
                train_dataset = dataset_dict["train"]

                # Create validation and test datasets
                if validation_split > 0 or test_split > 0:
                    # Split dataset with fixed seed for reproducibility
                    splits = train_dataset.train_test_split(
                        test_size=validation_split + test_split,
                        shuffle=True,
                        seed=42  # Fixed seed for reproducibility
                    )
                    train_dataset = splits["train"]

                    if validation_split > 0 and test_split > 0:
                        # Split the test portion into validation and test with fixed seed
                        # Calculate ratio of test to validation from the combined test+validation split
                        test_val_ratio = test_split / (validation_split + test_split)
                        val_test_splits = splits["test"].train_test_split(
                            test_size=test_val_ratio,
                            shuffle=True,
                            seed=43  # Use a different fixed seed for this split
                        )
                        val_dataset = val_test_splits["train"]
                        test_dataset = val_test_splits["test"]
                    else:
                        # Only validation split, no test split
                        val_dataset = splits["test"]
                        test_dataset = val_dataset  # Use validation as test too
                else:
                    # No splits requested
                    val_dataset = train_dataset
                    test_dataset = train_dataset

                logger.info(f"Loaded HuggingFace dataset: {dataset_name} with {len(train_dataset)} training samples")

                # Convert to TextDataset format expected by this module
                train_text_dataset = self._convert_to_text_dataset(train_dataset, tokenizer)
                val_text_dataset = self._convert_to_text_dataset(val_dataset, tokenizer)
                test_text_dataset = self._convert_to_text_dataset(test_dataset, tokenizer)

                return train_text_dataset, val_text_dataset, test_text_dataset

            except Exception as e:
                logger.error(f"Error loading HuggingFace dataset {dataset_name}: {e}")
                raise ValueError(f"Failed to load HuggingFace dataset {dataset_name}: {e}")

        # Check if dataset is supported
        if dataset_name not in self.AVAILABLE_DATASETS:
            raise ValueError(f"Dataset {dataset_name} not supported. Available built-in datasets: {self.AVAILABLE_DATASETS}")

        # Load appropriate tokenizer
        _, tokenizer = self.load_model(dataset_name)

        # Placeholder for actual dataset loading logic
        # In a real implementation, you would load from disk or a dataset API

        # For demonstration, create synthetic data
        logger.info(f"Loading {dataset_name} dataset (max {max_samples} samples)")

        if dataset_name == "pile":
            # Code-heavy examples
            texts = [
                f"def fibonacci(n):\n    if n <= 1:\n        return n\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)\n\nresult = fibonacci(10)"
                for _ in range(max_samples)
            ]
        elif dataset_name == "openassistant":
            # Conversation examples
            texts = [
                f"User: How does photosynthesis work?\nJarvis: Photosynthesis is the process used by plants to convert light energy into chemical energy. Plants capture light using chlorophyll and use it to convert carbon dioxide and water into glucose and oxygen."
                for _ in range(max_samples)
            ]
        elif dataset_name == "gpteacher":
            # Educational examples
            texts = [
                f"User: Explain Newton's laws of motion\nJarvis: Newton's First Law: An object at rest stays at rest, and an object in motion stays in motion unless acted upon by an external force. Newton's Second Law: Force equals mass times acceleration (F=ma). Newton's Third Law: For every action, there is an equal and opposite reaction."
                for _ in range(max_samples)
            ]

        # Create dataset
        full_dataset = TextDataset(texts, tokenizer)

        # Split into train, validation, and test
        val_size = int(len(full_dataset) * validation_split)
        test_size = int(len(full_dataset) * test_split)
        train_size = len(full_dataset) - val_size - test_size

        # CRITICAL FIX: Always use CPU generator for dataset operations
        # even when CUDA is available, and this causes device mismatch errors

        # Use a try-except block to handle any device mismatch errors
        try:
            # Create a fresh CPU generator for this operation
            cpu_generator = torch.Generator().manual_seed(42)

            # Split the dataset using CPU generator to avoid device mismatch errors
            train_dataset, val_dataset, test_dataset = random_split(
                full_dataset,
                [train_size, val_size, test_size],
                generator=cpu_generator
            )
        except RuntimeError as e:
            # Check if it's a device mismatch error
            if "Expected a 'cuda' device type for generator but found 'cpu'" in str(e):
                logger.warning("CUDA generator device mismatch detected! Applying advanced fix...")

                # Apply a direct fix by creating our own split with indices
                indices = list(range(len(full_dataset)))
                random.seed(42)  # Use random.shuffle instead of torch's randomness
                random.shuffle(indices)

                # Split indices manually
                train_idx = indices[:train_size]
                val_idx = indices[train_size:train_size+val_size]
                test_idx = indices[train_size+val_size:]

                # Create Subset datasets
                from torch.utils.data import Subset
                train_dataset = Subset(full_dataset, train_idx)
                val_dataset = Subset(full_dataset, val_idx)
                test_dataset = Subset(full_dataset, test_idx)

                logger.info("Successfully created dataset splits using manual indices")
            else:
                # Re-raise if it's not the specific error we're handling
                raise

        logger.info(f"Dataset split: {train_size} train, {val_size} validation, {test_size} test")

        return train_dataset, val_dataset, test_dataset

    def _convert_to_text_dataset(self, hf_dataset, tokenizer):
        """
        Convert a HuggingFace dataset to TextDataset format.

        Args:
            hf_dataset: HuggingFace dataset
            tokenizer: Tokenizer to use

        Returns:
            TextDataset object
        """
        # Extract text from HuggingFace dataset
        texts = []

        # Check for common text fields
        text_fields = ["text", "content", "instruction", "input", "prompt"]
        response_fields = ["response", "output", "completion", "answer"]

        # Find the text field
        text_field = None
        for field in text_fields:
            if field in hf_dataset.features:
                text_field = field
                break

        # Find response field if available
        response_field = None
        for field in response_fields:
            if field in hf_dataset.features:
                response_field = field
                break

        # Extract text
        if text_field and response_field:
            # Both input and response available - combine them
            for item in hf_dataset:
                text = f"{item[text_field]}\n{item[response_field]}"
                texts.append(text)
        elif text_field:
            # Only input available
            for item in hf_dataset:
                texts.append(item[text_field])
        else:
            # No standard fields found - try first column
            first_col = list(hf_dataset.features.keys())[0]
            for item in hf_dataset:
                texts.append(str(item[first_col]))

        # Create dataset
        return TextDataset(texts, tokenizer, max_length=self.default_seq_length)

    def train_models(
        self,
        datasets: List[str] = None,
        max_samples: int = None,
        epochs: int = 10,
        batch_size: int = None,
        learning_rate: float = 1e-4,
        validation_split: float = 0.1,
        test_split: float = 0.1,
        early_stopping: int = 3,
        visualization_dir: Optional[str] = None,
        use_lora: bool = True,  # GPU optimization
        lora_r: int = 64,       # GPU optimization
        lora_alpha: int = 16,   # GPU optimization
        gradient_accumulation_steps: int = None,  # Added for RTX GPUs
        sequence_length: int = None  # Added for control over sequence length
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Train models on datasets with GPU optimizations.

        Args:
            datasets: List of datasets to train on
            max_samples: Maximum number of samples per dataset
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            validation_split: Fraction of data to use for validation
            test_split: Fraction of data to use for testing
            early_stopping: Number of epochs to wait for improvement before stopping
            visualization_dir: Directory to save visualizations
            use_lora: Whether to use LoRA for efficient fine-tuning (GPU optimization)
            lora_r: LoRA rank parameter
            lora_alpha: LoRA alpha parameter
            gradient_accumulation_steps: Number of gradient accumulation steps
            sequence_length: Maximum sequence length for tokenization

        Returns:
            Dictionary of training metrics per dataset
        """
        # Set GPU consistency settings to avoid random generator issues
        if self.device == 'cuda':
            # Set consistency environment variables for PyTorch
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            os.environ['PYTHONHASHSEED'] = '42'

            # Set deterministic operations
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            # Set all seeds
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)

            logger.info("Set deterministic mode for CUDA operations to ensure consistent results")

        # Use all datasets if none specified
        if datasets is None:
            datasets = self.AVAILABLE_DATASETS

        # Use default values if not specified
        if max_samples is None:
            max_samples = 500 if "RTX5000" in torch.cuda.get_device_name(0) or "RTX 5000" in torch.cuda.get_device_name(0) else 200 if "RTX4000" in torch.cuda.get_device_name(0) or "RTX 4000" in torch.cuda.get_device_name(0) else 300 if torch.cuda.is_available() else 200

        if batch_size is None:
            batch_size = self.default_batch_size

        if gradient_accumulation_steps is None:
            gradient_accumulation_steps = self.gradient_accumulation_steps

        if sequence_length is None:
            sequence_length = self.default_seq_length

        # Create visualization directory if needed
        if visualization_dir:
            os.makedirs(visualization_dir, exist_ok=True)

        # Store training metrics
        all_metrics = {}

        # Check for HuggingFace datasets - handle separately
        huggingface_datasets = []
        regular_datasets = []

        for dataset in datasets:
            if self.is_huggingface_dataset(dataset):
                huggingface_datasets.append(dataset)
            elif dataset in self.AVAILABLE_DATASETS:
                regular_datasets.append(dataset)
            else:
                logger.warning(f"Dataset {dataset} not recognized as a built-in or HuggingFace dataset. Skipping.")

        # Process HuggingFace datasets first if any
        if huggingface_datasets:
            try:
                from .train_models import train_text_model

                # Group HuggingFace datasets with commas for the train_models.py script
                hf_dataset_str = ",".join(huggingface_datasets)
                logger.info(f"Training on HuggingFace datasets: {hf_dataset_str}")

                # Setup output directory
                hf_output_dir = os.path.join(str(self.models_dir), "huggingface_model")
                os.makedirs(hf_output_dir, exist_ok=True)

                # Use train_text_model function from train_models.py with fixed seeds
                # First ensure all random seeds are set for reproducibility
                if self.device == 'cuda':
                    torch.cuda.manual_seed(42)
                    torch.cuda.manual_seed_all(42)
                random.seed(42)
                np.random.seed(42)
                torch.manual_seed(42)
                os.environ['PYTHONHASHSEED'] = '42'

                model, tokenizer, training_args = train_text_model(
                    dataset=hf_dataset_str,
                    model_name_or_path="gpt2",  # Use a default model
                    batch_size=batch_size,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    weight_decay=0.01,
                    max_length=sequence_length,
                    output_dir=hf_output_dir,
                    eval_metrics_dir=visualization_dir or os.path.join(hf_output_dir, "metrics"),
                    dataset_subset=None,
                    max_samples=max_samples,
                    evaluation_strategy="steps",
                    save_strategy="steps",
                    logging_steps=50,
                    eval_steps=100,
                    visualize_metrics=visualization_dir is not None,
                    use_deepspeed=False,
                    use_8bit=False,
                    use_4bit=self.load_in_4bit,
                    use_qlora=use_lora,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    fp16=not self.use_bf16,
                    bf16=self.use_bf16,
                    temperature=0.7,
                    resume_from_checkpoint=False,
                    use_mps=False,
                    use_flash_attn=False,
                    use_unsloth=self.use_unsloth,
                    cache_dir=None
                )

                # Store metrics for HuggingFace datasets
                all_metrics["huggingface"] = {
                    "train_loss": [log["loss"] for log in training_args.logging_history if "loss" in log],
                    "eval_loss": [log["eval_loss"] for log in training_args.logging_history if "eval_loss" in log],
                    "epochs": [log.get("epoch", 0) for log in training_args.logging_history if "eval_loss" in log]
                }

                # Create visualizations if requested
                if visualization_dir:
                    self._visualize_training("huggingface", all_metrics["huggingface"], visualization_dir)

                logger.info(f"Successfully trained on HuggingFace datasets")

            except Exception as e:
                logger.error(f"Error training on HuggingFace datasets: {e}")
                logger.exception(e)

        # Train on each regular dataset
        for dataset in regular_datasets:
            logger.info(f"Training model for {dataset} dataset")

            # Load dataset
            train_dataset, val_dataset, _ = self.load_dataset(
                dataset,
                max_samples=max_samples,
                validation_split=validation_split,
                test_split=test_split
            )

            # Load model and tokenizer
            model, tokenizer = self.load_model(dataset)

            # Apply LoRA if using Unsloth
            if use_lora and self.use_unsloth and unsloth_available and "deepseek" in str(model):
                try:
                    logger.info(f"Applying LoRA for efficient fine-tuning (r={lora_r}, alpha={lora_alpha})")
                    # Setup LoRA config
                    lora_config = LoraConfig(
                        r=lora_r,
                        lora_alpha=lora_alpha,
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                        lora_dropout=0.05,
                        bias="none",
                        task_type="CAUSAL_LM"
                    )

                    # Apply LoRA using Unsloth
                    model = FastLanguageModel.get_peft_model(
                        model,
                        lora_config,
                        use_gradient_checkpointing=True
                    )
                    logger.info("LoRA applied successfully")
                except Exception as e:
                    logger.error(f"Failed to apply LoRA: {e}")

            # Configure output path
            output_dir = self.models_dir / f"{dataset}_final"
            best_model_dir = self.models_dir / f"{dataset}_best"
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(best_model_dir, exist_ok=True)

            # Set up training arguments with GPU optimizations
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                overwrite_output_dir=True,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                eval_steps=100,
                save_steps=100,
                warmup_steps=100,
                logging_dir=str(output_dir / "logs"),
                logging_steps=10,
                evaluation_strategy="steps",
                save_strategy="steps",
                save_total_limit=2,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                bf16=self.use_bf16,  # Use BF16 on A100
                fp16=not self.use_bf16,  # Use FP16 when not using BF16
                optim="adamw_torch_fused",  # Use fused optimizer on NVIDIA GPUs
                gradient_accumulation_steps=gradient_accumulation_steps,  # More steps for RTX GPUs
                seed=42,  # Set fixed seed for reproducibility
                dataloader_drop_last=False,  # Keep all samples even if last batch is smaller
                dataloader_num_workers=4,  # Optimize number of workers for RTX5000
                torch_compile=False  # Disable torch compile as it can cause issues with CUDA generators
            )

            # Set up early stopping
            early_stopping_callback = EarlyStoppingCallback(
                early_stopping_patience=early_stopping
            )

            # Force PyTorch to use the right seed for both CPU and CUDA operations
            # This is needed to avoid generator device mismatch errors
            if self.device == 'cuda':
                torch.manual_seed(42)
                torch.cuda.manual_seed(42)
                torch.cuda.manual_seed_all(42)
            else:
                torch.manual_seed(42)

            # CRITICAL FIX: Patch or override PyTorch's random_split/Sampler to use CPU generator
            # The core issue is that PyTorch's DataLoader creates samplers with CPU generators
            # even when CUDA is available, and this causes device mismatch errors

            # Override the default PyTorch sampler behavior to always use CPU generators
            def create_cpu_generator():
                """Create a CPU generator regardless of device being used"""
                return torch.Generator().manual_seed(42)

            # Monkey-patch torch.utils.data.random_split during training to avoid CUDA generator issues
            original_random_split = torch.utils.data.random_split

            def safe_random_split(dataset, lengths, generator=None):
                """Safe version of random_split that ensures CPU generator"""
                # Always use CPU generator to avoid device mismatch
                cpu_generator = create_cpu_generator()
                return original_random_split(dataset, lengths, generator=cpu_generator)

            # Apply the monkey patch
            torch.utils.data.random_split = safe_random_split

            # Train model with device-appropriate settings
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=tokenizer,
                callbacks=[early_stopping_callback],
                data_collator=default_data_collator  # Use default collator to avoid custom samplers
            )

            # Start training
            logger.info(f"Starting training for {dataset}")
            try:
                # Monitor memory usage before training
                if torch.cuda.is_available():
                    memory_before = torch.cuda.memory_allocated() / 1e9
                    logger.info(f"GPU memory allocated before training: {memory_before:.2f} GB")

                train_result = trainer.train()

                # Monitor memory usage after training
                if torch.cuda.is_available():
                    memory_after = torch.cuda.memory_allocated() / 1e9
                    memory_peak = torch.cuda.max_memory_allocated() / 1e9
                    logger.info(f"GPU memory allocated after training: {memory_after:.2f} GB")
                    logger.info(f"GPU memory peak during training: {memory_peak:.2f} GB")
            except Exception as e:
                logger.error(f"Training error: {e}")

                # Get CUDA OOM error details
                if "CUDA out of memory" in str(e):
                    if torch.cuda.is_available():
                        memory_info = torch.cuda.memory_summary(abbreviated=True)
                        logger.error(f"Memory summary during OOM error: {memory_info}")

                    # Provide suggestions based on GPU type
                    if "RTX4000" in torch.cuda.get_device_name(0) or "RTX 4000" in torch.cuda.get_device_name(0):
                        logger.error(
                            "RTX4000 GPU (8GB) encountered memory error. Try: \n"
                            "1. Reduce batch size to 1\n"
                            "2. Increase gradient_accumulation_steps to 8\n"
                            "3. Reduce sequence_length to 512\n"
                            "4. Use a smaller model (e.g., deepseek-coder-1.3b)"
                        )
                    elif "RTX5000" in torch.cuda.get_device_name(0) or "RTX 5000" in torch.cuda.get_device_name(0):
                        logger.error(
                            "RTX5000 GPU (16GB) encountered memory error. Try: \n"
                            "1. Reduce batch size to 2\n"
                            "2. Increase gradient_accumulation_steps to 8\n"
                            "3. Reduce sequence_length to 1024"
                        )

                # Skip to next dataset
                continue

            # Save final model
            trainer.save_model(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))

            # Save best model separately
            if os.path.exists(str(training_args.output_dir / "checkpoint-best")):
                best_model = AutoModelForCausalLM.from_pretrained(
                    str(training_args.output_dir / "checkpoint-best")
                )
                best_model.save_pretrained(str(best_model_dir))
                tokenizer.save_pretrained(str(best_model_dir))
                logger.info(f"Saved best model to {best_model_dir}")

            # Log and visualize metrics
            metrics = {
                "train_loss": trainer.state.log_history[-1]["train_loss"],
                "eval_loss": trainer.state.log_history[-1]["eval_loss"],
                "epoch": trainer.state.epoch
            }

            all_metrics[dataset] = {
                "train_loss": [log["train_loss"] for log in trainer.state.log_history if "train_loss" in log],
                "eval_loss": [log["eval_loss"] for log in trainer.state.log_history if "eval_loss" in log],
                "epochs": [log.get("epoch", 0) for log in trainer.state.log_history if "eval_loss" in log]
            }

            # Visualize training if requested
            if visualization_dir:
                self._visualize_training(dataset, all_metrics[dataset], visualization_dir)

            logger.info(f"Training completed for {dataset}: {metrics}")

        return all_metrics

    def _visualize_training(
        self,
        dataset: str,
        metrics: Dict[str, Any],
        output_dir: str
    ) -> None:
        """
        Visualize training metrics.

        Args:
            dataset: Name of the dataset
            metrics: Training metrics
            output_dir: Directory to save visualizations
        """
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(metrics["epochs"], metrics["train_loss"], label="Train Loss")
            plt.plot(metrics["epochs"], metrics["eval_loss"], label="Eval Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title(f"Training and Evaluation Loss for {dataset.capitalize()}")
            plt.legend()
            plt.grid(True)

            # Save plot
            output_path = os.path.join(output_dir, f"{dataset}_training_loss.png")
            plt.savefig(output_path)
            plt.close()

            logger.info(f"Saved training visualization to {output_path}")
        except Exception as e:
            logger.error(f"Failed to visualize training metrics: {e}")

    def run_interactive(self, load_path: Optional[str] = None) -> None:
        """
        Run an interactive session with the AI.

        Args:
            load_path: Path to a model to load for the session
        """
        print("=== Jarvis AI Interactive Session ===")
        print("Type 'exit' or 'quit' to end the session")

        while True:
            try:
                user_input = input("\nYou: ")

                if user_input.lower() in ["exit", "quit"]:
                    print("Exiting interactive session")
                    break

                print("\nJarvis is thinking...")

                if load_path:
                    response = self.generate_response(user_input, from_path=load_path)
                else:
                    response = self.generate_response(user_input)

                print(f"\nJarvis: {response}")

            except KeyboardInterrupt:
                print("\nExiting interactive session")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Continuing session...")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Jarvis AI Unified")

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "interactive", "generate"],
        default="interactive",
        help="Mode to run (train, interactive, generate)"
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/deepseek-coder-6.7b-instruct",
        help="The model name to use"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to a model to use"
    )

    # Training arguments
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Datasets to train on. Can include built-in datasets (pile, openassistant, gpteacher) and HuggingFace datasets like 'google/Synthetic-Persona-Chat'"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples per dataset (auto-determined by GPU type if None)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for training (auto-determined by GPU type if None)"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Number of gradient accumulation steps (auto-determined by GPU type if None)"
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="Maximum sequence length for tokenization (auto-determined by GPU type if None)"
    )

    # Generation arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt to generate from"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=200,
        help="Maximum length of generated text"
    )

    # GPU optimization arguments
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=True,
        help="Load model in 4-bit precision (GPU optimization)"
    )
    parser.add_argument(
        "--use-unsloth",
        action="store_true",
        default=True,
        help="Use Unsloth for optimized training (GPU optimization)"
    )

    # Memory arguments
    parser.add_argument(
        "--memory-file",
        type=str,
        default=None,
        help="Path to memory file"
    )
    parser.add_argument(
        "--history",
        type=str,
        default=None,
        help="Path to previous chat history JSON file to load"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save chat history after completion"
    )

    # Visualization arguments
    parser.add_argument(
        "--visualize-metrics",
        action="store_true",
        default=False,
        help="Visualize training metrics and save plots"
    )
    parser.add_argument(
        "--visualization-dir",
        type=str,
        default="visualizations",
        help="Directory to save visualizations"
    )

    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_arguments()

    # Initialize Jarvis AI
    jarvis = JarvisAI(
        memory_file=args.memory_file or args.history,
        load_in_4bit=args.load_in_4bit,
        use_unsloth=args.use_unsloth
    )

    # Run in appropriate mode
    if args.mode == "train":
        # Print available dataset types
        print("\n===== Available Dataset Types =====")
        print(f"Built-in datasets: {', '.join(jarvis.AVAILABLE_DATASETS)}")
        print(f"HuggingFace datasets: Any dataset with '/' in the name or starting with: {', '.join(jarvis.HUGGINGFACE_PREFIXES)}")

        # Set up visualization directory if needed
        visualization_dir = args.visualization_dir if args.visualize_metrics else None
        if visualization_dir:
            os.makedirs(visualization_dir, exist_ok=True)
            print(f"Visualizations will be saved to: {visualization_dir}")

        # Train models
        jarvis.train_models(
            datasets=args.datasets,
            max_samples=args.max_samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            sequence_length=args.sequence_length,
            visualization_dir=visualization_dir
        )

        print("\n===== Training Complete =====")
        print(f"Models saved to: {jarvis.models_dir}")

    elif args.mode == "interactive":
        # Run interactive session
        jarvis.run_interactive(load_path=args.model_path or args.model)
    elif args.mode == "generate":
        # Generate text from prompt
        if args.prompt:
            response = jarvis.generate_response(
                args.prompt,
                temperature=args.temperature,
                max_length=args.max_length,
                from_path=args.model_path or args.model
            )
            print(response)
        else:
            print("Error: Please provide a prompt for generation mode")

    # Save memory file if output specified
    if args.output and args.memory_file:
        # Copy memory file to output
        import shutil
        shutil.copy(args.memory_file, args.output)
        print(f"Chat history saved to {args.output}")

if __name__ == "__main__":
    main()