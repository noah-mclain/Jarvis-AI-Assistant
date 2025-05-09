import zipfile
import os
import sys
import json
import logging
from pathlib import Path
import datetime
import shutil
from typing import Optional, List, Dict, Any, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_project_root():
    """Returns the absolute path to the project root directory."""
    # Adjust this path based on your project structure
    current_file = Path(__file__).resolve()
    # Go up two levels: from utils.py to generative_ai_module to src
    return str(current_file.parent.parent.parent)

def is_paperspace_environment():
    """Checks if code is running in Paperspace Gradient environment."""
    return os.path.exists('/notebooks') and (
        os.environ.get('PAPERSPACE') == 'true' or
        os.path.exists('/etc/paperspace') or
        os.path.exists('notebooks/Jarvis_AI_Assistant')
    )

def ensure_directory_exists(folder_type, relative_path=""):
    """
    Ensure that a directory exists, syncing from Google Drive first if needed.
    This handles cases where local directories have been deleted.

    Args:
        folder_type (str): One of "metrics", "models", "datasets", "checkpoints",
                           "preprocessed_data", "logs"
        relative_path (str, optional): Relative path within the folder

    Returns:
        str: Full path to the directory
    """
    # Get the appropriate path
    directory = get_storage_path(folder_type, relative_path)

    # If the directory doesn't exist and we're in Paperspace, try to sync from Google Drive
    if not os.path.exists(directory) and is_paperspace_environment():
        try:
            # Create parent directory if it doesn't exist
            os.makedirs(os.path.dirname(directory), exist_ok=True)

            # Try to sync from Google Drive to get the directory contents
            logger.info(f"Directory {directory} doesn't exist. Attempting to sync from Google Drive...")
            sync_from_gdrive(folder_type)
        except Exception as e:
            logger.warning(f"Error syncing from Google Drive: {str(e)}")

    # Create the directory if it still doesn't exist
    os.makedirs(directory, exist_ok=True)

    return directory

def get_storage_path(folder_type, relative_path=""):
    """
    Get the appropriate storage path based on environment.
    On Paperspace, this will return the local path but also ensures Google Drive sync is configured.

    Args:
        folder_type (str): One of "metrics", "models", "datasets", "checkpoints",
                           "preprocessed_data", "logs"
        relative_path (str, optional): Relative path within the folder

    Returns:
        str: Full path to the storage location
    """
    # Import here to avoid circular imports
    if is_paperspace_environment():
        try:
            from .google_drive_storage import GoogleDriveSync
            path = GoogleDriveSync.get_local_path(folder_type, relative_path)

            # Ensure the directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)

            return path
        except ImportError:
            # Fallback if GoogleDriveSync is not available
            path = os.path.join('notebooks/Jarvis_AI_Assistant', folder_type, relative_path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            return path
    else:
        # For local development or other environments
        root_dir = get_project_root()
        path = os.path.join(root_dir, folder_type, relative_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

def sync_to_gdrive(folder_type=None):
    """
    Sync data to Google Drive if running in Paperspace environment.

    Args:
        folder_type (str, optional): One of "metrics", "models", "datasets", "checkpoints", "preprocessed_data"
                                     or None to sync all folders
    """
    if is_paperspace_environment():
        try:
            from .google_drive_storage import GoogleDriveSync
            GoogleDriveSync.sync_to_gdrive(folder_type)
        except ImportError:
            logger.warning("GoogleDriveSync module not available. No sync performed.")
    else:
        logger.info("Not running in Paperspace environment, skipping Google Drive sync.")

def sync_from_gdrive(folder_type=None):
    """
    Sync data from Google Drive if running in Paperspace environment.

    Args:
        folder_type (str, optional): One of "metrics", "models", "datasets", "checkpoints", "preprocessed_data"
                                     or None to sync all folders
    """
    if is_paperspace_environment():
        try:
            from .google_drive_storage import GoogleDriveSync
            GoogleDriveSync.sync_from_gdrive(folder_type)
        except ImportError:
            logger.warning("GoogleDriveSync module not available. No sync performed.")
    else:
        logger.info("Not running in Paperspace environment, skipping Google Drive sync.")

def is_zipfile(file_path: str) -> bool:
    """
    Check if a file is a valid ZIP file.

    Args:
        file_path: Path to the file to check

    Returns:
        bool: True if the file is a valid ZIP file, False otherwise
    """
    return zipfile.is_zipfile(file_path) if os.path.exists(file_path) else False

def process_zip(zip_path: str, extract_to: str) -> bool:
    """
    Process a ZIP file by extracting its contents.

    Args:
        zip_path: Path to the ZIP file
        extract_to: Directory to extract the files to

    Returns:
        bool: True if extraction was successful, False otherwise
    """
    try:
        if not is_zipfile(zip_path):
            logger.error(f"File {zip_path} is not a valid ZIP file")
            return False

        # Create extraction directory if it doesn't exist
        os.makedirs(extract_to, exist_ok=True)

        # Extract the ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

        logger.info(f"Successfully extracted {zip_path} to {extract_to}")
        return True
    except Exception as e:
        logger.error(f"Error extracting ZIP file {zip_path}: {e}")
        return False

def setup_logging(log_filename=None):
    """
    Set up logging configuration to log to both console and a file.
    If running in Paperspace, this will store logs in the logs directory
    that syncs to Google Drive.

    Args:
        log_filename (str, optional): Name of the log file. If not provided,
                                     a timestamp-based name will be used.

    Returns:
        str: Path to the log file
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = log_filename or f"jarvis_log_{timestamp}.log"

    # Get the logs directory path and ensure it exists
    log_dir = ensure_directory_exists("logs")
    log_path = os.path.join(log_dir, log_filename)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_path}")

    return log_path

def sync_logs():
    """
    Sync log files to Google Drive.
    """
    if is_paperspace_environment():
        try:
            from .google_drive_storage import GoogleDriveSync
            GoogleDriveSync.sync_to_gdrive("logs")
            logger = logging.getLogger(__name__)
            logger.info("Synced logs to Google Drive")
        except ImportError:
            logger = logging.getLogger(__name__)
            logger.warning("GoogleDriveSync module not available. No sync performed.")
    else:
        logger = logging.getLogger(__name__)
        logger.info("Not running in Paperspace environment, skipping Google Drive sync for logs.")

def save_log_file(content, filename=None):
    """
    Save content to a log file and sync to Google Drive.

    Args:
        content (str): Content to save
        filename (str, optional): Name of the log file. If not provided,
                                 a timestamp-based name will be used.

    Returns:
        str: Path to the log file
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = filename or f"custom_log_{timestamp}.log"

    # Get the logs directory path and ensure it exists
    log_dir = ensure_directory_exists("logs")
    log_path = os.path.join(log_dir, filename)

    # Save content to log file
    with open(log_path, 'w') as f:
        f.write(content)

    # Sync logs to Google Drive
    sync_logs()

    return log_path

# Add spaCy utilities with Paperspace compatibility
def is_spacy_available():
    """Check if spaCy is available in the current environment"""
    try:
        # Check if minimal tokenizer is available first
        try:
            from .minimal_spacy_tokenizer import tokenize as minimal_tokenize
            return True, "minimal-tokenizer"
        except ImportError:
            pass

        # Try importing regular spaCy
        import spacy
        return True, spacy.__version__
    except ImportError:
        return False, None
    except Exception as e:
        # Handle ParametricAttention_v2 error or other issues
        if "ParametricAttention_v2" in str(e):
            # Try to use minimal tokenizer instead
            try:
                from .minimal_spacy_tokenizer import tokenize as minimal_tokenize
                return True, "minimal-tokenizer"
            except ImportError:
                pass
        return False, str(e)

def is_spacy_model_loaded(model_name="en_core_web_sm"):
    """Check if a specific spaCy model is available and can be loaded"""
    # First check if minimal tokenizer is available
    try:
        from .minimal_spacy_tokenizer import tokenizer
        if tokenizer.is_available:
            return True, f"Minimal tokenizer loaded successfully (Paperspace-safe)"
    except ImportError:
        pass

    # Try regular spaCy model
    try:
        import spacy
        # Only try this on non-Paperspace environments to avoid segfaults
        if not is_paperspace_environment():
            nlp = spacy.load(model_name)
            # Test the model with a simple sentence
            doc = nlp("This is a test sentence.")
            return True, f"Model {model_name} loaded successfully"
        else:
            # On Paperspace, we don't want to load the full pipeline
            try:
                # Just try a very minimal test to see if the model can be found
                nlp = spacy.load(model_name, disable=["ner", "parser", "attribute_ruler", "lemmatizer"])
                # Only test tokenization which is safe
                tokens = [t.text for t in nlp.tokenizer("Test sentence")]
                return True, f"Model {model_name} tokenizer available (Paperspace-safe mode)"
            except Exception as e:
                return False, f"Error loading model in Paperspace-safe mode: {str(e)}"
    except ImportError:
        return False, "spaCy not installed"
    except OSError:
        return False, f"Model {model_name} not found"
    except Exception as e:
        if "ParametricAttention_v2" in str(e):
            return False, "Paperspace compatibility issue: ParametricAttention_v2 error"
        return False, f"Error loading model: {str(e)}"

def initialize_spacy(fallback_to_basic=True, log_errors=True, paperspace_safe=None):
    """
    Initialize spaCy with the en_core_web_sm model if available.

    Args:
        fallback_to_basic: If True, will not raise errors but return None if spaCy is unavailable
        log_errors: If True, will log errors and warnings
        paperspace_safe: Override for Paperspace detection - if None, will auto-detect

    Returns:
        nlp: The spaCy NLP object if available, None otherwise
    """
    import logging

    # Auto-detect Paperspace if not specified
    if paperspace_safe is None:
        paperspace_safe = is_paperspace_environment()

    # Try using minimal tokenizer first if in Paperspace
    if paperspace_safe:
        try:
            from .minimal_spacy_tokenizer import tokenizer
            if tokenizer.is_available:
                if log_errors:
                    logging.info("Using minimal spaCy tokenizer (Paperspace-safe)")

                # Return a simplified object with just tokenizer functionality
                class MinimalNLP:
                    def __init__(self, tokenizer):
                        self.tokenizer = tokenizer

                    def __call__(self, text):
                        tokens = self.tokenizer.tokenize(text)
                        # Return a dummy doc object with just the tokens
                        class DummyDoc:
                            def __init__(self, tokens):
                                self.tokens = tokens
                                self.ents = []
                            def __iter__(self):
                                for t in self.tokens:
                                    yield type('DummyToken', (), {'text': t, 'pos_': "UNKNOWN", 'dep_': "UNKNOWN", 'head': type('DummyHead', (), {'text': ""})})
                            def __getitem__(self, i):
                                if isinstance(i, slice):
                                    return [type('DummyToken', (), {'text': t}) for t in self.tokens[i]]
                                return type('DummyToken', (), {'text': self.tokens[i]})
                        return DummyDoc(tokens)

                return MinimalNLP(tokenizer)
        except ImportError:
            if log_errors:
                logging.warning("Minimal tokenizer not available, trying regular spaCy")

    # For non-Paperspace or if minimal tokenizer failed, try regular spaCy
    try:
        import spacy
        try:
            # In Paperspace, load with minimal components to avoid segfaults
            if paperspace_safe:
                if log_errors:
                    logging.info("Loading spaCy in Paperspace-safe mode")
                nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "attribute_ruler", "lemmatizer"])
            else:
                nlp = spacy.load("en_core_web_sm")

            if log_errors:
                logging.info("spaCy initialized with en_core_web_sm model")
            return nlp
        except OSError as e:
            if log_errors:
                logging.warning(f"spaCy model not found: {str(e)}")
                logging.warning("To install the model, run: python -m spacy download en_core_web_sm")
            if not fallback_to_basic:
                raise e
            return None
        except Exception as e:
            if "ParametricAttention_v2" in str(e):
                if log_errors:
                    logging.error(f"Paperspace compatibility issue with spaCy: {str(e)}")
                try:
                    # Final attempt - create blank model which should work
                    if log_errors:
                        logging.info("Trying to create blank model as last resort")
                    nlp = spacy.blank("en")
                    return nlp
                except Exception:
                    pass
            if log_errors:
                logging.error(f"Error initializing spaCy: {str(e)}")
            if not fallback_to_basic:
                raise e
            return None
    except ImportError:
        if log_errors:
            logging.warning("spaCy not installed")
            logging.warning("For better text processing, install spaCy with: pip install spacy")
        if not fallback_to_basic:
            raise ImportError("spaCy not installed")
        return None

def process_text_with_spacy_or_fallback(text, nlp=None):
    """
    Process text with spaCy if available, or use a simple fallback method.

    Args:
        text: The text to process
        nlp: Optional spaCy NLP object. If None, will try to initialize

    Returns:
        dict: Processed text information (entities, tokens, etc.)
    """
    # Check if we're in Paperspace
    paperspace_mode = is_paperspace_environment()

    # Check for minimal tokenizer first
    try:
        from .minimal_spacy_tokenizer import tokenize as minimal_tokenize
        if paperspace_mode:
            tokens = minimal_tokenize(text)
            return {
                "entities": [],  # Empty since we can't detect entities in minimal mode
                "tokens": tokens,
                "pos_tags": [],  # Empty in minimal mode
                "nouns": [t for t in tokens if t[0].isupper() and len(t) > 3],  # Simple heuristic
                "verbs": [],  # Empty in minimal mode
                "adjectives": [],  # Empty in minimal mode
                "dependency_tree": [],  # Empty in minimal mode
                "sentences": [s.strip() for s in text.split('.') if s.strip()],  # Simple period splitting
                "minimal_mode": True
            }
    except ImportError:
        pass  # Continue with standard approach if minimal_tokenize not available

    # If nlp is not provided, try to initialize spaCy
    if nlp is None:
        nlp = initialize_spacy(fallback_to_basic=True, log_errors=False, paperspace_safe=paperspace_mode)

    # If spaCy is available, use it for processing
    if nlp is not None:
        try:
            doc = nlp(text)

            # In Paperspace/minimal mode, only return tokens to avoid segfaults
            if paperspace_mode or hasattr(nlp, 'tokenizer') and not hasattr(doc, 'ents'):
                tokens = [token.text for token in doc]
                return {
                    "entities": [],  # Empty since we can't detect entities in minimal mode
                    "tokens": tokens,
                    "pos_tags": [],  # Empty in minimal mode
                    "nouns": [t for t in tokens if t[0].isupper() and len(t) > 3],  # Simple heuristic
                    "verbs": [],  # Empty in minimal mode
                    "adjectives": [],  # Empty in minimal mode
                    "dependency_tree": [],  # Empty in minimal mode
                    "sentences": [s.strip() for s in text.split('.') if s.strip()],  # Simple period splitting
                    "minimal_mode": True
                }

            # Full mode for non-Paperspace environments
            return {
                "entities": [(ent.text, ent.label_) for ent in doc.ents],
                "tokens": [token.text for token in doc],
                "pos_tags": [(token.text, token.pos_) for token in doc],
                "nouns": [token.text for token in doc if token.pos_ == "NOUN"],
                "verbs": [token.text for token in doc if token.pos_ == "VERB"],
                "adjectives": [token.text for token in doc if token.pos_ == "ADJ"],
                "dependency_tree": [(token.text, token.dep_, token.head.text) for token in doc],
                "sentences": [sent.text for sent in doc.sents],
                "minimal_mode": False
            }
        except Exception:
            # Fall back to basic processing if spaCy fails
            pass

    # Basic fallback processing
    import re
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Basic POS guesses - very rudimentary!
    nouns = [word for word in words if word[0].isupper() and len(word) > 3]
    potential_verbs = [word for word in words if word.lower() in {
        'is', 'are', 'was', 'were', 'have', 'has', 'had', 'do', 'does', 'did',
        'can', 'could', 'will', 'would', 'shall', 'should', 'may', 'might',
        'make', 'create', 'generate', 'write', 'build', 'design', 'develop'
    }]

    return {
        "entities": [],  # Empty since we can't detect entities
        "tokens": words,
        "pos_tags": [],  # Empty since we can't determine POS
        "nouns": nouns,
        "verbs": potential_verbs,
        "adjectives": [],  # Empty since we can't determine adjectives
        "dependency_tree": [],  # Empty since we can't determine dependencies
        "sentences": sentences,
        "minimal_mode": True  # Using basic mode
    }

def setup_gpu_for_training(force_gpu=True):
    """
    Configure GPU usage for model training/inference, with special handling for RTX5000 GPU on Paperspace.

    Args:
        force_gpu (bool): If True, will raise an error if GPU is not available when specifically requested

    Returns:
        torch.device: The device to use for model training/inference
        dict: Additional configuration parameters for specific GPU optimizations
    """
    import torch
    import subprocess
    import re
    import logging
    import os

    logger = logging.getLogger("jarvis_unified")

    # Default optimization parameters
    config = {
        "use_fp16": False,
        "use_bf16": False,
        "use_4bit": True,  # Default to 4-bit for RTX5000
        "gradient_checkpointing": True,
        "gradient_accumulation_steps": 8,  # Default for RTX5000 with 16GB VRAM
        "attention_implementation": "sdpa",  # Default for LLMs on newer GPUs
    }

    # Always force GPU usage
    force_gpu = True

    # Ensure CUDA devices are visible
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Set memory allocation config for better efficiency
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    # Set CUDA benchmark for faster training with fixed input sizes
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.benchmark = True

    # Try to use CUDA if available
    if not torch.cuda.is_available():
        if force_gpu:
            # Try to initialize CUDA more aggressively
            try:
                # Force a small CUDA tensor allocation to trigger initialization
                dummy_tensor = torch.zeros(1, device="cuda")
                logger.info(f"CUDA initialized successfully with dummy tensor: {dummy_tensor.device}")
                # If we reach here, CUDA is actually available despite torch.cuda.is_available() reporting False
                return torch.device("cuda"), config
            except Exception as e:
                logger.error(f"Attempted to force CUDA initialization but failed: {e}")

            # Check if this is Paperspace and we're missing CUDA
            if is_paperspace_environment():
                logger.error("GPU requested but CUDA not available on Paperspace. This may indicate a configuration issue.")
                # Try to get system GPU info through subprocess to diagnose
                try:
                    gpu_info = subprocess.check_output("nvidia-smi", shell=True).decode("utf-8")
                    logger.error(f"GPU info from nvidia-smi: {gpu_info}")
                except Exception as cmd_error:
                    logger.error(f"Couldn't get GPU info via nvidia-smi. NVIDIA drivers may not be installed correctly: {cmd_error}")

                # Since user specifically requested GPU enforcement, this is an error condition
                raise RuntimeError("Failed to find CUDA GPU despite being requested. Please check your system configuration.")

            # Check for Apple Silicon as fallback
            try:
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    logger.info("CUDA not available, but Apple Silicon GPU (MPS) is available. Using MPS.")
                    return torch.device("mps"), config
            except Exception as mps_error:
                logger.error(f"Error checking for MPS availability: {mps_error}")

            logger.error("GPU was requested but no CUDA or MPS device is available.")
            raise RuntimeError("GPU was specifically requested but no GPU is available on this system. Check your configuration.")
        else:
            logger.warning("No GPU available, falling back to CPU despite preference for GPU.")
            return torch.device("cpu"), config

    # If we reach here, CUDA is available. Now check for RTX5000 specifically
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)

    # Check for RTX5000 in Paperspace environment
    is_rtx5000 = "RTX5000" in gpu_name or "RTX 5000" in gpu_name

    # Get GPU memory
    try:
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU Memory: {gpu_memory_gb:.2f} GB")
    except Exception as memory_error:
        gpu_memory_gb = None
        logger.warning(f"Could not determine GPU memory size: {memory_error}")

    # If this is Paperspace + RTX5000, apply specific optimizations
    if is_paperspace_environment() and is_rtx5000:
        # Log that we've detected the specific setup
        logger.info(f"Using GPU: {gpu_name}")

        # Get CUDA version
        cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else "Unknown"
        logger.info(f"CUDA Version: {cuda_version}")

        # RTX5000 has 16GB VRAM, optimize accordingly
        logger.info("RTX 5000 GPU detected - applying optimized settings")

        # Check for BF16 support (not available on RTX5000 but check anyway for future compatibility)
        bf16_support = torch.cuda.is_bf16_supported() if hasattr(torch.cuda, 'is_bf16_supported') else False
        logger.info(f"BF16 support: {bf16_support}")

        config.update({
            "use_fp16": True,         # Use FP16 precision
            "use_bf16": bf16_support, # Use BF16 if supported
            "use_4bit": True,         # Use 4-bit quantization for maximum memory efficiency
            "attention_implementation": "flash_attention_2" if cuda_version >= "11.8" else "sdpa",
            "gradient_accumulation_steps": 8,
            "gradient_checkpointing": True,
            "low_cpu_mem_usage": True
        })

        # Set environment variables to optimize GPU memory usage
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Better performance but less debugging info

        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Ensure GPU is visible

        # Log availability of Unsloth optimizations
        try:
            import unsloth
            logger.info("Unsloth optimizations available")
        except ImportError:
            logger.info("Unsloth not available - consider installing for faster training")
    else:
        # General optimizations for other NVIDIA GPUs
        logger.info(f"Using GPU: {gpu_name}")

        # Force default tensor type to CUDA
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Force PyTorch to use CUDA for all operations where possible
    torch.set_default_device('cuda')

    return device, config

def force_cuda_device():
    """
    Force all PyTorch operations to use CUDA device if available,
    with special handling for RTX5000 on Paperspace.
    This is a simpler version of setup_gpu_for_training when
    you just need to ensure GPU usage without detailed configuration.

    Returns:
        torch.device: The CUDA device if available, otherwise CPU
    """
    import torch
    import os
    import logging

    logger = logging.getLogger("jarvis_unified")

    # Ensure CUDA devices are visible
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Set memory allocation config for better efficiency
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    # Set CUDA benchmark for faster training with fixed input sizes
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.benchmark = True

    # Check for CUDA availability
    if torch.cuda.is_available():
        try:
            # Modern approach (PyTorch 2.0+)
            if hasattr(torch, 'set_default_device'):
                torch.set_default_device('cuda')
                logger.info("Using modern device setting approach (PyTorch 2.0+)")
            else:
                # Legacy approach - will show deprecation warnings
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
                logger.info("Using legacy tensor type setting approach")

            # For model inference
            device = torch.device('cuda')

            # Get device properties for optimizations
            device_props = torch.cuda.get_device_properties(0)
            total_memory_gb = device_props.total_memory / (1024**3)
            gpu_name = torch.cuda.get_device_name(0)

            # Add to config
            config = {
                "gpu_memory": f"{total_memory_gb:.2f} GB",
                "gpu_name": gpu_name,
                "cuda_version": torch.version.cuda
            }

            return device, config
        except Exception as e:
            logger.error(f"Error setting up CUDA device: {e}")
            return torch.device("cpu"), {}
    else:
        logger.warning("No GPU available. Falling back to CPU.")
        return torch.device("cpu"), {}
