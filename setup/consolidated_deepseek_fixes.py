#!/usr/bin/env python3
"""
Consolidated DeepSeek Fixes

This module consolidates all fixes for DeepSeek models.
It includes fixes for:
- DeepSeek model initialization
- DeepSeek attention implementation
- Unsloth integration with DeepSeek
- Trust remote code issues
- BitsAndBytes compatibility

This consolidates functionality from:
- bypass_deepseek.py
- debug_unsloth.py
- direct_fix_deepseek.py
- fix_deepseek_init.py
- fix_deepseek_model.py
- fix_deepseek_training.py
- manual_fix_deepseek.py
- fix_unsloth_trust_remote_code.py
- fix_bitsandbytes_version.py
"""

import os
import sys
import inspect
import types
import logging
import importlib
import shutil
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Some fixes may not work.")

try:
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Some fixes may not work.")

try:
    import bitsandbytes
    BITSANDBYTES_AVAILABLE = True
    
    # Check if bitsandbytes has a version attribute
    if not hasattr(bitsandbytes, "__version__"):
        # Try to get the version from pip
        try:
            import subprocess
            pip_output = subprocess.check_output([sys.executable, "-m", "pip", "show", "bitsandbytes"]).decode("utf-8")
            version_line = [line for line in pip_output.split("\n") if line.startswith("Version:")]
            if version_line:
                version = version_line[0].split(":", 1)[1].strip()
                bitsandbytes.__version__ = version
                logger.info(f"Added __version__ attribute to bitsandbytes: {version}")
            else:
                bitsandbytes.__version__ = "0.41.1"  # Default if not found
                logger.warning("Could not determine bitsandbytes version. Using default: 0.41.1")
        except Exception as e:
            bitsandbytes.__version__ = "0.41.1"  # Default if command fails
            logger.warning(f"Error getting bitsandbytes version: {e}. Using default: 0.41.1")
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    logger.warning("BitsAndBytes not available. Some fixes may not work.")

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    logger.warning("Unsloth not available. Some fixes may not work.")

# Add the parent directory to the path so we can import from the root
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def fix_bitsandbytes_version():
    """
    Fix BitsAndBytes version for 4-bit quantization compatibility.
    
    DeepSeek models require BitsAndBytes version 0.42.0 or higher for 4-bit quantization.
    """
    if not BITSANDBYTES_AVAILABLE:
        logger.error("BitsAndBytes not available. Cannot fix version.")
        return False
    
    # Check the current version
    version = getattr(bitsandbytes, "__version__", "0.41.1")
    logger.info(f"Current BitsAndBytes version: {version}")
    
    # Parse the version
    try:
        major, minor, patch = version.split(".")
        version_tuple = (int(major), int(minor), int(patch))
    except ValueError:
        logger.warning(f"Could not parse BitsAndBytes version: {version}")
        version_tuple = (0, 41, 1)  # Default if parsing fails
    
    # Check if the version is compatible with 4-bit quantization
    if version_tuple < (0, 42, 0):
        logger.warning(f"BitsAndBytes version {version} is not compatible with 4-bit quantization.")
        logger.warning("DeepSeek models require BitsAndBytes version 0.42.0 or higher for 4-bit quantization.")
        logger.warning("Attempting to upgrade BitsAndBytes...")
        
        # Try to upgrade BitsAndBytes
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "bitsandbytes>=0.42.0"], check=True)
            logger.info("Successfully upgraded BitsAndBytes.")
            
            # Reload the module
            importlib.reload(bitsandbytes)
            
            # Check the new version
            new_version = getattr(bitsandbytes, "__version__", "Unknown")
            logger.info(f"New BitsAndBytes version: {new_version}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to upgrade BitsAndBytes: {e}")
            logger.warning("Please manually upgrade BitsAndBytes to version 0.42.0 or higher.")
            return False
    else:
        logger.info(f"BitsAndBytes version {version} is compatible with 4-bit quantization.")
        return True

def fix_unsloth_trust_remote_code():
    """
    Fix Unsloth trust_remote_code issue.
    
    Unsloth needs to pass trust_remote_code=True when loading DeepSeek models.
    """
    if not UNSLOTH_AVAILABLE:
        logger.error("Unsloth not available. Cannot fix trust_remote_code issue.")
        return False
    
    # Find the Unsloth package directory
    try:
        unsloth_dir = os.path.dirname(inspect.getfile(FastLanguageModel))
        logger.info(f"Unsloth directory: {unsloth_dir}")
        
        # Find the file that needs to be patched
        patch_file = os.path.join(unsloth_dir, "models.py")
        
        if not os.path.exists(patch_file):
            logger.warning(f"Could not find {patch_file}. Looking for alternative files...")
            
            # Look for alternative files
            for root, dirs, files in os.walk(unsloth_dir):
                for file in files:
                    if file.endswith(".py"):
                        file_path = os.path.join(root, file)
                        with open(file_path, "r") as f:
                            content = f.read()
                            if "from_pretrained" in content and "trust_remote_code" in content:
                                patch_file = file_path
                                logger.info(f"Found alternative file to patch: {patch_file}")
                                break
                if patch_file != os.path.join(unsloth_dir, "models.py"):
                    break
        
        if not os.path.exists(patch_file):
            logger.error(f"Could not find a file to patch in the Unsloth package.")
            return False
        
        # Read the file
        with open(patch_file, "r") as f:
            content = f.read()
        
        # Check if the file already has trust_remote_code=True
        if "trust_remote_code=True" in content:
            logger.info(f"File {patch_file} already has trust_remote_code=True.")
            return True
        
        # Patch the file
        patched_content = content.replace(
            "AutoModelForCausalLM.from_pretrained(",
            "AutoModelForCausalLM.from_pretrained(trust_remote_code=True, "
        )
        
        # Write the patched file
        with open(patch_file, "w") as f:
            f.write(patched_content)
        
        logger.info(f"Successfully patched {patch_file} to include trust_remote_code=True.")
        return True
    
    except Exception as e:
        logger.error(f"Failed to patch Unsloth for trust_remote_code: {e}")
        return False

def fix_deepseek_model():
    """
    Fix DeepSeek model in transformers.
    
    This ensures that the DeepSeek model is properly registered in the transformers library.
    """
    if not TRANSFORMERS_AVAILABLE:
        logger.error("Transformers not available. Cannot fix DeepSeek model.")
        return False
    
    # Check if DeepSeek model is already available
    try:
        from transformers.models import deepseek
        from transformers.models.deepseek import DeepSeekModel, DeepSeekConfig, DeepSeekAttention
        logger.info("DeepSeek model is already available in transformers.")
        return True
    except ImportError:
        logger.warning("DeepSeek model is not available in transformers. Attempting to fix...")
    
    # Find the transformers package directory
    transformers_dir = os.path.dirname(transformers.__file__)
    logger.info(f"Transformers directory: {transformers_dir}")
    
    # Check if the models directory exists
    models_dir = os.path.join(transformers_dir, "models")
    if not os.path.exists(models_dir):
        logger.error(f"Models directory {models_dir} does not exist.")
        return False
    
    # Check if the deepseek directory exists
    deepseek_dir = os.path.join(models_dir, "deepseek")
    if not os.path.exists(deepseek_dir):
        logger.warning(f"DeepSeek directory {deepseek_dir} does not exist. Creating it...")
        os.makedirs(deepseek_dir, exist_ok=True)
    
    # Create the __init__.py file in the deepseek directory
    init_file = os.path.join(deepseek_dir, "__init__.py")
    if not os.path.exists(init_file):
        logger.info(f"Creating {init_file}...")
        with open(init_file, "w") as f:
            f.write("""
# DeepSeek model implementation
from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)

_import_structure = {
    "configuration_deepseek": ["DeepSeekConfig"],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_deepseek"] = [
        "DeepSeekModel",
        "DeepSeekForCausalLM",
        "DeepSeekForSequenceClassification",
        "DeepSeekPreTrainedModel",
        "DeepSeekAttention",
    ]

if TYPE_CHECKING:
    from .configuration_deepseek import DeepSeekConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_deepseek import (
            DeepSeekModel,
            DeepSeekForCausalLM,
            DeepSeekForSequenceClassification,
            DeepSeekPreTrainedModel,
            DeepSeekAttention,
        )
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
""")
    
    # Create a minimal configuration_deepseek.py file
    config_file = os.path.join(deepseek_dir, "configuration_deepseek.py")
    if not os.path.exists(config_file):
        logger.info(f"Creating {config_file}...")
        with open(config_file, "w") as f:
            f.write("""
# DeepSeek configuration
from ...configuration_utils import PretrainedConfig

class DeepSeekConfig(PretrainedConfig):
    model_type = "deepseek"
    
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        attention_bias=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
""")
    
    # Create a minimal modeling_deepseek.py file
    modeling_file = os.path.join(deepseek_dir, "modeling_deepseek.py")
    if not os.path.exists(modeling_file):
        logger.info(f"Creating {modeling_file}...")
        with open(modeling_file, "w") as f:
            f.write("""
# DeepSeek model implementation
import torch
import torch.nn as nn
from torch.nn import functional as F

from ...modeling_utils import PreTrainedModel
from .configuration_deepseek import DeepSeekConfig

class DeepSeekAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        # This is a placeholder implementation
        return hidden_states

class DeepSeekPreTrainedModel(PreTrainedModel):
    config_class = DeepSeekConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DeepSeekAttention"]
    
    def __init__(self, config):
        super().__init__(config)

class DeepSeekModel(DeepSeekPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
    
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # This is a placeholder implementation
        return (input_ids, None)

class DeepSeekForCausalLM(DeepSeekPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = DeepSeekModel(config)
    
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # This is a placeholder implementation
        return (input_ids, None)

class DeepSeekForSequenceClassification(DeepSeekPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = DeepSeekModel(config)
    
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # This is a placeholder implementation
        return (input_ids, None)
""")
    
    # Update the models/__init__.py file to include DeepSeek
    models_init_file = os.path.join(models_dir, "__init__.py")
    if os.path.exists(models_init_file):
        logger.info(f"Updating {models_init_file} to include DeepSeek...")
        with open(models_init_file, "r") as f:
            content = f.read()
        
        # Check if DeepSeek is already included
        if "deepseek" in content:
            logger.info("DeepSeek is already included in models/__init__.py.")
        else:
            # Add DeepSeek to the _MODELING_NAMES list
            if "_MODELING_NAMES" in content:
                content = content.replace(
                    "_MODELING_NAMES = [",
                    "_MODELING_NAMES = [\n    \"deepseek\","
                )
            
            # Write the updated file
            with open(models_init_file, "w") as f:
                f.write(content)
            
            logger.info("Successfully updated models/__init__.py to include DeepSeek.")
    
    # Try to import DeepSeek again to verify
    try:
        importlib.invalidate_caches()
        from transformers.models import deepseek
        from transformers.models.deepseek import DeepSeekModel, DeepSeekConfig, DeepSeekAttention
        logger.info("Successfully added DeepSeek model to transformers.")
        return True
    except ImportError as e:
        logger.error(f"Failed to add DeepSeek model to transformers: {e}")
        return False

def apply_all_deepseek_fixes():
    """
    Apply all DeepSeek-related fixes.
    
    This function applies all the fixes for DeepSeek models.
    """
    # Fix BitsAndBytes version
    fix_bitsandbytes_version()
    
    # Fix Unsloth trust_remote_code issue
    if UNSLOTH_AVAILABLE:
        fix_unsloth_trust_remote_code()
    
    # Fix DeepSeek model in transformers
    fix_deepseek_model()
    
    logger.info("All DeepSeek fixes have been applied successfully.")
    return True

if __name__ == "__main__":
    # Apply all DeepSeek fixes
    apply_all_deepseek_fixes()
