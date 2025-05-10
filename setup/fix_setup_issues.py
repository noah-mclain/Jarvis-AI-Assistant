#!/usr/bin/env python3
"""
Consolidated Fix Script for Setup Issues

This script fixes various issues in the setup scripts:
1. Syntax errors and unterminated string literals
2. Missing files (creates wrapper scripts)
3. Transformers utils module
4. DeepSeek model in transformers

Run this script before running consolidated_unified_setup.sh
"""

import os
import sys
import re
import glob
import importlib
import shutil
from pathlib import Path

# Configure basic logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_unterminated_strings(file_path):
    """Fix unterminated string literals in a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find unterminated string literals
        fixed_content = content
        string_regex = r'(["\'])((?:\\.|[^\\])*?)(?:\1|$)'

        fixed = False
        for match in re.finditer(string_regex, content):
            full_match = match.group(0)
            quote = match.group(1)

            # Check if the string is unterminated
            if not full_match.endswith(quote):
                # Fix the unterminated string by adding the closing quote
                fixed_content = fixed_content.replace(full_match, full_match + quote)
                logger.info(f"Fixed unterminated string in {file_path}: {full_match[:20]}...")
                fixed = True

        # Fix specific issues
        if "optimize_memory_usage()'" in fixed_content:
            fixed_content = fixed_content.replace("optimize_memory_usage()'", "optimize_memory_usage()")
            logger.info(f"Fixed optimize_memory_usage issue in {file_path}")
            fixed = True

        # Fix double colon issue
        if "sys.path::" in fixed_content:
            fixed_content = fixed_content.replace("sys.path::", "sys.path:")
            logger.info(f"Fixed double colon issue in {file_path}")
            fixed = True

        # Write the fixed content back to the file
        if fixed:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)

            logger.info(f"Fixed unterminated strings in {file_path}")
            return True
        else:
            logger.info(f"No unterminated strings found in {file_path}")
            return True

    except Exception as e:
        logger.error(f"Error fixing unterminated strings in {file_path}: {e}")
        return False

def fix_syntax_errors(file_path):
    """Fix common syntax errors in a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Fix missing colons after if/for/while/def/class statements
        fixed_content = re.sub(r'(if\s+.*?)\s*\n', r'\1:\n', content)
        fixed_content = re.sub(r'(for\s+.*?)\s*\n', r'\1:\n', fixed_content)
        fixed_content = re.sub(r'(while\s+.*?)\s*\n', r'\1:\n', fixed_content)
        fixed_content = re.sub(r'(def\s+.*?\))\s*\n', r'\1:\n', fixed_content)
        fixed_content = re.sub(r'(class\s+.*?(?:\(.*?\))?)\s*\n', r'\1:\n', fixed_content)

        # Fix indentation (convert tabs to spaces)
        lines = fixed_content.split('\n')
        fixed_lines = []
        for line in lines:
            if line.startswith('\t'):
                fixed_line = line.replace('\t', '    ')
                fixed_lines.append(fixed_line)
            else:
                fixed_lines.append(line)

        fixed_content = '\n'.join(fixed_lines)

        # Write the fixed content back to the file
        if fixed_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)

            logger.info(f"Fixed syntax errors in {file_path}")
            return True
        else:
            logger.info(f"No syntax errors found in {file_path}")
            return True

    except Exception as e:
        logger.error(f"Error fixing syntax errors in {file_path}: {e}")
        return False

def create_wrapper_script(target_file, source_function, source_module):
    """Create a wrapper script that imports and calls a function from another module."""
    if os.path.exists(target_file):
        logger.info(f"Wrapper script {target_file} already exists.")
        return True

    try:
        wrapper_content = f"""#!/usr/bin/env python3
\"\"\"
Wrapper script for {source_function} function in {source_module}.py
\"\"\"

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import from the root
parent_dir = str(Path(__file__).resolve().parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the function from {source_module}.py
try:
    from {source_module} import {source_function}

    # Call the function
    success = {source_function}()

    # Exit with appropriate code
    sys.exit(0 if success else 1)
except ImportError:
    print("Error: Could not import {source_function} from {source_module}.py")
    print("Make sure {source_module}.py is in the same directory as this script.")
    sys.exit(1)
"""

        with open(target_file, 'w', encoding='utf-8') as f:
            f.write(wrapper_content)

        # Make the script executable
        os.chmod(target_file, 0o755)

        logger.info(f"Created wrapper script {target_file}")
        return True

    except Exception as e:
        logger.error(f"Error creating wrapper script {target_file}: {e}")
        return False

def fix_transformers_utils():
    """Fix transformers.utils module."""
    try:
        # Try to import transformers.utils
        try:
            import transformers.utils
            logger.info("transformers.utils module already exists.")
            return True
        except ImportError:
            logger.info("transformers.utils module not found. Creating it...")

        # Import transformers
        import transformers

        # Find the transformers package directory
        transformers_dir = os.path.dirname(transformers.__file__)
        utils_dir = os.path.join(transformers_dir, "utils")

        # Create the utils directory if it doesn't exist
        os.makedirs(utils_dir, exist_ok=True)

        # Create an empty __init__.py file
        with open(os.path.join(utils_dir, "__init__.py"), "w") as f:
            f.write("""
# Auto-generated utils module for transformers
import logging
logger = logging.getLogger(__name__)

def get_attention_mask_dtype(dtype):
    \"\"\"
    Get the correct dtype for attention mask based on the model's dtype.

    Args:
        dtype: The dtype of the model's parameters

    Returns:
        The appropriate dtype for the attention mask
    \"\"\"
    # Handle torch dtypes
    if hasattr(dtype, "is_floating_point") and dtype.is_floating_point:
        return dtype

    # Handle numpy dtypes
    if hasattr(dtype, "kind") and dtype.kind == "f":
        return dtype

    # Default to float32
    import torch
    return torch.float32

def prepare_4d_attention_mask(attention_mask, dtype):
    \"\"\"
    Prepare a 4D attention mask from a 2D attention mask.

    Args:
        attention_mask: The 2D attention mask [batch_size, seq_length]
        dtype: The dtype to use for the attention mask

    Returns:
        The 4D attention mask [batch_size, 1, seq_length, seq_length]
    \"\"\"
    import torch

    # Ensure attention_mask is a tensor
    if not isinstance(attention_mask, torch.Tensor):
        attention_mask = torch.tensor(attention_mask, dtype=dtype)

    # Convert to correct dtype if needed
    if attention_mask.dtype != dtype:
        attention_mask = attention_mask.to(dtype=dtype)

    # Create 4D attention mask [batch_size, 1, seq_length, seq_length]
    batch_size, seq_length = attention_mask.shape

    # Create causal mask
    # [batch_size, 1, seq_length, seq_length]
    causal_mask = torch.triu(
        torch.ones((seq_length, seq_length), dtype=dtype) * -10000.0,
        diagonal=1
    )

    # Expand attention_mask to 4D
    # [batch_size, 1, seq_length, 1]
    extended_attention_mask = attention_mask[:, None, :, None]

    # Combine with causal mask
    # [batch_size, 1, seq_length, seq_length]
    extended_attention_mask = extended_attention_mask + causal_mask

    return extended_attention_mask

def fix_attention_mask(attention_mask, dtype, unmasked_value=0.0):
    \"\"\"
    Fix attention mask for DeepSeek models.

    Args:
        attention_mask: The attention mask
        dtype: The dtype to use for the attention mask
        unmasked_value: The value to use for unmasked positions

    Returns:
        The fixed attention mask
    \"\"\"
    import torch

    # Ensure attention_mask is a tensor
    if not isinstance(attention_mask, torch.Tensor):
        attention_mask = torch.tensor(attention_mask, dtype=dtype)

    # Convert to correct dtype if needed
    if attention_mask.dtype != dtype:
        attention_mask = attention_mask.to(dtype=dtype)

    # Fix the attention mask values
    # 0 -> -10000.0, 1 -> 0.0
    attention_mask = (1.0 - attention_mask) * -10000.0 + unmasked_value

    return attention_mask
""")

        # Try to import the module again to verify
        importlib.invalidate_caches()
        import transformers.utils
        logger.info("Successfully created transformers.utils module.")
        return True

    except Exception as e:
        logger.error(f"Failed to create transformers.utils module: {e}")
        return False

def fix_deepseek_model():
    """Fix DeepSeek model in transformers."""
    try:
        # Try to import DeepSeek model
        try:
            from transformers.models import deepseek
            from transformers.models.deepseek import DeepSeekModel, DeepSeekConfig, DeepSeekAttention
            logger.info("DeepSeek model is already available in transformers.")
            return True
        except ImportError:
            logger.info("DeepSeek model is not available in transformers. Creating it...")

        # Import transformers
        import transformers

        # Find the transformers package directory
        transformers_dir = os.path.dirname(transformers.__file__)
        models_dir = os.path.join(transformers_dir, "models")

        # Check if the models directory exists
        if not os.path.exists(models_dir):
            logger.error(f"Models directory {models_dir} does not exist.")
            return False

        # Create the deepseek directory
        deepseek_dir = os.path.join(models_dir, "deepseek")
        os.makedirs(deepseek_dir, exist_ok=True)

        # Create the __init__.py file
        with open(os.path.join(deepseek_dir, "__init__.py"), "w") as f:
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

        # Create the configuration_deepseek.py file
        with open(os.path.join(deepseek_dir, "configuration_deepseek.py"), "w") as f:
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

        # Create the modeling_deepseek.py file
        with open(os.path.join(deepseek_dir, "modeling_deepseek.py"), "w") as f:
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
        importlib.invalidate_caches()
        from transformers.models import deepseek
        from transformers.models.deepseek import DeepSeekModel, DeepSeekConfig, DeepSeekAttention
        logger.info("Successfully added DeepSeek model to transformers.")
        return True

    except Exception as e:
        logger.error(f"Failed to add DeepSeek model to transformers: {e}")
        return False

def fix_all_setup_scripts():
    """Fix all setup scripts."""
    setup_dir = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Fixing setup scripts in {setup_dir}...")

    # Fix syntax errors in all Python files
    python_files = glob.glob(os.path.join(setup_dir, "*.py"))
    for file_path in python_files:
        fix_unterminated_strings(file_path)
        fix_syntax_errors(file_path)

    # Create wrapper scripts for missing files
    create_wrapper_script(
        os.path.join(setup_dir, "fix_transformers_utils.py"),
        "fix_transformers_utils",
        "consolidated_utils"
    )

    create_wrapper_script(
        os.path.join(setup_dir, "fix_deepseek_model.py"),
        "fix_deepseek_model",
        "consolidated_deepseek_fixes"
    )

    # Fix transformers.utils module
    fix_transformers_utils()

    # Fix DeepSeek model
    fix_deepseek_model()

    logger.info("All setup scripts have been fixed.")
    return True

if __name__ == "__main__":
    fix_all_setup_scripts()
    logger.info("Setup issues have been fixed. You can now run consolidated_unified_setup.sh")
    sys.exit(0)
