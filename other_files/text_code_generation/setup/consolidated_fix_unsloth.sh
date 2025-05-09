#!/bin/bash

echo "================================================================"
echo "Jarvis AI Assistant - Consolidated Unsloth Fix"
echo "================================================================"

# Directory paths
SRC_DIR="/notebooks/src/generative_ai_module"
CUSTOM_UNSLOTH_DIR="/notebooks/custom_unsloth"

# Function to create minimal unsloth implementation
create_minimal_unsloth() {
    echo "Creating Minimal Unsloth Implementation (No Dependencies)"
    
    # First, clean up any existing unsloth installations
    echo "Removing existing unsloth packages..."
    pip uninstall -y unsloth unsloth_zoo
    
    # Create directory for our custom implementation
    echo "Creating custom unsloth directory..."
    mkdir -p "$CUSTOM_UNSLOTH_DIR/unsloth/models"
    
    # Create the __init__.py file in the root directory
    cat > "$CUSTOM_UNSLOTH_DIR/unsloth/__init__.py" << 'EOF'
"""
Minimal Unsloth Implementation - No dependencies on unsloth_zoo
This is a stripped-down version of unsloth that provides basic functionality
without requiring unsloth_zoo or patching.
"""

import os
import sys
import warnings

__version__ = "minimal.1.0"

print(f"🦥 Using Minimal Unsloth {__version__} (No unsloth_zoo dependency)")

# Import the minimal components
from .models import FastLanguageModel, ModelAdapter

# Create stub for PatchDynamicLora
class PatchDynamicLora:
    """Compatibility stub for PatchDynamicLora"""
    def __init__(self, *args, **kwargs):
        print("⚠️ PatchDynamicLora running in compatibility mode (no-op).")
    
    def __call__(self, *args, **kwargs):
        # Just return the first argument (model) unchanged
        return args[0] if args else None

print("✅ Minimal Unsloth loaded successfully!")
EOF
    
    # Create models/__init__.py
    cat > "$CUSTOM_UNSLOTH_DIR/unsloth/models/__init__.py" << 'EOF'
"""
Minimal implementation of unsloth.models
"""
import os
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Adapter class for different model types
class ModelAdapter:
    """Simplified adapter for different model types"""
    @staticmethod
    def get_model_and_tokenizer(model_name, *args, **kwargs):
        """Get model and tokenizer for the given model name"""
        # Handle common parameters
        load_in_4bit = kwargs.pop('load_in_4bit', False)
        load_in_8bit = kwargs.pop('load_in_8bit', False)
        
        # Basic quantization config
        if load_in_4bit:
            print("Loading model in 4-bit quantization")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        elif load_in_8bit:
            print("Loading model in 8-bit quantization")
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            bnb_config = None
        
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        # Load model with quantization if specified
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            **kwargs
        )
        
        return model, tokenizer

# Main entry point with simplified interface
class FastLanguageModel:
    """Minimal implementation of FastLanguageModel"""
    
    @staticmethod
    def from_pretrained(model_name, **kwargs):
        """Load a model and tokenizer with simplified interface"""
        print(f"🦥 Loading model {model_name} with minimal unsloth")
        
        # Get adapter
        adapter = ModelAdapter()
        
        # Load model and tokenizer
        model, tokenizer = adapter.get_model_and_tokenizer(
            model_name, **kwargs
        )
        
        print(f"✅ Successfully loaded {model_name}")
        return model, tokenizer
    
    @staticmethod
    def get_peft_model(model, **kwargs):
        """Get a PEFT model with specified parameters"""
        from peft import LoraConfig, get_peft_model
        
        # Extract arguments
        r = kwargs.pop('r', 16)
        lora_alpha = kwargs.pop('lora_alpha', 32)
        lora_dropout = kwargs.pop('lora_dropout', 0.05)
        target_modules = kwargs.pop('target_modules', None)
        bias = kwargs.pop('bias', "none")
        
        # If target_modules not specified, make a reasonable guess
        if target_modules is None:
            # Look for likely module names
            param_names = [name for name, _ in model.named_parameters()]
            if any("q_proj" in name for name in param_names):
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
                print(f"Auto-detected target modules: {target_modules}")
            else:
                target_modules = ["query_key_value"]
                print(f"Using default target modules: {target_modules}")
        
        # Create LoRA config
        peft_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias=bias,
            task_type="CAUSAL_LM",
            **kwargs
        )
        
        # Get PEFT model
        print(f"🦥 Creating PEFT model with config: {peft_config}")
        peft_model = get_peft_model(model, peft_config)
        
        return peft_model
EOF
    
    # Create a helper script to use this version
    cat > "$CUSTOM_UNSLOTH_DIR/use_minimal_unsloth.py" << 'EOF'
"""
Example of using minimal unsloth
"""
import os
import sys

# Add this script's directory to path so we can import our minimal unsloth
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Now import minimal unsloth
import unsloth
from unsloth.models import FastLanguageModel

print(f"Minimal Unsloth version: {unsloth.__version__}")

# Example usage - uncomment to test with a small model
"""
model_name = "HuggingFaceH4/tiny-random-LlamaForCausalLM"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name,
    load_in_4bit=True,
)

# Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

print("Model successfully loaded and prepared with LoRA!")
"""

print("\nTo use this version, add this to your Python code:")
print("import sys")
print(f"sys.path.insert(0, '{script_dir}')")
print("import unsloth")
print("from unsloth.models import FastLanguageModel")
EOF
    
    # Create an activation script
    cat > "$CUSTOM_UNSLOTH_DIR/activate_minimal_unsloth.sh" << EOF
#!/bin/bash

# Add minimal unsloth to PYTHONPATH
export PYTHONPATH="$CUSTOM_UNSLOTH_DIR:\$PYTHONPATH"
echo "Minimal Unsloth activated at $CUSTOM_UNSLOTH_DIR"
echo "PYTHONPATH now includes minimal unsloth"
echo ""
echo "To test, run: python $CUSTOM_UNSLOTH_DIR/use_minimal_unsloth.py"
EOF
    
    chmod +x "$CUSTOM_UNSLOTH_DIR/activate_minimal_unsloth.sh"
}

# Function to apply fixed unsloth to Python files
apply_fixed_unsloth() {
    echo "Applying Fixed Minimal Unsloth to Python Files"
    
    # Make sure the src directory exists
    if [ ! -d "$SRC_DIR" ]; then
        echo "⚠️ Warning: Source directory not found at $SRC_DIR"
        echo "Please enter the path to your code directory:"
        read -p "> " SRC_DIR
        
        if [ ! -d "$SRC_DIR" ]; then
            echo "❌ Error: Directory not found at $SRC_DIR"
            return 1
        fi
    fi
    
    # Find files that use unsloth
    echo "Finding files that use unsloth..."
    UNSLOTH_FILES=$(grep -l -r "unsloth" --include="*.py" "$SRC_DIR" || echo "")
    
    if [ -z "$UNSLOTH_FILES" ]; then
        echo "⚠️ Warning: No files found that use unsloth. Trying to scan all Python files."
        find "$SRC_DIR" -name "*.py" > /tmp/python_files.txt
    else
        echo "$UNSLOTH_FILES" > /tmp/python_files.txt
        echo "Found $(wc -l < /tmp/python_files.txt) files that use unsloth"
    fi
    
    # Process each file individually
    while read -r file; do
        echo "Processing $file..."
        
        # Create backup if it doesn't exist
        if [ ! -f "${file}.bak" ]; then
            cp "$file" "${file}.bak"
            echo "Created backup: ${file}.bak"
        fi
        
        # Fix the file by adding the necessary imports at the top
        cat > /tmp/temp_header.py << EOF
import sys
import os
# Use custom minimal unsloth implementation
if "$CUSTOM_UNSLOTH_DIR" not in sys.path:
    sys.path.insert(0, "$CUSTOM_UNSLOTH_DIR")

EOF
        
        # Check if the file already has our custom path
        if ! grep -q "$CUSTOM_UNSLOTH_DIR" "$file"; then
            # Add it to the top of the file (after any possible shebang or docstring)
            if head -1 "$file" | grep -q "^#!"; then
                # File has shebang, insert after it
                sed -i.tmp '1r /tmp/temp_header.py' "$file"
            elif head -10 "$file" | grep -q '"""'; then
                # File has docstring, need to insert after it
                # This is more complex, use a Python script
                python -c "
import re
with open('$file', 'r') as f:
    content = f.read()
if content.startswith('\"\"\"'):
    match = re.search(r'^(\"\"\".*?\"\"\")(.*)$', content, re.DOTALL)
    if match:
        with open('$file', 'w') as f:
            docstring, rest = match.groups()
            with open('/tmp/temp_header.py', 'r') as header:
                header_content = header.read()
            f.write(docstring + '\n' + header_content + rest)
else:
    with open('$file', 'r') as f:
        content = f.read()
    with open('$file', 'w') as f:
        with open('/tmp/temp_header.py', 'r') as header:
            header_content = header.read()
        f.write(header_content + content)
"
            else
                # No shebang or docstring, insert at the top
                cat /tmp/temp_header.py "$file" > "${file}.new"
                mv "${file}.new" "$file"
            fi
        fi
        
        echo "✅ Modified $file to use minimal unsloth"
    done < /tmp/python_files.txt
    
    # Clean up
    rm -f /tmp/temp_header.py /tmp/python_files.txt
}

# Main execution flow
echo "Starting consolidated Unsloth fix..."

# Create minimal unsloth implementation
if [ ! -d "$CUSTOM_UNSLOTH_DIR" ]; then
    create_minimal_unsloth
else
    echo "Custom unsloth directory already exists at $CUSTOM_UNSLOTH_DIR"
    echo "Skipping creation step"
fi

# Apply fixed unsloth to Python files
apply_fixed_unsloth

# Add the directory to PYTHONPATH
echo "Adding custom unsloth to PYTHONPATH..."
export PYTHONPATH="$CUSTOM_UNSLOTH_DIR:$PYTHONPATH"
echo 'export PYTHONPATH="'$CUSTOM_UNSLOTH_DIR':$PYTHONPATH"' >> ~/.bashrc

# Verify unsloth minimal import
echo "Verifying minimal unsloth functionality..."
python -c "
try:
    import unsloth
    print(f'✅ Basic unsloth imports working in compatibility mode')
    print(f'Unsloth version: {unsloth.__version__}')
    
    from unsloth.models import FastLanguageModel
    print(f'✅ FastLanguageModel imported successfully')
    
    # Show available functionality
    print('\nAvailable unsloth functionality:')
    funcs = [attr for attr in dir(unsloth) if not attr.startswith('_')]
    for func in funcs:
        print(f'- {func}')
except Exception as e:
    print(f'❌ Error importing unsloth: {e}')
    import traceback
    traceback.print_exc()
"

echo "================================================================"
echo "Unsloth fix complete!"
echo ""
echo "To use this unsloth version:"
echo ""
echo "1. Add to your PYTHONPATH:"
echo "   export PYTHONPATH=\"$CUSTOM_UNSLOTH_DIR:\$PYTHONPATH\""
echo ""
echo "2. Or use the activation script:"
echo "   source $CUSTOM_UNSLOTH_DIR/activate_minimal_unsloth.sh"
echo ""
echo "3. Then in your Python code:"
echo "   import unsloth"
echo "   from unsloth.models import FastLanguageModel"
echo "================================================================"
