#!/bin/bash

echo "================================================================"
echo "Final Unsloth Fix - Bypassing Patching Issues"
echo "================================================================"

# First remove both packages
echo "Removing existing unsloth packages..."
pip uninstall -y unsloth unsloth_zoo

# Use a minimal approach - install only what we need
echo "Installing minimal unsloth version..."
pip install unsloth==2025.2.15 --no-deps

# Now we'll patch the unsloth initialization to bypass the zoo requirement
UNSLOTH_INIT=$(python -c "import unsloth; print(unsloth.__file__)")
echo "Patching unsloth initialization file at $UNSLOTH_INIT"

# Create a backup
cp "$UNSLOTH_INIT" "${UNSLOTH_INIT}.backup"

# Modify the initialization file to bypass the unsloth_zoo checks and patch
cat > "$UNSLOTH_INIT" << 'EOF'
# Patched unsloth file - bypasses dynamic patching requirements

print("🦥 Unsloth: Running in minimal compatibility mode (patched for transformers 4.36.2)")

# Standard Python imports
import os
import sys
import platform
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any

# Don't try to import unsloth_zoo
# try:
#     import unsloth_zoo
# except ImportError:
#     unsloth_zoo = None

# Avoid trying to do the dynamic patching that causes errors
try:
    from .models import PatchHuggingFaceModels
except ImportError:
    print("⚠️ Using limited unsloth functionality.")

# Define version attributes
__version__ = "2025.2.15"
_zoo_version = "none"

# Expose minimal API
class PatchDynamicLora:
    """Compatibility stub for PatchDynamicLora"""
    def __init__(self, *args, **kwargs):
        print("⚠️ PatchDynamicLora running in compatibility mode.")
    
    def __call__(self, *args, **kwargs):
        return args[0] if args else None

# Define minimal imports
from .models import FastLanguageModel
# Import what we can safely
try:
    from .models import FastLanguageModel, patch_unsloth_qlora, patch_unsloth_bnb
except ImportError:
    print("⚠️ Some unsloth models could not be imported.")

# Success message
print("✅ Unsloth loaded in minimal compatibility mode!")
EOF

echo "Patched unsloth initialization file"

# Create stub module for unsloth_zoo to prevent import errors
echo "Creating stub for unsloth_zoo..."
UNSLOTH_PATH=$(dirname "$UNSLOTH_INIT")
SITE_PACKAGES=$(dirname "$UNSLOTH_PATH")
mkdir -p "$SITE_PACKAGES/unsloth_zoo"

# Create a minimal __init__.py for unsloth_zoo
cat > "$SITE_PACKAGES/unsloth_zoo/__init__.py" << 'EOF'
# Stub module for unsloth_zoo
print("🦥 Unsloth zoo: Using minimal compatibility stub")

__version__ = "stub"

# Define empty functions that might be imported
def patch_attention(*args, **kwargs):
    print("⚠️ Using unsloth_zoo stub: patch_attention")
    return args[0] if args else None

def patch_for_peft(*args, **kwargs):
    print("⚠️ Using unsloth_zoo stub: patch_for_peft")
    return args[0] if args else None
EOF

# Create stub patching_utils.py
cat > "$SITE_PACKAGES/unsloth_zoo/patching_utils.py" << 'EOF'
# Stub module for unsloth_zoo.patching_utils

# Define functions that unsloth might try to import
def patch_model_with_peft_config(*args, **kwargs):
    return args[0] if args else None

def undo_peft_config_patch(*args, **kwargs):
    return args[0] if args else None

def replace_linear_with_dynamic_qlora(*args, **kwargs):
    return args[0] if args else None

def create_dynamic_module(*args, **kwargs):
    return None
EOF

echo "Created compatibility stubs"

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
echo "Unsloth minimal mode setup complete!"
echo ""
echo "NOTE: This is a minimal compatibility mode with limited functionality."
echo "You should be able to use basic FastLanguageModel features, but"
echo "advanced features requiring unsloth_zoo patching may not work."
echo "================================================================" 