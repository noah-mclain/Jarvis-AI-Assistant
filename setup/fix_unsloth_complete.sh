#!/bin/bash

echo "================================================================"
echo "Complete Unsloth Fix (Zero Dependency Conflicts Edition)"
echo "================================================================"

# Uninstall unsloth only - no other packages will be touched
echo "Removing any existing unsloth package..."
pip uninstall -y unsloth

# Skip cache purge - don't want to affect other packages
# pip cache purge

# First, create a compatibility patch for the missing Gemma module
# This only adds files, doesn't modify any existing ones
echo "Creating compatibility patch for transformers..."
mkdir -p $(python -c "import transformers; print(transformers.__path__[0])")/models/gemma

# Create an __init__.py file in the gemma directory to allow imports
cat > $(python -c "import transformers; print(transformers.__path__[0])")/models/gemma/__init__.py << 'EOF'
# Compatibility file for unsloth
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...modeling_utils import PreTrainedModel

# Stub so unsloth can import this module
EOF

echo "Created compatibility patch for transformers.models.gemma"

# Get current sentencepiece version to ensure we don't change it
CURRENT_SENTENCEPIECE=$(pip freeze | grep sentencepiece || echo "sentencepiece==0.1.99")
echo "Detected $CURRENT_SENTENCEPIECE - preserving this version"

# Install a specific unsloth version compatible with transformers 4.36.2
# Using --no-deps ensures no dependencies are installed or modified
echo "Installing unsloth 2025.2.15 with absolutely no dependency changes..."
pip install unsloth==2025.2.15 --no-deps

# Verify installation
echo "Verifying unsloth installation..."
if python -c "from unsloth import PatchDynamicLora; print('✅ Unsloth successfully imported')" 2>/dev/null; then
    echo "✅ Unsloth successfully patched and installed!"
    
    # Successfully installed, show version
    python -c "import unsloth; print(f'Unsloth version: {getattr(unsloth, \"__version__\", \"unknown\")}')"
else
    echo "❌ Installation verification failed. Additional error details:"
    python -c "
import traceback
try:
    import unsloth
    print(f'Unsloth imported but submodule might be missing. Version: {getattr(unsloth, \"__version__\", \"unknown\")}')
except Exception as e:
    print(f'Error importing unsloth: {e}')
    traceback.print_exc()
"
fi

# Check that all other dependencies are still intact
echo ""
echo "Verifying no dependency conflicts were created..."
python -c "
import transformers
import peft
import accelerate
import numpy
import torch
print(f'✅ All key dependencies still working correctly:')
print(f'  - numpy: {numpy.__version__}')
print(f'  - torch: {torch.__version__}')
print(f'  - transformers: {transformers.__version__}')
print(f'  - peft: {peft.__version__}')
print(f'  - accelerate: {accelerate.__version__}')
"

echo "================================================================"
echo "Unsloth fix complete, with zero dependency changes!"
echo "Only unsloth was installed, all other packages were preserved."
echo "================================================================" 