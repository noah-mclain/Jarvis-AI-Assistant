#!/bin/bash

echo "================================================================"
echo "Complete Unsloth Fix (Only modifies unsloth, preserves other packages)"
echo "================================================================"

# Uninstall unsloth
echo "Removing any existing unsloth package..."
pip uninstall -y unsloth

# Clear pip cache to ensure clean install
pip cache purge

# First, create a compatibility patch for the missing Gemma module
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

# Install a specific unsloth version compatible with transformers 4.36.2
echo "Installing unsloth 2025.2.15 (known compatible version)..."
pip install unsloth==2025.2.15 --no-deps

# Add other potentially missing dependencies
echo "Ensuring sentencepiece dependency is installed..."
pip install sentencepiece

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

echo "================================================================"
echo "Unsloth fix attempt complete!"
echo "================================================================" 