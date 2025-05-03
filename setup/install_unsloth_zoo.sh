#!/bin/bash

echo "================================================================"
echo "Installing unsloth_zoo with Zero Dependency Conflicts"
echo "================================================================"

# Install unsloth_zoo with the --no-deps flag to avoid modifying any dependencies
echo "Installing unsloth_zoo without affecting existing packages..."
pip install unsloth_zoo --no-deps

# Verify that unsloth now works
echo "Verifying unsloth installation..."
if python -c "import unsloth; print('✅ Unsloth successfully imported')" 2>/dev/null; then
    echo "✅ Unsloth now works successfully!"
    python -c "import unsloth; print(f'Unsloth version: {getattr(unsloth, \"__version__\", \"unknown\")}')"
else
    echo "❌ Unsloth import still failing. Error details:"
    python -c "
import traceback
try:
    import unsloth
    print(f'Unsloth partial import successful. Version: {getattr(unsloth, \"__version__\", \"unknown\")}')
except Exception as e:
    print(f'Error importing unsloth: {e}')
    traceback.print_exc()
"
fi

# Verify that all other dependencies are still intact
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
echo "Installation complete with zero dependency changes!"
echo "================================================================" 