#!/bin/bash

echo "================================================================"
echo "Unsloth-Only Fix Script (Preserves Working Dependencies)"
echo "================================================================"

# Uninstall unsloth
echo "Removing unsloth package..."
pip uninstall -y unsloth

# Try an older version first that might be compatible with transformers 4.36.2
echo "Installing unsloth (compatible version attempt 1)..."
pip install unsloth==2025.3.3 --no-deps

# Verify installation
echo "Verifying unsloth installation..."
if python -c "import unsloth; print('Unsloth successfully imported')" 2>/dev/null; then
    echo "✅ Unsloth 2025.3.3 successfully installed!"
else
    echo "❌ First attempt failed, trying alternative version..."
    
    # Try a newer version that might also be compatible
    pip install unsloth==2025.4.4 --no-deps
    
    if python -c "import unsloth; print('Unsloth successfully imported')" 2>/dev/null; then
        echo "✅ Unsloth 2025.4.4 successfully installed as fallback!"
    else
        echo "❌ Second attempt failed, trying yet another version..."
        
        # Try an older version as last attempt
        pip install unsloth==2025.2.15 --no-deps
        
        if python -c "import unsloth; print('Unsloth successfully imported')" 2>/dev/null; then
            echo "✅ Unsloth 2025.2.15 successfully installed as second fallback!"
        else
            echo "❌ All unsloth installation attempts failed."
            echo "Please consider updating your transformers version if you need unsloth functionality."
            echo "Current transformers version: $(python -c "import transformers; print(transformers.__version__)")"
        fi
    fi
fi

echo "================================================================"
echo "Unsloth fix attempt complete!"
echo "================================================================" 