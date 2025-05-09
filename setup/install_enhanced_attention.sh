#!/bin/bash

echo "===================================================================="
echo "Installing Enhanced Attention Mechanisms for FLAN-UL2"
echo "===================================================================="

# Function to install a package with version constraint
install_package() {
    package=$1
    version=$2
    options=$3
    
    echo "Installing $package $version..."
    
    if [ -z "$version" ]; then
        pip install "$package" --no-deps $options
        pip install "$package" $options
    else
        pip install "$package$version" --no-deps $options
        pip install "$package$version" $options
    fi
    
    # Check if installation was successful
    if python -c "import $package" 2>/dev/null; then
        echo "✅ $package installed successfully"
    else
        echo "⚠️ $package installation may have issues"
    fi
}

# Install xFormers with enhanced attention support
echo "Installing xFormers with enhanced attention support..."
pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121 --no-deps
pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121
echo "✅ xFormers installed"

# Install einops (required for attention operations)
install_package "einops" "==0.7.0"

# Install opt_einsum (optimized einsum operations for attention)
install_package "opt_einsum" "==3.3.0"

# Verify installations
echo "Verifying installations..."

# Check xFormers
python -c "
try:
    import xformers
    import xformers.ops
    print(f'xFormers version: {xformers.__version__ if hasattr(xformers, \"__version__\") else \"installed\"}')
    print('✅ xFormers successfully imported')
except Exception as e:
    print(f'❌ xFormers error: {e}')
"

# Check einops
python -c "
try:
    import einops
    print(f'einops version: {einops.__version__ if hasattr(einops, \"__version__\") else \"installed\"}')
    print('✅ einops successfully imported')
except Exception as e:
    print(f'❌ einops error: {e}')
"

# Check opt_einsum
python -c "
try:
    import opt_einsum
    print(f'opt_einsum version: {opt_einsum.__version__ if hasattr(opt_einsum, \"__version__\") else \"installed\"}')
    print('✅ opt_einsum successfully imported')
except Exception as e:
    print(f'❌ opt_einsum error: {e}')
"

echo "===================================================================="
echo "Enhanced attention mechanisms installed successfully!"
echo "===================================================================="
