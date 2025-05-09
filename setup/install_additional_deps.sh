#!/bin/bash

echo "===================================================================="
echo "Installing additional dependencies without conflicts"
echo "===================================================================="

# Function to install a package with version constraint
install_package() {
    package=$1
    version=$2
    
    echo "Installing $package $version..."
    
    if [ -z "$version" ]; then
        pip install "$package" --no-deps
        pip install "$package"
    else
        pip install "$package$version" --no-deps
        pip install "$package$version"
    fi
    
    # Check if installation was successful
    if python -c "import $package" 2>/dev/null; then
        echo "✅ $package installed successfully"
    else
        echo "⚠️ $package installation may have issues"
    fi
}

# Install markdown
install_package "markdown" ""

# Install protobuf with version constraint
pip install "protobuf<4.24" --no-deps
pip install "protobuf<4.24"
echo "✅ protobuf<4.24 installed"

# Install werkzeug
install_package "werkzeug" ""

# Install pandas (already in requirements but ensuring it's installed)
install_package "pandas" "==2.2.0"

# Install huggingface-hub (already in requirements but ensuring it's installed)
install_package "huggingface_hub" "==0.19.4"

echo "===================================================================="
echo "Additional dependencies installed successfully!"
echo "===================================================================="
