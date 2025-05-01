#!/bin/bash

# Setup script for storage optimization requirements

echo "Setting up storage optimization for DeepSeek fine-tuning..."

# Detect environment (Gradient, Colab, or standard)
if [ -n "$GRADIENT_NODE_ID" ]; then
    ENVIRONMENT="gradient"
    echo "Detected Gradient environment"
elif [ -n "$COLAB_GPU" ]; then
    ENVIRONMENT="colab"
    echo "Detected Google Colab environment"
else
    ENVIRONMENT="standard"
    echo "Standard environment detected"
fi

# Install base requirements
echo "Installing base requirements..."

# Install unsloth for efficient fine-tuning
pip install unsloth

# Install additional packages for storage optimization
echo "Installing storage optimization requirements..."
pip install -q boto3 gdown

# Setup environment-specific optimizations
if [ "$ENVIRONMENT" = "gradient" ]; then
    echo "Setting up Gradient-specific optimizations..."
    
    # Set environment variables for better memory management
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
    
    # Create persistent storage directories
    mkdir -p /storage/models
    mkdir -p /storage/datasets
    mkdir -p /storage/checkpoints
    
    echo "Created persistent storage directories in /storage/"
    
elif [ "$ENVIRONMENT" = "colab" ]; then
    echo "Setting up Colab-specific optimizations..."
    
    # Mount Google Drive for external storage
    python -c "from google.colab import drive; drive.mount('/content/drive')"
    
    # Create directories in Google Drive
    mkdir -p /content/drive/MyDrive/DeepSeek_Models
    mkdir -p /content/drive/MyDrive/DeepSeek_Datasets
    
    echo "Created directories in Google Drive"
fi

# Clone the repository if not already present
if [ ! -d "Jarvis-AI-Assistant" ]; then
    echo "Cloning repository..."
    git clone https://github.com/your-username/Jarvis-AI-Assistant.git
    cd Jarvis-AI-Assistant
else
    echo "Repository already exists"
    cd Jarvis-AI-Assistant
fi

# Install project requirements
pip install -e .

echo "Storage optimization setup complete!"

# Print usage information
echo ""
echo "To fine-tune DeepSeek with storage optimization:"
echo "================================================"
echo ""

if [ "$ENVIRONMENT" = "gradient" ]; then
    echo "# On Gradient:"
    echo "python src/generative_ai_module/optimize_deepseek_storage.py \\"
    echo "    --storage-type local \\"
    echo "    --output-dir /storage/models/deepseek_optimized \\"
    echo "    --quantize 4 \\"
    echo "    --max-steps 200 \\"
    echo "    --batch-size 2 \\"
    echo "    --use-mini-dataset"
elif [ "$ENVIRONMENT" = "colab" ]; then
    echo "# On Google Colab:"
    echo "python src/generative_ai_module/optimize_deepseek_storage.py \\"
    echo "    --storage-type gdrive \\"
    echo "    --remote-path DeepSeek_Models \\"
    echo "    --quantize 4 \\"
    echo "    --max-steps 200 \\"
    echo "    --batch-size 2 \\"
    echo "    --use-mini-dataset"
else
    echo "# Standard usage:"
    echo "python src/generative_ai_module/optimize_deepseek_storage.py \\"
    echo "    --storage-type local \\"
    echo "    --quantize 4 \\"
    echo "    --max-steps 200 \\"
    echo "    --batch-size 2 \\"
    echo "    --use-mini-dataset"
fi

echo ""
echo "For AWS S3 storage:"
echo "python src/generative_ai_module/optimize_deepseek_storage.py \\"
echo "    --storage-type s3 \\"
echo "    --s3-bucket your-bucket-name \\"
echo "    --aws-access-key-id YOUR_ACCESS_KEY \\"
echo "    --aws-secret-access-key YOUR_SECRET_KEY \\"
echo "    --quantize 4"

echo ""
echo "For more options, run:"
echo "python src/generative_ai_module/optimize_deepseek_storage.py --help" 