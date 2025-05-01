#!/bin/bash
# Google Drive Integration Setup for DeepSeek-Coder on Paperspace Gradient
# This script sets up the environment for fine-tuning DeepSeek-Coder with
# Google Drive integration to handle Paperspace Gradient's 15GB storage limit.

set -e  # Exit on error

echo "=== Setting up Google Drive integration for DeepSeek-Coder on Paperspace Gradient ==="
echo "This script will install the necessary dependencies and configure the environment."

# Check if running on Paperspace Gradient
if [ -d "/storage" ]; then
    echo "Detected Paperspace Gradient environment"
    STORAGE_DIR="/storage"
else
    echo "Not running on Paperspace Gradient, using local storage"
    STORAGE_DIR="./storage"
    mkdir -p $STORAGE_DIR
fi

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "pip is not installed. Installing pip..."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python get-pip.py
    rm get-pip.py
fi

# Install dependencies
echo "Installing required packages..."
pip install -U pip

# Install Google Drive integration dependencies
echo "Installing Google Drive integration dependencies..."
pip install gdown google-auth google-auth-oauthlib google-auth-httplib2

# Install optimized training dependencies
echo "Installing optimized training dependencies..."
pip install unsloth bitsandbytes accelerate

# Install storage optimization dependencies
echo "Installing storage optimization dependencies..."
pip install fsspec boto3 psutil

# Create the models directory in persistent storage
echo "Creating models directory in $STORAGE_DIR..."
mkdir -p $STORAGE_DIR/models

# Setup Google Drive authentication
echo "Setting up Google Drive authentication..."

# Check if service account credentials exist
if [ -f "service-account.json" ]; then
    echo "Using existing service account credentials"
    cp service-account.json $STORAGE_DIR/service-account.json
else
    echo "No service account credentials found."
    echo "You will need to authenticate via browser when first using Google Drive."
    
    # Instructions for creating service account credentials
    echo ""
    echo "To use service account credentials for headless authentication:"
    echo "1. Go to https://console.cloud.google.com/apis/credentials"
    echo "2. Create a service account and download the JSON key"
    echo "3. Rename it to 'service-account.json' and place it in this directory"
    echo "4. Re-run this script"
    echo ""
fi

# Create a Google Drive folder placeholder file
cat > google_drive_setup.txt << EOF
To use Google Drive integration:

1. Create a folder in Google Drive to store your models and checkpoints
2. Open that folder and copy the folder ID from the URL
   (the part after "folders/" in the URL: https://drive.google.com/drive/folders/YOUR_FOLDER_ID)
3. Use this folder ID when running fine-tuning:
   python src/generative_ai_module/optimize_deepseek_gdrive.py --gdrive-folder-id YOUR_FOLDER_ID
EOF

echo "Created Google Drive setup instructions in google_drive_setup.txt"

# Create symlinks for convenient access
if [ -d "/storage" ]; then
    echo "Creating symlinks for convenient access..."
    ln -sf /storage/models ./models
    
    # Create a script to clean up local storage
    cat > cleanup_storage.sh << EOF
#!/bin/bash
# Cleanup script for Paperspace Gradient storage
echo "Cleaning up local storage in /storage/models..."
rm -rf /storage/models/*
echo "Storage cleaned up. You can now download your models from Google Drive."
EOF
    
    chmod +x cleanup_storage.sh
    echo "Created cleanup_storage.sh script to free up space when needed"
fi

# Create example run script
cat > run_gdrive_finetune.sh << EOF
#!/bin/bash
# Example script to run fine-tuning with Google Drive integration

# Replace with your Google Drive folder ID
GDRIVE_FOLDER_ID="your_folder_id_here"

# Run fine-tuning
python src/generative_ai_module/optimize_deepseek_gdrive.py \\
    --gdrive-folder-id \$GDRIVE_FOLDER_ID \\
    --output-dir $STORAGE_DIR/models/deepseek_optimized \\
    --quantize 4 \\
    --max-steps 500 \\
    --batch-size 4 \\
    --checkpoint-strategy improvement \\
    --max-checkpoints 2
EOF

chmod +x run_gdrive_finetune.sh
echo "Created example run script: run_gdrive_finetune.sh"

echo ""
echo "=== Setup Complete ==="
echo "To start fine-tuning with Google Drive integration:"
echo "1. Edit run_gdrive_finetune.sh and replace 'your_folder_id_here' with your Google Drive folder ID"
echo "2. Run: ./run_gdrive_finetune.sh"
echo ""
echo "Note: The first time you use Google Drive, you may need to authenticate via browser."