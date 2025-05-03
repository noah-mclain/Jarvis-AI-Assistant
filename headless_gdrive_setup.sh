#!/bin/bash

# Headless Google Drive Setup for Paperspace
# This script provides a method to authenticate with Google Drive without requiring a browser

echo "=== Setting up Google Drive for Paperspace (Headless Mode) ==="

# Install rclone if not already installed
if ! command -v rclone &> /dev/null; then
    echo "Installing rclone..."
    curl https://rclone.org/install.sh | sudo bash
fi

# Create necessary directories
mkdir -p ~/.config/rclone
mkdir -p /content/drive/MyDrive

# Generate a basic rclone config
cat > ~/.config/rclone/rclone.conf << EOL
[gdrive]
type = drive
client_id = 202264815644.apps.googleusercontent.com
client_secret = X4Z3ca8xfWDb1Voo-F9a7ZxJ
scope = drive
root_folder_id =
EOL

# Create the headless auth script
cat > headless_auth.py << EOL
#!/usr/bin/env python3
"""
Headless Google Drive authentication for Paperspace environments.
This script generates an auth URL and allows you to paste the authorization code.
"""
import os
import json
import subprocess
from urllib.parse import urlparse, parse_qs

# Run rclone config to generate auth URL
print("Generating Google Drive authentication URL...")
result = subprocess.run(
    ["rclone", "config", "reconnect", "gdrive:", "--no-browser"],
    capture_output=True,
    text=True
)

# Extract the auth URL from rclone output
auth_lines = result.stderr.split("\\n")
auth_url = None
for line in auth_lines:
    if "https://accounts.google.com/o/oauth2/auth" in line:
        auth_url = line.strip()
        break

if not auth_url:
    print("ERROR: Could not generate authentication URL")
    print("rclone output:", result.stderr)
    exit(1)

# Display the URL for the user
print("\n" + "="*80)
print("GOOGLE DRIVE AUTHENTICATION REQUIRED")
print("="*80)
print("\n1. Open this URL in a browser on your local machine:")
print("\n" + auth_url + "\n")
print("2. Log in with your Google account and grant access")
print("3. Copy the authorization code from the browser")
print("4. Paste the authorization code below:")
print("="*80 + "\n")

# Get the auth code from user input
auth_code = input("Enter the authorization code: ")

# Complete the authentication process
print("Completing authentication...")
subprocess.run(
    ["rclone", "config", "reconnect", "gdrive:", "--code", auth_code],
    check=True
)

print("\nGoogle Drive authentication completed successfully!")
EOL

chmod +x headless_auth.py

# Create mount script
cat > mount_gdrive_headless.sh << EOL
#!/bin/bash

# Check if already mounted
if mountpoint -q /content/drive/MyDrive; then
    echo "Google Drive is already mounted at /content/drive/MyDrive"
    exit 0
fi

# Mount Google Drive in the background
echo "Mounting Google Drive at /content/drive/MyDrive..."
rclone mount gdrive: /content/drive/MyDrive --daemon --vfs-cache-mode writes

# Wait for mount to be ready
echo "Waiting for mount to be ready..."
sleep 3

if mountpoint -q /content/drive/MyDrive; then
    echo "Google Drive successfully mounted at /content/drive/MyDrive"
    
    # Create Jarvis directories
    mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant
    mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant/models
    mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant/datasets
    mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant/checkpoints
    
    # Create symlink from Paperspace storage to Google Drive
    ln -sf /content/drive/MyDrive/Jarvis_AI_Assistant /notebooks/google_drive_storage
    echo "Created symlink to Google Drive storage in /notebooks/google_drive_storage"
    
    echo ""
    echo "You can use Google Drive paths exactly as you would in Google Colab:"
    echo "  - Main folder: /content/drive/MyDrive/Jarvis_AI_Assistant"
    echo "  - Models: /content/drive/MyDrive/Jarvis_AI_Assistant/models"
    echo "  - Datasets: /content/drive/MyDrive/Jarvis_AI_Assistant/datasets"
    echo "  - Checkpoints: /content/drive/MyDrive/Jarvis_AI_Assistant/checkpoints"
else
    echo "Failed to mount Google Drive. Please check if authentication is completed."
    echo "Run './headless_auth.py' to authenticate."
fi
EOL

chmod +x mount_gdrive_headless.sh

# Run headless authentication
echo "Starting Google Drive authentication process..."
python headless_auth.py

# Prompt to mount
echo ""
echo "Authentication completed! Would you like to mount Google Drive now? [y/N]"
read -n 1 -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ./mount_gdrive_headless.sh
fi

echo ""
echo "=== Headless Google Drive setup complete! ==="
echo ""
echo "To mount Google Drive in the future, run: ./mount_gdrive_headless.sh"
echo ""
echo "The Jarvis directories are now available at:"
echo "/content/drive/MyDrive/Jarvis_AI_Assistant"
echo "" 