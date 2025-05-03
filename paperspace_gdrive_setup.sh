#!/bin/bash

# Paperspace Google Drive Integration Script
# This script sets up rclone to connect to Google Drive from Paperspace
# and creates the same directory structure that would be used in Google Colab

echo "=== Setting up Google Drive integration for Paperspace ==="
echo "This will allow you to use the same directory structure as Google Colab"

# Install rclone
echo "Installing rclone..."
curl https://rclone.org/install.sh | sudo bash

# Install dependencies for authentication
pip install -q pydrive2 google-auth google-auth-oauthlib google-auth-httplib2

# Create config directories
mkdir -p ~/.config/rclone
mkdir -p /content

# Create mount point for Google Drive (same as Colab)
echo "Creating mount point at /content/drive..."
mkdir -p /content/drive/MyDrive

# Create the basic rclone config
cat > ~/.config/rclone/rclone.conf << EOF
[gdrive]
type = drive
scope = drive
token = {"access_token":"","token_type":"Bearer","refresh_token":"","expiry":"2023-01-01T00:00:00.000000000Z"}
EOF

# Create the authentication helper script
cat > setup_gdrive_auth.py << EOF
import os
import json
import webbrowser
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

# OAuth client config for rclone
CLIENT_CONFIG = {
    "installed": {
        "client_id": "202264815644.apps.googleusercontent.com",
        "client_secret": "X4Z3ca8xfWDb1Voo-F9a7ZxJ",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob", "http://localhost"]
    }
}

def main():
    print("Setting up Google Drive authentication...")
    
    # Create credentials flow
    flow = InstalledAppFlow.from_client_config(
        CLIENT_CONFIG,
        scopes=['https://www.googleapis.com/auth/drive']
    )
    
    # Open browser for authentication
    credentials = flow.run_local_server(port=0)
    
    # Get the token
    token_data = {
        "access_token": credentials.token,
        "token_type": "Bearer",
        "refresh_token": credentials.refresh_token,
        "expiry": credentials.expiry.isoformat() + "Z"
    }
    
    # Update rclone config
    rclone_conf_path = os.path.expanduser("~/.config/rclone/rclone.conf")
    with open(rclone_conf_path, 'r') as f:
        config = f.read()
    
    # Replace token in config
    token_str = json.dumps(token_data)
    config = config.replace('token = {"access_token":"","token_type":"Bearer","refresh_token":"","expiry":"2023-01-01T00:00:00.000000000Z"}', f'token = {token_str}')
    
    # Write updated config
    with open(rclone_conf_path, 'w') as f:
        f.write(config)
    
    print("Google Drive authentication completed!")
    print("You can now mount Google Drive with: rclone mount gdrive: /content/drive/MyDrive --daemon")

if __name__ == "__main__":
    main()
EOF

# Create the mount script
cat > mount_google_drive.sh << 'EOF'
#!/bin/bash

# Check if already mounted
if mountpoint -q /content/drive/MyDrive; then
    echo "Google Drive is already mounted at /content/drive/MyDrive"
    exit 0
fi

# Mount Google Drive in the background
echo "Mounting Google Drive at /content/drive/MyDrive..."
rclone mount gdrive: /content/drive/MyDrive --daemon --vfs-cache-mode writes

# Wait for mount to be available
echo "Waiting for mount to be ready..."
sleep 3

if mountpoint -q /content/drive/MyDrive; then
    echo "Google Drive successfully mounted at /content/drive/MyDrive"
    
    # Create Jarvis directories (same as Colab would)
    mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant
    mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant/models
    mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant/datasets
    mkdir -p /content/drive/MyDrive/Jarvis_AI_Assistant/checkpoints
    
    # Create symlink from Paperspace storage to Google Drive (for compatibility)
    ln -sf /content/drive/MyDrive/Jarvis_AI_Assistant /notebooks/google_drive_storage
    echo "Created symlink to Google Drive storage in /notebooks/google_drive_storage"
    
    # Print usage info
    echo ""
    echo "You can use Google Drive paths exactly as you would in Google Colab:"
    echo "  - Main folder: /content/drive/MyDrive/Jarvis_AI_Assistant"
    echo "  - Models: /content/drive/MyDrive/Jarvis_AI_Assistant/models"
    echo "  - Datasets: /content/drive/MyDrive/Jarvis_AI_Assistant/datasets"
    echo "  - Checkpoints: /content/drive/MyDrive/Jarvis_AI_Assistant/checkpoints"
    echo ""
    echo "Remember: This mount will not persist after restart."
    echo "You'll need to run mount_google_drive.sh again if you restart your Paperspace instance."
else
    echo "Failed to mount Google Drive. Please check if authentication is completed."
    echo "Run 'python setup_gdrive_auth.py' to authenticate."
fi
EOF

chmod +x mount_google_drive.sh

# Create automatic startup script
cat > add_to_bashrc.sh << 'EOF'
#!/bin/bash

# Add mount script to bashrc to prompt on startup
if ! grep -q "mount_google_drive.sh" ~/.bashrc; then
    echo '
# Google Drive integration
echo "Would you like to mount Google Drive? [y/N]"
read -n 1 -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ~/mount_google_drive.sh
fi
' >> ~/.bashrc
    echo "Added Google Drive mount prompt to ~/.bashrc"
fi
EOF

chmod +x add_to_bashrc.sh

# Create environment variable setup
cat > setup_gdrive_env.sh << 'EOF'
#!/bin/bash

# Set environment variables to use Google Drive paths
export JARVIS_STORAGE_PATH="/content/drive/MyDrive/Jarvis_AI_Assistant"
export JARVIS_MODELS_PATH="/content/drive/MyDrive/Jarvis_AI_Assistant/models"
export JARVIS_DATASETS_PATH="/content/drive/MyDrive/Jarvis_AI_Assistant/datasets"
export JARVIS_CHECKPOINTS_PATH="/content/drive/MyDrive/Jarvis_AI_Assistant/checkpoints"

# Add to bashrc for persistence
if ! grep -q "JARVIS_STORAGE_PATH" ~/.bashrc; then
    echo 'export JARVIS_STORAGE_PATH="/content/drive/MyDrive/Jarvis_AI_Assistant"' >> ~/.bashrc
    echo 'export JARVIS_MODELS_PATH="/content/drive/MyDrive/Jarvis_AI_Assistant/models"' >> ~/.bashrc
    echo 'export JARVIS_DATASETS_PATH="/content/drive/MyDrive/Jarvis_AI_Assistant/datasets"' >> ~/.bashrc
    echo 'export JARVIS_CHECKPOINTS_PATH="/content/drive/MyDrive/Jarvis_AI_Assistant/checkpoints"' >> ~/.bashrc
    echo "Added Jarvis Google Drive paths to environment variables"
fi
EOF

chmod +x setup_gdrive_env.sh

# Run authentication setup
echo "Setting up Google Drive authentication..."
echo "Please follow the prompts to authenticate with Google Drive"
echo ""
echo "IMPORTANT: You'll need to authorize access in a web browser."
echo "After authentication is complete, run './mount_google_drive.sh' to mount Google Drive"
echo ""

python setup_gdrive_auth.py

# Add to bashrc
./add_to_bashrc.sh

# Source environment variables
./setup_gdrive_env.sh

echo ""
echo "=== Google Drive integration setup complete! ==="
echo ""
echo "Your Google Drive is now available at: /content/drive/MyDrive"
echo "The Jarvis directory is at: /content/drive/MyDrive/Jarvis_AI_Assistant"
echo ""
echo "To manually mount Google Drive, run: ./mount_google_drive.sh"
echo ""
echo "Usage:"
echo "  1. Use the same paths as in Google Colab"
echo "     Examples:"
echo "       - Save models to: /content/drive/MyDrive/Jarvis_AI_Assistant/models"
echo "       - Load from: /content/drive/MyDrive/Jarvis_AI_Assistant/models/my_model"
echo ""
echo "  2. Environment variables are available:"
echo "     - \$JARVIS_STORAGE_PATH"
echo "     - \$JARVIS_MODELS_PATH"
echo "     - \$JARVIS_DATASETS_PATH"
echo "     - \$JARVIS_CHECKPOINTS_PATH"
echo ""
echo "Google Drive will be automatically mounted on login (after confirmation)" 