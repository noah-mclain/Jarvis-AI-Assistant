#!/bin/bash

echo "===================================================================="
echo "Manual Google Drive Setup for Paperspace - Simple Version"
echo "===================================================================="

# Create directories
mkdir -p /content/drive
mkdir -p /content/drive/MyDrive

# Check if rclone is installed
if ! command -v rclone &> /dev/null; then
    echo "Installing rclone..."
    apt-get update -q
    apt-get install -y rclone
fi

echo
echo "First, let's configure rclone for Google Drive"
echo "--------------------------------------------------------------------"
echo "When prompted, please follow these steps:"
echo "1. Press 'n' for new remote"
echo "2. Name: gdrive"
echo "3. Type: Select Google Drive (typically option number 18)"
echo "4. For client_id and client_secret, just press Enter to use defaults"
echo "5. Scope: Select 1 (full access)"
echo "6. Root folder: Just press Enter (default)"
echo "7. Service account file: Press Enter (leave empty)"
echo "8. Edit advanced config: n (no)"
echo "9. Use auto config: n (no, we're on a remote server)"
echo "10. Copy the URL shown and paste it into a browser"
echo "11. Authenticate with your Google account"
echo "12. Copy the verification code and paste it back in the terminal"
echo "13. Configure as team drive: n (no)"
echo "14. Press 'y' to confirm the configuration is correct"
echo "15. Press 'q' to quit the config tool"
echo "--------------------------------------------------------------------"
echo

read -p "Press Enter to start rclone configuration..." 

# Start rclone config
rclone config

# Check if rclone is now configured for Google Drive
if rclone listremotes | grep -q "gdrive:"; then
    echo "✅ Google Drive configuration found!"
    
    # Mount Google Drive
    echo "Mounting Google Drive to /content/drive..."
    
    # Kill any existing rclone processes
    pkill -f "rclone mount" || true
    sleep 1
    
    # Mount with optimized parameters
    rclone mount gdrive: /content/drive \
        --daemon \
        --vfs-cache-mode=full \
        --allow-other \
        --buffer-size=256M \
        --transfers=4 \
        --checkers=8 \
        --dir-cache-time=24h \
        --vfs-read-chunk-size=128M \
        --vfs-read-chunk-size-limit=1G
    
    # Wait for mount to complete
    echo "Waiting for mount to complete..."
    sleep 5
    
    # Verify mount worked
    if mountpoint -q /content/drive; then
        echo "✅ Google Drive successfully mounted to /content/drive!"
        
        # Create Jarvis directories on Google Drive
        JARVIS_DIR="/content/drive/MyDrive/Jarvis_AI_Assistant"
        mkdir -p "$JARVIS_DIR/models"
        mkdir -p "$JARVIS_DIR/datasets"
        mkdir -p "$JARVIS_DIR/checkpoints"
        mkdir -p "$JARVIS_DIR/metrics"
        
        # Create symlink for convenient access
        ln -sf "$JARVIS_DIR" /notebooks/google_drive_jarvis
        
        # Add environment variables
        cat >> ~/.bashrc << EOL

# Jarvis AI Assistant paths (Google Drive)
export JARVIS_STORAGE_PATH="$JARVIS_DIR"
export JARVIS_MODELS_PATH="$JARVIS_DIR/models"
export JARVIS_DATA_PATH="$JARVIS_DIR/datasets"
export JARVIS_CHECKPOINTS_PATH="$JARVIS_DIR/checkpoints"
export JARVIS_METRICS_PATH="$JARVIS_DIR/metrics"
EOL

        # Create test file
        echo "Testing write access to Google Drive..."
        echo "Mount test file created $(date)" > "$JARVIS_DIR/mount_test.txt"
        
        echo
        echo "===================================================================="
        echo "✅ SUCCESS: Google Drive mounted and Jarvis directories created!"
        echo "Your data will be stored at: $JARVIS_DIR"
        echo "Symlink created at: /notebooks/google_drive_jarvis"
        echo
        echo "To use the environment variables in your current session, run:"
        echo "source ~/.bashrc"
        echo "===================================================================="
    else
        echo "❌ Failed to mount Google Drive"
        create_local_storage
    fi
else
    echo "❌ Google Drive configuration not found or incomplete"
    create_local_storage
fi

# Function to create local storage as fallback
create_local_storage() {
    echo "Setting up local storage instead..."
    
    # Create local directories
    JARVIS_DIR="/notebooks/Jarvis_AI_Assistant"
    mkdir -p "$JARVIS_DIR/models"
    mkdir -p "$JARVIS_DIR/datasets"
    mkdir -p "$JARVIS_DIR/checkpoints"
    mkdir -p "$JARVIS_DIR/metrics"
    
    # Create symlink
    ln -sf "$JARVIS_DIR" /notebooks/local_jarvis
    
    # Add environment variables
    cat >> ~/.bashrc << EOL

# Jarvis AI Assistant paths (Local Storage)
export JARVIS_STORAGE_PATH="$JARVIS_DIR"
export JARVIS_MODELS_PATH="$JARVIS_DIR/models"
export JARVIS_DATA_PATH="$JARVIS_DIR/datasets"
export JARVIS_CHECKPOINTS_PATH="$JARVIS_DIR/checkpoints"
export JARVIS_METRICS_PATH="$JARVIS_DIR/metrics"
EOL

    echo
    echo "===================================================================="
    echo "⚠️ WARNING: Using local storage instead of Google Drive."
    echo "Your data will be stored at: $JARVIS_DIR"
    echo "THIS DATA WILL BE LOST when your Paperspace session ends!"
    echo
    echo "To use the environment variables in your current session, run:"
    echo "source ~/.bashrc"
    echo "===================================================================="
} 