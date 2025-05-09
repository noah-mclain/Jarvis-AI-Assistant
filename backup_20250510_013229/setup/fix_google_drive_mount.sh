#!/bin/bash
# Fix Google Drive mounting issues
# This script fixes Google Drive mounting issues in Paperspace

echo "===================================================================="
echo "Fixing Google Drive mounting issues"
echo "===================================================================="

# Check if rclone is installed
if ! command -v rclone &> /dev/null; then
    echo "Installing rclone..."
    apt-get update -q
    apt-get install -y rclone fuse
else
    echo "✅ rclone is already installed"
fi

# Check if rclone is configured
if ! rclone listremotes | grep -q "gdrive:"; then
    echo "⚠️ Google Drive remote 'gdrive:' not found in rclone config"
    echo "Running rclone config to set up Google Drive..."
    
    echo "===================================================================="
    echo "Interactive rclone configuration for Google Drive"
    echo "===================================================================="
    echo "Please follow these steps:"
    echo "1. Select 'n' for New remote"
    echo "2. Enter 'gdrive' as the name"
    echo "3. Select the number for 'Google Drive'"
    echo "4. For client_id and client_secret, just press Enter to use the defaults"
    echo "5. Select 'scope' option 1 (full access)"
    echo "6. For root_folder_id, just press Enter"
    echo "7. For service_account_file, just press Enter"
    echo "8. Select 'y' to edit advanced config if you need to, otherwise 'n'"
    echo "9. Select 'y' to use auto config"
    echo "10. Follow the browser authentication steps when prompted"
    echo "11. Select 'y' to confirm the configuration is correct"
    echo "12. Select 'q' to quit the config process when done"
    echo "===================================================================="
    
    # Run rclone config interactively
    rclone config
    
    # Verify rclone configuration
    if rclone listremotes | grep -q "gdrive:"; then
        echo "✅ Google Drive remote 'gdrive:' configured successfully"
    else
        echo "⚠️ Google Drive remote 'gdrive:' still not found in rclone config"
        echo "Please run 'rclone config' manually to set up Google Drive"
        exit 1
    fi
else
    echo "✅ Google Drive remote 'gdrive:' already configured"
fi

# Create mount point for Google Drive
DRIVE_MOUNT_POINT="/notebooks/drive"
mkdir -p "$DRIVE_MOUNT_POINT"

# Check if already mounted
if mountpoint -q "$DRIVE_MOUNT_POINT"; then
    echo "✅ Google Drive is already mounted at $DRIVE_MOUNT_POINT"
else
    echo "Mounting Google Drive..."
    
    # Unmount if there's a stale mount
    fusermount -uz "$DRIVE_MOUNT_POINT" 2>/dev/null
    
    # Mount with better options for Paperspace
    rclone mount gdrive: "$DRIVE_MOUNT_POINT" \
        --daemon \
        --vfs-cache-mode=full \
        --vfs-cache-max-size=1G \
        --dir-cache-time=1h \
        --buffer-size=64M \
        --transfers=8 \
        --checkers=16 \
        --drive-chunk-size=32M \
        --timeout=1h \
        --umask=000 \
        --allow-non-empty \
        --allow-other \
        --log-level=INFO \
        --log-file=/tmp/rclone_mount.log
    
    # Wait for the mount to be ready
    echo "Waiting for Google Drive to be mounted..."
    for i in {1..30}; do
        if mountpoint -q "$DRIVE_MOUNT_POINT"; then
            echo "✅ Google Drive mounted successfully at $DRIVE_MOUNT_POINT"
            break
        fi
        echo "Waiting... ($i/30)"
        sleep 1
    done
    
    if ! mountpoint -q "$DRIVE_MOUNT_POINT"; then
        echo "⚠️ Google Drive mount not detected after waiting"
        echo "Checking rclone mount log..."
        tail -n 20 /tmp/rclone_mount.log
        
        echo "Trying alternative mounting method..."
        # Kill any existing rclone processes
        pkill -f "rclone mount"
        
        # Try alternative mounting method
        nohup rclone mount gdrive: "$DRIVE_MOUNT_POINT" \
            --vfs-cache-mode=full \
            --vfs-cache-max-size=1G \
            --dir-cache-time=1h \
            --buffer-size=64M \
            --transfers=8 \
            --checkers=16 \
            --drive-chunk-size=32M \
            --timeout=1h \
            --umask=000 \
            --allow-non-empty \
            --allow-other \
            > /tmp/rclone_mount_alt.log 2>&1 &
        
        # Wait for the mount to be ready
        echo "Waiting for Google Drive to be mounted (alternative method)..."
        for i in {1..30}; do
            if mountpoint -q "$DRIVE_MOUNT_POINT"; then
                echo "✅ Google Drive mounted successfully at $DRIVE_MOUNT_POINT (alternative method)"
                break
            fi
            echo "Waiting... ($i/30)"
            sleep 1
        done
        
        if ! mountpoint -q "$DRIVE_MOUNT_POINT"; then
            echo "⚠️ Google Drive mount still not detected"
            echo "Checking alternative rclone mount log..."
            tail -n 20 /tmp/rclone_mount_alt.log
            
            echo "⚠️ Failed to mount Google Drive. Will set up local storage instead."
            setup_local_storage
            exit 1
        fi
    fi
fi

# Define Jarvis AI Assistant directory structure
JARVIS_DIR="$DRIVE_MOUNT_POINT/My Drive/Jarvis_AI_Assistant"

# Create Jarvis directory structure in Google Drive
mkdir -p "$JARVIS_DIR/checkpoints"
mkdir -p "$JARVIS_DIR/datasets"
mkdir -p "$JARVIS_DIR/models"
mkdir -p "$JARVIS_DIR/logs"
mkdir -p "$JARVIS_DIR/metrics"
mkdir -p "$JARVIS_DIR/preprocessed_data"
mkdir -p "$JARVIS_DIR/visualizations"

# Create symbolic links to the Jarvis directories
echo "Creating symbolic links to Google Drive directories..."
ln -sf "$JARVIS_DIR/checkpoints" /notebooks/Jarvis_AI_Assistant/checkpoints
ln -sf "$JARVIS_DIR/datasets" /notebooks/Jarvis_AI_Assistant/datasets
ln -sf "$JARVIS_DIR/models" /notebooks/Jarvis_AI_Assistant/models
ln -sf "$JARVIS_DIR/logs" /notebooks/Jarvis_AI_Assistant/logs
ln -sf "$JARVIS_DIR/metrics" /notebooks/Jarvis_AI_Assistant/metrics
ln -sf "$JARVIS_DIR/preprocessed_data" /notebooks/Jarvis_AI_Assistant/preprocessed_data
ln -sf "$JARVIS_DIR/visualizations" /notebooks/Jarvis_AI_Assistant/visualizations

# Create a test file to verify the mount is working
echo "Testing Google Drive mount..." > "$JARVIS_DIR/mount_test.txt"
if [ -f "$JARVIS_DIR/mount_test.txt" ]; then
    echo "✅ Successfully wrote test file to Google Drive"
    cat "$JARVIS_DIR/mount_test.txt"
else
    echo "⚠️ Could not write test file to Google Drive"
    setup_local_storage
    exit 1
fi

# Set up environment variables
echo "Setting up environment variables..."
cat >> ~/.bashrc << EOF

# Jarvis AI Assistant Google Drive paths
export JARVIS_DRIVE_DIR="$JARVIS_DIR"
export JARVIS_CHECKPOINTS_DIR="$JARVIS_DIR/checkpoints"
export JARVIS_DATASETS_DIR="$JARVIS_DIR/datasets"
export JARVIS_MODELS_DIR="$JARVIS_DIR/models"
export JARVIS_LOGS_DIR="$JARVIS_DIR/logs"
export JARVIS_METRICS_DIR="$JARVIS_DIR/metrics"
export JARVIS_PREPROCESSED_DATA_DIR="$JARVIS_DIR/preprocessed_data"
export JARVIS_VISUALIZATIONS_DIR="$JARVIS_DIR/visualizations"
EOF

# Export variables for current session
export JARVIS_DRIVE_DIR="$JARVIS_DIR"
export JARVIS_CHECKPOINTS_DIR="$JARVIS_DIR/checkpoints"
export JARVIS_DATASETS_DIR="$JARVIS_DIR/datasets"
export JARVIS_MODELS_DIR="$JARVIS_DIR/models"
export JARVIS_LOGS_DIR="$JARVIS_DIR/logs"
export JARVIS_METRICS_DIR="$JARVIS_DIR/metrics"
export JARVIS_PREPROCESSED_DATA_DIR="$JARVIS_DIR/preprocessed_data"
export JARVIS_VISUALIZATIONS_DIR="$JARVIS_DIR/visualizations"

echo "===================================================================="
echo "Google Drive mounted successfully!"
echo "Your Jarvis AI Assistant storage is at: $JARVIS_DIR"
echo "===================================================================="

# Function to set up local storage if Google Drive mounting fails
setup_local_storage() {
    echo "Setting up local storage in /notebooks/Jarvis_AI_Assistant..."
    
    # Create directories
    mkdir -p /notebooks/Jarvis_AI_Assistant/checkpoints
    mkdir -p /notebooks/Jarvis_AI_Assistant/datasets
    mkdir -p /notebooks/Jarvis_AI_Assistant/models
    mkdir -p /notebooks/Jarvis_AI_Assistant/logs
    mkdir -p /notebooks/Jarvis_AI_Assistant/metrics
    mkdir -p /notebooks/Jarvis_AI_Assistant/preprocessed_data
    mkdir -p /notebooks/Jarvis_AI_Assistant/visualizations
    
    # Set up environment variables
    cat >> ~/.bashrc << EOF

# Jarvis AI Assistant local paths
export JARVIS_DRIVE_DIR="/notebooks/Jarvis_AI_Assistant"
export JARVIS_CHECKPOINTS_DIR="/notebooks/Jarvis_AI_Assistant/checkpoints"
export JARVIS_DATASETS_DIR="/notebooks/Jarvis_AI_Assistant/datasets"
export JARVIS_MODELS_DIR="/notebooks/Jarvis_AI_Assistant/models"
export JARVIS_LOGS_DIR="/notebooks/Jarvis_AI_Assistant/logs"
export JARVIS_METRICS_DIR="/notebooks/Jarvis_AI_Assistant/metrics"
export JARVIS_PREPROCESSED_DATA_DIR="/notebooks/Jarvis_AI_Assistant/preprocessed_data"
export JARVIS_VISUALIZATIONS_DIR="/notebooks/Jarvis_AI_Assistant/visualizations"
EOF

    # Export variables for current session
    export JARVIS_DRIVE_DIR="/notebooks/Jarvis_AI_Assistant"
    export JARVIS_CHECKPOINTS_DIR="/notebooks/Jarvis_AI_Assistant/checkpoints"
    export JARVIS_DATASETS_DIR="/notebooks/Jarvis_AI_Assistant/datasets"
    export JARVIS_MODELS_DIR="/notebooks/Jarvis_AI_Assistant/models"
    export JARVIS_LOGS_DIR="/notebooks/Jarvis_AI_Assistant/logs"
    export JARVIS_METRICS_DIR="/notebooks/Jarvis_AI_Assistant/metrics"
    export JARVIS_PREPROCESSED_DATA_DIR="/notebooks/Jarvis_AI_Assistant/preprocessed_data"
    export JARVIS_VISUALIZATIONS_DIR="/notebooks/Jarvis_AI_Assistant/visualizations"
    
    echo "===================================================================="
    echo "WARNING: Using local storage instead of Google Drive."
    echo "Data will not persist across sessions!"
    echo "===================================================================="
}
