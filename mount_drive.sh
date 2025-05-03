#!/bin/bash

echo "===================================================================="
echo "Google Drive mounting for Paperspace (RTX5000 environment)"
echo "===================================================================="

# Check if running as root/sudo
if [ "$EUID" -ne 0 ]; then
  echo "This script needs root privileges to install packages and mount drives."
  echo "Please run with sudo or as root."
  exit 1
fi

# Make sure the content directory exists
mkdir -p /content

# Install apt dependencies for mounting
echo "Installing necessary packages for mounting..."
apt-get update -q
apt-get install -y fuse rclone

# Run the Python mounting script
echo "Running the Google Drive mounting script..."
python mount_drive_paperspace.py

echo "===================================================================="
echo "If your drive was not mounted successfully, you can try running:"
echo "rclone config"
echo "# Then follow the prompts to setup your Google Drive"
echo "====================================================================" 