#!/bin/bash

echo "===================================================================="
echo "Google Drive mounting for Paperspace (RTX5000 environment)"
echo "===================================================================="

# Check if running as root/sudo
if [ "$EUID" -ne 0 ]; then
  echo "Attempting to use sudo to run this script..."
  sudo "$0" "$@"
  exit $?
fi

# Make sure the content directory exists
mkdir -p /content

# Install apt dependencies for mounting
echo "Installing necessary packages for mounting..."
apt-get update -q
apt-get install -y fuse rclone

# Determine the script's directory to find the Python script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run the Python mounting script with full path
echo "Running the Google Drive mounting script..."
python "${SCRIPT_DIR}/mount_drive_paperspace.py"

# Source bashrc to apply environment variables
echo "Updating environment variables..."
source ~/.bashrc

echo "===================================================================="
echo "If your drive was not mounted successfully, you can try running:"
echo "rclone config"
echo "# Then follow the prompts to setup your Google Drive"
echo "====================================================================" 