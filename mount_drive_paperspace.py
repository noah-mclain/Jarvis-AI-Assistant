#!/usr/bin/env python3
"""
Direct Google Drive mounting script for Paperspace.
This script attempts multiple mounting methods to handle Paperspace environments.
"""
import os
import sys
import subprocess

def install_dependencies():
    """Install required packages for Google Drive mounting."""
    print("Installing required packages...")
    subprocess.run([
        "pip", "install", "-q", 
        "gdown", "pydrive2", "google-auth", "google-auth-oauthlib"
    ])
    
    # Install rclone if not present
    try:
        subprocess.run(["rclone", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("rclone is already installed")
    except FileNotFoundError:
        print("Installing rclone...")
        subprocess.run([
            "curl", "https://rclone.org/install.sh", 
            "|", "sudo", "bash"
        ], shell=True)

def try_colab_mounting():
    """Try mounting with Google Colab's method."""
    print("\nAttempting to mount via Google Colab method...")
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        return True
    except (ImportError, Exception) as e:
        print(f"Colab mounting failed: {e}")
        return False

def try_rclone_mounting():
    """Try mounting with rclone."""
    print("\nAttempting to mount via rclone...")
    
    # Check if rclone is configured
    result = subprocess.run(["rclone", "listremotes"], capture_output=True, text=True)
    
    if "gdrive:" not in result.stdout:
        print("Setting up rclone for Google Drive...")
        print("\n\033[1;32mIMPORTANT: You'll need to complete authentication in your browser.\033[0m")
        print("Follow these steps:")
        print("1. Run: rclone config")
        print("2. Select 'n' for new remote")
        print("3. Name it 'gdrive'")
        print("4. Select Google Drive (option number varies)")
        print("5. Accept the defaults for most options")
        print("6. Choose option to open browser for authentication")
        print("7. Complete the authentication process\n")
        
        input("Press Enter after you've manually configured rclone...")
        
    # Create mount point
    os.makedirs('/content/drive', exist_ok=True)
    
    # Mount drive
    process = subprocess.Popen(
        ["rclone", "mount", "gdrive:", "/content/drive", 
         "--daemon", "--vfs-cache-mode", "writes"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    
    # Wait briefly to see if mount fails immediately
    import time
    time.sleep(3)
    
    # Check if mount succeeded
    if os.path.exists('/content/drive/MyDrive'):
        print("Successfully mounted Google Drive via rclone")
        return True
    else:
        print("Failed to mount Google Drive via rclone")
        return False

def setup_directories():
    """Create necessary directories in Google Drive or locally."""
    # Check if drive is mounted
    if os.path.exists('/content/drive/MyDrive'):
        print("\nSetting up Google Drive directories...")
        base_path = '/content/drive/MyDrive/Jarvis_AI_Assistant'
        symlink_path = '/notebooks/google_drive_jarvis'
        env_var_prefix = 'export JARVIS_STORAGE_PATH="/content/drive/MyDrive/Jarvis_AI_Assistant"'
    else:
        print("\nSetting up local directories (Google Drive mount failed)...")
        base_path = '/notebooks/Jarvis_AI_Assistant'
        symlink_path = '/notebooks/local_jarvis'
        env_var_prefix = 'export JARVIS_STORAGE_PATH="/notebooks/Jarvis_AI_Assistant"'
    
    # Create directories
    for subdir in ['models', 'datasets', 'checkpoints', 'metrics']:
        os.makedirs(f"{base_path}/{subdir}", exist_ok=True)
    
    print(f"Created directories in {base_path}")
    
    # Create symlink for easier access
    if os.path.exists(symlink_path):
        os.remove(symlink_path)
    os.symlink(base_path, symlink_path)
    print(f"Created symlink at {symlink_path}")
    
    # Set environment variables
    with open(os.path.expanduser("~/.bashrc"), "a") as bashrc:
        bashrc.write('\n# Jarvis AI Assistant paths\n')
        bashrc.write(f'{env_var_prefix}\n')
        bashrc.write(f'export JARVIS_MODELS_PATH="{base_path}/models"\n')
        bashrc.write(f'export JARVIS_DATA_PATH="{base_path}/datasets"\n')
    
    print("Added environment variables to ~/.bashrc")
    print(f"Run 'source ~/.bashrc' to apply them to current session")

def main():
    """Main function to coordinate mounting attempts."""
    print("=" * 60)
    print("Google Drive Mounting for Paperspace")
    print("=" * 60)
    
    install_dependencies()
    
    # Try mounting methods
    mounted = try_colab_mounting() or try_rclone_mounting()
    
    # Setup directories whether mounted or not
    setup_directories()
    
    print("\n" + "=" * 60)
    if mounted:
        print("Google Drive successfully mounted!")
        print("Your Jarvis AI Assistant storage is now in Google Drive")
    else:
        print("WARNING: Google Drive could not be mounted.")
        print("Using local storage instead.")
        print("To try again manually, run: rclone config")
    print("=" * 60)

if __name__ == "__main__":
    main() 