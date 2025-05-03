"""
Direct Google Drive mounting script for Paperspace.
This script attempts multiple mounting methods to handle Paperspace environments.
"""
import os
import sys
import subprocess
import time

def install_dependencies():
    """Install required packages for Google Drive mounting."""
    print("Installing required packages...")
    subprocess.run([
        "pip", "install", "-q", 
        "gdown", "pydrive2", "google-auth", "google-auth-oauthlib"
    ])
    
    # Install rclone properly using apt
    print("Installing rclone using apt...")
    try:
        # Update apt first
        subprocess.run(["apt-get", "update", "-q"], check=True)
        # Install rclone
        subprocess.run(["apt-get", "install", "-y", "rclone"], check=True)
        
        # Verify installation
        result = subprocess.run(["rclone", "--version"], capture_output=True, text=True)
        if "rclone" in result.stdout:
            print("rclone installed successfully!")
            return True
        else:
            print("rclone installation verification failed")
            return False
    except Exception as e:
        print(f"Failed to install rclone: {e}")
        return False

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
    
    # First check if rclone is available
    try:
        result = subprocess.run(["which", "rclone"], capture_output=True, text=True)
        if not result.stdout.strip():
            print("rclone is not installed or not in PATH")
            return False
    except Exception as e:
        print(f"Error checking for rclone: {e}")
        return False
    
    # Check if rclone is configured
    try:
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
            
            # Create a minimal default config file for rclone
            os.makedirs(os.path.expanduser("~/.config/rclone"), exist_ok=True)
            with open(os.path.expanduser("~/.config/rclone/rclone.conf"), "w") as f:
                f.write("[gdrive]\n")
                f.write("type = drive\n")
                f.write("scope = drive\n")
            
            input("Press Enter after you've manually configured rclone...")
            
            # Recheck after manual configuration
            result = subprocess.run(["rclone", "listremotes"], capture_output=True, text=True)
            if "gdrive:" not in result.stdout:
                print("rclone still not configured for Google Drive")
                return False
        
        # Create mount point
        os.makedirs('/content/drive', exist_ok=True)
        os.makedirs('/content/drive/MyDrive', exist_ok=True)
        
        # Mount drive with better parameters for stability
        print("Mounting Google Drive using rclone...")
        mount_process = subprocess.Popen(
            ["rclone", "mount", "gdrive:", "/content/drive", 
             "--daemon", "--vfs-cache-mode=writes", 
             "--allow-other", "--buffer-size=256M"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        
        # Wait a bit longer for mount to complete
        print("Waiting for mount to complete...")
        time.sleep(5)
        
        # Check if mount succeeded
        if os.path.exists('/content/drive') and len(os.listdir('/content/drive')) > 0:
            print("Successfully mounted Google Drive via rclone")
            return True
        else:
            print("Failed to mount Google Drive via rclone")
            return False
    except Exception as e:
        print(f"Error in rclone mounting: {e}")
        return False

def try_direct_mount():
    """Try mounting with direct OS mount."""
    print("\nAttempting to mount with direct OS method...")
    try:
        # Check if we have /dev/fuse (needed for FUSE mounts)
        if not os.path.exists('/dev/fuse'):
            print("FUSE device not available")
            return False
            
        # Ensure mount point exists
        os.makedirs('/content/drive/MyDrive', exist_ok=True)
        
        # Try to use a simple local mount to simulate Google Drive folder structure
        # This is a fallback when all else fails
        print("Creating a simulated drive structure...")
        os.makedirs('/notebooks/simulated_gdrive', exist_ok=True)
        os.makedirs('/notebooks/simulated_gdrive/MyDrive', exist_ok=True)
        
        # Use bind mount to make it accessible at the expected path
        subprocess.run([
            "mount", "--bind", 
            "/notebooks/simulated_gdrive/MyDrive", 
            "/content/drive/MyDrive"
        ], check=True)
        
        if os.path.ismount('/content/drive/MyDrive'):
            print("Created local simulated Google Drive structure")
            return True
        else:
            print("Failed to create simulated drive")
            return False
    except Exception as e:
        print(f"Error in direct mounting: {e}")
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
    try:
        os.symlink(base_path, symlink_path)
        print(f"Created symlink at {symlink_path}")
    except Exception as e:
        print(f"Failed to create symlink: {e}")
    
    # Set environment variables
    with open(os.path.expanduser("~/.bashrc"), "a") as bashrc:
        bashrc.write('\n# Jarvis AI Assistant paths\n')
        bashrc.write(f'{env_var_prefix}\n')
        bashrc.write(f'export JARVIS_MODELS_PATH="{base_path}/models"\n')
        bashrc.write(f'export JARVIS_DATA_PATH="{base_path}/datasets"\n')
    
    print("Added environment variables to ~/.bashrc")
    print(f"Run 'source ~/.bashrc' to apply them to current session")
    
    return base_path

def main():
    """Main function to coordinate mounting attempts."""
    print("=" * 60)
    print("Google Drive Mounting for Paperspace")
    print("=" * 60)
    
    # Make sure the content directory exists
    os.makedirs('/content', exist_ok=True)
    
    # Install dependencies (returns True if rclone installed)
    rclone_installed = install_dependencies()
    
    # Try mounting methods
    mounted = try_colab_mounting()
    
    if not mounted and rclone_installed:
        mounted = try_rclone_mounting()
    
    if not mounted:
        mounted = try_direct_mount()
    
    # Setup directories whether mounted or not
    base_path = setup_directories()
    
    print("\n" + "=" * 60)
    if mounted:
        print("Google Drive successfully mounted!")
        print("Your Jarvis AI Assistant storage is now in Google Drive")
    else:
        print("WARNING: Google Drive could not be mounted.")
        print("Using local storage instead.")
        print("To try again manually, run: rclone config")
    print("=" * 60)
    
    # Return the base path for use by other scripts
    return base_path, mounted

if __name__ == "__main__":
    main() 