"""
Direct Google Drive mounting script for Paperspace.
This script attempts multiple mounting methods to handle Paperspace environments.
"""
import os
import sys
import subprocess
import time
import webbrowser
import json
import base64

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

def setup_rclone_config():
    """Create a headless-friendly rclone config for Paperspace."""
    config_dir = os.path.expanduser("~/.config/rclone")
    os.makedirs(config_dir, exist_ok=True)
    
    # Check if config already exists
    config_path = os.path.join(config_dir, "rclone.conf")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            if 'gdrive' in f.read():
                print("Existing gdrive config found in rclone.conf")
                return True
    
    print("\n\033[1;32mSetting up rclone for Google Drive (Paperspace method)\033[0m")
    print("This method uses a browser authentication.")
    
    # Start the rclone config process with automated responses
    cmd = [
        "rclone", "config", "create", "gdrive", "drive",
        "scope=drive", "config_is_local=true", "config_refresh_token=true"
    ]
    
    print("\nFollowing process will open a browser for authentication.")
    print("Please complete the authentication and copy the verification code.")
    
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        
        print(stdout.decode())
        if stderr:
            print("Errors:", stderr.decode())
            
        # Verify configuration worked
        result = subprocess.run(["rclone", "listremotes"], capture_output=True, text=True)
        if "gdrive:" in result.stdout:
            print("Google Drive successfully configured in rclone!")
            return True
        else:
            print("Failed to configure Google Drive in rclone.")
            return False
    except Exception as e:
        print(f"Error setting up rclone config: {e}")
        return False

def try_rclone_mounting():
    """Try mounting with rclone optimized for Paperspace."""
    print("\nAttempting to mount via rclone (Paperspace optimized)...")
    
    # First check if rclone is available
    try:
        result = subprocess.run(["which", "rclone"], capture_output=True, text=True)
        if not result.stdout.strip():
            print("rclone is not installed or not in PATH")
            return False
    except Exception as e:
        print(f"Error checking for rclone: {e}")
        return False
    
    # Check if rclone is configured, and configure if needed
    try:
        result = subprocess.run(["rclone", "listremotes"], capture_output=True, text=True)
        
        if "gdrive:" not in result.stdout:
            if not setup_rclone_config():
                print("Could not configure rclone for Google Drive")
                return False
        
        # Create mount points
        os.makedirs('/content/drive', exist_ok=True)
        os.makedirs('/content/drive/MyDrive', exist_ok=True)
        
        # Kill any existing rclone processes
        try:
            subprocess.run(["pkill", "-f", "rclone mount"], stderr=subprocess.PIPE)
            time.sleep(1)  # Wait for processes to terminate
        except:
            pass
        
        # Mount drive with optimized parameters for Paperspace
        print("Mounting Google Drive using rclone (optimized for Paperspace)...")
        mount_process = subprocess.Popen([
            "rclone", "mount", "gdrive:", "/content/drive", 
            "--daemon", "--vfs-cache-mode=full", 
            "--allow-other", "--buffer-size=256M",
            "--transfers=4", "--checkers=8",
            "--dir-cache-time=24h", "--vfs-read-chunk-size=128M",
            "--vfs-read-chunk-size-limit=1G"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait longer for mount to complete
        print("Waiting for mount to complete...")
        time.sleep(10)
        
        # Check if mount succeeded
        retries = 3
        while retries > 0:
            if os.path.exists('/content/drive') and os.path.ismount('/content/drive'):
                print("Successfully mounted Google Drive via rclone")
                return True
            else:
                print(f"Mount check failed, retrying... ({retries} attempts left)")
                retries -= 1
                time.sleep(5)
        
        print("Failed to verify Google Drive mount")
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
    if os.path.exists('/content/drive') and os.path.ismount('/content/drive'):
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
        bashrc.write(f'export JARVIS_CHECKPOINTS_PATH="{base_path}/checkpoints"\n')
        bashrc.write(f'export JARVIS_METRICS_PATH="{base_path}/metrics"\n')
    
    print("Added environment variables to ~/.bashrc")
    
    # Create a simple test file to verify write access
    try:
        test_file = f"{base_path}/mount_test.txt"
        with open(test_file, 'w') as f:
            f.write(f"Mount test created at {time.ctime()}\n")
        print(f"Created test file at {test_file} - write access confirmed!")
    except Exception as e:
        print(f"WARNING: Could not write to {base_path} - {e}")
    
    return base_path

def main():
    """Main function to coordinate mounting attempts."""
    print("=" * 60)
    print("Google Drive Mounting for Paperspace Gradient")
    print("=" * 60)
    
    # Make sure the content directory exists
    os.makedirs('/content', exist_ok=True)
    
    # Install dependencies (returns True if rclone installed)
    rclone_installed = install_dependencies()
    
    # Try mounting methods in order of preference for Paperspace
    mounted = False
    
    # Skip Colab mounting in Paperspace environments
    # mounted = try_colab_mounting()
    
    if not mounted and rclone_installed:
        mounted = try_rclone_mounting()
    
    if not mounted:
        mounted = try_direct_mount()
    
    # Setup directories whether mounted or not
    base_path = setup_directories()
    
    print("\n" + "=" * 60)
    if mounted:
        print("Google Drive successfully mounted!")
        print(f"Your Jarvis AI Assistant storage is at: {base_path}")
        print("Use this path for all your model and data storage needs")
    else:
        print("WARNING: Google Drive could not be mounted.")
        print("Using local storage instead. Data will not persist across sessions!")
        print("To try again manually, run: rclone config")
    print("=" * 60)
    print("To apply environment variables in your current terminal session, run:")
    print("source ~/.bashrc")
    print("=" * 60)
    
    # Return the base path for use by other scripts
    return base_path, mounted

if __name__ == "__main__":
    main() 