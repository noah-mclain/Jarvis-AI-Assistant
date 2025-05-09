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
    """Try mounting with Google Colab's method."""'
    print("\nAttempting to mount via Google Colab method...")
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        return True
    except (ImportError, Exception) as e:
        print(f"Colab mounting failed: {e}")
        return False

def setup_rclone_config_manual():
    """Setup rclone configuration manually, guiding the user through the process."""
    config_dir = os.path.expanduser("~/.config/rclone")
    os.makedirs(config_dir, exist_ok=True)
    
    # Check if config already exists
    config_path = os.path.join(config_dir, "rclone.conf")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            if 'gdrive' in f.read():
                print("Existing gdrive config found in rclone.conf")
                return True
    
    print("\n" + "=" * 70)
    print("MANUAL RCLONE CONFIGURATION REQUIRED")
    print("=" * 70)
    print("Please follow these steps to configure Google Drive:")
    print("1. In your terminal, run: rclone config")
    print("2. Press 'n' for new remote")
    print("3. Name: gdrive")
    print("4. Type: drive (Google Drive)")
    print("5. For client_id and client_secret, just press Enter to use defaults")
    print("6. Scope: 1 (full access)")
    print("7. Root folder: Just press Enter for default")
    print("8. Service account file: Press Enter (leave empty)")
    print("9. Edit advanced config: n (no)")
    print("10. Use auto config: n (no, since we're on a remote server)")'
    print("11. Copy the URL shown and paste it into a browser")
    print("12. Authenticate with your Google account")
    print("13. Copy the verification code and paste it back in the terminal")
    print("14. Configure as team drive: n (no)")
    print("15. Press 'y' to confirm the configuration is correct")
    print("16. Press 'q' to quit the config tool")
    print("=" * 70)
    
    response = input("Have you completed the manual rclone configuration? (y/n): ")
    if response.lower() == 'y':
        # Verify configuration worked
        result = subprocess.run(["rclone", "listremotes"], capture_output=True, text=True)
        if "gdrive:" in result.stdout:
            print("Google Drive successfully configured in rclone!")
            return True
        else:
            print("Failed to verify Google Drive configuration.")
            return False
    else:
        print("Please complete the rclone configuration before continuing.")
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
            if not setup_rclone_config_manual():
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

def create_simple_local_storage():
    """Create a simple local storage structure for when Google Drive mounting fails."""
    print("\nSetting up local storage in /notebooks/Jarvis_AI_Assistant...")
    base_path = '/notebooks/Jarvis_AI_Assistant'
    symlink_path = '/notebooks/local_jarvis'
    
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
        bashrc.write(f'export JARVIS_STORAGE_PATH="{base_path}"\n')
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

def setup_directories():
    """Create necessary directories in Google Drive or locally."""
    # Check if drive is mounted
    if os.path.exists('/content/drive') and os.path.ismount('/content/drive'):
        print("\nSetting up Google Drive directories...")
        base_path = '/content/drive/MyDrive/Jarvis_AI_Assistant'
        symlink_path = '/notebooks/google_drive_jarvis'
        
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
            bashrc.write(f'export JARVIS_STORAGE_PATH="{base_path}"\n')
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
    else:
        # Fall back to local storage
        return create_simple_local_storage()

def main():
    """Main function to coordinate mounting attempts."""
    print("=" * 60)
    print("Google Drive Mounting for Paperspace Gradient")
    print("=" * 60)
    
    # Make sure the content directory exists
    os.makedirs('/content', exist_ok=True)
    
    # Install dependencies (returns True if rclone installed)
    rclone_installed = install_dependencies()
    
    # Skip Colab mounting in Paperspace environments
    mounted = False
    
    if rclone_installed:
        mounted = try_rclone_mounting()
    
    # Skip direct mount - it doesn't work in Paperspace due to FUSE limitations
    
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
        print("\nTo manually mount Google Drive later:")
        print("1. Run: rclone config")
        print("2. Follow the setup steps for Google Drive")
        print("3. Run: rclone mount gdrive: /content/drive --daemon --vfs-cache-mode=full")
    print("=" * 60)
    print("To apply environment variables in your current terminal session, run:")
    print("source ~/.bashrc")
    print("=" * 60)
    
    # Return the base path for use by other scripts
    return base_path, mounted

if __name__ == "__main__":
    main() 