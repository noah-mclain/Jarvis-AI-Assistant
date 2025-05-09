#!/usr/bin/env python3
"""
Test script to verify Google Drive mounting is working correctly.
Run this after running mount_drive.sh to verify the mount is functioning.
"""
import os
import sys
import time
import datetime

def test_mount():
    """Test if Google Drive is properly mounted and we can read/write to it."""
    print("=" * 60)
    print("Google Drive Mount Test")
    print("=" * 60)
    
    # Check if mount point exists
    if not os.path.exists('/content/drive'):
        print("ERROR: /content/drive directory does not exist!")
        return False
    
    # Check if it's actually mounted
    if not os.path.ismount('/content/drive'):
        print("ERROR: /content/drive exists but is not a mount point!")
        return False
        
    # Check if MyDrive is accessible
    if not os.path.exists('/content/drive/MyDrive'):
        print("ERROR: /content/drive/MyDrive directory does not exist!")
        return False
    
    # Try writing to the mount
    test_file = '/content/drive/MyDrive/jarvis_mount_test.txt'
    try:
        with open(test_file, 'w') as f:
            f.write(f"Jarvis mount test file created at {datetime.datetime.now()}\n")
        print(f"✅ Successfully wrote test file to {test_file}")
        
        # Try reading from the mount
        with open(test_file, 'r') as f:
            content = f.read()
        print(f"✅ Successfully read from test file: {content.strip()}")
        
        return True
    except Exception as e:
        print(f"ERROR: Failed to read/write to Google Drive: {e}")
        return False

def check_env_vars():
    """Check if environment variables are properly set."""
    print("\nChecking environment variables:")
    expected_vars = [
        'JARVIS_STORAGE_PATH',
        'JARVIS_MODELS_PATH',
        'JARVIS_DATA_PATH',
        'JARVIS_CHECKPOINTS_PATH',
        'JARVIS_METRICS_PATH'
    ]
    
    all_present = True
    for var in expected_vars:
        if var in os.environ:
            print(f"✅ {var} = {os.environ[var]}")
        else:
            print(f"❌ {var} is not set!")
            all_present = False
    
    if not all_present:
        print("\nEnvironment variables are missing. Try running: source ~/.bashrc")
    
    return all_present

def main():
    """Main function to run mount tests."""
    mount_success = test_mount()
    env_success = check_env_vars()
    
    print("\n" + "=" * 60)
    if mount_success and env_success:
        print("🎉 SUCCESS: Google Drive is properly mounted and ready to use!")
        print("All environment variables are properly set.")
        print("\nYou can now use the following paths in your code:")
        print("- Models: $JARVIS_MODELS_PATH")
        print("- Datasets: $JARVIS_DATA_PATH")
        print("- Checkpoints: $JARVIS_CHECKPOINTS_PATH")
        print("- Metrics: $JARVIS_METRICS_PATH")
    else:
        print("⚠️ There were issues with the Google Drive mount or environment setup.")
        print("Please check the error messages above and try again.")
    print("=" * 60)

if __name__ == "__main__":
    main() 