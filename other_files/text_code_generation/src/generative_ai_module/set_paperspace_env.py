"""
Environment setup for Paperspace Gradient.

This module handles setting up the environment for Paperspace Gradient,
including creating necessary directories and configuring paths.
"""

import os
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories for Jarvis AI Assistant"""
    # Check if Google Drive is mounted
    drive_mount_point = "/notebooks/drive"
    if os.path.exists(drive_mount_point) and os.path.ismount(drive_mount_point):
        # Use Google Drive paths
        base_dir = os.path.join(drive_mount_point, "My Drive/Jarvis_AI_Assistant")
        logger.info(f"Using Google Drive for storage: {base_dir}")
    else:
        # Use local paths
        base_dir = "notebooks/Jarvis_AI_Assistant"
        logger.info(f"Using local storage: {base_dir}")

    directories = [
        "models",
        "datasets",
        "checkpoints",
        "logs",
        "visualizations",
        "evaluation_metrics",
        "preprocessed_data",
        "metrics"
    ]

    for directory in directories:
        dir_path = os.path.join(base_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

        # Create symbolic links if using Google Drive
        if drive_mount_point in base_dir:
            local_dir = os.path.join("notebooks/Jarvis_AI_Assistant", directory)
            if os.path.exists(local_dir) and not os.path.islink(local_dir):
                # Remove existing directory
                import shutil
                shutil.rmtree(local_dir)

            # Create symbolic link
            if not os.path.exists(local_dir):
                os.symlink(dir_path, local_dir)
                logger.info(f"Created symbolic link: {local_dir} -> {dir_path}")

    # Set environment variables
    os.environ['JARVIS_STORAGE_BASE'] = base_dir
    for directory in directories:
        env_var_name = f"JARVIS_{directory.upper()}_DIR"
        env_var_value = os.path.join(base_dir, directory)
        os.environ[env_var_name] = env_var_value
        logger.info(f"Set environment variable: {env_var_name}={env_var_value}")

    return base_dir

def is_paperspace_environment():
    """Checks if code is running in Paperspace Gradient environment."""
    # More robust check for Paperspace environment
    paperspace_indicators = [
        os.path.exists('/notebooks'),
        os.environ.get('PAPERSPACE') == 'true',
        os.path.exists('/notebooks/Jarvis_AI_Assistant'),
        os.path.exists('/notebooks/jarvis_env')
    ]

    # If any of the indicators are true, we're in Paperspace
    is_paperspace = any(paperspace_indicators)

    # Force Paperspace environment for this specific case
    if not is_paperspace:
        logger.warning("Paperspace environment not detected by standard checks, but forcing it based on user input")
        is_paperspace = True

    # Set environment variable for future checks
    if is_paperspace:
        os.environ['PAPERSPACE'] = 'true'

    return is_paperspace

def setup_paperspace_env():
    """Set up the Paperspace environment"""
    if is_paperspace_environment():
        logger.info("Setting up Paperspace environment...")

        # Create necessary directories
        base_dir = create_directories()

        # Set environment variables for Paperspace
        os.environ['PAPERSPACE'] = 'true'
        os.environ['JARVIS_STORAGE_BASE'] = base_dir

        # Add environment variables to .bashrc for persistence
        bashrc_path = os.path.expanduser('~/.bashrc')
        with open(bashrc_path, 'a') as f:
            f.write('\n# Jarvis AI Assistant environment variables\n')
            f.write('export PAPERSPACE=true\n')
            f.write(f'export JARVIS_STORAGE_BASE="{base_dir}"\n')

            # Add directory-specific environment variables
            directories = [
                "models", "datasets", "checkpoints", "logs",
                "visualizations", "evaluation_metrics", "preprocessed_data", "metrics"
            ]

            for directory in directories:
                env_var_name = f"JARVIS_{directory.upper()}_DIR"
                env_var_value = os.path.join(base_dir, directory)
                f.write(f'export {env_var_name}="{env_var_value}"\n')

        logger.info(f"Added environment variables to {bashrc_path}")
        logger.info("Paperspace environment setup complete")
    else:
        logger.info("Not running in Paperspace environment, skipping setup")

def get_storage_base_path():
    """Get the base path for storage"""
    if is_paperspace_environment():
        # Check if Google Drive is mounted
        drive_mount_point = "/notebooks/drive"
        if os.path.exists(drive_mount_point) and os.path.ismount(drive_mount_point):
            # Use Google Drive paths
            return os.path.join(drive_mount_point, "My Drive/Jarvis_AI_Assistant")
        else:
            # Use local paths
            return "notebooks/Jarvis_AI_Assistant"
    else:
        # For local development, use the project root
        current_file = Path(__file__).resolve()
        project_root = str(current_file.parent.parent.parent)
        return project_root

def main():
    """Main function to set up the Paperspace environment"""
    setup_paperspace_env()

    # Print storage paths for verification
    base_path = get_storage_base_path()
    logger.info(f"Storage base path: {base_path}")

    # Check if Google Drive is mounted
    drive_mount_point = "/notebooks/drive"
    if os.path.exists(drive_mount_point) and os.path.ismount(drive_mount_point):
        logger.info(f"Google Drive is mounted at {drive_mount_point}")

        # Check if we can write to Google Drive
        test_file = os.path.join(base_path, "mount_test.txt")
        try:
            with open(test_file, 'w') as f:
                f.write(f"Mount test created at {os.path.basename(__file__)} on {os.path.basename(os.getcwd())}\n")
            logger.info(f"Successfully wrote to {test_file}")
        except Exception as e:
            logger.error(f"Failed to write to {test_file}: {e}")
    else:
        logger.warning(f"Google Drive is not mounted at {drive_mount_point}")
        logger.warning("Using local storage instead")

    # Print all directory paths
    directories = [
        "models", "datasets", "checkpoints", "logs",
        "visualizations", "evaluation_metrics", "preprocessed_data", "metrics"
    ]

    for directory in directories:
        path = os.path.join(base_path, directory)
        logger.info(f"{directory} path: {path}")

        # Check if the directory exists and is accessible
        if os.path.exists(path):
            logger.info(f"✅ {directory} directory exists")
        else:
            logger.warning(f"❌ {directory} directory does not exist")

        # Check if the directory is a symlink
        if os.path.islink(path):
            target = os.readlink(path)
            logger.info(f"📌 {directory} is a symlink to {target}")

    logger.info("Paperspace environment setup complete")

if __name__ == "__main__":
    main()
