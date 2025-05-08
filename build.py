#!/usr/bin/env python3
import os
import subprocess
import sys
import platform
import shutil

def run_command(command, cwd=None):
    """Run a command and return its output"""
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}")
        print(f"Error output: {e.stderr}")
        sys.exit(1)

def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")

    # Check Python version
    python_version = platform.python_version()
    print(f"Python version: {python_version}")
    if int(python_version.split('.')[0]) < 3 or int(python_version.split('.')[1]) < 7:
        print("Python 3.7 or higher is required")
        sys.exit(1)

    # Check Node.js
    try:
        node_version = run_command("node --version")
        print(f"Node.js version: {node_version.strip()}")
    except:
        print("Node.js is not installed. Please install Node.js 14 or higher.")
        sys.exit(1)

    # Check npm
    try:
        npm_version = run_command("npm --version")
        print(f"npm version: {npm_version.strip()}")
    except:
        print("npm is not installed. Please install npm.")
        sys.exit(1)

    print("All dependencies are installed.")

def install_python_dependencies():
    """Install Python dependencies"""
    print("Installing Python dependencies...")
    run_command("pip install -r requirements.txt")
    print("Python dependencies installed.")

def install_node_dependencies():
    """Install Node.js dependencies"""
    print("Installing Node.js dependencies...")
    run_command("npm install")
    print("Node.js dependencies installed.")

def build_frontend():
    """Build the React frontend"""
    print("Building frontend...")

    # Set environment variables to ensure offline build
    os.environ["VITE_OFFLINE_MODE"] = "true"
    os.environ["VITE_BASE_URL"] = "/"

    # Clean the dist directory if it exists
    dist_dir = "dist"
    if os.path.exists(dist_dir):
        print(f"Cleaning {dist_dir} directory...")
        shutil.rmtree(dist_dir)

    # Build with offline mode enabled
    run_command("npm run build")

    # Create a .nojekyll file to ensure all assets are served correctly
    with open(os.path.join("dist", ".nojekyll"), "w") as f:
        f.write("")

    # Verify the build
    if os.path.exists(os.path.join("dist", "index.html")):
        print("Frontend built successfully for offline use.")
    else:
        print("WARNING: index.html not found in dist directory. Build may have failed.")

    # Copy the dist directory to the server's static folder
    server_static = os.path.join("server", "static")

    # Remove existing directory or symlink
    if os.path.exists(server_static):
        if os.path.islink(server_static):
            # If it's a symlink, just remove the link
            os.unlink(server_static)
            print(f"Removed symbolic link: {server_static}")
        else:
            # If it's a directory, remove it recursively
            shutil.rmtree(server_static)
            print(f"Removed directory: {server_static}")

    # Create a symbolic link or copy the files
    try:
        # Try to create a symbolic link first (works on Unix-like systems)
        os.symlink(os.path.abspath("dist"), os.path.abspath(server_static))
        print(f"Created symbolic link from {dist_dir} to {server_static}")
    except (OSError, AttributeError):
        # If symbolic link fails, copy the files
        print(f"Copying files from {dist_dir} to {server_static}...")
        shutil.copytree("dist", server_static)
        print(f"Files copied successfully.")

def setup_jarvis_integration():
    """Set up integration with Jarvis AI assistant"""
    print("Setting up Jarvis AI integration...")

    # Create a models directory if it doesn't exist
    models_dir = os.path.join("data", "models")
    os.makedirs(models_dir, exist_ok=True)

    # Ask the user for the path to the Jarvis AI model files
    jarvis_path = input("Enter the path to your Jarvis AI project (leave empty to skip): ").strip()

    if jarvis_path and os.path.exists(jarvis_path):
        # Copy necessary files from the Jarvis project
        try:
            # Create a jarvis directory in our project
            jarvis_dir = os.path.join("server", "jarvis")
            os.makedirs(jarvis_dir, exist_ok=True)

            # Copy the necessary Python files
            for file in ["__init__.py", "assistant.py", "model.py"]:
                src_file = os.path.join(jarvis_path, file)
                if os.path.exists(src_file):
                    shutil.copy2(src_file, os.path.join(jarvis_dir, file))

            # Copy model files to the models directory
            model_files = [f for f in os.listdir(os.path.join(jarvis_path, "models"))
                          if f.endswith(".bin") or f.endswith(".onnx") or f.endswith(".pt")]

            for model_file in model_files:
                src_file = os.path.join(jarvis_path, "models", model_file)
                dst_file = os.path.join(models_dir, model_file)
                shutil.copy2(src_file, dst_file)
                print(f"Copied model file: {model_file}")

            print("Jarvis AI integration set up successfully!")
        except Exception as e:
            print(f"Error setting up Jarvis AI integration: {e}")
            print("You will need to manually set up the integration.")
    else:
        print("Skipping Jarvis AI integration setup.")
        print("You will need to manually set up the integration later.")

def main():
    """Main function"""
    print("Building Jarvis AI Assistant...")

    # Check dependencies
    check_dependencies()

    # Install dependencies
    install_python_dependencies()
    install_node_dependencies()

    # Build frontend
    build_frontend()

    # Set up Jarvis AI integration
    setup_jarvis_integration()

    print("\nBuild completed successfully!")
    print("\nTo run the application, use: python main.py")

if __name__ == "__main__":
    main()
