"""
Jarvis AI Assistant Desktop Application

This is the main entry point for the Jarvis AI Assistant desktop application.
It creates a desktop window using PyWebView that loads a Flask web application
served locally. The application provides a user interface for interacting with
the Jarvis AI Assistant.

The application architecture consists of:
1. A Flask web server (server/app.py) that serves the UI and handles API requests
2. A PyWebView window that displays the web UI in a desktop application
3. The Jarvis AI Assistant integration (server/jarvis/assistant.py)

Author: Nada Mohamed
License: MIT
"""

import webview  # Used to create the desktop window
import threading  # Used for running the server in a background thread
import sys  # Used for system operations like exit
import os  # Used for file and directory operations
import logging  # Used for application logging
from waitress import serve  # Production-ready WSGI server for Flask
from server.app import app  # Import the Flask application

# Set up logging configuration for the application
# This configures both console output and file logging
logging.basicConfig(
    level=logging.INFO,  # Set the minimum log level to INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Define log format with timestamp
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler('jarvis_app.log')  # Log to file
    ]
)
logger = logging.getLogger('jarvis_desktop')  # Create a logger for this module

def start_server():
    """
    Start the Flask server in a separate thread.

    This function starts the Flask application using the Waitress WSGI server,
    which is more production-ready than Flask's built-in development server.
    The server runs on localhost (127.0.0.1) port 5000 with an appropriate
    number of worker threads based on the system's capabilities.

    The function handles platform-specific configurations to ensure the server
    works properly on Windows, macOS, and Linux environments.

    If the server fails to start, the application will exit with an error code.
    """
    try:
        # Determine the optimal number of threads based on CPU cores
        # This ensures good performance across different hardware
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        thread_count = max(2, min(cpu_count, 8))  # Between 2 and 8 threads

        # Log server startup information
        logger.info(f"Starting Jarvis AI Assistant server on http://127.0.0.1:5000")
        logger.info(f"Using {thread_count} worker threads (detected {cpu_count} CPU cores)")

        # Platform-specific server configurations
        if sys.platform == 'win32':
            # Windows-specific server settings
            logger.info("Using Windows-optimized server configuration")
            serve(app, host='127.0.0.1', port=5000, threads=thread_count,
                  connection_limit=100, cleanup_interval=30)
        elif sys.platform == 'darwin':
            # macOS-specific server settings
            logger.info("Using macOS-optimized server configuration")
            serve(app, host='127.0.0.1', port=5000, threads=thread_count,
                  connection_limit=200, cleanup_interval=60)
        else:
            # Linux and other platforms
            logger.info("Using standard server configuration")
            serve(app, host='127.0.0.1', port=5000, threads=thread_count)

    except Exception as e:
        # Log the error and exit if the server fails to start
        logger.error(f"Error starting server: {e}")
        logger.error("Please check if port 5000 is already in use by another application")
        sys.exit(1)  # Exit with error code

def setup_window_events(window):
    """
    Set up event handlers for the PyWebView window.

    This function attaches event handlers to the window events:
    - closed: Called when the window is closed by the user
    - shown: Called when the window is first displayed
    - loaded: Called when the window content is fully loaded

    Args:
        window: The PyWebView window object to attach events to

    Note:
        If event handlers fail to set up, the application will continue running
        as these handlers are primarily for logging and clean shutdown.
    """
    try:
        # Handle window close event - exit the application cleanly when window is closed
        window.events.closed += lambda: (
            logger.info("Application window closed"),
            sys.exit(0)  # Exit with success code when window is closed normally
        )

        # Handle window shown event - log when the window is first displayed
        window.events.shown += lambda: logger.info("Application window shown")

        # Handle window loaded event - log when the window content is fully loaded
        window.events.loaded += lambda: logger.info("Application window loaded")

        logger.info("Window event handlers set up successfully")
    except Exception as e:
        # Log the error but continue - the application can work without these handlers
        logger.error(f"Error setting up window event handlers: {e}")
        # Continue anyway - the application will still work without event handlers

def main():
    """
    Main application entry point.

    This function:
    1. Determines the application path (handles both development and bundled modes)
    2. Creates necessary directories for offline operation
    3. Starts the Flask server in a background thread
    4. Waits for the server to be ready
    5. Creates the PyWebView window to display the UI
    6. Sets up window event handlers
    7. Starts the PyWebView event loop

    The function handles various edge cases and provides fallbacks for error conditions.
    It works across different operating systems (Windows, macOS, Linux) by using
    platform-independent path handling and appropriate fallback mechanisms.

    The application is designed to work fully offline, with all resources
    stored locally. No internet connection is required for operation.
    """
    try:
        # Determine if we're running in a bundled app (e.g., PyInstaller) or in development mode
        # This handles different operating systems (Windows, macOS, Linux)
        if getattr(sys, 'frozen', False):
            # If we're running in a bundled app, use the directory the executable is in
            # This ensures resources are found relative to the executable in bundled mode
            application_path = os.path.dirname(sys.executable)
        else:
            # If we're running in a normal Python environment, use the script's directory
            # This ensures resources are found relative to this script in development mode
            application_path = os.path.dirname(os.path.abspath(__file__))

        logger.info(f"Application path: {application_path}")
        logger.info(f"Operating system: {os.name} - {sys.platform}")

        # Create data directory if it doesn't exist
        # This directory stores application data like models, conversation history, etc.
        data_dir = os.path.join(application_path, 'data')
        os.makedirs(data_dir, exist_ok=True)

        # Create platform-specific directories if needed
        # Different operating systems may require different paths
        if sys.platform == 'win32':
            # Windows-specific setup
            logger.info("Setting up for Windows environment")
            # Ensure Windows temp directory exists and is writable
            win_temp = os.path.join(data_dir, 'temp')
            os.makedirs(win_temp, exist_ok=True)
            # Set environment variable for temporary files
            os.environ['TEMP'] = win_temp
        elif sys.platform == 'darwin':
            # macOS-specific setup
            logger.info("Setting up for macOS environment")
            # Create macOS cache directory
            macos_cache = os.path.join(data_dir, 'cache')
            os.makedirs(macos_cache, exist_ok=True)
        elif sys.platform.startswith('linux'):
            # Linux-specific setup
            logger.info("Setting up for Linux environment")
            # Create Linux config directory
            linux_config = os.path.join(data_dir, 'config')
            os.makedirs(linux_config, exist_ok=True)

        # Start the Flask server in a separate daemon thread
        # Using a daemon thread ensures the server will be terminated when the main thread exits
        server_thread = threading.Thread(target=start_server)
        server_thread.daemon = True  # Mark as daemon so it terminates when main thread exits
        server_thread.start()  # Start the server thread

        # Import additional modules needed for server readiness check
        import time
        import socket

        # Wait for the server to be ready by attempting to connect to it
        max_retries = 10  # Maximum number of connection attempts
        retry_count = 0  # Current attempt counter
        server_ready = False  # Flag to indicate if server is ready
        retry_delay = 0.5  # Delay between retry attempts in seconds

        logger.info("Waiting for server to be ready...")

        # Try to connect to the server multiple times until it's ready or max retries reached
        while retry_count < max_retries and not server_ready:
            try:
                # Try to establish a TCP connection to the server
                # If successful, the server is accepting connections and likely ready
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect(('127.0.0.1', 5000))  # Connect to localhost port 5000
                    server_ready = True  # Mark server as ready if connection succeeds
                    logger.info("Server is ready")
            except socket.error:
                # Connection failed, server not ready yet
                retry_count += 1
                logger.info(f"Waiting for server to start (attempt {retry_count}/{max_retries})...")
                time.sleep(retry_delay)  # Wait before trying again

        # If we couldn't connect to the server after all retries, log a warning but continue
        # The application might still work if the server starts later
        if not server_ready:
            logger.warning("Server may not be ready, but continuing anyway...")

        # Add an additional delay to ensure the server is fully initialized
        # This helps prevent race conditions where the server is accepting connections
        # but not yet fully ready to handle requests
        time.sleep(1)

        # Try different approaches to create the window
        # This section handles the creation of the PyWebView window with appropriate error handling
        try:
            # First, verify that the index.html file exists by checking possible locations
            # The UI can be served from either the dist directory (after build) or server/static
            dist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dist')
            server_static = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'server', 'static')

            # Check both possible locations for index.html
            # The application can work with either location depending on how it was built
            index_paths = [
                os.path.join(dist_dir, 'index.html'),  # Primary location after build
                os.path.join(server_static, 'index.html')  # Alternative location
            ]

            # Check if index.html exists in any of the expected locations
            index_exists = False
            for path in index_paths:
                if os.path.exists(path):
                    logger.info(f"Found index.html at: {path}")
                    index_exists = True
                    break

            # Handle the case where index.html is not found
            if not index_exists:
                logger.error("index.html not found in any expected location")
                # Create a simple HTML error page to display to the user
                # This provides a clear message about what went wrong and how to fix it
                error_html = """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Error</title>
                    <style>
                        body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
                        h1 { color: red; }
                    </style>
                </head>
                <body>
                    <h1>Error: index.html not found</h1>
                    <p>The application could not find the required files.</p>
                    <p>Please run 'python build.py' to rebuild the application.</p>
                </body>
                </html>
                """
                # Create a window with the error message
                # This ensures the user sees something rather than the application silently failing
                window = webview.create_window(
                    title="Jarvis AI Assistant - Error",
                    html=error_html,  # Use the error HTML content
                    width=800,
                    height=600,
                    resizable=True,
                    text_select=True,  # Allow text selection for copying error messages
                    confirm_close=False,  # Don't ask for confirmation when closing error window
                    background_color='#ffffff'  # White background
                )
            else:
                # If index.html exists, create the main application window
                logger.info("Creating window with root URL")

                # Set platform-specific window options
                window_options = {
                    "title": "Jarvis AI Assistant",  # Window title
                    "url": "http://127.0.0.1:5000/",  # URL to the Flask server
                    "text_select": True,  # Allow text selection in the window
                    "confirm_close": False,  # Don't ask for confirmation when closing
                }

                # Platform-specific window configurations
                if sys.platform == 'win32':
                    # Windows-specific window settings
                    logger.info("Using Windows-optimized window configuration")
                    window_options.update({
                        "width": 1200,
                        "height": 800,
                        "min_size": (800, 600),
                        "resizable": True,
                        "background_color": '#ffffff',
                        "easy_drag": True,  # Makes window dragging smoother on Windows
                        "frameless": False,  # Use standard Windows frame
                    })
                elif sys.platform == 'darwin':
                    # macOS-specific window settings
                    logger.info("Using macOS-optimized window configuration")
                    window_options.update({
                        "width": 1200,
                        "height": 800,
                        "min_size": (800, 600),
                        "resizable": True,
                        "background_color": '#ffffff',
                        "frameless": False,  # Use standard macOS frame
                    })
                else:
                    # Linux and other platforms
                    logger.info("Using standard window configuration")
                    window_options.update({
                        "width": 1200,
                        "height": 800,
                        "min_size": (800, 600),
                        "resizable": True,
                        "background_color": '#ffffff',
                    })

                # Create the window with platform-specific options
                window = webview.create_window(**window_options)

                # Add a function to reload the window if it's blank or showing an error
                # This is a fallback mechanism to handle cases where the initial page load fails
                def check_and_reload():
                    """
                    Check if the window loaded correctly and reload if necessary.

                    This function:
                    1. Waits for the window to load initially
                    2. Checks if the page is blank or showing a 404 error
                    3. Tries to load a fallback page if there's an issue
                    4. If the fallback also fails, tries the root URL again

                    This helps recover from transient loading issues that can occur
                    when the server is still initializing.
                    """
                    # Wait for the initial page load attempt to complete
                    time.sleep(2)  # Wait for the window to load

                    logger.info("Checking if window needs to be reloaded")

                    # Execute JavaScript in the window to check the page content
                    # and reload if necessary
                    window.evaluate_js("""
                        // Check if the page is blank or showing an error
                        if (document.body.innerHTML.trim() === '' ||
                            document.body.innerHTML.includes('404') ||
                            document.body.innerHTML.includes('Page not found')) {
                            console.log('Window is blank or showing 404, trying fallback...');

                            // First try the fallback page
                            // The fallback page is a simpler version that might load
                            // when the main page fails
                            window.location.href = 'http://127.0.0.1:5000/fallback';

                            // Set a timeout to try the root URL if fallback doesn't work
                            setTimeout(function() {
                                // Check if the fallback page also failed
                                if (document.body.innerHTML.includes('404') ||
                                    document.body.innerHTML.includes('Page not found')) {
                                    console.log('Fallback failed, trying root URL...');
                                    // Try the root URL again as a last resort
                                    window.location.href = 'http://127.0.0.1:5000/';
                                }
                            }, 2000);  // Wait 2 seconds before checking fallback
                        } else {
                            console.log('Window loaded successfully');
                        }
                    """)

                # Start the check in a separate daemon thread
                # Using a daemon thread ensures it won't prevent application exit
                reload_thread = threading.Thread(target=check_and_reload)
                reload_thread.daemon = True  # Mark as daemon so it won't prevent application exit
                reload_thread.start()  # Start the thread

        except Exception as e:
            # Handle any exceptions that occur during window creation
            logger.error(f"Error creating window: {e}")
            sys.exit(1)  # Exit with error code

        # Set up window event handlers for logging and clean shutdown
        setup_window_events(window)

        # Configure the window for offline operation
        # This is done in a separate function to avoid conflicts with other event handlers
        def setup_offline_mode(window):
            """
            Configure the window for offline operation across all platforms.

            This function:
            1. Overrides the service worker registration to make the app work offline
            2. Prevents unnecessary network requests for service worker registration
            3. Applies platform-specific optimizations for offline functionality

            Args:
                window: The PyWebView window object to configure
            """
            try:
                # Platform-specific offline mode configurations
                if sys.platform == 'win32':
                    # Windows-specific offline configuration
                    logger.info("Setting up Windows-optimized offline mode")
                    # Execute JavaScript to override service worker registration
                    window.evaluate_js(
                        """
                        // Mark as offline-capable application by providing a mock service worker registration
                        if ('serviceWorker' in navigator) {
                            // Override the service worker registration function with a mock that always succeeds
                            navigator.serviceWorker.register = function() {
                                return new Promise((resolve) => {
                                    resolve({scope: '/'});  // Return a mock registration object
                                });
                            };
                        }

                        // Windows-specific: Disable fetch requests to external domains
                        const originalFetch = window.fetch;
                        window.fetch = function(url, options) {
                            if (url.toString().startsWith('http://127.0.0.1') || url.toString().startsWith('/')) {
                                return originalFetch(url, options);
                            } else {
                                console.log('Blocked external fetch to: ' + url);
                                return Promise.resolve(new Response('', {status: 200}));
                            }
                        };
                        """
                    )
                elif sys.platform == 'darwin':
                    # macOS-specific offline configuration
                    logger.info("Setting up macOS-optimized offline mode")
                    window.evaluate_js(
                        """
                        // Mark as offline-capable application by providing a mock service worker registration
                        if ('serviceWorker' in navigator) {
                            // Override the service worker registration function with a mock that always succeeds
                            navigator.serviceWorker.register = function() {
                                return new Promise((resolve) => {
                                    resolve({scope: '/'});  // Return a mock registration object
                                });
                            };
                        }

                        // macOS-specific: Set application cache mode
                        if ('applicationCache' in window) {
                            window.applicationCache.oncached = function() {
                                console.log('Application cached successfully');
                            };
                        }
                        """
                    )
                else:
                    # Linux and other platforms
                    logger.info("Setting up standard offline mode")
                    window.evaluate_js(
                        """
                        // Mark as offline-capable application by providing a mock service worker registration
                        if ('serviceWorker' in navigator) {
                            // Override the service worker registration function with a mock that always succeeds
                            navigator.serviceWorker.register = function() {
                                return new Promise((resolve) => {
                                    resolve({scope: '/'});  // Return a mock registration object
                                });
                            };
                        }
                        """
                    )

                logger.info("Offline mode configured successfully")
            except Exception as e:
                # Log the error but continue - this is not critical functionality
                logger.error(f"Error configuring offline mode: {e}")

        # Add a handler to run our offline mode setup function when the window is fully loaded
        # This ensures the JavaScript environment is ready before we try to modify it
        window.events.loaded += lambda: setup_offline_mode(window)

        # Start the PyWebView event loop
        # This is the main application loop that processes window events and keeps the UI responsive
        logger.info("Starting PyWebView event loop")
        webview.start(debug=False)  # Start with debug=False for production use
    except Exception as e:
        # Handle any uncaught exceptions in the main function
        logger.error(f"Error in main application: {e}")
        sys.exit(1)  # Exit with error code

# Standard Python idiom to ensure main() is only called when this script is run directly
if __name__ == '__main__':
    main()  # Start the application
