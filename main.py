import webview
import threading
import sys
import os
import logging
from waitress import serve
from server.app import app

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('jarvis_app.log')
    ]
)
logger = logging.getLogger('jarvis_desktop')

def start_server():
    """Start the server in a separate thread"""
    try:
        # Use waitress for production-ready serving
        logger.info("Starting Jarvis AI Assistant server on http://127.0.0.1:5000")
        serve(app, host='127.0.0.1', port=5000, threads=4)
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        sys.exit(1)

def setup_window_events(window):
    """Set up window event handlers"""
    try:
        # Handle window close event
        window.events.closed += lambda: (
            logger.info("Application window closed"),
            sys.exit(0)
        )

        # Handle window shown event
        window.events.shown += lambda: logger.info("Application window shown")

        # Handle window loaded event
        window.events.loaded += lambda: logger.info("Application window loaded")

        logger.info("Window event handlers set up successfully")
    except Exception as e:
        logger.error(f"Error setting up window event handlers: {e}")
        # Continue anyway - the application will still work without event handlers

def main():
    """Main application entry point"""
    try:
        # Determine if we're running in a bundled app or in development
        if getattr(sys, 'frozen', False):
            # If we're running in a bundled app, use the directory the executable is in
            application_path = os.path.dirname(sys.executable)
        else:
            # If we're running in a normal Python environment, use the script's directory
            application_path = os.path.dirname(os.path.abspath(__file__))

        logger.info(f"Application path: {application_path}")

        # Create data directory if it doesn't exist
        data_dir = os.path.join(application_path, 'data')
        os.makedirs(data_dir, exist_ok=True)

        # Start the server in a separate thread
        server_thread = threading.Thread(target=start_server)
        server_thread.daemon = True
        server_thread.start()

        # Give the server a moment to start
        import time
        import socket

        # Wait for the server to be ready
        max_retries = 10
        retry_count = 0
        server_ready = False

        logger.info("Waiting for server to be ready...")

        while retry_count < max_retries and not server_ready:
            try:
                # Try to connect to the server
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect(('127.0.0.1', 5000))
                    server_ready = True
                    logger.info("Server is ready")
            except socket.error:
                retry_count += 1
                logger.info(f"Waiting for server to start (attempt {retry_count}/{max_retries})...")
                time.sleep(0.5)

        if not server_ready:
            logger.warning("Server may not be ready, but continuing anyway...")

        # Create a window with PyWebView
        # Add a delay to ensure the server is fully ready
        time.sleep(1)

        # Try different approaches to create the window
        try:
            # First, verify that the index.html file exists
            dist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dist')
            server_static = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'server', 'static')

            # Check both possible locations for index.html
            index_paths = [
                os.path.join(dist_dir, 'index.html'),
                os.path.join(server_static, 'index.html')
            ]

            index_exists = False
            for path in index_paths:
                if os.path.exists(path):
                    logger.info(f"Found index.html at: {path}")
                    index_exists = True
                    break

            if not index_exists:
                logger.error("index.html not found in any expected location")
                # Create a simple HTML file to display in case of error
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
                window = webview.create_window(
                    title="Jarvis AI Assistant - Error",
                    html=error_html,
                    width=800,
                    height=600,
                    resizable=True,
                    text_select=True,
                    confirm_close=False,
                    background_color='#ffffff'
                )
            else:
                # Create a window with the root URL
                logger.info("Creating window with root URL")
                window = webview.create_window(
                    title="Jarvis AI Assistant",
                    url="http://127.0.0.1:5000/",
                    width=1200,
                    height=800,
                    min_size=(800, 600),
                    resizable=True,
                    text_select=True,
                    confirm_close=False,
                    background_color='#ffffff'
                )

                # Add a function to reload the window if it's blank
                def check_and_reload():
                    time.sleep(2)  # Wait for the window to load
                    logger.info("Checking if window needs to be reloaded")
                    window.evaluate_js("""
                        if (document.body.innerHTML.trim() === '' ||
                            document.body.innerHTML.includes('404') ||
                            document.body.innerHTML.includes('Page not found')) {
                            console.log('Window is blank or showing 404, trying fallback...');
                            // First try the fallback page
                            window.location.href = 'http://127.0.0.1:5000/fallback';

                            // Set a timeout to try the root URL if fallback doesn't work
                            setTimeout(function() {
                                if (document.body.innerHTML.includes('404') ||
                                    document.body.innerHTML.includes('Page not found')) {
                                    console.log('Fallback failed, trying root URL...');
                                    window.location.href = 'http://127.0.0.1:5000/';
                                }
                            }, 2000);
                        } else {
                            console.log('Window loaded successfully');
                        }
                    """)

                # Start the check in a separate thread
                reload_thread = threading.Thread(target=check_and_reload)
                reload_thread.daemon = True
                reload_thread.start()

        except Exception as e:
            logger.error(f"Error creating window: {e}")
            sys.exit(1)

        # Set up window event handlers
        setup_window_events(window)

        # Set window attributes to indicate this is a local application
        # We'll do this in a separate function to avoid conflicts with the event handler
        def setup_offline_mode(window):
            try:
                window.evaluate_js(
                    """
                    // Mark as offline-capable application
                    if ('serviceWorker' in navigator) {
                        navigator.serviceWorker.register = function() {
                            return new Promise((resolve) => {
                                resolve({scope: '/'});
                            });
                        };
                    }
                    """
                )
                logger.info("Offline mode configured successfully")
            except Exception as e:
                logger.error(f"Error configuring offline mode: {e}")

        # Add a handler to run our setup function when the window is loaded
        window.events.loaded += lambda: setup_offline_mode(window)

        # Start the PyWebView event loop
        logger.info("Starting PyWebView event loop")
        webview.start(debug=False)
    except Exception as e:
        logger.error(f"Error in main application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
