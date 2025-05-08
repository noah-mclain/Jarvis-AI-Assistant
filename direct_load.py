import webview
import os
import threading
import sys
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

def main():
    """Main application entry point"""
    try:
        # Start the server in a separate thread
        server_thread = threading.Thread(target=start_server)
        server_thread.daemon = True
        server_thread.start()

        # Give the server a moment to start
        import time
        time.sleep(1)

        # Get the path to the index.html file
        dist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dist')
        index_path = os.path.join(dist_dir, 'index.html')

        if os.path.exists(index_path):
            # Read the index.html file
            with open(index_path, 'r') as f:
                html_content = f.read()

            # Create a window with PyWebView using the HTML content directly
            window = webview.create_window(
                title="Jarvis AI Assistant (Direct Load)",
                html=html_content,
                width=1200,
                height=800,
                min_size=(800, 600),
                resizable=True,
                text_select=True,
                confirm_close=False,
                background_color='#ffffff'
            )

            # Start the PyWebView event loop
            logger.info("Starting PyWebView event loop with direct HTML content")
            webview.start(debug=True)
        else:
            logger.error(f"Index.html not found at {index_path}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error in main application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
