import webview
import os
import sys
import logging
import threading
import time
import socket
from flask import Flask, send_from_directory

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('alternative_app.log')
    ]
)
logger = logging.getLogger('alternative_app')

# Create a simple Flask app that just serves static files
app = Flask(__name__)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_static(path):
    dist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dist')
    if path != "" and os.path.exists(os.path.join(dist_dir, path)):
        return send_from_directory(dist_dir, path)
    else:
        return send_from_directory(dist_dir, 'index.html')

def start_server():
    """Start the Flask server"""
    try:
        app.run(host='127.0.0.1', port=5001, threaded=True)
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

        # Wait for the server to be ready
        max_retries = 10
        retry_count = 0
        server_ready = False

        logger.info("Waiting for server to be ready...")

        while retry_count < max_retries and not server_ready:
            try:
                # Try to connect to the server
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect(('127.0.0.1', 5001))
                    server_ready = True
                    logger.info("Server is ready")
            except socket.error:
                retry_count += 1
                logger.info(f"Waiting for server to start (attempt {retry_count}/{max_retries})...")
                time.sleep(0.5)

        if not server_ready:
            logger.warning("Server may not be ready, but continuing anyway...")

        # Create a window with PyWebView
        window = webview.create_window(
            title="Jarvis AI Assistant (Alternative)",
            url="http://127.0.0.1:5001",
            width=1200,
            height=800,
            min_size=(800, 600),
            resizable=True,
            text_select=True,
            confirm_close=False,
            background_color='#ffffff'
        )

        # Start the PyWebView event loop
        logger.info("Starting PyWebView event loop")
        webview.start(debug=True)
    except Exception as e:
        logger.error(f"Error in main application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
