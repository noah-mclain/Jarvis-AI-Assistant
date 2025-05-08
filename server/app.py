from flask import Flask, request, jsonify, send_from_directory
import os
import time
import json
import logging
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('jarvis_app.log')
    ]
)
logger = logging.getLogger('jarvis_app')

# Create Flask app
# Try to use the static folder if it exists, otherwise fall back to dist
app_dir = Path(os.path.dirname(os.path.abspath(__file__)))
static_folder = app_dir / 'static'
dist_folder = app_dir.parent / 'dist'

if static_folder.exists() and static_folder.is_dir():
    app = Flask(__name__, static_folder=str(static_folder))
    logger.info(f"Using static folder: {static_folder}")
elif dist_folder.exists() and dist_folder.is_dir():
    app = Flask(__name__, static_folder=str(dist_folder))
    logger.info(f"Using dist folder: {dist_folder}")
else:
    # If neither exists, create the static folder and use it
    static_folder.mkdir(exist_ok=True)
    app = Flask(__name__, static_folder=str(static_folder))
    logger.info(f"Created and using static folder: {static_folder}")

# Get the application root directory
root_dir = app_dir.parent

# Create data directory if it doesn't exist
data_dir = root_dir / 'data'
data_dir.mkdir(exist_ok=True)
chats_file = data_dir / 'chats.json'
messages_file = data_dir / 'messages.json'

logger.info(f"Using data directory: {data_dir}")
logger.info(f"Chats file: {chats_file}")
logger.info(f"Messages file: {messages_file}")

# Jarvis AI integration
# Initialize the Jarvis AI assistant
try:
    from server.jarvis.assistant import JarvisAI
    jarvis_ai = JarvisAI()
    logger.info("Jarvis AI assistant initialized successfully")
    JARVIS_AVAILABLE = True
except Exception as e:
    logger.error(f"Error initializing Jarvis AI assistant: {e}")
    logger.warning("Using fallback response generation")
    jarvis_ai = None
    JARVIS_AVAILABLE = False

def get_jarvis_response(message_text):
    """
    This function integrates with the Jarvis AI assistant.
    It takes a message text and returns a response from the AI assistant.

    Args:
        message_text (str): The user's message

    Returns:
        str: The AI assistant's response
    """
    try:
        if JARVIS_AVAILABLE and jarvis_ai:
            # Use the actual Jarvis AI assistant
            logger.info(f"Sending query to Jarvis AI: {message_text[:50]}...")
            response = jarvis_ai.process_query(message_text)
            logger.info(f"Received response from Jarvis AI: {response[:50]}...")
            return response
        else:
            # Fallback response if Jarvis AI is not available
            logger.warning(f"Using fallback response for query: {message_text[:50]}...")
            return f"I've processed your request: \"{message_text}\"\n\nThis is a simulated response. The Jarvis AI integration is not available."
    except Exception as e:
        logger.error(f"Error getting response from Jarvis AI: {e}")
        return "I'm sorry, I encountered an error processing your request. Please try again."

# Load or initialize chats and messages data
def load_data():
    global chats, messages

    # Default data if files don't exist
    default_chats = [
        {"id": "1", "title": "Introduction to Jarvis", "date": datetime.now().strftime("%b %d, %Y")},
        {"id": "2", "title": "Help with coding", "date": datetime.now().strftime("%b %d, %Y")},
        {"id": "3", "title": "Weather and news", "date": datetime.now().strftime("%b %d, %Y")},
    ]

    default_messages = {
        "1": [
            {
                "id": "1",
                "role": "assistant",
                "content": "Hello! I'm Jarvis, your AI assistant. How can I help you today?",
                "timestamp": datetime.now().strftime("%I:%M %p"),
            }
        ],
        "2": [
            {
                "id": "1",
                "role": "assistant",
                "content": "I can help you with coding questions. What language are you working with?",
                "timestamp": datetime.now().strftime("%I:%M %p"),
            }
        ],
        "3": [
            {
                "id": "1",
                "role": "assistant",
                "content": "I can provide weather updates and news summaries. What would you like to know?",
                "timestamp": datetime.now().strftime("%I:%M %p"),
            }
        ]
    }

    # Try to load data from files
    try:
        if chats_file.exists():
            with open(chats_file, 'r') as f:
                chats = json.load(f)
            logger.info(f"Loaded {len(chats)} chats from {chats_file}")
        else:
            chats = default_chats
            logger.info(f"Initialized with {len(chats)} default chats")

        if messages_file.exists():
            with open(messages_file, 'r') as f:
                messages = json.load(f)
            logger.info(f"Loaded messages for {len(messages)} chats from {messages_file}")
        else:
            messages = default_messages
            logger.info(f"Initialized with default messages for {len(messages)} chats")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        chats = default_chats
        messages = default_messages

# Save data to files
def save_data():
    try:
        with open(chats_file, 'w') as f:
            json.dump(chats, f, indent=2)
        with open(messages_file, 'w') as f:
            json.dump(messages, f, indent=2)
        logger.info("Data saved successfully")
    except Exception as e:
        logger.error(f"Error saving data: {e}")

# Initialize data
chats = []
messages = {}
load_data()

# API Routes
@app.route('/api/chats', methods=['GET'])
def get_chats():
    try:
        return jsonify(chats)
    except Exception as e:
        logger.error(f"Error getting chats: {e}")
        return jsonify({"error": "Failed to get chats"}), 500

@app.route('/api/chats/<chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    try:
        # Find the chat to delete
        chat_to_delete = None
        for i, chat in enumerate(chats):
            if chat['id'] == chat_id:
                chat_to_delete = chats.pop(i)
                break

        if not chat_to_delete:
            return jsonify({"error": "Chat not found"}), 404

        # Remove messages for this chat
        if chat_id in messages:
            del messages[chat_id]

        # Save data to persistent storage
        save_data()

        logger.info(f"Deleted chat: {chat_id}")
        return jsonify({"success": True, "message": "Chat deleted successfully"})
    except Exception as e:
        logger.error(f"Error deleting chat {chat_id}: {e}")
        return jsonify({"error": "Failed to delete chat"}), 500

@app.route('/api/chats', methods=['POST'])
def create_chat():
    try:
        new_chat = {
            "id": f"new-chat-{int(time.time())}",
            "title": "New Conversation",
            "date": datetime.now().strftime("%b %d, %Y")
        }
        chats.append(new_chat)
        messages[new_chat["id"]] = [
            {
                "id": "welcome",
                "role": "assistant",
                "content": "How can I help you with this new conversation?",
                "timestamp": datetime.now().strftime("%I:%M %p"),
            }
        ]

        # Save data to persistent storage
        save_data()

        logger.info(f"Created new chat: {new_chat['id']}")
        return jsonify(new_chat)
    except Exception as e:
        logger.error(f"Error creating chat: {e}")
        return jsonify({"error": "Failed to create chat"}), 500

@app.route('/api/chats/<chat_id>/messages', methods=['GET'])
def get_messages(chat_id):
    try:
        chat_messages = messages.get(chat_id, [])
        return jsonify(chat_messages)
    except Exception as e:
        logger.error(f"Error getting messages for chat {chat_id}: {e}")
        return jsonify({"error": "Failed to get messages"}), 500

@app.route('/api/chats/<chat_id>/messages', methods=['POST'])
def send_message(chat_id):
    try:
        data = request.json
        message_content = data.get('content', '')

        if not message_content.strip():
            return jsonify({"error": "Message content cannot be empty"}), 400

        # Create user message
        user_message = {
            "id": f"msg-{int(time.time())}",
            "role": "user",
            "content": message_content,
            "timestamp": datetime.now().strftime("%I:%M %p"),
        }

        # Add user message to chat
        if chat_id not in messages:
            messages[chat_id] = []
        messages[chat_id].append(user_message)

        # Update chat title if it's a new chat
        chat_title_updated = False
        new_title = None

        if any(chat["id"] == chat_id and chat["title"] == "New Conversation" for chat in chats):
            new_title = message_content[:30] + "..." if len(message_content) > 30 else message_content
            for chat in chats:
                if chat["id"] == chat_id:
                    chat["title"] = new_title
                    chat_title_updated = True
                    break

        # Get response from Jarvis AI
        try:
            ai_response_content = get_jarvis_response(message_content)
        except Exception as e:
            logger.error(f"Error getting AI response: {e}")
            ai_response_content = "I'm sorry, I encountered an error processing your request. Please try again."

        # Create AI response message
        ai_response = {
            "id": f"msg-{int(time.time()) + 1}",
            "role": "assistant",
            "content": ai_response_content,
            "timestamp": datetime.now().strftime("%I:%M %p"),
        }

        # Add AI response to chat
        messages[chat_id].append(ai_response)

        # Save data to persistent storage
        save_data()

        logger.info(f"Processed message in chat {chat_id}")
        return jsonify({
            "userMessage": user_message,
            "aiResponse": ai_response,
            "chatTitleUpdated": chat_title_updated,
            "newTitle": new_title
        })
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return jsonify({"error": "Failed to process message"}), 500

# Serve the static files from the React app
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    try:
        logger.info(f"Serving path: {path}")

        # If path is empty, serve index.html
        if path == "":
            logger.info("Serving index.html for root path")
            return send_from_directory(app.static_folder, 'index.html')

        # Special case for assets directory
        if path.startswith('assets/') or path.startswith('assets\\'):
            asset_path = path[7:]
            asset_dir = Path(app.static_folder) / 'assets'
            logger.info(f"Serving asset: {asset_path} from {asset_dir}")
            asset_full_path = asset_dir / asset_path
            if asset_full_path.exists():
                return send_from_directory(str(asset_dir), asset_path)
            else:
                logger.error(f"Asset not found: {asset_path}")

        # Check if the path exists as a file
        full_path = Path(app.static_folder) / path
        if full_path.exists() and full_path.is_file():
            logger.info(f"Serving file: {path}")
            return send_from_directory(app.static_folder, path)

        # For all other paths, serve index.html (SPA routing)
        logger.info(f"Path {path} not found, serving index.html (SPA routing)")
        return send_from_directory(app.static_folder, 'index.html')
    except Exception as e:
        logger.error(f"Error serving path {path}: {e}")
        # Return a custom 404 page
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Page Not Found</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
                h1 { font-size: 36px; margin-bottom: 20px; }
                p { font-size: 18px; margin-bottom: 20px; }
                a { color: #0066cc; text-decoration: none; }
                a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <h1>404 - Page Not Found</h1>
            <p>The page you are looking for does not exist.</p>
            <p><a href="/">Return to Home</a></p>
            <p><small>Error details: """ + str(e) + """</small></p>
        </body>
        </html>
        """, 404

# Add a specific route for the index.html file
@app.route('/index.html')
def serve_index():
    try:
        logger.info("Serving index.html directly")
        index_path = Path(app.static_folder) / 'index.html'
        if index_path.exists():
            return send_from_directory(app.static_folder, 'index.html')
        else:
            logger.error(f"index.html not found at {index_path}")
            # Try to serve the fallback.html file
            fallback_path = Path(app.static_folder) / 'fallback.html'
            if fallback_path.exists():
                logger.info(f"Serving fallback.html instead")
                return send_from_directory(app.static_folder, 'fallback.html')
            else:
                return "index.html not found", 404
    except Exception as e:
        logger.error(f"Error serving index.html: {e}")
        return f"Error: {str(e)}", 500

@app.route('/fallback')
def serve_fallback():
    """Serve the fallback page"""
    try:
        logger.info("Serving fallback.html")
        fallback_path = Path(app.static_folder) / 'fallback.html'
        if fallback_path.exists():
            return send_from_directory(app.static_folder, 'fallback.html')
        else:
            logger.error(f"fallback.html not found at {fallback_path}")
            return "Fallback page not found", 404
    except Exception as e:
        logger.error(f"Error serving fallback.html: {e}")
        return f"Error: {str(e)}", 500

# Add a route for handling 404 errors
@app.errorhandler(404)
def page_not_found(e):
    logger.error(f"404 error: {e}")
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Page Not Found</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
            h1 { font-size: 36px; margin-bottom: 20px; }
            p { font-size: 18px; margin-bottom: 20px; }
            a { color: #0066cc; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <h1>404 - Page Not Found</h1>
        <p>The page you are looking for does not exist.</p>
        <p><a href="/">Return to Home</a></p>
    </body>
    </html>
    """, 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)
