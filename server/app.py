"""
Jarvis AI Assistant - Flask Server Application

This module implements the backend server for the Jarvis AI Assistant application.
It provides a RESTful API for the frontend to interact with the Jarvis AI system,
as well as serving the static files for the web interface.

The server handles:
1. Chat management (creating, retrieving, and deleting conversations)
2. Message processing (sending user messages to Jarvis AI and returning responses)
3. Static file serving for the web interface
4. Persistent storage of conversations and messages

Author: Nada Mohamed
License: MIT
"""

from flask import Flask, request, jsonify, send_from_directory  # Flask web framework
import os  # Operating system utilities
import time  # Time utilities for timestamps
import json  # JSON handling for data storage
import logging  # Logging utilities
from datetime import datetime  # Date and time handling
from pathlib import Path  # Object-oriented filesystem paths

# Set up logging configuration for the server application
# This configures both console output and file logging
logging.basicConfig(
    level=logging.INFO,  # Set the minimum log level to INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Define log format with timestamp
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler('jarvis_app.log')  # Log to file
    ]
)
logger = logging.getLogger('jarvis_app')  # Create a logger for this module

# Create Flask app with appropriate static file serving configuration
# The application can serve static files from either the server/static directory
# or the dist directory (created during the build process)
app_dir = Path(os.path.dirname(os.path.abspath(__file__)))  # Get the server directory path
static_folder = app_dir / 'static'  # Path to server/static directory
dist_folder = app_dir.parent / 'dist'  # Path to dist directory (created by build process)

# Determine which static folder to use based on what exists
# Priority: 1. server/static if it exists, 2. dist if it exists, 3. create server/static
if static_folder.is_dir():
    # Use the server/static directory if it exists
    app = Flask(__name__, static_folder=str(static_folder))
    logger.info(f"Using static folder: {static_folder}")
elif dist_folder.is_dir():
    # Fall back to the dist directory if server/static doesn't exist
    app = Flask(__name__, static_folder=str(dist_folder))
    logger.info(f"Using dist folder: {dist_folder}")
else:
    # If neither exists, create the static folder and use it
    # This ensures the application can always start, even without a build
    static_folder.mkdir(exist_ok=True)
    app = Flask(__name__, static_folder=str(static_folder))
    logger.info(f"Created and using static folder: {static_folder}")

# Get the application root directory (one level up from the server directory)
root_dir = app_dir.parent

# Set up data storage directories and files
# The application stores chat history and messages in JSON files
data_dir = root_dir / 'data'  # Path to data directory
data_dir.mkdir(exist_ok=True)  # Create the data directory if it doesn't exist
chats_file = data_dir / 'chats.json'  # File to store chat metadata
messages_file = data_dir / 'messages.json'  # File to store chat messages

# Log the data storage locations for debugging
logger.info(f"Using data directory: {data_dir}")
logger.info(f"Chats file: {chats_file}")
logger.info(f"Messages file: {messages_file}")

# Jarvis AI Integration
# This section handles the integration with the Jarvis AI assistant module
# It attempts to import and initialize the JarvisAI class from server/jarvis/assistant.py
# If the import fails or initialization fails, it falls back to a simulated response mode
try:
    # Import the JarvisAI class from the assistant module
    from server.jarvis.assistant import JarvisAI

    # Initialize the Jarvis AI assistant
    # This will load any necessary models and prepare the assistant for processing queries
    global jarvis_ai

    jarvis_ai = JarvisAI()
    logger.info("Jarvis AI assistant initialized successfully")
    JARVIS_AVAILABLE = True  # Flag to indicate that Jarvis AI is available
except Exception as e:
    # Handle any errors during import or initialization
    logger.error(f"Error initializing Jarvis AI assistant: {e}")
    logger.warning("Using fallback response generation")
    jarvis_ai = None  # Set to None to indicate that Jarvis AI is not available
    JARVIS_AVAILABLE = False  # Flag to indicate that Jarvis AI is not available

def get_jarvis_response(message_text, model_type=None):
    """
    Process a user message through the Jarvis AI assistant and get a response.

    This function serves as the integration point between the web API and the
    Jarvis AI assistant. It handles:
    1. Sending the user's message to the Jarvis AI assistant
    2. Getting the response from the assistant
    3. Providing a fallback response if the assistant is not available
    4. Error handling for any issues during processing

    Args:
        message_text (str): The user's message text to process
        model_type (str, optional): The type of model to use. If None, the current model type will be used.

    Returns:
        str: The AI assistant's response text, or a fallback response if the
             assistant is not available or encounters an error
    """
    try:
        # Check if Jarvis AI is available and initialized
        if JARVIS_AVAILABLE and jarvis_ai:
            # Use the actual Jarvis AI assistant to process the query
            # Log only the first 50 characters of the message for brevity
            logger.info(f"Sending query to Jarvis AI: {message_text[:50]}...")

            # Process the query through the Jarvis AI assistant
            response = jarvis_ai.process_query(message_text, model_type)

            # Log the response (first 50 characters) for debugging
            logger.info(f"Received response from Jarvis AI: {response[:50]}...")

            return response
        else:
            # Provide a fallback response if Jarvis AI is not available
            # This ensures the application can still function without the AI component
            logger.warning(f"Using fallback response for query: {message_text[:50]}...")
            return f"I've processed your request: \"{message_text}\"\n\nThis is a simulated response. The Jarvis AI integration is not available."
    except Exception as e:
        # Handle any errors that occur during processing
        logger.error(f"Error getting response from Jarvis AI: {e}")
        return "I'm sorry, I encountered an error processing your request. Please try again."

# Data Management Functions
# These functions handle loading and saving chat data to/from persistent storage

def load_data():
    """
    Load chat and message data from JSON files or initialize with defaults.

    This function:
    1. Attempts to load existing chat and message data from JSON files
    2. Falls back to default sample data if the files don't exist
    3. Handles any errors during loading by using default data

    The function modifies the global chats and messages variables.
    """
    global chats, messages

    # Define default sample data to use if files don't exist
    # This provides a better first-time user experience than empty chats
    default_chats = [
        {"id": "1", "title": "Introduction to Jarvis", "date": datetime.now().strftime("%b %d, %Y")},
        {"id": "2", "title": "Help with coding", "date": datetime.now().strftime("%b %d, %Y")},
        {"id": "3", "title": "Weather and news", "date": datetime.now().strftime("%b %d, %Y")},
    ]

    # Default messages for each sample chat
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

    # Try to load data from files, with fallback to defaults
    try:
        # Load chats data if the file exists, otherwise use defaults
        if chats_file.exists():
            with open(chats_file, 'r') as f:
                chats = json.load(f)
            logger.info(f"Loaded {len(chats)} chats from {chats_file}")
        else:
            # Use default chats for first-time users
            chats = default_chats
            logger.info(f"Initialized with {len(chats)} default chats")

        # Load messages data if the file exists, otherwise use defaults
        if messages_file.exists():
            with open(messages_file, 'r') as f:
                messages = json.load(f)
            logger.info(f"Loaded messages for {len(messages)} chats from {messages_file}")
        else:
            # Use default messages for first-time users
            messages = default_messages
            logger.info(f"Initialized with default messages for {len(messages)} chats")
    except Exception as e:
        # Handle any errors during loading by using default data
        logger.error(f"Error loading data: {e}")
        chats = default_chats
        messages = default_messages
        logger.info("Using default data due to loading error")

def save_data():
    """
    Save chat and message data to JSON files for persistence.

    This function:
    1. Writes the current chats data to the chats JSON file
    2. Writes the current messages data to the messages JSON file
    3. Handles any errors during saving

    The data is saved with indentation for better readability if manually inspected.
    """
    global chats, messages
    try:
        # Save chats data to JSON file
        with open(chats_file, 'w') as f:
            json.dump(chats, f, indent=2)  # Use indentation for readability

        # Save messages data to JSON file
        with open(messages_file, 'w') as f:
            json.dump(messages, f, indent=2)  # Use indentation for readability

        logger.info("Data saved successfully")
    except Exception as e:
        # Log any errors during saving but continue execution
        # This prevents data saving errors from breaking the application
        logger.error(f"Error saving data: {e}")

# Initialize global data structures
chats = []  # List to store chat metadata
messages = {}  # Dictionary to store messages for each chat
load_data()  # Load data from files or initialize with defaults

# API Routes
# These routes define the REST API endpoints for the frontend to interact with

@app.route('/api/execute', methods=['POST'])
def execute_command():
    """
    API endpoint to execute a command.

    This endpoint:
    1. Receives a command from the frontend
    2. Processes the command through the Jarvis AI assistant
    3. Returns the response

    Request Body:
        JSON object with 'command' field containing the command text
        and optional 'modelType' field specifying the model to use

    Returns:
        JSON response with the command result
    """
    try:
        # Get the command and model type from the request body
        data = request.json
        command_text = data.get('command', '')
        model_type = data.get('modelType', None)

        # Validate that the command is not empty
        if not command_text.strip():
            return jsonify({"error": "Command cannot be empty"}), 400

        # Process the command through the Jarvis AI assistant
        try:
            # Use the same get_jarvis_response function that handles messages
            response_content = get_jarvis_response(command_text, model_type)
        except Exception as e:
            # Handle any errors in AI processing with a fallback response
            logger.error(f"Error processing command: {e}")
            response_content = "I'm sorry, I encountered an error processing your command. Please try again."

        # Return the response
        return jsonify({
            "success": True,
            "result": response_content
        })
    except Exception as e:
        # Log and return any errors that occur
        logger.error(f"Error executing command: {e}")
        return jsonify({"error": "Failed to execute command"}), 500

@app.route('/api/image/', methods=['POST'])
def generate_image():
    """
    API endpoint to generate an image based on a prompt.

    This endpoint:
    1. Receives a prompt from the frontend
    2. Processes the prompt through the Jarvis AI assistant
    3. Returns the generated image

    Request Body:
        JSON object with 'prompt' field containing the image generation prompt

    Returns:
        JSON response with the generated image or error message
    """
    try:
        # Get the prompt from the request body
        data = request.json
        prompt = data.get('prompt', '')

        # Validate that the prompt is not empty
        if not prompt.strip():
            return jsonify({"error": "Prompt cannot be empty"}), 400

        # Process the prompt through the Jarvis AI assistant
        try:
            # Use the same get_jarvis_response function that handles messages
            response_content = get_jarvis_response(prompt)
        except Exception as e:
            # Handle any errors in AI processing with a fallback response
            logger.error(f"Error generating image: {e}")
            response_content = "I'm sorry, I encountered an error generating your image. Please try again."

        # Return the response
        return jsonify({
            "success": True,
            "result": response_content
        })
    except Exception as e:
        # Log and return any errors that occur
        logger.error(f"Error generating image: {e}")
        return jsonify({"error": "Failed to generate image"}), 500

@app.route('/api/speech-to-text', methods=['POST'])
def process_speech():
    """
    API endpoint to process speech-to-text input.

    This endpoint:
    1. Receives audio data or a transcription from the frontend
    2. If it's a transcription, processes it through the Jarvis AI assistant
    3. Returns the response

    Request Body:
        JSON object with 'text' field containing the transcribed text
        and optional 'modelType' field specifying the model to use

    Returns:
        JSON response with the processing result
    """
    try:
        # Get the transcribed text and model type from the request body
        data = request.json
        text = data.get('text', '')
        model_type = data.get('modelType', 'speechToText')  # Default to speechToText model

        # Validate that the text is not empty
        if not text.strip():
            return jsonify({"error": "Transcribed text cannot be empty"}), 400

        # Process the text through the Jarvis AI assistant
        try:
            # Use the same get_jarvis_response function that handles messages
            response_content = get_jarvis_response(text, model_type)
        except Exception as e:
            # Handle any errors in AI processing with a fallback response
            logger.error(f"Error processing speech: {e}")
            response_content = "I'm sorry, I encountered an error processing your speech. Please try again."

        # Return the response
        return jsonify({
            "success": True,
            "result": response_content
        })
    except Exception as e:
        # Log and return any errors that occur
        logger.error(f"Error processing speech: {e}")
        return jsonify({"error": "Failed to process speech"}), 500

@app.route('/api/models/select', methods=['POST'])
def select_model():
    """
    API endpoint to select a specific model type.

    This endpoint:
    1. Receives a model type from the frontend
    2. Sets the current model type in the Jarvis AI assistant
    3. Returns a success or error response

    Request Body:
        JSON object with 'modelType' field containing the model type

    Returns:
        JSON response with success message or error
    """
    try:
        # Check if Jarvis AI is available
        if not JARVIS_AVAILABLE or not jarvis_ai:
            return jsonify({"error": "Jarvis AI is not available"}), 503

        # Get the model type from the request body
        data = request.json
        model_type = data.get('modelType', '')

        # Validate that the model type is not empty
        if not model_type.strip():
            return jsonify({"error": "Model type cannot be empty"}), 400

        # Set the model type in the Jarvis AI assistant
        success = jarvis_ai.set_model_type(model_type)

        if success:
            # Log the model selection and return success response
            logger.info(f"Selected model type: {model_type}")
            return jsonify({"success": True, "message": f"Model type set to {model_type}"})
        else:
            # Return error if the model type is not recognized
            return jsonify({"error": f"Unknown model type: {model_type}"}), 400
    except Exception as e:
        # Log and return any errors that occur
        logger.error(f"Error selecting model type: {e}")
        return jsonify({"error": "Failed to select model type"}), 500

@app.route('/api/chats', methods=['GET'])
def get_chats():
    """
    API endpoint to get all chats.

    This endpoint returns a JSON array of all chat objects, each containing:
    - id: Unique identifier for the chat
    - title: Display title for the chat
    - date: Formatted date when the chat was created

    Returns:
        JSON response with array of chat objects, or error message
    """
    try:
        # Return the list of chats as JSON
        return jsonify(chats)
    except Exception as e:
        # Log and return any errors that occur
        logger.error(f"Error getting chats: {e}")
        return jsonify({"error": "Failed to get chats"}), 500

@app.route('/api/chats/<chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    """
    API endpoint to delete a specific chat.

    This endpoint:
    1. Removes the specified chat from the chats list
    2. Removes all messages associated with the chat
    3. Saves the updated data to persistent storage

    Args:
        chat_id (str): The unique identifier of the chat to delete

    Returns:
        JSON response with success message or error
    """
    try:
        # Find the chat to delete by its ID
        chat_to_delete = None
        for i, chat in enumerate(chats):
            if chat['id'] == chat_id:
                # Remove the chat from the list and store it
                chat_to_delete = chats.pop(i)
                break

        # If the chat wasn't found, return a 404 error
        if not chat_to_delete:
            return jsonify({"error": "Chat not found"}), 404

        # Remove all messages associated with this chat
        if chat_id in messages:
            del messages[chat_id]

        # Save the updated data to persistent storage
        save_data()

        # Log the deletion and return success response
        logger.info(f"Deleted chat: {chat_id}")
        return jsonify({"success": True, "message": "Chat deleted successfully"})
    except Exception as e:
        # Log and return any errors that occur
        logger.error(f"Error deleting chat {chat_id}: {e}")
        return jsonify({"error": "Failed to delete chat"}), 500

@app.route('/api/chats', methods=['POST'])
def create_chat():
    """
    API endpoint to create a new chat.

    This endpoint:
    1. Creates a new chat with a unique ID based on the current timestamp
    2. Initializes the chat with a default title
    3. Creates an initial welcome message from the assistant
    4. Saves the new chat data to persistent storage

    Returns:
        JSON response with the newly created chat object or error
    """
    try:
        # Create a new chat object with a unique ID based on timestamp
        new_chat = {
            "id": f"new-chat-{int(time.time())}",  # Generate unique ID using timestamp
            "title": "New Conversation",  # Default title (will be updated with first message)
            "date": datetime.now().strftime("%b %d, %Y")  # Current date in readable format
        }

        # Add the new chat to the chats list
        chats.append(new_chat)

        # Initialize the chat with a welcome message from the assistant
        messages[new_chat["id"]] = [
            {
                "id": "welcome",  # Special ID for the welcome message
                "role": "assistant",  # Message is from the assistant
                "content": "How can I help you with this new conversation?",  # Welcome message
                "timestamp": datetime.now().strftime("%I:%M %p"),  # Current time
            }
        ]

        # Save the updated data to persistent storage
        save_data()

        # Log the creation and return the new chat object
        logger.info(f"Created new chat: {new_chat['id']}")
        return jsonify(new_chat)
    except Exception as e:
        # Log and return any errors that occur
        logger.error(f"Error creating chat: {e}")
        return jsonify({"error": "Failed to create chat"}), 500

@app.route('/api/chats/<chat_id>/messages', methods=['GET'])
def get_messages(chat_id):
    """
    API endpoint to get all messages for a specific chat.

    This endpoint retrieves all messages associated with the specified chat ID.
    If the chat ID doesn't exist, it returns an empty array rather than an error.

    Args:
        chat_id (str): The unique identifier of the chat

    Returns:
        JSON response with array of message objects or error
    """
    try:
        # Get messages for the specified chat ID, or empty list if not found
        chat_messages = messages.get(chat_id, [])
        return jsonify(chat_messages)
    except Exception as e:
        # Log and return any errors that occur
        logger.error(f"Error getting messages for chat {chat_id}: {e}")
        return jsonify({"error": "Failed to get messages"}), 500

@app.route('/api/chats/<chat_id>/messages', methods=['POST'])
def send_message(chat_id):
    """
    API endpoint to send a message in a specific chat.

    This endpoint:
    1. Receives a user message from the frontend
    2. Adds the user message to the chat
    3. Updates the chat title if it's a new chat
    4. Gets a response from the Jarvis AI assistant
    5. Adds the AI response to the chat
    6. Saves the updated data to persistent storage

    Special handling for 'fallback' chat_id which is used by the fallback.html
    interface when the main UI cannot be loaded.

    Args:
        chat_id (str): The unique identifier of the chat

    Request Body:
        JSON object with 'content' field containing the message text

    Returns:
        JSON response with user message, AI response, and chat title update info
    """
    try:
        # Get the message content and model type from the request body
        data = request.json
        message_content = data.get('content', '')
        model_type = data.get('modelType', None)

        # Validate that the message is not empty
        if not message_content.strip():
            return jsonify({"error": "Message content cannot be empty"}), 400

        # Special handling for fallback mode
        # The fallback.html interface uses a special chat_id 'fallback'
        if chat_id == 'fallback':
            # Create a simple response without storing in chat history
            # This is used when the main UI cannot be loaded
            logger.info("Processing message in fallback mode")

            # Create user message
            user_message = {
                "id": f"fallback-{int(time.time())}",
                "role": "user",
                "content": message_content,
                "timestamp": datetime.now().strftime("%I:%M %p"),
            }

            # Get response from Jarvis AI
            try:
                global jarvis_ai
                jarvis_ai = JarvisAI()

                tg = jarvis_ai.model_handlers['textGeneration']
                tg.is_initialized = False
                tg.chat           = None

                ai_response_content = get_jarvis_response(message_content, model_type)
            except Exception as e:
                logger.error(f"Error getting AI response in fallback mode: {e}")
                ai_response_content = "I'm sorry, I encountered an error processing your request. Please try again."

            # Create AI response message
            ai_response = {
                "id": f"fallback-{int(time.time()) + 1}",
                "role": "assistant",
                "content": ai_response_content,
                "timestamp": datetime.now().strftime("%I:%M %p"),
            }

            # Return the response without saving to persistent storage
            return jsonify({
                "userMessage": user_message,
                "aiResponse": ai_response,
                "chatTitleUpdated": False,
                "newTitle": None
            })

        # Normal processing for regular chats
        # Create a user message object with unique ID and timestamp
        user_message = {
            "id": f"msg-{int(time.time())}",  # Generate unique ID using timestamp
            "role": "user",  # Message is from the user
            "content": message_content,  # The actual message text
            "timestamp": datetime.now().strftime("%I:%M %p"),  # Current time
        }

        # Add user message to the chat, creating the message array if needed
        if chat_id not in messages:
            messages[chat_id] = []
        messages[chat_id].append(user_message)

        # Update chat title if it's a new chat (still has default title)
        # This uses the first user message as the chat title
        chat_title_updated = False
        new_title = None

        # Check if this chat has the default title "New Conversation"
        if any(chat["id"] == chat_id and chat["title"] == "New Conversation" for chat in chats):
            # Create a title from the first 30 chars of the message
            new_title = f"{message_content[:30]}..." if len(message_content) > 30 else message_content

            # Find the chat and update its title
            for chat in chats:
                if chat["id"] == chat_id:
                    chat["title"] = new_title
                    chat_title_updated = True
                    break
        tg = jarvis_ai.model_handlers['textGeneration']
        tg.is_initialized = False
        tg.active_chat_id = chat_id
        tg.chat           = None
        # Get response from Jarvis AI assistant
        try:
            # Process the user message through the Jarvis AI assistant
            ai_response_content = get_jarvis_response(message_content, model_type)
        except Exception as e:
            # Handle any errors in AI processing with a fallback response
            logger.error(f"Error getting AI response: {e}")
            ai_response_content = "I'm sorry, I encountered an error processing your request. Please try again."

        # Create an AI response message object
        ai_response = {
            "id": f"msg-{int(time.time()) + 1}",  # Generate unique ID using timestamp + 1
            "role": "assistant",  # Message is from the assistant
            "content": ai_response_content,  # The AI-generated response
            "timestamp": datetime.now().strftime("%I:%M %p"),  # Current time
        }

        # Add the AI response to the chat
        messages[chat_id].append(ai_response)

        # Save the updated data to persistent storage
        save_data()

        # Log the successful processing
        logger.info(f"Processed message in chat {chat_id}")

        # Return both messages and title update info to the frontend
        return jsonify({
            "userMessage": user_message,  # The user message object
            "aiResponse": ai_response,  # The AI response object
            "chatTitleUpdated": chat_title_updated,  # Whether the title was updated
            "newTitle": new_title  # The new title if updated
        })
    except Exception as e:
        # Log and return any errors that occur
        logger.error(f"Error processing message: {e}")
        return jsonify({"error": "Failed to process message"}), 500

# Static File Serving Routes
# These routes handle serving the frontend static files and implementing SPA routing

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    """
    Main route handler for serving static files and implementing SPA routing.

    This function:
    1. Serves index.html for the root path
    2. Handles special cases for asset files
    3. Serves specific files if they exist
    4. Falls back to index.html for client-side routing (SPA pattern)
    5. Returns a custom 404 page if an error occurs

    Args:
        path (str): The requested path

    Returns:
        The requested file or index.html for SPA routing
    """
    try:
        logger.info(f"Serving path: {path}")

        # If path is empty (root URL), serve index.html
        if path == "":
            logger.info("Serving index.html for root path")
            return send_from_directory(app.static_folder, 'index.html')

        # Special case for assets directory (images, fonts, etc.)
        # This handles both forward slash and backslash path separators
        if path.startswith('assets/') or path.startswith('assets\\'):
            # Extract the asset path without the 'assets/' prefix
            asset_path = path[7:]  # Remove 'assets/' prefix
            asset_dir = Path(app.static_folder) / 'assets'
            logger.info(f"Serving asset: {asset_path} from {asset_dir}")

            # Check if the asset file exists
            asset_full_path = asset_dir / asset_path
            if asset_full_path.is_file():
                return send_from_directory(str(asset_dir), asset_path)
            else:
                logger.error(f"Asset not found: {asset_path}")

        # Check if the requested path exists as a file
        full_path = Path(app.static_folder) / path
        if full_path.is_file():
            logger.info(f"Serving file: {path}")
            return send_from_directory(app.static_folder, path)

        # For all other paths, serve index.html to support client-side routing (SPA)
        # This allows the frontend router to handle the path
        logger.info(f"Path {path} not found, serving index.html (SPA routing)")
        return send_from_directory(app.static_folder, 'index.html')
    except Exception as e:
        # Log the error and return a custom 404 page
        logger.error(f"Error serving path {path}: {e}")

        # Return a custom 404 page with error details
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Page Not Found</title>
            <style>
                body {{ font-family: Arial, sans-serif; text-align: center; padding: 50px; }}
                h1 {{ font-size: 36px; margin-bottom: 20px; }}
                p {{ font-size: 18px; margin-bottom: 20px; }}
                a {{ color: #0066cc; text-decoration: none; }}
                a:hover {{ text-decoration: underline; }}
            </style>
        </head>
        <body>
            <h1>404 - Page Not Found</h1>
            <p>The page you are looking for does not exist.</p>
            <p><a href="/">Return to Home</a></p>
            <p><small>Error details: {str(e)}</small></p>
        </body>
        </html>
        """, 404

# Additional Static File Routes
# These routes provide specific handlers for direct file requests

@app.route('/index.html')
def serve_index():
    """
    Specific route handler for direct requests to index.html.

    This function:
    1. Attempts to serve index.html directly
    2. Falls back to fallback.html if index.html is not found
    3. Returns a 404 error if neither file is found

    Returns:
        The index.html file, fallback.html, or an error response
    """
    try:
        logger.info("Serving index.html directly")
        index_path = Path(app.static_folder) / 'index.html'

        # Check if index.html exists and serve it
        if index_path.is_file():
            return send_from_directory(app.static_folder, 'index.html')

        # If index.html doesn't exist, try to serve fallback.html
        logger.error(f"index.html not found at {index_path}")
        fallback_path = Path(app.static_folder) / 'fallback.html'

        if fallback_path.is_file():
            logger.info(f"Serving fallback.html instead")
            return send_from_directory(app.static_folder, 'fallback.html')

        # If neither file exists, return a 404 error
        return "index.html not found", 404
    except Exception as e:
        # Log and return any errors that occur
        logger.error(f"Error serving index.html: {e}")
        return f"Error: {str(e)}", 500

@app.route('/fallback')
def serve_fallback():
    """
    Route handler for the fallback page.

    This function serves a simplified fallback page that can be used
    when the main application fails to load. This provides a minimal
    interface for the user to interact with the assistant.

    Returns:
        The fallback.html file or an error response
    """
    try:
        logger.info("Serving fallback.html")
        fallback_path = Path(app.static_folder) / 'fallback.html'

        # Check if fallback.html exists and serve it
        if fallback_path.is_file():
            return send_from_directory(app.static_folder, 'fallback.html')

        # If fallback.html doesn't exist, return a 404 error
        logger.error(f"fallback.html not found at {fallback_path}")
        return "Fallback page not found", 404
    except Exception as e:
        # Log and return any errors that occur
        logger.error(f"Error serving fallback.html: {e}")
        return f"Error: {str(e)}", 500

# Error Handlers
# These handlers provide custom responses for various HTTP errors

@app.errorhandler(404)
def page_not_found(e):
    """
    Custom handler for 404 (Not Found) errors.

    This function returns a user-friendly 404 page with a link
    to return to the home page.

    Args:
        e: The error object

    Returns:
        A custom 404 error page
    """
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

# Application Entry Point
# This section is only executed when the script is run directly (not imported)

if __name__ == '__main__':
    # Start the Flask development server
    # Note: In production, the app is served by waitress (see main.py)
    app.run(debug=True, port=5000)
