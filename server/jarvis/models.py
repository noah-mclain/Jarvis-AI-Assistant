"""
Jarvis AI Assistant Model Handlers

This module provides handlers for different AI model functionalities:
- Speech-to-Text: For voice input and transcription
- Code Generation: For generating code based on prompts
- Text Generation/Conversation: For general text conversations
- NLP: For natural language processing tasks
- Story Generation: For creative writing and storytelling
- Image Generation: For creating images from text descriptions

Each handler loads the appropriate model and provides a process_query method
that takes a query string and returns a response string.

Author: Nada Mohamed
License: MIT
"""

import os
import logging
from typing import Optional, Dict, Any
import json

global messages
# Note: The actual model implementations will need to import the necessary libraries
# such as torch, transformers, etc. These imports are left as TODOs for now.

# Configure logger for this module
logger = logging.getLogger('jarvis_models')

class ModelHandler:
    """Base class for all model handlers"""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the model handler

        Args:
            model_path: Path to the model files. If None, a default path will be used.
        """
        # Determine the model path - either use the provided path or calculate the default
        self.model_path = model_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'data',
            'models'
        )

        # Initialize the model
        self.model = None
        self.is_initialized = False

        from transformers import pipeline

        from huggingface_hub import login

        import soundfile as sf  # For handling audio files

    def load_model(self) -> bool:
        """
        Load the model

        Returns:
            bool: True if the model was loaded successfully, False otherwise
        """
        # This method should be implemented by subclasses
        return False

    def process_query(self, query: str) -> str:
        """
        Process a query and return a response

        Args:
            query: The query string

        Returns:
            str: The response string
        """
        # This method should be implemented by subclasses
        return f"Base model handler received: {query}"


class SpeechToTextHandler(ModelHandler):
    """Handler for speech-to-text functionality"""

    def load_model(self) -> bool:
        """Load the speech-to-text model"""
        try:
            logger.info("Loading speech-to-text model...")



            self.is_initialized = True
            logger.info("Speech-to-text model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading speech-to-text model: {e}")
            return False

    def process_query(self, query: str) -> str:
        """Process a speech-to-text query"""
        import sounddevice as sd
        from scipy.io.wavfile import write
        import requests

        # Set recording parameters
        fs = 44100  # Sample rate (Hz)
        duration = 30  # Recording duration (seconds)
        filename = "output.wav"  # Output file name

        # Record audio
        print("Recording...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()  # Wait until recording is finished

        # Save to WAV file
        write(filename, fs, recording)
        print(f"Saved to {filename}")

        json_query = {
            "audio": recording,
        }
        response = requests.post("https://6b1c-35-240-193-205.ngrok-free.app/generate", json=json_query)

        if response.status_code == 200:
            return "Audio sent successfully."
        else:
            logger.error(f"Error generating image: {response.status_code} - {response.text}")
            return "Audio failed. Please check the logs for details."


class CodeGenerationHandler(ModelHandler):
    """Handler for code generation functionality"""

    def load_model(self) -> bool:
        """Load the code generation model"""
        try:
            logger.info("Loading code generation model (DeepSeek Coder)...")

            # TODO: Implement actual model loading
            # This should import and initialize the CodeGenerator from the generative_ai_module
            # Example:
            # from src.generative_ai_module.code_generator import CodeGenerator
            # self.generator = CodeGenerator(use_deepseek=True, load_in_4bit=True)

            # For now, just simulate successful loading
            self.is_initialized = True
            logger.info("Code generation model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading code generation model: {e}")
            return False

    def process_query(self, query: str) -> str:
        """Process a code generation query"""
        if not self.is_initialized and not self.load_model():
            return "Code generation model could not be loaded. Please check the logs for details."

        # TODO: Implement actual code generation
        # This should use the DeepSeek Coder model to generate code based on the query
        # Example:
        # return self.generator.generate_code(prompt=query, length=500, temperature=0.7)

        # For now, just return a placeholder response
        return f"Code generation model would process: {query}\n\n```python\n# Example generated code\ndef hello_world():\n    print('Hello, world!')\n\nhello_world()\n```"


class TextGenerationHandler(ModelHandler):
    """Handler for text generation/conversation functionality"""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize with messages as instance variable"""
        super().__init__(model_path)
        self.messages = None  # Initialize messages as an instance variable
        self.client = None  # Initialize client as an instance variable
        self.chat = None  # Initialize chat as an instance variable

    def load_model(self) -> bool:
        """Load the text generation model"""
        try:
            from pathlib import Path

            # Get the directory of the current script
            current_dir = Path(__file__).parent.resolve()

            # Navigate up two parents, then into 'data/messages.json'
            messages_file = current_dir.parent.parent / 'data' / 'messages.json'
            logger.info(f"Attempting to load messages from {messages_file}")
            
            try:
                with open(messages_file, 'r') as f:
                    self.messages = json.load(f)  # Store messages as instance variable
                logger.info(f"Loaded messages for {len(self.messages)} chats from {messages_file}")
            except FileNotFoundError:
                logger.warning(f"Messages file not found at {messages_file}. Creating empty messages dictionary.")
                self.messages = {}
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in messages file at {messages_file}. Creating empty messages dictionary.")
                self.messages = {}

            # Initialize the Google Generative AI client
            try:
                from google import genai
                self.client = genai.client.Client(api_key="AIzaSyB7qZ_-4axHQElsgZjqIXfWKSsVBL3BXdA")
                logger.info("Successfully initialized Google Generative AI client")
            except ImportError:
                logger.error("Google Generative AI package not installed. Please install with: pip install google-generativeai")
                return False
            except Exception as e:
                logger.error(f"Failed to initialize Google Generative AI client: {e}")
                return False
            
            # Create a new chat if none exists
            try:
                # Extract messages for the active chat
                chat_messages = []
                
                # If self.messages is a dictionary with chat IDs as keys

                if isinstance(self.messages, dict) and self.messages:
                   # Use the chat_id that was set externally, or fall back to most recent
                    chat_id = getattr(self, "active_chat_id", None)
                    if chat_id and chat_id in self.messages:
                       chat_messages = self.messages[chat_id]
                       logger.info(f"Using messages from chat ID: {chat_id}")
                    else:
                        recent = list(self.messages.keys())[-1]
                        chat_messages = self.messages[recent]
                        logger.info(f"Using messages from chat ID: {recent}")
                print(chat_messages)

                # Create formatted history for the conversation
                if chat_messages:
                    # If messages have role/content format
                    if isinstance(chat_messages[0], dict) and 'role' in chat_messages[0]:
                        formatted_history = [
                            {
                                "role": "user" if msg["role"] == "user" else "model",  # Gemini uses "model", not "assistant"
                                "parts": [{"text": msg["content"]}]
                            }
                            for msg in chat_messages
                            if "role" in msg and "content" in msg
                        ]

                        while formatted_history and formatted_history[0]["role"] != "user":
                            logger.warning("Removed model message at start of history (Gemini requires user to start)")
                            formatted_history.pop(0)
                        logger.info(f"Formatted {len(formatted_history)} messages for Gemini chat history")

                    # Limit the context to avoid exceeding token limits
                    context_window = formatted_history[-10:] if formatted_history else []

                self.chat = self.client.chats.create(model = "gemini-2.0-flash", history=context_window,)
                
                logger.info("Successfully created new chat session")
            except Exception as e:
                logger.error(f"Chat creation failed: {e}")
                return False

            self.is_initialized = True
            logger.info("Text generation model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading text generation model: {e}")
            return False

    def process_query(self, query: str) -> str:
        """Process a text generation query"""
        if not self.is_initialized:
            success = self.load_model()
            if not success:
                return "Text generation model could not be loaded. Please check the logs for details."
        
        # Send the latest user message
        try:
            if self.chat is None:
                logger.error("Chat session is None. Attempting to reload model.")
                success = self.load_model()
                if not success or self.chat is None:
                    return "Failed to create chat session. Please check the logs for details."
            
            response = self.chat.send_message(query)
            return response.text
        except Exception as e:
            logger.error(f"API request failed: {e}")
            return f"Error generating response: {str(e)}"


class NLPHandler(ModelHandler):
    """Handler for natural language processing functionality"""

    def load_model(self) -> bool:
        """Load the NLP model"""
        try:
            logger.info("Loading NLP model...")

            # TODO: Implement actual model loading
            # This should import and initialize NLP components
            # Example:
            # from src.generative_ai_module.nlp_utils import NLPProcessor
            # self.nlp_processor = NLPProcessor()
            #
            # Or use transformers directly:
            # from transformers import pipeline
            # self.sentiment_analyzer = pipeline("sentiment-analysis", ...)
            # self.ner = pipeline("ner", ...)
            # self.classifier = pipeline("text-classification", ...)

            # For now, just simulate successful loading
            self.is_initialized = True
            logger.info("NLP model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading NLP model: {e}")
            return False

    def process_query(self, query: str) -> str:
        """Process an NLP query"""
        if not self.is_initialized and not self.load_model():
            return "NLP model could not be loaded. Please check the logs for details."

        # TODO: Implement actual NLP processing
        # This should analyze the text for sentiment, entities, and intent
        # Example:
        # return self.nlp_processor.analyze_text(query)
        #
        # Or use transformers pipelines:
        # sentiment_result = self.sentiment_analyzer(query)
        # entities_result = self.ner(query)
        # intent_results = self.classifier(query, candidate_labels=["question", "request", "statement", "command"])

        # For now, just return a placeholder response
        return f"NLP model would analyze: {query}\n\nSentiment: Positive\nEntities: None detected\nIntent: Information request"


class StoryGenerationHandler(ModelHandler):
    """Handler for story generation functionality"""

    def load_model(self) -> bool:
        """Load the story generation model"""
        try:
            logger.info("Loading story generation model...")

            # TODO: Implement actual model loading
            # This should import and initialize the TextGenerator and UnifiedDatasetHandler
            # Example:
            # from src.generative_ai_module.text_generator import TextGenerator
            # from src.generative_ai_module.unified_dataset_handler import UnifiedDatasetHandler
            # self.generator = TextGenerator(force_gpu=True)
            # self.dataset_handler = UnifiedDatasetHandler()
            #
            # If a specific model path is provided:
            # if self.model_path:
            #     self.generator.load_model(self.model_path)
            #
            # Try to load a story-specific model:
            # model_path = os.path.join(self.model_path, "story_generator_model.pt")
            # if os.path.exists(model_path):
            #     self.generator.load_model(model_path)

            # For now, just simulate successful loading
            self.is_initialized = True
            logger.info("Story generation model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading story generation model: {e}")
            return False

    def process_query(self, query: str) -> str:
        """Process a story generation query"""
        if not self.is_initialized and not self.load_model():
            return "Story generation model could not be loaded. Please check the logs for details."

        # TODO: Implement actual story generation
        # This should use the TextGenerator and UnifiedDatasetHandler to generate a story
        # Example:
        # prompt = f"Write a creative story about: {query}\n\n"
        # return self.dataset_handler.generate_with_context(
        #     self.generator,
        #     prompt,
        #     temperature=0.8,
        #     max_length=500
        # )

        # For now, just return a placeholder response
        return f"Story generation model would create a story about: {query}\n\nOnce upon a time, in a land far away, there was a kingdom of wonder and magic. The people lived in harmony with nature, and the kingdom prospered under the wise rule of its benevolent monarch..."


class ImageGenerationHandler(ModelHandler): 
    """Handler for image generation functionality"""

    def load_model(self) -> bool:
        """Load the image generation model"""
        try:
            logger.info("Image generation model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading image generation model: {e}")
            return False

    def process_query(self, query: str) -> str:
        """Process an image generation query"""
        import requests
        from PIL import Image
        from io import BytesIO
        print(query)
        json_query = {
            "prompt": query,
            "negative_prompt": "",
            "width": 1024,
            "height": 1024,
            "guidance_scale": 9.0,
            "seed": 12345
        }
        response = requests.post("https://f20d-34-125-240-149.ngrok-free.app/generate", json=json_query)

        if response.status_code == 200:

            img = Image.open(BytesIO(response.content))
            img.show()  # Opens the image in your default viewer
            img.save("generated_image.png")
            return "Image generated successfully and saved as 'generated_image.png'."
        else:
            logger.error(f"Error generating image: {response.status_code} - {response.text}")
            return "Image generation failed. Please check the logs for details."


# Dictionary mapping model types to their handlers
MODEL_HANDLERS: Dict[str, ModelHandler] = {
    "speechToText": SpeechToTextHandler(),
    "codeGeneration": CodeGenerationHandler(),
    "textGeneration": TextGenerationHandler(),
    "nlp": NLPHandler(),
    "storyGeneration": StoryGenerationHandler(),
    "generativeImage": ImageGenerationHandler()
}
