#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Jarvis AI Assistant - Full Pipeline Runner
This script handles the complete pipeline from user prompt to AI response.
Optimized for Google Colab with A100 GPU.
"""

import os
import sys
import torch
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging
from datetime import datetime
from .utils import setup_logging, sync_logs, sync_from_gdrive, is_paperspace_environment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("jarvis")

# Check for CUDA
if torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA Version: {torch.version.cuda}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    logger.info(f"BF16 support: {torch.cuda.is_bf16_supported()}")
else:
    device = torch.device("cpu")
    logger.warning("CUDA not available, using CPU")

# Try to import the necessary modules with helpful error messages
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    import bitsandbytes as bnb
except ImportError as e:
    logger.error(f"Error importing required modules: {e}")
    logger.error("Please run the setup script first: !bash colab_setup.sh")
    sys.exit(1)

class ChatHistory:
    """Manages conversation history and context"""
    
    def __init__(self, max_history: int = 10):
        self.history: List[Dict[str, str]] = []
        self.max_history = max_history
    
    def add(self, role: str, content: str):
        """Add a message to the history"""
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def get_formatted_history(self) -> str:
        """Format the history for the model prompt"""
        formatted = ""
        for message in self.history:
            if message["role"] == "user":
                formatted += f"USER: {message['content']}\n"
            else:
                formatted += f"ASSISTANT: {message['content']}\n"
        return formatted
    
    def save(self, filepath: str):
        """Save the chat history to a file"""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load(self, filepath: str):
        """Load chat history from a file"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.history = json.load(f)
        else:
            logger.warning(f"Chat history file {filepath} not found")

class JarvisAssistant:
    """Main Jarvis AI Assistant class for handling the full pipeline"""
    
    def __init__(
        self, 
        model_name: str = "deepseek-ai/deepseek-coder-6.7b-instruct",
        model_path: Optional[str] = None,
        load_in_4bit: bool = True,
        max_new_tokens: int = 1024,
        history_file: Optional[str] = None
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.chat_history = ChatHistory()
        
        # Load chat history if provided
        if history_file:
            self.chat_history.load(history_file)
        
        # Load the model and tokenizer
        self.load_model(load_in_4bit)
    
    def load_model(self, load_in_4bit: bool = True):
        """Load the model and tokenizer with appropriate optimizations for A100"""
        logger.info(f"Loading model: {self.model_name if not self.model_path else self.model_path}")
        
        # Configure quantization for efficiency
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        else:
            quantization_config = None
        
        # Choose the correct path/name for loading
        model_id = self.model_path if self.model_path else self.model_name
        
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        
        # Load the model with appropriate optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            trust_remote_code=True
        )
        
        logger.info("Model loaded successfully")
    
    def generate_prompt(self, user_input: str) -> str:
        """Format the prompt for the model based on the model type"""
        history = self.chat_history.get_formatted_history()
        
        # Optimize prompt format based on the model type
        if "deepseek" in self.model_name.lower():
            # DeepSeek-specific prompt format
            prompt = f"{history}USER: {user_input}\nASSISTANT:"
        else:
            # Generic prompt format
            prompt = f"{history}USER: {user_input}\nASSISTANT:"
            
        return prompt
    
    def generate_response(self, user_input: str) -> str:
        """Process user input and generate a response"""
        # Add user input to history
        self.chat_history.add("user", user_input)
        
        # Generate the prompt
        prompt = self.generate_prompt(user_input)
        
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate the response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                repetition_penalty=1.1
            )
        
        # Decode the response
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the model's response (not the prompt)
        response = full_output[len(prompt):].strip()
        
        # Add the response to the chat history
        self.chat_history.add("assistant", response)
        
        return response
    
    def save_chat_history(self, filepath: str):
        """Save the current chat history to a file"""
        self.chat_history.save(filepath)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Jarvis AI Assistant")
    parser.add_argument(
        "--model", 
        type=str, 
        default="deepseek-ai/deepseek-coder-6.7b-instruct",
        help="The model name or path to use"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to a fine-tuned model (takes precedence over --model)"
    )
    parser.add_argument(
        "--history",
        type=str,
        default=None,
        help="Path to a chat history JSON file to load"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="A single prompt to process (non-interactive mode)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the chat history after completion"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate"
    )
    
    return parser.parse_args()

def interactive_mode(assistant: JarvisAssistant, output_file: Optional[str] = None):
    """Run Jarvis in interactive mode"""
    print("Jarvis AI Assistant - Interactive Mode")
    print("Type 'exit' or 'quit' to end the conversation")
    print("=" * 50)
    
    try:
        while True:
            user_input = input("\nYou: ")
            
            if user_input.lower() in ['exit', 'quit']:
                break
                
            print("\nJarvis is thinking...")
            response = assistant.generate_response(user_input)
            print(f"\nJarvis: {response}")
            
            # Save history after each interaction if output file is specified
            if output_file:
                assistant.save_chat_history(output_file)
                
    except KeyboardInterrupt:
        print("\nExiting interactive mode.")
    finally:
        if output_file:
            assistant.save_chat_history(output_file)
            print(f"Chat history saved to {output_file}")

def main():
    """
    Main entry point for running the Jarvis AI Assistant.
    This configures and starts the assistant with the specified settings.
    """
    # Set up logging to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"jarvis_run_{timestamp}.log"
    setup_logging(log_file)
    
    logger.info("Starting Jarvis AI Assistant")
    
    # Enable cuDNN benchmark for better GPU performance
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logger.info("Enabled cuDNN benchmark for better GPU performance")
    
    # If we're in Paperspace, sync models from Google Drive
    if is_paperspace_environment():
        try:
            logger.info("Running in Paperspace environment, syncing models from Google Drive...")
            sync_from_gdrive("models")
            logger.info("Synced latest models from Google Drive")
        except Exception as e:
            logger.warning(f"Error syncing models from Google Drive: {str(e)}")
    
    args = parse_arguments()
    
    # Initialize the assistant
    assistant = JarvisAssistant(
        model_name=args.model,
        model_path=args.model_path,
        max_new_tokens=args.max_tokens,
        history_file=args.history
    )
    
    # Handle different modes
    if args.interactive:
        interactive_mode(assistant, args.output)
    elif args.prompt:
        # Single prompt mode
        response = assistant.generate_response(args.prompt)
        print(response)
        
        # Save history if requested
        if args.output:
            assistant.save_chat_history(args.output)
    else:
        # No mode specified, default to interactive
        interactive_mode(assistant, args.output)

    # Sync logs at the end of the run
    try:
        sync_logs()
    except Exception as e:
        logger.warning(f"Failed to sync logs to Google Drive: {str(e)}")
    
    logger.info("Jarvis AI Assistant run completed")

if __name__ == "__main__":
    main() 