#!/usr/bin/env python3
"""
Use a trained model with the ImprovedPreprocessor for inference.
This script demonstrates how to use a model trained with dataset-specific settings
for generating text.
"""

import os
import sys
import torch
import random
import logging
from tqdm import tqdm

# Add the parent directory to the Python path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Import the ImprovedPreprocessor class and SimpleModel
from src.generative_ai_module.improved_preprocessing import ImprovedPreprocessor
from src.generative_ai_module.train_with_preprocessor import SimpleModel

def load_model(model_path, vocab_size):
    """Load a trained model"""
    model = SimpleModel(vocab_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    return model

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8):
    """Generate text using the trained model"""
    device = next(model.parameters()).device
    
    # Tokenize the prompt
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    
    # Generate text
    generated = list(input_ids[0].cpu().numpy())
    
    with torch.no_grad():
        for _ in range(max_length):
            # Prepare input for the model
            inputs = torch.tensor([generated[-min(len(generated), 100):]], dtype=torch.long).to(device)
            
            # Get predictions
            outputs = model(inputs)
            next_token_logits = outputs[0, -1, :] / temperature
            
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Add the token to the generated sequence
            generated.append(next_token)
            
            # Stop if we generate an EOS token
            if next_token == tokenizer.eos_idx:
                break
    
    # Decode the generated tokens
    return tokenizer.decode(generated)

def main():
    """Main function to demonstrate inference with dataset-specific settings"""
    logger.info("Starting inference with dataset-specific settings")
    
    # Create the preprocessor
    preprocessor = ImprovedPreprocessor()
    
    # Check if models exist
    writing_model_path = "models/writing_prompts_model.pt"
    persona_model_path = "models/persona_chat_model.pt"
    
    if not os.path.exists(writing_model_path) or not os.path.exists(persona_model_path):
        logger.error("Models not found. Please run train_with_preprocessor.py first.")
        logger.info("Simulating inference with untrained models for demonstration...")
        
        # Create untrained models for demonstration
        vocab_size = preprocessor.tokenizer.vocab_size
        writing_model = SimpleModel(vocab_size)
        persona_model = SimpleModel(vocab_size)
    else:
        # Load trained models
        vocab_size = preprocessor.tokenizer.vocab_size
        writing_model = load_model(writing_model_path, vocab_size)
        persona_model = load_model(persona_model_path, vocab_size)
    
    # Move models to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writing_model = writing_model.to(device)
    persona_model = persona_model.to(device)
    
    # Generate text with writing_prompts model
    logger.info("Generating text with writing_prompts model:")
    writing_prompt = "<PROMPT>\nYou wake up one day with the ability to see 10 seconds into the future.\n<STORY>\n"
    writing_generated = generate_text(writing_model, preprocessor.tokenizer, writing_prompt, max_length=50)
    logger.info(f"Prompt: {writing_prompt}")
    logger.info(f"Generated: {writing_generated}")
    
    # Generate text with persona_chat model
    logger.info("\nGenerating text with persona_chat model:")
    persona_prompt = "<PERSONA>\n- I love hiking in the mountains.\n<DIALOGUE>\nUSER: Hi there! Do you like outdoor activities?\nASSISTANT: "
    persona_generated = generate_text(persona_model, preprocessor.tokenizer, persona_prompt, max_length=50)
    logger.info(f"Prompt: {persona_prompt}")
    logger.info(f"Generated: {persona_generated}")
    
    logger.info("\nInference complete!")

if __name__ == "__main__":
    main()
