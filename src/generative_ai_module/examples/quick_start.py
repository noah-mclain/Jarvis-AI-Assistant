"""
Quick Start Example for Generative AI Module

This script demonstrates how to train and generate text using the unified pipeline
in just a few steps.
"""

import os
import sys

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, module_dir)

# Import the unified pipeline functions
from generative_ai_module.unified_generation_pipeline import (
    train_text_generator,
    generate_with_char_model
)
from generative_ai_module.dataset_processor import DatasetProcessor

def main():
    # Step 1: Show available datasets
    print("======= Generative AI Quick Start =======")
    print("\nStep 1: Available Datasets")
    print("- persona_chat: Conversational dialogue dataset")
    print("- writing_prompts: Creative writing dataset")
    
    # Step 2: Train a simple model
    print("\nStep 2: Training a simple model (this will take a few minutes)")
    dataset_name = "persona_chat"  # Use persona_chat for faster training
    epochs = 3  # Use fewer epochs for quick demonstration
    
    print(f"Training on {dataset_name} dataset for {epochs} epochs...")
    model, vocab_size = train_text_generator(dataset_name, epochs=epochs)
    
    if model is None:
        print("Error: Training failed. Please check that the preprocessed data exists.")
        return
    
    # Step 3: Generate text
    print("\nStep 3: Generating text with the trained model")
    
    prompts = [
        "Hello, how are you?",
        "<PERSONA>\n- I am a",
        "USER: What's your favorite hobby?\nASSISTANT:"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        print("-" * 40)
        
        # Generate text with the character model
        generated_text = generate_with_char_model(
            model,
            prompt,
            vocab_size,
            max_length=100,
            temperature=0.7
        )
        
        print(generated_text)
        print("-" * 40)
    
    # Step 4: Show how to use DatasetProcessor directly
    print("\nStep 4: Using DatasetProcessor directly")
    
    try:
        processor = DatasetProcessor()
        data = processor.load_preprocessed_data(dataset_name)
        print(f"Successfully loaded {dataset_name} dataset")
        print(f"Vocabulary size: {data['vocab_size']}")
        print(f"Number of batches: {len(data['batches']) if 'batches' in data else 'N/A'}")
    except Exception as e:
        print(f"Note: Could not load preprocessed data directly: {e}")
    
    print("\n======= Quick Start Complete =======")
    print("To train and save models for later use, try the unified pipeline script:")
    print("python src/generative_ai_module/unified_generation_pipeline.py --mode train --save-model")
    print("\nFor preprocessing, try:")
    print("python src/generative_ai_module/unified_generation_pipeline.py --mode preprocess --dataset persona_chat --analyze")
    
if __name__ == "__main__":
    main() 