"""
Simple script to fine-tune a pre-trained model on a very small dataset
for more coherent text generation.
"""

import os
import sys
import torch

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, module_dir)

from generative_ai_module.text_generator import TextGenerator

def load_model(model_path):
    """Load a model from disk"""
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
        
    print(f"Loading model from {model_path}")
    text_gen = TextGenerator()
    
    try:
        # Load the checkpoint
        checkpoint = torch.load(model_path)
        
        # Check if the checkpoint has model_state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Create a new model with correct vocab size
            vocab_size = checkpoint.get('vocab_size', 104)
            
            # Create a new CombinedModel with the right dimensions
            from generative_ai_module.text_generator import CombinedModel
            text_gen.model = CombinedModel(
                input_size=vocab_size,
                hidden_size=128,
                output_size=vocab_size,
                num_layers=2
            )
            
            # Load the state dict
            text_gen.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded with vocabulary size: {vocab_size}")
        else:
            # Direct load attempt
            text_gen.load_model(model_path)
            print("Model loaded directly")
            
        return text_gen
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_model_file():
    """Find a suitable model file to use"""
    # Get project root
    project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
    
    # List of potential model files to try
    model_options = [
        os.path.join(project_root, "models", "text_gen_model.pt"),
        os.path.join(project_root, "models", "text_generator_model.pt"),
        os.path.join(current_dir, "best_model.pt"),
        os.path.join(current_dir, "models", "persona_chat_model.pt")
    ]
    
    # Find the first existing model file
    for model_file in model_options:
        if os.path.exists(model_file):
            print(f"Found model file: {model_file}")
            return model_file
            
    return None

def main():
    # Load the mini dataset
    dataset_path = os.path.join(current_dir, "smaller_dataset.txt")
    
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset_text = f.read()
            
        print(f"Loaded dataset ({len(dataset_text)} characters)")
        
        # Find and load a pre-trained model
        model_path = get_model_file()
        if not model_path:
            print("No model file found. Cannot proceed with fine-tuning.")
            return
            
        text_gen = load_model(model_path)
        if not text_gen:
            return
            
        # Fine-tune the model using the character-by-character method
        print("Fine-tuning model...")
        losses = text_gen.train_character_model_step(dataset_text, [])
        
        print(f"Fine-tuning complete. Final loss: {losses[-1] if losses else 'N/A'}")
        
        # Save the fine-tuned model
        output_model = os.path.join(current_dir, "models", "simple_finetuned_model.pt")
        os.makedirs(os.path.dirname(output_model), exist_ok=True)
        text_gen.save_model(output_model)
        print(f"Saved fine-tuned model to {output_model}")
        
        # Generate samples with different temperatures
        prompts = [
            "<PERSONA>\n- I am a chef\n<DIALOGUE>\nUSER: What do you do for a living?\nASSISTANT:",
            "<PERSONA>\n- I have a cat\n<DIALOGUE>\nUSER: Do you have any pets?\nASSISTANT:"
        ]
        
        temperatures = [0.3, 0.5]
        
        for prompt in prompts:
            for temp in temperatures:
                print(f"\nPrompt: '{prompt}', Temperature: {temp}")
                print("-" * 40)
                
                generated_text = text_gen.generate(
                    initial_str=prompt,
                    pred_len=50,
                    temperature=temp
                )
                
                print(generated_text)
                print("-" * 40)
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 