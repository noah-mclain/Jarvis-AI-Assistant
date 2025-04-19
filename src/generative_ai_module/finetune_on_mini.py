"""
Script to fine-tune a pre-trained model on a smaller, focused dataset
for more coherent text generation.
"""

import os
import sys
import torch
from tqdm import tqdm

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, module_dir)

from generative_ai_module.text_generator import TextGenerator, CombinedModel
from generative_ai_module.dataset_processor import DatasetProcessor

def load_model(model_path):
    """Load a pre-trained model"""
    print(f"Loading model from {model_path}")
    
    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found")
        return None, None
    
    # Load the checkpoint
    checkpoint = torch.load(model_path)
    
    # If it's a state dict directly, not a checkpoint dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        vocab_size = checkpoint.get('vocab_size', 104)
    else:
        state_dict = checkpoint
        vocab_size = 104  # Default vocabulary size
    
    # Create and initialize the model
    model = CombinedModel(
        input_size=vocab_size,
        hidden_size=128,
        output_size=vocab_size,
        num_layers=2
    )
    
    # Load the state dict
    model.load_state_dict(state_dict)
    print(f"Model loaded with vocabulary size: {vocab_size}")
    
    return model, vocab_size

def prepare_mini_dataset(file_path, processor, seq_length=50, batch_size=2):
    """Prepare the mini dataset for training"""
    print(f"Preparing mini dataset from {file_path}")

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: Mini dataset file not found at {file_path}")
        return []

    try:
        return load_clean_data(
            file_path, processor, seq_length, batch_size
        )
    except Exception as e:
        print(f"Error preparing mini dataset: {e}")
        import traceback
        traceback.print_exc()
        return []


# TODO Rename this here and in `prepare_mini_dataset`
def load_clean_data(file_path, processor, seq_length, batch_size):
    # Load and clean the data
    with open(file_path, 'r', encoding='utf-8') as f:
        text_data = f.read()

    print(f"Loaded text data: {len(text_data)} characters")

    # Clean the text
    cleaned_text = processor.clean_text(text_data)
    print(f"Cleaned text: {len(cleaned_text)} characters")

    # Print a sample of the cleaned text
    print(f"Sample of cleaned text: '{cleaned_text[:100]}...'")

    # Create sequences
    sequences = processor.create_sequences(cleaned_text, seq_length)
    print(f"Created {len(sequences)} sequences with length {seq_length}")

    # Create batches
    batches = processor.create_batches(sequences, batch_size)
    print(f"Created {len(batches)} batches with batch size {batch_size}")

    # Check batch structure
    if batches:
        inputs, targets = batches[0]
        print(f"First batch - Inputs shape: {inputs.shape}, Targets shape: {targets.shape}")

    return batches

def finetune_model(model, batches, epochs=10, learning_rate=0.0005):
    """Fine-tune the model on the mini dataset"""
    print(f"Fine-tuning model for {epochs} epochs")
    
    # Check if batches exist
    if not batches or len(batches) == 0:
        print("Error: No batches provided for fine-tuning")
        return model
        
    # Set up training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    try:
        for epoch in range(epochs):
            epoch_loss = 0
            batch_count = 0
            
            for i, (input_batch, target_batch) in enumerate(tqdm(batches, desc=f"Epoch {epoch+1}/{epochs}")):
                try:
                    # Print batch information for the first few batches
                    if epoch == 0 and i < 2:
                        print(f"Batch {i} - Input shape: {input_batch.shape}, Target shape: {target_batch.shape}")
                    
                    # Move to device
                    input_batch = input_batch.to(device)
                    target_batch = target_batch.to(device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    output, _ = model(input_batch)
                    
                    # Calculate loss
                    loss = criterion(output, target_batch)
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optimizer.step()
                    
                    # Update metrics
                    epoch_loss += loss.item()
                    batch_count += 1
                    
                except Exception as e:
                    print(f"Error in batch {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Print epoch metrics
            if batch_count > 0:
                avg_loss = epoch_loss / batch_count
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}: No valid batches processed")
                
    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        import traceback
        traceback.print_exc()
    
    print("Fine-tuning complete")
    return model

def save_finetuned_model(model, vocab_size, output_path):
    """Save the fine-tuned model"""
    print(f"Saving fine-tuned model to {output_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size
    }, output_path)
    
    print(f"Fine-tuned model saved to {output_path}")

def generate_sample(model, vocab_size, prompt, max_length=100, temperature=0.4):
    """Generate a sample text using the fine-tuned model"""
    print(f"Generating sample with prompt: '{prompt}'")
    
    # Create a TextGenerator with the fine-tuned model
    text_gen = TextGenerator()
    text_gen.model = model
    
    # Generate text
    generated_text = text_gen.generate(
        initial_str=prompt,
        pred_len=max_length,
        temperature=temperature
    )
    
    print("\nGenerated text:")
    print("=" * 60)
    print(generated_text)
    print("=" * 60)

def main():
    # Get the current directory and project root for absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
    
    # Paths with absolute references - using model files we know exist
    model_dir = os.path.join(project_root, "models")
    
    # List of potential model files to try (in order of preference)
    model_options = [
        os.path.join(model_dir, "text_gen_model.pt"),
        os.path.join(model_dir, "text_generator_model.pt"),
        os.path.join(model_dir, "best_model.pt"), 
        os.path.join(model_dir, "persona_chat_model.pt")
    ]
    
    # Find the first existing model file
    input_model = next((model for model in model_options if os.path.exists(model)), None)
    
    # Output model path
    output_model = os.path.join(model_dir, "finetuned_model.pt")
    
    # Use absolute path for the mini dataset
    mini_dataset = os.path.join(current_dir, "mini_dataset.txt")
    
    print(f"Mini dataset path: {mini_dataset}")
    print(f"Selected input model: {input_model}")
    print(f"Output model path: {output_model}")
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Load the pre-trained model
    if input_model is None:
        print("\nERROR: Could not find any trained model file.")
        print("Please make sure you have trained the model first using:")
        print("  python3 src/generative_ai_module/unified_generation_pipeline.py --mode train --save-model")
        return
        
    model, vocab_size = load_model(input_model)
    if model is None:
        return
    
    # Create a dataset processor
    processor = DatasetProcessor()
    
    # Prepare the mini dataset
    batches = prepare_mini_dataset(mini_dataset, processor)
    
    # Check if we have batches
    if not batches:
        print("No training batches were created. Fine-tuning cannot proceed.")
        return
        
    # Fine-tune the model
    model = finetune_model(model, batches, epochs=20, learning_rate=0.0005)
    
    # Save the fine-tuned model
    save_finetuned_model(model, vocab_size, output_model)
    
    # Generate sample texts with different prompts
    prompts = [
        "<PERSONA>\n- I am a chef\n<DIALOGUE>\nUSER: What do you do for a living?\nASSISTANT:",
        "<PERSONA>\n- I have a cat named Whiskers\n<DIALOGUE>\nUSER: Do you have any pets?\nASSISTANT:"
    ]
    
    for prompt in prompts:
        generate_sample(model, vocab_size, prompt, temperature=0.3)

if __name__ == "__main__":
    main() 