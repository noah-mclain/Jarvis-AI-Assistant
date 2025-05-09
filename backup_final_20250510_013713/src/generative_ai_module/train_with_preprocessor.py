#!/usr/bin/env python3
"""
Train a model using the ImprovedPreprocessor with dataset-specific settings.
This script demonstrates how to use the ImprovedPreprocessor class for training
with special settings for the writing_prompts dataset.
"""

import os
import sys
import torch
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

# Import the ImprovedPreprocessor class
from src.generative_ai_module.improved_preprocessing import ImprovedPreprocessor

class SimpleModel(torch.nn.Module):
    """Simple character-level language model for demonstration"""
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        return self.fc(output)

def train_model(preprocessor, dataset_name, num_epochs=1, max_batches=10):
    """Train a model using the preprocessor with dataset-specific settings"""
    logger.info(f"Training model on {dataset_name} dataset")
    
    # Process dataset with dataset-specific settings
    data = preprocessor.process_dataset(dataset_name)
    
    # Log the dataset-specific parameters used
    logger.info(f"{dataset_name} parameters:")
    logger.info(f"  - Batch size: {data['params']['batch_size']}")
    logger.info(f"  - Sequence length: {data['params']['max_sequence_length']}")
    logger.info(f"  - Stride: {data['params']['stride']}")
    logger.info(f"  - Gradient accumulation steps: {data['params']['grad_accum_steps']}")
    
    # Create model and optimizer
    vocab_size = preprocessor.tokenizer.vocab_size
    model = SimpleModel(vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    model = model.to(device)
    
    # Train the model
    model.train()
    
    # Limit the number of batches for demonstration
    batches = data['batches'][:min(len(data['batches']), max_batches)]
    
    # Training loop
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        epoch_loss = 0.0
        for i, batch in enumerate(tqdm(batches, desc=f"Training on {dataset_name}")):
            # Train on batch with dataset-specific settings
            loss = preprocessor.train_batch(batch, model, optimizer)
            epoch_loss += loss
            
            # Log progress
            if (i + 1) % 5 == 0:
                logger.info(f"Batch {i+1}/{len(batches)}, Loss: {loss:.4f}")
        
        # Log epoch results
        avg_loss = epoch_loss / len(batches)
        logger.info(f"Epoch {epoch+1} complete, Average loss: {avg_loss:.4f}")
    
    # Save the model
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{dataset_name}_model.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    return model, data

def main():
    """Main function to demonstrate training with dataset-specific settings"""
    logger.info("Starting training with dataset-specific settings")
    
    # Create the preprocessor
    preprocessor = ImprovedPreprocessor()
    
    # Train on writing_prompts with special settings
    writing_model, writing_data = train_model(
        preprocessor, 
        "writing_prompts", 
        num_epochs=1, 
        max_batches=5
    )
    
    # Train on persona_chat with default settings
    persona_model, persona_data = train_model(
        preprocessor, 
        "persona_chat", 
        num_epochs=1, 
        max_batches=5
    )
    
    # Compare training results
    logger.info("\nTraining Results Comparison:")
    logger.info(f"writing_prompts (Special Settings):")
    logger.info(f"  - Batch size: {writing_data['params']['batch_size']}")
    logger.info(f"  - Gradient accumulation steps: {writing_data['params']['grad_accum_steps']}")
    logger.info(f"  - Effective batch size: {writing_data['params']['batch_size'] * writing_data['params']['grad_accum_steps']}")
    
    logger.info(f"\npersona_chat (Default Settings):")
    logger.info(f"  - Batch size: {persona_data['params']['batch_size']}")
    logger.info(f"  - Gradient accumulation steps: {persona_data['params']['grad_accum_steps']}")
    logger.info(f"  - Effective batch size: {persona_data['params']['batch_size'] * persona_data['params']['grad_accum_steps']}")
    
    logger.info("\nTraining complete!")

if __name__ == "__main__":
    main()
