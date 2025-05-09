#!/usr/bin/env python3
"""
Use the ImprovedPreprocessor with dataset-specific settings.
This script demonstrates how to use the ImprovedPreprocessor class
with special settings for the writing_prompts dataset.
"""

import os
import sys
import torch
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

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
from src.generative_ai_module.improved_preprocessing import (
    ImprovedPreprocessor, 
    ImprovedCharTokenizer,
    clean_and_normalize_text
)

def main():
    """Main function to demonstrate the usage of ImprovedPreprocessor"""
    logger.info("Starting preprocessing with dataset-specific settings")
    
    # Create output directories
    os.makedirs("preprocessed_data", exist_ok=True)
    os.makedirs("preprocessing_analysis", exist_ok=True)
    
    # Create the preprocessor
    preprocessor = ImprovedPreprocessor()
    
    # Process writing_prompts with special settings
    logger.info("Processing writing_prompts dataset with special settings")
    writing_data = preprocessor.process_dataset("writing_prompts")
    
    # Log the dataset-specific parameters used
    logger.info(f"writing_prompts parameters: {writing_data['params']}")
    
    # Analyze token distribution for writing_prompts
    logger.info("Analyzing token distribution for writing_prompts")
    writing_analysis = preprocessor.analyze_token_distribution(writing_data)
    
    # Save the preprocessed data and analysis
    logger.info("Saving writing_prompts preprocessed data and analysis")
    preprocessor.save_tokenized_data(writing_data, "preprocessed_data", "writing_prompts")
    
    # Process persona_chat with default settings
    logger.info("Processing persona_chat dataset with default settings")
    persona_data = preprocessor.process_dataset("persona_chat")
    
    # Log the dataset-specific parameters used
    logger.info(f"persona_chat parameters: {persona_data['params']}")
    
    # Analyze token distribution for persona_chat
    logger.info("Analyzing token distribution for persona_chat")
    persona_analysis = preprocessor.analyze_token_distribution(persona_data)
    
    # Save the preprocessed data and analysis
    logger.info("Saving persona_chat preprocessed data and analysis")
    preprocessor.save_tokenized_data(persona_data, "preprocessed_data", "persona_chat")
    
    # Compare the datasets
    compare_datasets(writing_data, persona_data)
    
    logger.info("Preprocessing complete!")

def compare_datasets(writing_data, persona_data):
    """Compare the two datasets and visualize differences"""
    logger.info("Comparing datasets")
    
    # Create a comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot batch sizes
    datasets = ['writing_prompts', 'persona_chat']
    batch_sizes = [
        writing_data['params']['batch_size'],
        persona_data['params']['batch_size']
    ]
    
    ax1.bar(datasets, batch_sizes)
    ax1.set_title('Batch Size Comparison')
    ax1.set_ylabel('Batch Size')
    
    # Plot sequence lengths
    seq_lengths = [
        writing_data['params']['max_sequence_length'],
        persona_data['params']['max_sequence_length']
    ]
    
    ax2.bar(datasets, seq_lengths)
    ax2.set_title('Sequence Length Comparison')
    ax2.set_ylabel('Sequence Length')
    
    plt.tight_layout()
    plt.savefig("preprocessing_analysis/dataset_comparison.png")
    plt.close()
    
    # Print comparison summary
    logger.info("\nDataset Comparison Summary:")
    logger.info(f"writing_prompts:")
    logger.info(f"  - Batch size: {writing_data['params']['batch_size']}")
    logger.info(f"  - Sequence length: {writing_data['params']['max_sequence_length']}")
    logger.info(f"  - Stride: {writing_data['params']['stride']}")
    logger.info(f"  - Gradient accumulation steps: {writing_data['params']['grad_accum_steps']}")
    logger.info(f"  - Number of batches: {len(writing_data['batches'])}")
    
    logger.info(f"\npersona_chat:")
    logger.info(f"  - Batch size: {persona_data['params']['batch_size']}")
    logger.info(f"  - Sequence length: {persona_data['params']['max_sequence_length']}")
    logger.info(f"  - Stride: {persona_data['params']['stride']}")
    logger.info(f"  - Gradient accumulation steps: {persona_data['params']['grad_accum_steps']}")
    logger.info(f"  - Number of batches: {len(persona_data['batches'])}")

def train_model_example():
    """Example of how to use the preprocessor for training"""
    # This is just an example and won't be executed
    
    # Create a simple model for demonstration
    class SimpleModel(torch.nn.Module):
        def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
            self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
            self.fc = torch.nn.Linear(hidden_dim, vocab_size)
            
        def forward(self, x):
            embedded = self.embedding(x)
            output, _ = self.lstm(embedded)
            return self.fc(output)
    
    # Create preprocessor
    preprocessor = ImprovedPreprocessor()
    
    # Process datasets
    writing_data = preprocessor.process_dataset("writing_prompts")
    persona_data = preprocessor.process_dataset("persona_chat")
    
    # Create model and optimizer
    vocab_size = preprocessor.tokenizer.vocab_size
    model = SimpleModel(vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Train on writing_prompts with special settings
    preprocessor.current_dataset = "writing_prompts"
    for batch in writing_data['batches']:
        loss = preprocessor.train_batch(batch, model, optimizer)
        print(f"Writing prompts batch loss: {loss}")
    
    # Train on persona_chat with default settings
    preprocessor.current_dataset = "persona_chat"
    for batch in persona_data['batches']:
        loss = preprocessor.train_batch(batch, model, optimizer)
        print(f"Persona chat batch loss: {loss}")

if __name__ == "__main__":
    main()
