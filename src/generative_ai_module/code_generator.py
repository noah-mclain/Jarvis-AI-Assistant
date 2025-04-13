from .text_generator import TextGenerator
from .prompt_enhancer import PromptEnhancer
from .dataset_processor import DatasetProcessor

class CodeGenerator:
    def __init__(self):
        self.text_generator = TextGenerator()
        self.prompt_enhancer = PromptEnhancer()
        self.dataset_processor = DatasetProcessor(self.text_generator)

    def generate_code(self, prompt, length=100):
        enhanced_prompt = self.prompt_enhancer.enhance_prompt(prompt)
        return self.text_generator.generate(
            initial_str=enhanced_prompt, pred_len=length
        )
    
    def train_on_codebase(self, source_path, epochs=10, sequence_length=100, batch_size=64):
        """
        Train the code generator on a specific codebase
        
        Args:
            source_path: Path to codebase (file, directory, or zip)
            epochs: Number of training epochs
            sequence_length: Sequence length for training
            batch_size: Batch size for training
            
        Returns:
            Training loss history
        """
        # Prepare code-specific dataset
        batched_data = self.dataset_processor.prepare_code_dataset(
            source_path, 
            sequence_length=sequence_length, 
            batch_size=batch_size
        )

        return self.text_generator.train(batched_data, epochs=epochs)
    
    def fine_tune(self, code_snippets, epochs=5):
        """
        Fine-tune the model on specific code snippets
        
        Args:
            code_snippets: List of code snippets or path to code files
            epochs: Number of fine-tuning epochs
            
        Returns:
            Fine-tuning loss history
        """
        if isinstance(code_snippets, list) and all(isinstance(snippet, str) for snippet in code_snippets):
            # Process list of code snippets
            combined_text = "\n\n".join(code_snippets)
            cleaned_text = self.dataset_processor.clean_text(combined_text)

            # Create sequences and batches
            sequences = self.dataset_processor.create_sequences(cleaned_text)
            batched_data = self.dataset_processor.create_batches(sequences)

        else:
            # Treat as path to code files
            batched_data = self.dataset_processor.prepare_code_dataset(code_snippets)

        return self.text_generator.train(batched_data, epochs=epochs)

    def train_from_preprocessed(self, dataset_name="writing_prompts", epochs=5):
        """
        Train the model using preprocessed data
        
        Args:
            dataset_name: Name of the preprocessed dataset to use
            epochs: Number of training epochs
            
        Returns:
            Training loss history
        """
        # Load preprocessed data using the dataset processor
        batched_data = self.dataset_processor.prepare_from_preprocessed(dataset_name)
        
        if not batched_data:
            raise ValueError(f"No valid batches found in preprocessed data: {dataset_name}")
            
        print(f"Training on {len(batched_data)} batches from preprocessed {dataset_name} dataset")
        return self.text_generator.train(batched_data, epochs=epochs)
        
    def save_model(self, path="models/code_generator_model.pt"):
        """Save the trained model to disk"""
        import os
        import torch
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model
        torch.save(self.text_generator.model.state_dict(), path)
        print(f"Model saved to {path}")
        
    def load_model(self, path="models/code_generator_model.pt"):
        """Load a trained model from disk"""
        import torch
        
        # Load the model
        self.text_generator.model.load_state_dict(torch.load(path))
        self.text_generator.model.eval()  # Set to evaluation mode
        print(f"Model loaded from {path}")