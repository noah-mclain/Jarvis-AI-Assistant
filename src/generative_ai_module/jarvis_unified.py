"""
Jarvis Unified AI Module

This module provides a unified interface for the Jarvis AI capabilities,
including dataset processing, model training, context-aware text generation,
and interactive sessions. This version focuses solely on The Pile, OpenAssistant,
and GPTeacher datasets.
"""

import os
import sys
import json
import time
import torch
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback,
    default_data_collator
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("jarvis_ai.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("jarvis_unified")

class ConversationMemory:
    """
    Manages conversation history and context for the AI assistant.
    
    This class provides methods for adding conversation exchanges,
    formatting conversation context for prompts, and saving/loading
    conversation memory.
    """
    
    def __init__(self, max_exchanges: int = 10, memory_file: Optional[str] = None):
        """
        Initialize conversation memory.
        
        Args:
            max_exchanges: Maximum number of exchanges to remember
            memory_file: File to save/load memory from
        """
        self.exchanges: List[Tuple[str, str]] = []
        self.max_exchanges = max_exchanges
        self.memory_file = memory_file
        self.user_preferences: Dict[str, Any] = {}
        
        # Load memory if file exists
        if memory_file and os.path.exists(memory_file):
            self.load_memory()
    
    def add_exchange(self, user_input: str, ai_response: str) -> None:
        """
        Add a conversation exchange to memory.
        
        Args:
            user_input: User's input text
            ai_response: AI's response text
        """
        self.exchanges.append((user_input, ai_response))
        
        # Trim to max exchanges
        if len(self.exchanges) > self.max_exchanges:
            self.exchanges = self.exchanges[-self.max_exchanges:]
        
        # Auto-save if memory file is set
        if self.memory_file:
            self.save_memory()
    
    def get_context(self, max_exchanges: Optional[int] = None) -> str:
        """
        Format conversation history as context for prompts.
        
        Args:
            max_exchanges: Maximum number of exchanges to include in context
            
        Returns:
            Formatted conversation context
        """
        if max_exchanges is None:
            max_exchanges = self.max_exchanges

        exchanges = self.exchanges[-max_exchanges:] if max_exchanges > 0 else self.exchanges

        return "".join(
            f"User: {user_input}\nJarvis: {ai_response}\n\n"
            for user_input, ai_response in exchanges
        )
    
    def save_memory(self) -> None:
        """Save conversation memory to file."""
        if not self.memory_file:
            return
        
        memory_data = {
            "exchanges": self.exchanges,
            "user_preferences": self.user_preferences,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            with open(self.memory_file, 'w') as f:
                json.dump(memory_data, f, indent=2)
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Error saving memory: {e}")
    
    def load_memory(self) -> None:
        """Load conversation memory from file."""
        if not self.memory_file or not os.path.exists(self.memory_file):
            return
        
        try:
            with open(self.memory_file, 'r') as f:
                memory_data = json.load(f)
            
            self.exchanges = memory_data.get("exchanges", [])
            self.user_preferences = memory_data.get("user_preferences", {})
            logger.info(f"Loaded {len(self.exchanges)} conversation exchanges from memory")
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Error loading memory: {e}")
    
    def update_preferences(self, preferences: Dict[str, Any]) -> None:
        """
        Update user preferences.
        
        Args:
            preferences: Dictionary of preference key-value pairs
        """
        self.user_preferences.update(preferences)
        if self.memory_file:
            self.save_memory()
    
    def clear(self) -> None:
        """Clear conversation memory."""
        self.exchanges = []
        if self.memory_file:
            self.save_memory()


class TextDataset(Dataset):
    """Dataset for training language models on text data."""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        """
        Initialize dataset.
        
        Args:
            texts: List of text samples
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
        """
        self.encodings = tokenizer(
            texts, 
            truncation=True, 
            padding="max_length", 
            max_length=max_length,
            return_tensors="pt"
        )
    
    def __getitem__(self, idx):
        """Get encoded item by index."""
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = item["input_ids"].clone()
        return item
    
    def __len__(self):
        """Get dataset length."""
        return len(self.encodings.input_ids)


class JarvisAI:
    """
    Unified Jarvis AI class that handles training, inference,
    and interaction across multiple datasets.
    """
    
    # Available datasets, focused on The Pile, OpenAssistant, and GPTeacher
    AVAILABLE_DATASETS = ["pile", "openassistant", "gpteacher"]
    
    def __init__(
        self,
        models_dir: str = "models",
        use_best_models: bool = True,
        device: Optional[str] = None,
        memory_file: Optional[str] = None
    ):
        """
        Initialize Jarvis AI.
        
        Args:
            models_dir: Directory for storing models
            use_best_models: Whether to use best models or final models
            device: Device to use (cpu or cuda)
            memory_file: File to save/load conversation memory
        """
        self.models_dir = Path(models_dir)
        self.use_best_models = use_best_models
        self.models = {}
        self.tokenizers = {}
        self.memory = ConversationMemory(memory_file=memory_file)
        
        # Determine device
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Available datasets: {', '.join(self.AVAILABLE_DATASETS)}")
        
        # Create models directory
        os.makedirs(self.models_dir, exist_ok=True)
    
    def load_model(self, dataset: str) -> Tuple[Any, Any]:
        """
        Load model and tokenizer for a specific dataset.
        
        Args:
            dataset: Name of the dataset
            
        Returns:
            Tuple of (model, tokenizer)
        """
        if dataset in self.models and dataset in self.tokenizers:
            return self.models[dataset], self.tokenizers[dataset]
        
        # Check if model exists
        model_type = "best" if self.use_best_models else "final"
        model_path = self.models_dir / f"{dataset}_{model_type}"
        
        if os.path.exists(model_path):
            logger.info(f"Loading {model_type} model for {dataset} from {model_path}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForCausalLM.from_pretrained(model_path)
                model.to(self.device)
                
                self.models[dataset] = model
                self.tokenizers[dataset] = tokenizer
                return model, tokenizer
            except Exception as e:
                logger.error(f"Error loading model from {model_path}: {e}")
        
        # If model doesn't exist or fails to load, use default GPT-2
        logger.info(f"No trained model found for {dataset}, using default GPT-2")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.to(self.device)
        
        self.models[dataset] = model
        self.tokenizers[dataset] = tokenizer
        return model, tokenizer
    
    def determine_best_dataset(self, prompt: str) -> str:
        """
        Determine the best dataset to use for a given prompt.
        
        Args:
            prompt: User input prompt
            
        Returns:
            Name of the best dataset
        """
        # Simple keyword-based routing for now
        prompt_lower = prompt.lower()
        
        # Keyword mappings for datasets
        keywords = {
            "pile": ["knowledge", "information", "fact", "data", "science", "history", "literature"],
            "openassistant": ["help", "assist", "support", "guide", "advice", "recommend"],
            "gpteacher": ["explain", "teach", "learn", "understand", "concept", "tutorial"]
        }
        
        # Count keyword matches
        scores = {dataset: 0 for dataset in self.AVAILABLE_DATASETS}
        for dataset, words in keywords.items():
            for word in words:
                if word in prompt_lower:
                    scores[dataset] += 1
        
        # Get dataset with highest score, or default to pile
        best_dataset = max(scores.items(), key=lambda x: x[1])[0] if any(scores.values()) else "pile"
        logger.info(f"Selected dataset '{best_dataset}' for prompt: {prompt[:50]}...")
        
        return best_dataset
    
    def generate_response(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_length: int = 200,
        dataset: Optional[str] = None
    ) -> str:
        """
        Generate a response to a user prompt.
        
        Args:
            prompt: User input prompt
            temperature: Temperature for text generation
            max_length: Maximum response length
            dataset: Specific dataset to use, or None to auto-determine
            
        Returns:
            Generated response text
        """
        # Determine dataset if not specified
        if dataset is None or dataset not in self.AVAILABLE_DATASETS:
            dataset = self.determine_best_dataset(prompt)
        
        # Load model and tokenizer
        model, tokenizer = self.load_model(dataset)
        
        # Prepare input with conversation context
        context = self.memory.get_context()
        input_text = f"{context}User: {prompt}\nJarvis:"
        
        # Encode input
        inputs = tokenizer(input_text, return_tensors="pt").to(self.device)
        
        # Generate response
        output = model.generate(
            inputs.input_ids,
            max_length=inputs.input_ids.shape[1] + max_length,
            temperature=temperature,
            top_p=0.92,
            top_k=50,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode response
        full_output = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract just the response part
        response = full_output.split("Jarvis:")[-1].strip()
        
        # Add to conversation memory
        self.memory.add_exchange(prompt, response)
        
        return response
    
    def load_dataset(
        self, 
        dataset_name: str,
        max_samples: int = 1000,
        validation_split: float = 0.1,
        test_split: float = 0.1
    ) -> Tuple[TextDataset, TextDataset, TextDataset]:
        """
        Load and prepare a dataset.
        
        Args:
            dataset_name: Name of the dataset to load
            max_samples: Maximum number of samples to use
            validation_split: Fraction of data to use for validation
            test_split: Fraction of data to use for testing
            
        Returns:
            Tuple of (train_dataset, validation_dataset, test_dataset)
        """
        # Dataset paths
        dataset_path = Path(f"datasets/{dataset_name}.json")
        
        # Check if dataset exists
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset file not found: {dataset_path}")
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        # Load dataset
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        # Get text samples
        if isinstance(data, list):
            # Direct list of texts
            texts = data
        elif isinstance(data, dict) and "texts" in data:
            # Dictionary with 'texts' key
            texts = data["texts"]
        else:
            logger.error(f"Unsupported dataset format: {dataset_path}")
            raise ValueError(f"Unsupported dataset format: {dataset_path}")
        
        # Limit samples
        if max_samples > 0 and max_samples < len(texts):
            texts = random.sample(texts, max_samples)
            logger.info(f"Using {max_samples} samples from {dataset_name}")
        else:
            logger.info(f"Using all {len(texts)} samples from {dataset_name}")
        
        # Initialize tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Create full dataset
        full_dataset = TextDataset(texts, tokenizer)
        
        # Split dataset
        val_size = int(len(full_dataset) * validation_split)
        test_size = int(len(full_dataset) * test_split)
        train_size = len(full_dataset) - val_size - test_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, 
            [train_size, val_size, test_size]
        )
        
        logger.info(f"Dataset {dataset_name} split: {train_size} train, {val_size} validation, {test_size} test")
        
        return train_dataset, val_dataset, test_dataset
    
    def train_models(
        self,
        datasets: List[str] = None,
        max_samples: int = 500,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        validation_split: float = 0.1,
        test_split: float = 0.1,
        early_stopping: int = 3,
        visualization_dir: Optional[str] = None
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Train models on specified datasets.
        
        Args:
            datasets: List of dataset names to train on
            max_samples: Maximum number of samples per dataset
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            validation_split: Fraction of data to use for validation
            test_split: Fraction of data to use for testing
            early_stopping: Number of epochs without improvement before stopping
            visualization_dir: Directory to save training visualizations
            
        Returns:
            Dictionary of training metrics for each dataset
        """
        # Default to all available datasets if none specified
        datasets = datasets or self.AVAILABLE_DATASETS
        
        # Filter to available datasets
        datasets = [d for d in datasets if d in self.AVAILABLE_DATASETS]
        
        # Create visualization directory if specified
        if visualization_dir:
            os.makedirs(visualization_dir, exist_ok=True)
        
        # Train on each dataset
        metrics = {}
        for dataset in datasets:
            logger.info(f"Training model for {dataset}")
            
            try:
                # Load dataset
                train_dataset, val_dataset, test_dataset = self.load_dataset(
                    dataset,
                    max_samples=max_samples,
                    validation_split=validation_split,
                    test_split=test_split
                )
                
                # Initialize model and tokenizer
                model = GPT2LMHeadModel.from_pretrained("gpt2")
                tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                tokenizer.pad_token = tokenizer.eos_token
                
                # Setup training arguments
                model_output_dir = self.models_dir / f"{dataset}_final"
                best_model_dir = self.models_dir / f"{dataset}_best"
                
                os.makedirs(model_output_dir, exist_ok=True)
                os.makedirs(best_model_dir, exist_ok=True)
                
                training_args = TrainingArguments(
                    output_dir=str(model_output_dir),
                    num_train_epochs=epochs,
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size,
                    evaluation_strategy="epoch",
                    save_strategy="epoch",
                    load_best_model_at_end=True,
                    metric_for_best_model="eval_loss",
                    greater_is_better=False,
                    learning_rate=learning_rate,
                    warmup_steps=100,
                    weight_decay=0.01,
                    logging_dir='./logs',
                    logging_steps=10
                )
                
                # Setup trainer with early stopping
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    data_collator=default_data_collator,
                    callbacks=[
                        EarlyStoppingCallback(early_stopping_patience=early_stopping)
                    ]
                )
                
                # Train model
                logger.info(f"Starting training for {dataset}")
                train_result = trainer.train()
                
                # Evaluate on test dataset
                logger.info(f"Evaluating {dataset} model on test dataset")
                test_result = trainer.evaluate(test_dataset)
                
                # Save metrics
                train_metrics = {
                    "train_loss": trainer.state.log_history[-epochs:],
                    "eval_loss": [log["eval_loss"] for log in trainer.state.log_history if "eval_loss" in log],
                    "test_loss": test_result["eval_loss"]
                }
                metrics[dataset] = train_metrics
                
                # Save final model and best model
                logger.info(f"Saving final model for {dataset}")
                trainer.save_model(str(model_output_dir))
                tokenizer.save_pretrained(str(model_output_dir))
                
                # Copy best model
                logger.info(f"Saving best model for {dataset}")
                trainer.save_model(str(best_model_dir))
                tokenizer.save_pretrained(str(best_model_dir))
                
                # Create visualizations if requested
                if visualization_dir:
                    self._visualize_training(
                        dataset,
                        train_metrics,
                        visualization_dir
                    )
                
                # Clear memory
                del model, tokenizer, trainer
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                logger.info(f"Completed training for {dataset}")
                
            except Exception as e:
                logger.error(f"Error training model for {dataset}: {e}")
                metrics[dataset] = {"error": str(e)}
        
        return metrics
    
    def _visualize_training(
        self,
        dataset: str,
        metrics: Dict[str, Any],
        output_dir: str
    ) -> None:
        """
        Create visualizations of training metrics.
        
        Args:
            dataset: Name of the dataset
            metrics: Training metrics dictionary
            output_dir: Directory to save visualizations
        """
        try:
            # Extract metrics
            train_loss = [log["loss"] for log in metrics["train_loss"] if "loss" in log]
            eval_loss = metrics["eval_loss"]
            
            # Plot training and validation loss
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
            plt.plot(range(1, len(eval_loss) + 1), eval_loss, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'{dataset.capitalize()} Model Training')
            plt.legend()
            plt.grid(True)
            
            # Save plot
            output_path = Path(output_dir) / f"{dataset}_training.png"
            plt.savefig(output_path)
            plt.close()
            
            # Also save metrics as JSON
            metrics_path = Path(output_dir) / f"{dataset}_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"Training visualizations saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error creating training visualizations: {e}")
    
    def run_interactive(self) -> None:
        """Run an interactive session with the assistant."""
        print("\nWelcome to Jarvis AI Assistant! Type 'exit' to quit.\n")
        
        while True:
            # Get user input
            user_input = input("You: ")
            
            # Check for exit command
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\nThank you for using Jarvis AI. Goodbye!")
                break
            
            # Generate response
            response = self.generate_response(user_input)
            
            # Print response
            print(f"\nJarvis: {response}\n")


def main():
    """
    Command-line interface for the Jarvis AI module.
    
    Usage:
        python jarvis_unified.py train  # Train models
        python jarvis_unified.py interactive  # Run interactive session
        python jarvis_unified.py generate "Your prompt here"  # Generate a response
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Jarvis AI Unified Module")
    
    parser.add_argument(
        "action",
        choices=["train", "interactive", "generate"],
        help="Action to perform"
    )
    
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Prompt for text generation (required for 'generate' action)"
    )
    
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=JarvisAI.AVAILABLE_DATASETS,
        help="Datasets to use"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Maximum number of samples to use per dataset"
    )
    
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory for model storage"
    )
    
    parser.add_argument(
        "--use-best-models",
        action="store_true",
        help="Use best models instead of final models"
    )
    
    parser.add_argument(
        "--memory-file",
        help="File to save/load conversation memory"
    )
    
    parser.add_argument(
        "--visualization-dir",
        default="visualizations",
        help="Directory for training visualizations"
    )
    
    args = parser.parse_args()
    
    # Initialize Jarvis AI
    jarvis = JarvisAI(
        models_dir=args.models_dir,
        use_best_models=args.use_best_models,
        memory_file=args.memory_file
    )
    
    # Perform requested action
    if args.action == "train":
        jarvis.train_models(
            datasets=args.datasets,
            max_samples=args.max_samples,
            visualization_dir=args.visualization_dir
        )
        
    elif args.action == "generate":
        if not args.prompt:
            print("Error: Prompt is required for 'generate' action")
            return
        
        response = jarvis.generate_response(args.prompt)
        print(f"\nJarvis: {response}")
        
    elif args.action == "interactive":
        jarvis.run_interactive()


if __name__ == "__main__":
    main() 