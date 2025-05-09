"""
Jarvis AI Integration Module

This module provides integration between the components of the Jarvis AI Assistant.
It manages:
- Input/output routing
- Response format conversion
- Dataset mixing for training
- Metrics collection and visualization
- Unified interface for all AI capabilities

The integration module serves as a central hub connecting various components
of the system while providing a consistent interface for applications.
"""

import os
import sys
import json
import torch
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple

# Import the unified module
from src.generative_ai_module.jarvis_unified import JarvisAI, ConversationMemory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("jarvis_integration.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("jarvis_integration")

class JarvisIntegration:
    """
    Integration class for Jarvis AI components.
    
    This class handles:
    - Configuring and loading models
    - Routing inputs to appropriate processing modules
    - Standardizing response formats
    - Providing metrics and visualization hooks
    - Managing conversation context and memory
    """
    
    def __init__(
        self, 
        models_dir: str = "models",
        data_dir: str = "datasets",
        config_path: Optional[str] = "config/integration.json",
        memory_file: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the Jarvis Integration module.
        
        Args:
            models_dir: Directory for model storage
            data_dir: Directory for dataset storage
            config_path: Path to configuration file
            memory_file: File to save/load conversation memory
            device: Device to use (cpu or cuda)
        """
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.config_path = config_path
        self.memory_file = memory_file
        
        # Create necessary directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize the unified AI module
        self.ai = JarvisAI(
            models_dir=str(self.models_dir),
            use_best_models=self.config["system"].get("use_best_models", True),
            device=device,
            memory_file=memory_file
        )
        
        logger.info("JarvisIntegration initialized successfully")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from file or create default.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            "datasets": {
                "enabled": ["pile", "openassistant", "gpteacher"],
                "max_samples": {
                    "pile": 500,
                    "openassistant": 500,
                    "gpteacher": 500
                },
                "validation_split": 0.1,
                "test_split": 0.1
            },
            "training": {
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 1e-4,
                "early_stopping": 3,
                "visualization_dir": "visualizations"
            },
            "generation": {
                "temperature": 0.7,
                "max_length": 200
            },
            "system": {
                "use_best_models": True,
                "log_level": "INFO"
            }
        }
        
        if not config_path or not os.path.exists(config_path):
            logger.info("Config file not found, using default configuration")
            # Create default config file if path is provided
            if config_path:
                try:
                    os.makedirs(os.path.dirname(config_path), exist_ok=True)
                    with open(config_path, 'w') as f:
                        json.dump(default_config, f, indent=2)
                    logger.info(f"Created default configuration at {config_path}")
                except Exception as e:
                    logger.error(f"Error creating default config: {e}")
            return default_config
        
        # Load existing config and update with any missing default values
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Ensure all required sections and values exist
            for section, values in default_config.items():
                if section not in config:
                    config[section] = values
                elif isinstance(values, dict):
                    for key, value in values.items():
                        if key not in config[section]:
                            config[section][key] = value
            
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            return default_config
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        if not self.config_path:
            logger.warning("No config path specified, cannot save configuration")
            return
        
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Saved configuration to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def update_config(self, section: str, key: str, value: Any) -> None:
        """
        Update a configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            value: New value
        """
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section][key] = value
        self.save_config()
        logger.info(f"Updated configuration: {section}.{key} = {value}")
    
    def get_enabled_datasets(self) -> List[str]:
        """
        Get list of enabled datasets.
        
        Returns:
            List of enabled dataset names
        """
        return self.config["datasets"].get("enabled", ["pile", "openassistant", "gpteacher"])
    
    def train_models(
        self,
        datasets: Optional[List[str]] = None,
        max_samples: Optional[Dict[str, int]] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train models using the unified AI module.
        
        Args:
            datasets: List of dataset names to train on
            max_samples: Maximum samples per dataset
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            
        Returns:
            Dictionary of training metrics per dataset
        """
        # Use configured datasets if none provided
        if datasets is None:
            datasets = self.get_enabled_datasets()
        
        # Get max samples for each dataset
        if max_samples is None:
            max_samples = self.config["datasets"].get("max_samples", {})
        
        # Use configured training parameters if not provided
        training_config = self.config["training"]
        if epochs is None:
            epochs = training_config.get("epochs", 10)
        if batch_size is None:
            batch_size = training_config.get("batch_size", 32)
        if learning_rate is None:
            learning_rate = training_config.get("learning_rate", 1e-4)
        
        # Train models with provided or configured parameters
        logger.info(f"Starting training for datasets: {', '.join(datasets)}")
        metrics = self.ai.train_models(
            datasets=datasets,
            max_samples=max(max_samples.values()) if max_samples else 500,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            validation_split=self.config["datasets"].get("validation_split", 0.1),
            test_split=self.config["datasets"].get("test_split", 0.1),
            early_stopping=training_config.get("early_stopping", 3),
            visualization_dir=training_config.get("visualization_dir", "visualizations")
        )
        
        logger.info("Training completed")
        return metrics
    
    def generate_response(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_length: Optional[int] = None,
        dataset: Optional[str] = None
    ) -> str:
        """
        Generate a response to a user prompt.
        
        Args:
            prompt: User input prompt
            temperature: Temperature for text generation
            max_length: Maximum response length
            dataset: Specific dataset to use
            
        Returns:
            Generated response text
        """
        # Use configured parameters if not provided
        generation_config = self.config["generation"]
        if temperature is None:
            temperature = generation_config.get("temperature", 0.7)
        if max_length is None:
            max_length = generation_config.get("max_length", 200)
        
        # Generate response
        logger.info(f"Generating response for prompt: {prompt[:50]}...")
        response = self.ai.generate_response(
            prompt=prompt,
            temperature=temperature,
            max_length=max_length,
            dataset=dataset
        )
        
        logger.info(f"Generated response of length {len(response)}")
        return response
    
    def run_interactive(self) -> None:
        """Run an interactive session with the assistant."""
        logger.info("Starting interactive session")
        self.ai.run_interactive()
    
    def update_dataset_config(
        self,
        dataset: str,
        enabled: Optional[bool] = None,
        max_samples: Optional[int] = None
    ) -> None:
        """
        Update dataset configuration.
        
        Args:
            dataset: Dataset name
            enabled: Whether the dataset is enabled
            max_samples: Maximum number of samples to use
        """
        # Update enabled status
        if enabled is not None:
            enabled_datasets = set(self.get_enabled_datasets())
            if enabled and dataset not in enabled_datasets:
                enabled_datasets.add(dataset)
            elif not enabled and dataset in enabled_datasets:
                enabled_datasets.remove(dataset)
            self.config["datasets"]["enabled"] = list(enabled_datasets)
            logger.info(f"Dataset {dataset} {'enabled' if enabled else 'disabled'}")
        
        # Update max samples
        if max_samples is not None:
            if "max_samples" not in self.config["datasets"]:
                self.config["datasets"]["max_samples"] = {}
            self.config["datasets"]["max_samples"][dataset] = max_samples
            logger.info(f"Set max samples for {dataset} to {max_samples}")
        
        self.save_config()
    
    def batch_process(
        self,
        prompts: List[str],
        output_file: Optional[str] = None,
        temperature: Optional[float] = None,
        max_length: Optional[int] = None
    ) -> List[Tuple[str, str]]:
        """
        Process a batch of prompts.
        
        Args:
            prompts: List of prompt strings
            output_file: File to save results
            temperature: Temperature for generation
            max_length: Maximum response length
            
        Returns:
            List of (prompt, response) tuples
        """
        logger.info(f"Batch processing {len(prompts)} prompts")
        results = []
        
        for i, prompt in enumerate(prompts):
            response = self.generate_response(
                prompt=prompt,
                temperature=temperature,
                max_length=max_length
            )
            results.append((prompt, response))
            logger.info(f"Processed prompt {i+1}/{len(prompts)}")
        
        # Save results to file if requested
        if output_file:
            try:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, 'w') as f:
                    for prompt, response in results:
                        f.write(f"Prompt: {prompt}\n\nResponse: {response}\n\n{'-'*80}\n\n")
                logger.info(f"Saved batch results to {output_file}")
            except Exception as e:
                logger.error(f"Error saving batch results: {e}")
        
        return results
    
    def analyze_dataset_usage(self) -> Dict[str, int]:
        """
        Analyze which datasets are being used for responses.
        
        Returns:
            Dictionary of dataset usage counts
        """
        # This would ideally be implemented with actual usage tracking
        # For now, just return configured datasets with mock counts
        enabled_datasets = self.get_enabled_datasets()
        usage = {dataset: 0 for dataset in enabled_datasets}
        
        logger.info("Dataset usage analysis requested (mock data)")
        return usage
    
    def clear_memory(self) -> None:
        """Clear conversation memory."""
        self.ai.memory.clear()
        logger.info("Conversation memory cleared")


def main():
    """Command-line interface for the Jarvis AI Integration module."""
    parser = argparse.ArgumentParser(description="Jarvis AI Integration")
    
    parser.add_argument(
        "action",
        choices=["train", "generate", "interactive", "configure"],
        help="Action to perform"
    )
    
    parser.add_argument(
        "--prompt",
        help="Prompt for text generation"
    )
    
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Datasets to use"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature for text generation"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        help="Maximum response length"
    )
    
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory for model storage"
    )
    
    parser.add_argument(
        "--data-dir",
        default="datasets",
        help="Directory for dataset storage"
    )
    
    parser.add_argument(
        "--config-path",
        default="config/integration.json",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--memory-file",
        help="File to save/load conversation memory"
    )
    
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        help="Device to use for inference"
    )
    
    args = parser.parse_args()
    
    # Initialize the integration module
    integration = JarvisIntegration(
        models_dir=args.models_dir,
        data_dir=args.data_dir,
        config_path=args.config_path,
        memory_file=args.memory_file,
        device=args.device
    )
    
    # Perform requested action
    if args.action == "train":
        integration.train_models(datasets=args.datasets)
        
    elif args.action == "generate":
        if not args.prompt:
            print("Error: --prompt is required for generate action")
            return
        
        response = integration.generate_response(
            prompt=args.prompt,
            temperature=args.temperature,
            max_length=args.max_length
        )
        print(f"\nJarvis: {response}")
        
    elif args.action == "interactive":
        integration.run_interactive()
        
    elif args.action == "configure":
        # Example configuration options
        if args.datasets:
            for dataset in args.datasets:
                integration.update_dataset_config(dataset, enabled=True)
        
        print("Current configuration:")
        print(json.dumps(integration.config, indent=2))


if __name__ == "__main__":
    main() 