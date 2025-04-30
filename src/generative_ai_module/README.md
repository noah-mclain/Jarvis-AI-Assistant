# Jarvis AI Assistant - Unified Module

This module provides a comprehensive, unified interface to all Jarvis AI capabilities including dataset processing, model training, context-aware text generation, and interactive sessions.

## Features

- **Unified Dataset Processing**: Load and preprocess data from The Pile, OpenAssistant, and GPTeacher datasets
- **Integrated Model Training**: Train models with automatic train/validation/test splits and early stopping
- **Context-Aware Generation**: Generate responses with conversation memory for consistent interactions
- **Smart Model Selection**: Automatically choose the most appropriate model based on prompt content
- **Visualization Tools**: Generate training and validation metrics visualizations
- **Interactive CLI**: Command-line interface for training, generation, and interactive sessions

## Installation

Ensure you have all required dependencies:

```bash
pip install torch tqdm matplotlib datasets nltk transformers
```

## Recent Code Optimizations

The codebase has recently undergone a comprehensive cleanup and optimization:

1. **Dataset Handling Consolidation**:

   - Enhanced `UnifiedDatasetHandler` with conversation context management
   - Merged functionality from `use_new_datasets.py` into core handlers
   - Improved dataset preprocessing and validation

2. **Training Pipeline Improvements**:

   - Added visualization capabilities to `unified_generation_pipeline.py`
   - Enhanced metric calculation and tracking during training
   - Added validation split support for more accurate model evaluation

3. **Code Structure Cleanup**:

   - Removed redundant files like `use_new_datasets.py`
   - Consolidated duplicated functionality
   - Updated centralized imports in `__init__.py`

4. **Performance Optimizations**:
   - Streamlined dataset loading process
   - Improved conversation context management
   - Enhanced memory usage through better data handling

## Quick Start

### Training Models

Train models on all supported datasets:

```bash
python src/generative_ai_module/jarvis_unified.py --action train
```

Train on specific datasets:

```bash
python src/generative_ai_module/jarvis_unified.py --action train --datasets pile openassistant
```

### Interactive Mode

Start an interactive chat session:

```bash
python src/generative_ai_module/jarvis_unified.py --action interactive
```

With persistent memory:

```bash
python src/generative_ai_module/jarvis_unified.py --action interactive --memory-file conversation.json
```

### Generate Single Response

Generate a response to a specific prompt:

```bash
python src/generative_ai_module/jarvis_unified.py --action generate --prompt "Explain how neural networks work"
```

## Command Line Options

### General Options

- `--action`: Action to perform (train, interactive, generate)
- `--models-dir`: Directory to load/save models (default: models)
- `--use-best-models`: Use best models instead of final models
- `--no-force-gpu`: Do not force GPU usage

### Training Options

- `--datasets`: Datasets to train on (all, pile, openassistant, gpteacher)
- `--max-samples`: Maximum number of samples per dataset (default: 500)
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size for training (default: 32)
- `--early-stopping`: Stop if validation loss doesn't improve (default: 3)
- `--visualization-dir`: Directory to save visualizations (default: visualizations)

### Generation Options

- `--prompt`: Prompt for text generation
- `--temperature`: Temperature for text generation (default: 0.7)
- `--max-history`: Maximum conversation exchanges to remember (default: 5)
- `--memory-file`: File to save/load conversation memory

## Using as a Library

You can import the module in your Python code:

```python
from src.generative_ai_module.jarvis_unified import JarvisAI

# Initialize
jarvis = JarvisAI(models_dir="my_models", use_best_models=True)

# Generate a response
response = jarvis.generate_response("Tell me about artificial intelligence")
print(response)

# Train models
jarvis.train_models(datasets=["pile", "openassistant"], epochs=5)
```

## Conversation Memory

The system maintains conversation history to provide context-aware responses:

```python
# Initialize with persistent memory
jarvis = JarvisAI(memory_file="conversation.json")

# Generate responses that are aware of conversation history
response1 = jarvis.generate_response("Who was Alan Turing?")
response2 = jarvis.generate_response("What was his most famous contribution?")
```

## Dataset-Specific Features

The system intelligently routes prompts to the most appropriate model:

- **The Pile**: Best for factual, knowledge-based queries
- **OpenAssistant**: Ideal for conversational or assistant-like interactions
- **GPTeacher**: Specialized for instructional or how-to content
- **Writing Prompts**: Optimized for creative writing and story generation
- **Persona Chat**: Enhanced for dialogue-based interactions with consistent persona

When training with the unified pipeline, you can specify which datasets to use:

```bash
# Train on all datasets
python src/generative_ai_module/train_unified_models.py

# Train on specific datasets
python src/generative_ai_module/train_unified_models.py --datasets writing_prompts persona_chat

# Train with custom settings
python src/generative_ai_module/train_unified_models.py --datasets pile --max-samples 1000 --epochs 20
```

For interactive generation, the system will automatically select the most appropriate model based on your prompt content.

## Visualizations

During training, the system generates visualizations for:

- Loss curves (training and validation)
- Accuracy metrics
- Perplexity over time
- Cross-dataset comparisons

## Advanced Usage

### Custom Training Loop

```python
from src.generative_ai_module.jarvis_unified import JarvisAI

jarvis = JarvisAI()
metrics = jarvis.train_models(
    datasets=["pile"],
    max_samples=1000,
    epochs=20,
    batch_size=64,
    validation_split=0.15,
    test_split=0.1,
    early_stopping=5
)

# Access training metrics
print(f"Final loss: {metrics['pile']['loss'][-1]}")
print(f"Best validation loss: {min(metrics['pile']['val_loss'])}")
```

### Extending the System

The modular design allows for extending the system with new datasets or models:

1. Update the `DatasetProcessor` class to handle the new dataset
2. Add the dataset name to the available choices in the argument parser
3. Update the `determine_best_dataset` method to include heuristics for the new dataset

## Troubleshooting

- **GPU Memory Issues**: Reduce batch size or max_samples
- **Model Not Found**: Ensure models are trained before using interactive or generate modes
- **Poor Generation Quality**: Try adjusting the temperature (higher for more creativity, lower for more focused responses)

## More Information

See additional documentation in the docs directory:

- `README_DATASETS.md` - Detailed information about available datasets
- `README_MODULE_INSTS.md` - Installation and usage instructions
- `UNSLOTH_GUIDE.md` - Guide for using Unsloth for efficient fine-tuning
