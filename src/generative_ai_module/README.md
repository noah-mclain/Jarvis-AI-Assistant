# Generative AI Module

This module contains components for text and code generation using preprocessed tokenized data.

## Components

- `text_generator.py`: Implements the `TextGenerator` and `CombinedModel` classes for text generation.
- `code_generator.py`: Implements the `CodeGenerator` class for code generation.
- `dataset_processor.py`: Handles loading, cleaning, and batching of text data.
- `improved_preprocessing.py`: Advanced preprocessing with tokenization and analysis tools.
- `prompt_enhancer.py`: Enhances prompts for better generation results.
- `unified_generation_pipeline.py`: **Main script** for training, generation, and preprocessing.
- `test_tools.py`: Testing utilities and data inspection tools.
- `examples/quick_start.py`: Simple demonstration script for training and generation.

## Quick Start

The fastest way to get started is to run the quick_start.py example:

```bash
# Navigate to the project root directory first
cd /path/to/Jarvis-AI-Assistant

# Run the quick start script
python3 src/generative_ai_module/examples/quick_start.py
```

This will:

1. Train a small model on the persona_chat dataset (just 3 epochs)
2. Generate sample text from several prompts
3. Show you how to use the unified pipeline for more options

## Unified Pipeline Usage

The unified pipeline script provides a command-line interface for training, generation, and preprocessing:

```bash
# Navigate to the project root directory first
cd /path/to/Jarvis-AI-Assistant

# Training: Train both text and code models (save to disk)
python3 src/generative_ai_module/unified_generation_pipeline.py --mode train --save-model

# Training: Train only text model with persona_chat dataset (5 epochs)
python3 src/generative_ai_module/unified_generation_pipeline.py --mode train --train-type text --dataset persona_chat --epochs 5

# Generation: Generate text using trained model
python3 src/generative_ai_module/unified_generation_pipeline.py --mode generate --gen-type text --prompt "Hello, how are you?"

# Generation: Generate code using trained model
python3 src/generative_ai_module/unified_generation_pipeline.py --mode generate --gen-type code --prompt "def fibonacci(n):"

# Generation: Use tokenizer for generation (instead of character-level)
python3 src/generative_ai_module/unified_generation_pipeline.py --mode generate --use-tokenizer --prompt "Hello"

# Preprocessing: Preprocess and save dataset
python3 src/generative_ai_module/unified_generation_pipeline.py --mode preprocess --dataset persona_chat --max-samples 200 --analyze
```

### Command Line Options

```bash
--mode {train,generate,preprocess}  Mode of operation: train models, generate text/code, or preprocess data
--train-type {text,code,both}       Which generator to train (text, code, or both)
--epochs EPOCHS                     Number of training epochs (default: 5)
--save-model                        Save the trained models
--dataset {persona_chat,writing_prompts} Which dataset to use for text training
--gen-type {text,code}              Type of content to generate
--prompt PROMPT                     Text prompt to start generation
--length LENGTH                     Number of tokens/characters to generate (default: 100)
--temperature TEMPERATURE           Sampling temperature - higher values give more random results (default: 0.7)
--use-tokenizer                     Use tokenizer for generation instead of character-level
--preprocess-output DIR             Directory to save preprocessed data
--analyze                           Analyze token distribution when preprocessing
--max-samples MAX_SAMPLES           Maximum number of samples to process (for preprocessing)
--model-dir MODEL_DIR               Directory for model files (default: "models")
--text-model TEXT_MODEL             Filename for text generator model
--code-model CODE_MODEL             Filename for code generator model
```

## Model Architecture

Both text and code generator models use the `CombinedModel` class, which is an RNN-based model with:

- Embedding layer
- LSTM layers (configurable number of layers)
- Linear output layer

Default settings:

- Hidden size: 128
- Number of layers: 2
- Vocabulary size: Determined by the dataset (typically around 100-150 tokens)

## Preprocessed Data

The module uses preprocessed datasets located in `examples/preprocessed_data/`:

- `persona_chat_preprocessed.pt`: Chat dialogue dataset
- `writing_prompts_preprocessed.pt`: Creative writing dataset

## Python API

If you want to use the individual components in your code:

```python
# Import the necessary modules
from generative_ai_module.text_generator import TextGenerator
from generative_ai_module.code_generator import CodeGenerator
from generative_ai_module.dataset_processor import DatasetProcessor

# Create generators
text_gen = TextGenerator()
code_gen = CodeGenerator()

# Load preprocessed data
processor = DatasetProcessor()
data = processor.load_preprocessed_data("persona_chat")

# Train model
text_gen.train(data['batches'], epochs=5)

# Generate text
generated_text = text_gen.generate(initial_str="Hello", pred_len=100)
```

## Note on Python Imports

When importing modules in your own scripts, ensure that the correct path is in your Python path:

```python
import os
import sys

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.abspath(os.path.join(current_dir, 'path/to/src'))
sys.path.insert(0, module_dir)

# Now you can import the modules
from generative_ai_module.text_generator import TextGenerator
```

## Testing and Debugging

For testing and debugging the module, use the test_tools.py script:

```bash
# Test module imports and structure
python3 src/generative_ai_module/test_tools.py --test-imports

# Test data loading and preprocessing
python3 src/generative_ai_module/test_tools.py --test-data

# Test the entire pipeline
python3 src/generative_ai_module/test_tools.py --test-all
```

## Limitations

- Models are small and trained for only a few epochs
- Limited vocabulary size (104 tokens)
- Might not generate coherent text due to limited training
- Character-level tokenization can be inefficient
