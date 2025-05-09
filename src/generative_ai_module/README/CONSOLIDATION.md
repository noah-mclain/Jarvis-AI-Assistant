# Generative AI Module Consolidation

This document describes the consolidation of the generative AI module files to reduce redundancy and improve maintainability.

## Consolidated Files

The following files have been consolidated into two main files:

### 1. `consolidated_dataset_processor.py`

This file combines the functionality of:
- `dataset_processor.py`
- `dataset_processor_fixed.py`
- `improved_preprocessing.py`
- `unified_dataset_handler.py`
- `dataset_demo.py`
- `test_dataset.py`

### 2. `consolidated_generation_pipeline.py`

This file combines the functionality of:
- `text_generator.py`
- `code_generator.py`
- `unified_generation_pipeline.py`

## Files to Remove

The following files can be safely removed as their functionality has been consolidated:

### Dataset Processing Files
- `dataset_processor.py`
- `dataset_processor_fixed.py`
- `improved_preprocessing.py`
- `unified_dataset_handler.py`
- `dataset_demo.py`
- `test_dataset.py`

### Generation Pipeline Files
- `text_generator.py`
- `code_generator.py`
- `unified_generation_pipeline.py`

### Backup Files
- `finetune_deepseek.py.bak`
- `train_models.py.bak`
- `unified_deepseek_training.py.bak3`

### One-time Fix Files
- `fix_tuple_unpacking.py`
- `direct_model_fix.py`

## Backward Compatibility

Backward compatibility is maintained through aliases in the `__init__.py` file:

```python
# For backward compatibility
try:
    from .consolidated_generation_pipeline import ConsolidatedGenerationPipeline as TextGenerator
    from .consolidated_generation_pipeline import ConsolidatedGenerationPipeline as CodeGenerator
    from .consolidated_generation_pipeline import ConsolidatedGenerationPipeline as UnifiedGenerationPipeline
    from .consolidated_dataset_processor import ConsolidatedDatasetProcessor as UnifiedDatasetHandler
except ImportError as e:
    logger.warning(f"Unable to set up backward compatibility classes: {e}")
    # Fallback classes...
```

## Benefits of Consolidation

1. **Reduced Code Duplication**: Similar functionality is now in one place
2. **Improved Maintainability**: Fewer files to maintain and update
3. **Clearer API**: Consolidated classes provide a more consistent interface
4. **Better Documentation**: Comprehensive docstrings in consolidated files
5. **Reduced Confusion**: Eliminates confusion about which file to use for a specific task

## Usage Examples

### Using the Consolidated Dataset Processor

```python
from generative_ai_module import ConsolidatedDatasetProcessor

# Create a processor
processor = ConsolidatedDatasetProcessor()

# Load and preprocess data
data = processor.load_data("path/to/data")
cleaned_data = processor.clean_text(data)
sequences = processor.create_sequences(cleaned_data, sequence_length=100)
batches = processor.create_batches(sequences, batch_size=64)

# Or use the dialogue dataset preparation
batches = processor.prepare_dialogue_dataset(
    source='persona_chat',
    sequence_length=100,
    batch_size=64
)
```

### Using the Consolidated Generation Pipeline

```python
from generative_ai_module import ConsolidatedGenerationPipeline

# Create a text generation pipeline
generator = ConsolidatedGenerationPipeline(model_type="text")

# Train the model
generator.train(dataset=batches, epochs=20)

# Generate text
generated_text = generator.generate_text(
    seed_text="Once upon a time",
    max_length=500,
    temperature=0.8
)

# Or create a code generation pipeline
code_generator = ConsolidatedGenerationPipeline(model_type="code")

# Generate code
generated_code = code_generator.generate_code(
    prompt="Write a function to calculate Fibonacci numbers",
    max_length=1000
)
```
