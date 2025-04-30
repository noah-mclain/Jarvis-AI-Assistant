# Jarvis AI Assistant Dataset Training

This document explains how to use the new dataset training capabilities in Jarvis AI Assistant.

## Supported Datasets

The system now supports the following datasets:

1. **The Pile** - A diverse 825 GiB dataset consisting of 22 smaller, high-quality datasets
2. **OpenAssistant** - Conversation dataset for training assistant-like AI
3. **GPTeacher** - High-quality instruction-response pairs for instruction-following AI
4. **Persona Chat** - Character-based dialogue dataset (already supported)
5. **Writing Prompts** - Creative writing dataset (already supported)

## Unified Training Script

The recommended way to train on all datasets is using the unified training script:

```bash
python src/generative_ai_module/train_unified_models.py
```

This script trains on all datasets by default, creates visualizations, and stores models in organized directories.

### Key Features:

- **Train on multiple datasets** simultaneously or selectively
- **Training visualization** for loss, accuracy, and perplexity
- **Automatic train/validation splitting** for better model evaluation
- **Early stopping** to prevent overfitting
- **Cross-dataset comparison** visualizations

### Command-line Options:

```
usage: train_unified_models.py [-h] [--datasets {all,pile,openassistant,gpteacher,persona_chat,writing_prompts} [{all,pile,openassistant,gpteacher,persona_chat,writing_prompts} ...]]
                              [--pile-subset PILE_SUBSET] [--max-samples MAX_SAMPLES] [--validation-split VALIDATION_SPLIT] [--epochs EPOCHS]
                              [--batch-size BATCH_SIZE] [--early-stopping EARLY_STOPPING] [--model-dir MODEL_DIR] [--visualization-dir VISUALIZATION_DIR]
                              [--no-force-gpu]
```

#### Examples:

1. Train on all datasets with default settings:

   ```bash
   python src/generative_ai_module/train_unified_models.py
   ```

2. Train only on specific datasets:

   ```bash
   python src/generative_ai_module/train_unified_models.py --datasets pile openassistant
   ```

3. Train with customized parameters:

   ```bash
   python src/generative_ai_module/train_unified_models.py --max-samples 1000 --epochs 20 --validation-split 0.15 --early-stopping 5
   ```

4. Train on a specific subset of The Pile:
   ```bash
   python src/generative_ai_module/train_unified_models.py --datasets pile --pile-subset pubmed
   ```

## Dataset Utility Script

For more specific dataset operations, you can use the dataset utility script:

```bash
python src/generative_ai_module/use_new_datasets.py
```

### Available Actions:

1. **sample** - View samples from datasets

   ```bash
   python src/generative_ai_module/use_new_datasets.py --action sample --dataset pile
   ```

2. **train** - Train a model on a specific dataset

   ```bash
   python src/generative_ai_module/use_new_datasets.py --action train --dataset openassistant --max-samples 500 --epochs 10
   ```

3. **test** - Test a trained model on a dataset

   ```bash
   python src/generative_ai_module/use_new_datasets.py --action test --model-path models/openassistant_model.pt --dataset openassistant
   ```

4. **generate** - Generate text using a trained model

   ```bash
   python src/generative_ai_module/use_new_datasets.py --action generate --model-path models/pile_model.pt --prompt "Explain quantum computing"
   ```

5. **unified-train** - Run the unified training script (shortcut)
   ```bash
   python src/generative_ai_module/use_new_datasets.py --action unified-train --dataset all
   ```

## Visualizations

Training visualizations are automatically generated and saved to the `visualizations` directory (customizable with `--visualization-dir`).

The following visualizations are created:

1. **Loss curves** - Training and validation loss over time
2. **Accuracy** - Model prediction accuracy over time
3. **Perplexity** - Model perplexity over time
4. **Cross-dataset comparisons** - Compare metrics across different datasets

## Dataset-Specific Preprocessing

Each dataset is preprocessed appropriately based on its specific structure:

- **The Pile**: Organized by source with metadata preserved
- **OpenAssistant**: Structured as conversations with USER/ASSISTANT roles
- **GPTeacher**: Preserved instruction/response structure
- **Persona Chat**: Maintained persona information and dialogue structure
- **Writing Prompts**: Preserved prompt/story format

## Requirements

To use these datasets, you need the Hugging Face datasets library:

```bash
pip install datasets transformers tqdm matplotlib
```

## Troubleshooting

### Common Issues:

1. **Out of memory errors**: Reduce batch size or max_samples

   ```bash
   python src/generative_ai_module/train_unified_models.py --batch-size 16 --max-samples 200
   ```

2. **Dataset not found**: Make sure you have internet access for downloading datasets

3. **Slow processing**: Use GPU acceleration (default) or reduce dataset size

   ```bash
   python src/generative_ai_module/train_unified_models.py --max-samples 100
   ```

4. **Model loading errors**: Ensure the model path is correct
   ```bash
   # Check if model exists
   ls -la models/
   ```

## Next Steps

After training models on these datasets, you can:

1. Compare model performance across different datasets
2. Fine-tune hyperparameters to improve results
3. Generate text with trained models
4. Create ensemble models combining knowledge from multiple datasets

For more information, please refer to the source code documentation.
