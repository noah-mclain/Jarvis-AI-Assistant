# Storage Management for Paperspace Gradient + Google Drive

This guide explains how to manage storage between Paperspace Gradient local storage and Google Drive. The Jarvis AI Assistant now stores all data files (models, datasets, checkpoints, metrics, logs) on Google Drive, while using local storage as temporary space.

## Directory Structure

All data is organized into the following directories:

- `models` - Trained and fine-tuned models
- `datasets` - Raw and processed datasets
- `checkpoints` - Training checkpoints
- `metrics` - Evaluation metrics and results
- `logs` - Log files
- `preprocessed_data` - Preprocessed data

## Automatic Syncing

The system automatically syncs data between Paperspace local storage and Google Drive. When you:

- Train a model → Models and checkpoints are synced to Google Drive
- Process a dataset → Dataset is synced to Google Drive
- Run evaluations → Metrics are synced to Google Drive
- Run any script → Log files are synced to Google Drive

## Manual Storage Management

You can also manually manage storage using the `manage_storage.py` utility:

```bash
# Check current storage status (both local and Google Drive)
python -m src.generative_ai_module.manage_storage status

# Sync all data to Google Drive
python -m src.generative_ai_module.manage_storage sync

# Clear local storage for a specific folder (after syncing)
python -m src.generative_ai_module.manage_storage clear --folder models --confirm

# Clear ALL local storage (after syncing)
python -m src.generative_ai_module.manage_storage clear --confirm
```

## How It Works

The system is designed to:

1. Create local directories as needed
2. Sync data to Google Drive automatically
3. Retrieve data from Google Drive when needed

Even if you delete local directories, the system will recreate them and sync data from Google Drive when needed.

## Individual Directory Syncing

You can also sync individual directories using the `sync_gdrive.py` utility:

```bash
# Sync models to Google Drive
python -m src.generative_ai_module.sync_gdrive to-gdrive --folder models

# Sync models from Google Drive
python -m src.generative_ai_module.sync_gdrive from-gdrive --folder models
```

## Storage Safety

Important notes about storage management:

1. **ALWAYS** sync data to Google Drive before clearing local storage
2. Use the `--confirm` flag when clearing storage to prevent accidental deletion
3. The system will try to retrieve data from Google Drive when needed, but it's best to ensure data is synced before deleting

## Using in Python Code

You can also manage storage from your Python code:

```python
from generative_ai_module.utils import sync_to_gdrive, sync_from_gdrive, ensure_directory_exists

# Sync all data to Google Drive
sync_to_gdrive()

# Sync specific data from Google Drive
sync_from_gdrive("models")

# Get path and ensure directory exists
models_dir = ensure_directory_exists("models", "my_model")
```

The `ensure_directory_exists` function will:

1. Check if the directory exists locally
2. If not, try to sync it from Google Drive
3. Create the directory if it doesn't exist on Google Drive
4. Return the path to the directory
