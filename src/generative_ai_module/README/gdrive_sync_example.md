# Google Drive Sync Functionality

This guide demonstrates how to use the Google Drive synchronization features that have been integrated into the Jarvis AI Assistant. These features allow you to automatically save and retrieve datasets, models, metrics, and checkpoints between your local Paperspace workspace and Google Drive.

## Prerequisites

1. You must have set up rclone to connect your Paperspace Gradient notebook to Google Drive
2. The rclone configuration should have a remote named "gdrive"

## Automatic Sync

The Jarvis AI Assistant now has automatic synchronization built into many operations:

- When training models, checkpoints and metrics are automatically synced to Google Drive
- When saving processed datasets, they are automatically synced to Google Drive
- When saving evaluation metrics, they are automatically synced to Google Drive

## Manual Sync Operations

### From Python Code

You can manually trigger sync operations from your Python code:

```python
from generative_ai_module import sync_to_gdrive, sync_from_gdrive

# Sync all models to Google Drive
sync_to_gdrive("models")

# Sync all datasets from Google Drive
sync_from_gdrive("datasets")

# Sync everything to Google Drive
sync_to_gdrive()  # None means sync all folders

# Sync everything from Google Drive
sync_from_gdrive()  # None means sync all folders
```

### From Command Line

You can also use the command-line tool to perform sync operations:

```bash
# Sync all models to Google Drive
python -m src.generative_ai_module.sync_gdrive to-gdrive --folder models

# Sync all datasets from Google Drive
python -m src.generative_ai_module.sync_gdrive from-gdrive --folder datasets

# Sync everything in both directions
python -m src.generative_ai_module.sync_gdrive all
```

## Using Correct File Paths

To ensure your code uses the correct file paths (that will be automatically synced), use the `get_storage_path` function:

```python
from generative_ai_module import get_storage_path

# Get the path to save model files
model_path = get_storage_path("models", "my_model_name")

# Get the path to save dataset files
dataset_path = get_storage_path("datasets", "my_dataset_name")

# Get the path to save metrics
metrics_path = get_storage_path("metrics", "my_evaluation_metrics.json")
```

This will return the correct path regardless of whether you're running in Paperspace or in a local development environment.

## Managing Large Files

When working with large models or datasets, it's advisable to:

1. Only sync what you need (use the folder-specific sync functions)
2. Consider scheduling syncs during breaks or after training sessions
3. Be mindful of your Google Drive storage limits

## Troubleshooting

If you encounter issues with syncing:

1. Check that rclone is correctly configured with `rclone config show`
2. Verify you have sufficient storage space on Google Drive
3. Look at the logs for detailed error messages (the sync functions log detailed information)
4. Try manually running rclone commands to debug: `rclone ls gdrive:Jarvis_AI_Assistant`
