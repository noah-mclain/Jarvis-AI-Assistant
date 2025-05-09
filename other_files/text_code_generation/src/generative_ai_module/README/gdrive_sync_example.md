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
- When saving log files, they are automatically synced to Google Drive

## Manual Sync Operations

### From Python Code

You can manually trigger sync operations from your Python code:

```python
from generative_ai_module import sync_to_gdrive, sync_from_gdrive, sync_logs

# Sync all models to Google Drive
sync_to_gdrive("models")

# Sync all datasets from Google Drive
sync_from_gdrive("datasets")

# Sync all logs to Google Drive
sync_logs()

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

# Sync all logs to Google Drive
python -m src.generative_ai_module.sync_gdrive to-gdrive --folder logs

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

# Get the path to save logs
logs_path = get_storage_path("logs", "my_log_file.log")
```

## Logging Setup

The module provides a dedicated logging setup function that creates logs in the synced directory:

```python
from generative_ai_module import setup_logging

# Set up logging with automatic file name
log_path = setup_logging()

# Set up logging with custom file name
log_path = setup_logging("my_custom_log.log")
```

This will configure logging to both console and a log file in the logs directory, which will be automatically synced to Google Drive.

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
