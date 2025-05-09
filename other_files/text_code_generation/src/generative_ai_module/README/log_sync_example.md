# Log File Syncing with Google Drive

This guide explains how to sync log files (including `finetune_output.log`) between your local Paperspace environment and Google Drive.

## Automatic Log Syncing

The Jarvis AI Assistant now includes automatic log syncing. When you run training, fine-tuning, or evaluation scripts, logs are automatically saved to the `logs` directory and synced to Google Drive.

## Syncing Existing Log Files

If you have existing log files (like `finetune_output.log`), you can sync them to Google Drive and optionally delete them from local storage:

```bash
# Sync finetune_output.log to Google Drive (keeping the local copy)
python -m src.generative_ai_module.sync_finetune_log --log-file finetune_output.log

# Sync and delete the local copy to free up storage
python -m src.generative_ai_module.sync_finetune_log --log-file finetune_output.log --delete-after-sync
```

## From Python Code

You can also sync logs from your Python code:

```python
from generative_ai_module import sync_logs, save_log_file

# Sync all logs to Google Drive
sync_logs()

# Save a specific log content and sync it
content = "This is my log content"
save_log_file(content, "my_custom_log.log")
```

## Logging Setup

To set up logging in your scripts that will automatically save to the logs directory and sync to Google Drive:

```python
from generative_ai_module import setup_logging

# Set up logging with automatic file name
log_path = setup_logging()

# Or specify a custom log file name
log_path = setup_logging("my_training_run.log")

# Now use regular logging
import logging
logger = logging.getLogger(__name__)
logger.info("This will be logged to both console and file")
```

## Sync All Logs Directory

To sync the entire logs directory:

```bash
# Sync logs to Google Drive
python -m src.generative_ai_module.sync_gdrive to-gdrive --folder logs

# Get logs from Google Drive
python -m src.generative_ai_module.sync_gdrive from-gdrive --folder logs
```

## Cleaning Up Local Storage

After syncing logs to Google Drive, you can safely delete them from your Paperspace storage to free up space:

```bash
# Delete local logs after confirming they've been synced
rm -rf /notebooks/Jarvis_AI_Assistant/logs/*.log
```

Remember to only delete logs after confirming they've been successfully synced to Google Drive.
