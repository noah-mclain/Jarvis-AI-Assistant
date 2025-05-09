# DeepSeek-Coder Fine-tuning on Paperspace Gradient

This guide explains how to fine-tune DeepSeek-Coder with Google Drive integration on Paperspace Gradient to overcome the 15GB storage limit.

## Setup Instructions

### 1. Create a Gradient Notebook

- Sign in to Paperspace Gradient
- Create a new Notebook
- Select the PyTorch 2.1 runtime
- Choose the RTX 5000 GPU option
- Set persistent storage to at least 15GB

### 2. Install Requirements

Run the following commands in your Gradient notebook:

```bash
# Clone this repository
git clone https://github.com/your-username/Jarvis-AI-Assistant.git
cd Jarvis-AI-Assistant

# Run the Google Drive setup script
bash setup_google_drive.sh
```

### 3. Set Up Google Drive

1. Create a folder in your Google Drive to store models and checkpoints
2. Open that folder and copy the folder ID from the URL
   - The folder ID is the part after "folders/" in the URL: `https://drive.google.com/drive/folders/YOUR_FOLDER_ID`
3. Edit `run_gdrive_finetune.sh` and replace `your_folder_id_here` with your actual Google Drive folder ID

## Fine-tuning with Google Drive Integration

Run the pre-configured script:

```bash
./run_gdrive_finetune.sh
```

Or customize your run with specific parameters:

```bash
python src/generative_ai_module/optimize_deepseek_gdrive.py \
    --gdrive-folder-id YOUR_FOLDER_ID \
    --output-dir /storage/models/deepseek_optimized \
    --quantize 4 \
    --max-steps 500 \
    --batch-size 4 \
    --sequence-length 1024 \
    --checkpoint-strategy improvement \
    --max-checkpoints 2
```

## Advanced Google Drive Integration

### Using Service Account for Headless Authentication

For Paperspace Gradient's headless environment, you can use a service account:

1. Go to the Google Cloud Console: https://console.cloud.google.com/apis/credentials
2. Create a project and enable the Google Drive API
3. Create a service account and download the JSON key
4. Rename it to `service-account.json` and upload it to your Gradient notebook
5. Run `setup_google_drive.sh` again to configure authentication

### Efficient Checkpoint Strategies

Choose from multiple checkpoint saving strategies to optimize storage:

- `improvement`: Only saves checkpoints that improve validation metrics (default)
- `regular`: Saves checkpoints at regular intervals
- `hybrid`: Saves both the best checkpoint and the latest checkpoint
- `all`: Saves all checkpoints (not recommended for limited storage)

Example:

```bash
python src/generative_ai_module/optimize_deepseek_gdrive.py \
    --gdrive-folder-id YOUR_FOLDER_ID \
    --checkpoint-strategy hybrid \
    --max-checkpoints 3
```

## Storage Management

### Cleaning Up Local Storage

After uploading models to Google Drive, you can free up local storage:

```bash
./cleanup_storage.sh
```

### Downloading Models from Google Drive

To use a previously uploaded model:

```python
from src.generative_ai_module.google_drive_storage import GoogleDriveStorage
from unsloth import FastLanguageModel

# Initialize Google Drive storage
gdrive = GoogleDriveStorage(folder_id="YOUR_FOLDER_ID")

# Download model from Google Drive
file_id = "YOUR_MODEL_FILE_ID"  # Get this from Google Drive
local_model_dir = gdrive.download_model(file_id, "/storage/models/downloaded_model")

# Load the downloaded model
base_model = "deepseek-ai/deepseek-coder-6.7b-base"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model,
    max_seq_length=2048,
    load_in_4bit=True
)
model = FastLanguageModel.get_peft_model(model)
model.load_adapter(local_model_dir)

# Generate code
prompt = "### Instruction: Write a function to calculate the factorial of a number.\n\n### Response:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Troubleshooting

### NumPy Version Conflicts

If you encounter NumPy version conflicts (common in Paperspace environments):

1. Run the emergency NumPy fix script:
   ```bash
   ./fix_numpy_errors.sh
   ```
2. If that doesn't work, run the more comprehensive dependency fix:

   ```bash
   ./fix_numpy.sh
   ```

3. If you still have issues, manually remove NumPy and reinstall:
   ```bash
   sudo rm -rf /usr/local/lib/python3.11/dist-packages/numpy*
   pip install numpy==1.26.4 --no-deps --force-reinstall
   ```

### Google Drive Authentication Issues

If you encounter authentication issues:

1. Try using a service account for headless authentication
2. Make sure the Google Drive API is enabled in your Google Cloud project
3. If using browser authentication, run the script locally first to authenticate, then upload the credentials

### Out-of-Memory Errors

If you encounter CUDA out-of-memory errors:

- Reduce batch size (try 1 or 2)
- Reduce sequence length (try 512 or 1024)
- Increase gradient accumulation steps (try 8 or 16)
- Use 4-bit quantization instead of 8-bit

### Storage Issues

If you encounter storage issues:

- Use the `cleanup_storage.sh` script to remove local models
- Reduce the number of checkpoints with `--max-checkpoints 1`
- Use the `improvement` checkpoint strategy to keep only the best model
