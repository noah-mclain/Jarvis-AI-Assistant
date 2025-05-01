# DeepSeek-Coder Fine-tuning on Paperspace Gradient

This guide explains how to fine-tune DeepSeek-Coder with storage optimizations on Paperspace Gradient using an RTX 5000 GPU.

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

# Run the storage optimization setup script
bash setup_storage_optimization.sh
```

## Fine-tuning with Storage Optimization

The following command will run fine-tuning with 4-bit quantization and storage optimization:

```bash
python src/generative_ai_module/optimize_deepseek_storage.py \
    --storage-type local \
    --output-dir /storage/models/deepseek_optimized \
    --quantize 4 \
    --max-steps 500 \
    --batch-size 4 \
    --sequence-length 1024 \
    --checkpoint-strategy improvement \
    --max-checkpoints 2
```

### Using S3 for External Storage

For larger datasets and models, you can use AWS S3 for storage:

```bash
python src/generative_ai_module/optimize_deepseek_storage.py \
    --storage-type s3 \
    --s3-bucket your-bucket-name \
    --aws-access-key-id YOUR_ACCESS_KEY \
    --aws-secret-access-key YOUR_SECRET_KEY \
    --output-dir /storage/models/deepseek_optimized \
    --quantize 4 \
    --max-steps 500
```

## Storage Optimization Features

This implementation includes several features to maximize storage efficiency:

1. **4-bit Quantization**: Reduces model size by ~87% compared to full precision
2. **LoRA Fine-tuning**: Only saves adapter weights (~10-100MB) instead of full model (13+GB)
3. **Efficient Checkpointing**: Only keeps the best-performing model checkpoints
4. **External Storage Integration**: Optional S3 or Google Drive integration
5. **Dataset Streaming**: Minimizes storage usage for training data

## Using the Fine-tuned Model

After fine-tuning, you can use the model as follows:

```python
from unsloth import FastLanguageModel

# Load the fine-tuned model
base_model = "deepseek-ai/deepseek-coder-6.7b-base"
adapter_path = "/storage/models/deepseek_optimized"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model,
    max_seq_length=2048,
    load_in_4bit=True
)
model = FastLanguageModel.get_peft_model(model)
model.load_adapter(adapter_path)

# Generate code
prompt = "### Instruction: Write a function to calculate the factorial of a number.\n\n### Response:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Troubleshooting

If you encounter CUDA out-of-memory errors:

- Reduce batch size (try 1 or 2)
- Reduce sequence length
- Increase gradient accumulation steps

For storage issues:

- Use 4-bit quantization instead of 8-bit
- Enable external storage with S3
- Reduce the number of checkpoints to keep
