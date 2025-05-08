# RTX 5000 GPU Optimized Training Guide

This guide provides optimized training configurations for the Jarvis AI Assistant on RTX 5000 GPUs (16GB VRAM).

## Quick Start

```bash
# Run the optimized training script with interactive model selection
./fix_and_run_training.sh
```

## Optimized Configuration Rationale

### Text Model Configuration

```bash
python -m src.generative_ai_module.train_models \
  --model_type text \
  --batch_size 8 \                # Maximize parallel processing
  --max_length 1024 \             # Ideal for narrative coherence
  --gradient_accumulation_steps 4 \  # Effective batch=32
  --max_samples 50000 \           # Quality > quantity
  --learning_rate 3e-5 \
  --weight_decay 0.05 \           # Regularization for creative tasks
  --use_4bit \                    # 4-bit QLoRA
  --use_flash_attention_2 \       # Critical for long sequences
  --gradient_checkpointing \      # Memory-for-compute tradeoff
  --optim adamw_bnb_8bit \        # 8-bit AdamW
  --eval_steps 500 \              # Frequent validation
  --save_steps 1000
```

#### Why This Works for Text

- Longer sequences (1024 tokens) capture story arcs
- Larger batches (8) improve throughput
- 4-bit + flash attention reduces memory by ~40%
- Sequence packing combines shorter samples to maximize efficiency

### Code Model Configuration

```bash
python -m src.generative_ai_module.train_models \
  --model_type code \
  --batch_size 2 \                # Handle long code sequences
  --max_length 2048 \             # Critical for code blocks/indentation
  --gradient_accumulation_steps 16 \  # Effective batch=32
  --max_samples 80000 \           # Needs more syntax diversity
  --learning_rate 2e-5 \
  --weight_decay 0.1 \            # Stronger regularization
  --use_4bit \
  --use_flash_attention_2 \
  --gradient_checkpointing \
  --optim adamw_bnb_8bit \
  --eval_steps 1000 \             # Code eval is slower
  --pad_token_id 50256 \          # GPT-2's <|endoftext|> token
  --fim_rate 0.5 \               # Fill-in-middle for code
  --num_workers 4                # Preprocess code in parallel
```

#### Why This Works for Code

- 2048 tokens handle complex code structures
- Smaller batches (2) prevent OOM with long sequences
- Higher gradient accumulation (16) maintains stability
- Fill-in-middle training improves code completion tasks

## Memory Monitoring

Monitor GPU memory usage during training:

```bash
# Start the GPU monitor in a separate terminal
python monitor_gpu.py --interval 2
```

If memory usage exceeds 14.5GB:

- For text models: Reduce `batch_size` to 6
- For code models: Reduce `max_length` to 1536

## CNN-Enhanced Text Model

For tasks requiring pattern recognition (e.g., writing generation), the CNN-enhanced text model combines convolutional layers with transformers:

```bash
python -m src.generative_ai_module.train_models \
  --model_type text \
  --use_cnn \
  --cnn_layers 2 \
  --sequence_packing \
  # ... use same parameters as text model
```

## Implementation Notes

### For Text Models

1. **Sequence Packing**  
   Combining short examples to fill token capacity:

   ```python
   # Example in code: Concatenate 3 stories
   input_str = story1 + "<|endoftext|>" + story2 + "<|endoftext|>" + story3
   ```

2. **Dynamic Padding**  
   Using `pad_sequence` with batch-first:
   ```python
   from torch.nn.utils.rnn import pad_sequence
   batch = pad_sequence(batch, batch_first=True, padding_value=tokenizer.pad_token_id)
   ```

### For Code Models

1. **Fill-in-Middle (FIM)**  
   Structure samples as:

   ```
   <PRE> function def() { ... <SUF> ... } <MID> // implementation
   ```

2. **Code-Specific Tokenization**  
   Adding special tokens:
   ```python
   tokenizer.add_tokens(["<FIM_PRE>", "<FIM_SUF>", "<FIM_MID>"])
   ```

## Validation Metrics

| Model Type | Key Metrics                              | Tools                       |
| ---------- | ---------------------------------------- | --------------------------- |
| **Text**   | Perplexity, BLEU, ROUGE-L                | `evaluate` + custom prompts |
| **Code**   | CodeBLEU, Compilation Rate, FIM Accuracy | `code-eval` + unit tests    |

## Troubleshooting

If you encounter OOM errors:

1. Check your VRAM usage:

   ```bash
   watch -n 1 "nvidia-smi --query-gpu=memory.used --format=csv"
   ```

2. Try reducing batch size or sequence length

3. Ensure environment variables are set:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
   ```
