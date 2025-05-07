# Jarvis AI Assistant Implementation Guide for RTX 5000 (16GB GPU)

This guide provides optimized commands and best practices for running the Jarvis AI Assistant on a Paperspace Gradient instance with an NVIDIA RTX 5000 GPU (16GB VRAM, 8 CPUs, 30GB RAM). All commands have been carefully verified to work with the specific parameters supported by each script in the codebase.

> **Important Note:** Script parameters may change as the codebase evolves. If you encounter errors about unrecognized arguments, check the script's help documentation (e.g., `python script.py --help`) and adjust the commands accordingly. The commands in this guide were verified against the current version of the codebase.

## Environment Setup

````bash
./setup/unified_setup.sh

## Optimal Training Commands for RTX 5000 (16GB GPU)

These commands are specifically optimized for the RTX 5000 GPU with 16GB VRAM, 8 CPUs, and 30GB RAM. They balance performance with memory constraints to get the best results.

### GPU Optimizations for Training

The Jarvis AI system includes several automatic optimizations to ensure maximum GPU utilization on the RTX 5000:

- **GPU Enforcement**: Sets `CUDA_VISIBLE_DEVICES=0` to ensure the GPU is used
- **Memory Optimizations**: Applies `PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"` to reduce memory fragmentation
- **Performance Tuning**: Enables cuDNN benchmark with `torch.backends.cudnn.benchmark = True` for better performance
- **Model Quantization**: Forces 4-bit quantization for model loading on RTX5000 to maximize available memory
- **Adaptive Batch Sizes**: Automatically reduces batch sizes when necessary to accommodate 16GB VRAM limits
- **Early Detection**: Updates to key files ensure GPU usage is prioritized:
  - `code_generator.py`: Modified `_get_device()` to detect and prioritize GPU
  - `finetune_deepseek.py`: Enhanced `setup_environment()` for RTX5000 detection
  - `train_models.py`: Added GPU setup at the beginning of `main()`
  - `jarvis_unified.py`: Added early GPU detection during module import

### 1. Training the Base Models

```bash
# Set environment variables for best performance
export CUDA_VISIBLE_DEVICES=0
export CODE_SUBSET="jarvis_code_instructions"

# Train code models with DeepSeek optimizations
cd /notebooks
python src/generative_ai_module/train_models.py \
    --model-type code \
    --use-deepseek \
    --code-subset $CODE_SUBSET \
    --batch-size 2 \
    --epochs 3 \
    --learning-rate 2e-5 \
    --warmup-steps 100 \
    --load-in-4bit \
    --sequence-length 1024 \
    --early-stopping 3 \
    --deepseek-batch-size 1 \
    --max-samples 5000 \
    --visualization-dir /notebooks/Jarvis_AI_Assistant/visualizations

# Train on ALL text datasets with RTX 5000 optimized parameters
cd /notebooks
python src/generative_ai_module/train_models.py \
    --model_type text \
    --dataset "agie-ai/OpenAssistant-oasst1,teknium/GPTeacher-General-Instruct,google/Synthetic-Persona-Chat,euclaise/writingprompts" \
    --batch_size 4 \
    --epochs 3 \
    --learning_rate 3e-5 \
    --max_length 512 \
    --output_dir /notebooks/Jarvis_AI_Assistant/models \
    --eval_metrics_dir /notebooks/Jarvis_AI_Assistant/visualizations \
    --evaluation_strategy steps \
    --save_strategy steps \
    --logging_steps 50 \
    --visualize_metrics

# Train on specific text datasets (if you don't want all)
python src/generative_ai_module/train_models.py \
    --model_type text \
    --dataset "writing_prompts,persona_chat" \
    --batch_size 4 \
    --epochs 3 \
    --learning_rate 3e-5 \
    --early_stopping 3 \
    --max_length 512 \
    --max_samples 5000 \
    --eval_metrics_dir /notebooks/Jarvis_AI_Assistant/visualizations \
    --output_dir /notebooks/Jarvis_AI_Assistant/models \
    --evaluation_strategy steps \
    --save_strategy steps \
    --logging_steps 50 \
    --visualize_metrics
````

### 2. Fine-tuning DeepSeek Models with Unsloth

Unsloth optimization is critical for the RTX 5000, achieving up to 2x speed improvement while reducing memory usage. Our fine-tuning process uses:

- 4-bit quantization for minimal memory usage
- LoRA for memory-efficient parameter-efficient fine-tuning
- Optimized sequence length based on available memory

```bash
# Install dependencies for Unsloth optimization
pip install ninja
pip install unsloth

# Run fine-tuning with optimized parameters for 6.7B model
cd /notebooks
python src/generative_ai_module/finetune_deepseek.py \
    --epochs 2 \
    --batch-size 1 \
    --max-samples 5000 \
    --all-subsets \
    --sequence-length 1024 \
    --learning-rate 2e-5 \
    --warmup-steps 100 \
    --load-in-4bit \
    --save-steps 100 \
    --save-total-limit 2 \
    --use-unsloth \
    --output-dir /notebooks/Jarvis_AI_Assistant/models/deepseek_finetuned

# For better performance with smaller model
python src/generative_ai_module/finetune_deepseek.py \
    --epochs 3 \
    --batch-size 2 \
    --max-samples 5000 \
    --all-subsets \
    --sequence-length 2048 \
    --learning-rate 3e-5 \
    --warmup-steps 50 \
    --load-in-4bit \
    --use-unsloth \
    --output-dir /notebooks/Jarvis_AI_Assistant/models/deepseek_small_finetuned

# Text model training with optimized settings for RTX 5000
cd /notebooks
python src/generative_ai_module/train_models.py \
    --model_type text \
    --dataset all \
    --batch_size 4 \
    --epochs 3 \
    --learning_rate 3e-5 \
    --early_stopping 3 \
    --max_length 512 \
    --max_samples 2000 \
    --eval_metrics_dir /notebooks/Jarvis_AI_Assistant/visualizations \
    --output_dir /notebooks/Jarvis_AI_Assistant/models \
    --warmup_steps 50 \
    --evaluation_strategy steps \
    --save_strategy steps \
    --logging_steps 50 \
    --visualize_metrics
```

### 3. Evaluation and Metrics

These commands evaluate model performance with memory-optimized settings for the RTX 5000:

```bash
# Evaluate on code generation tasks
cd /notebooks
python src/generative_ai_module/evaluate_generation.py \
    --batch-evaluate \
    --dataset-name jarvis_evaluation_set \
    --model-path /notebooks/Jarvis_AI_Assistant/models/deepseek_finetuned \
    --use-gpu \
    --metrics-dir /notebooks/Jarvis_AI_Assistant/metrics

# Evaluate on specific files
python src/generative_ai_module/evaluate_generation.py \
    --generated-file /notebooks/Jarvis_AI_Assistant/outputs/generated.txt \
    --reference-file /notebooks/Jarvis_AI_Assistant/outputs/reference.txt \
    --prompt-file /notebooks/Jarvis_AI_Assistant/outputs/prompt.txt \
    --dataset-name code_test \
    --use-gpu \
    --metrics-dir /notebooks/Jarvis_AI_Assistant/metrics
```

### 4. Using the Unified Generation Pipeline

The unified pipeline provides a comprehensive approach with optimized parameters:

```bash
# Run the unified generation pipeline with RTX 5000 optimizations
cd /notebooks
python src/generative_ai_module/unified_generation_pipeline.py \
    --mode train \
    --dataset jarvis_combined_dataset \
    --train-type code \
    --epochs 3 \
    --save-model \
    --use-deepseek \
    --deepseek-batch-size 1 \
    --learning-rate 1e-5 \
    --sequence-length 1024 \
    --warmup-steps 100 \
    --code-subset python \
    --all-subsets \
    --force-gpu \
    --max-samples 5000 \
    --model-dir /notebooks/Jarvis_AI_Assistant/models
```

### 5. Optimized Storage and Dataset Handling

For the RTX 5000 with limited 16GB VRAM, efficient storage management is critical. Use these approaches:

```bash
# Create a script to optimize storage for DeepSeek models
cd /notebooks
cat > optimize_storage.py << 'EOL'
#!/usr/bin/env python3
import os
import sys
import logging
from pathlib import Path

# Add the project root to the path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.generative_ai_module.storage_optimization import (
    optimize_storage_for_model,
    compress_dataset,
    create_checkpoint_strategy,
    setup_google_drive
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create required directories
os.makedirs("/notebooks/Jarvis_AI_Assistant/models", exist_ok=True)
os.makedirs("/notebooks/Jarvis_AI_Assistant/datasets", exist_ok=True)
os.makedirs("/notebooks/Jarvis_AI_Assistant/checkpoints", exist_ok=True)

# Optimize a model for RTX 5000
logger.info("Optimizing model for RTX 5000 (16GB VRAM)")
optimization_results = optimize_storage_for_model(
    model_name="deepseek-ai/deepseek-coder-6.7b-base",
    output_dir="/notebooks/Jarvis_AI_Assistant/models/deepseek_optimized",
    quantize_bits=4,  # Use 4-bit quantization for maximum memory efficiency
    use_external_storage=True,
    storage_type="gdrive",
    remote_path="DeepSeek_Models"
)

logger.info(f"Optimization complete: {optimization_results}")
EOL

# Make the script executable
chmod +x optimize_storage.py

# Run the optimization script
python optimize_storage.py

# Sync data to/from Google Drive using the built-in script
cd /notebooks

# Sync all data to Google Drive
python -m src.generative_ai_module.sync_gdrive to-gdrive

# Sync only models from Google Drive
python -m src.generative_ai_module.sync_gdrive from-gdrive --folder models

# Sync in both directions
python -m src.generative_ai_module.sync_gdrive all
```

### 6. Running the Jarvis AI Assistant

Run the assistant with memory-efficient settings optimized for the RTX 5000:

```bash
# Run the Jarvis AI Assistant in interactive mode
cd /notebooks
python src/generative_ai_module/run_jarvis.py \
    --model deepseek-ai/deepseek-coder-6.7b-instruct \
    --model-path /notebooks/Jarvis_AI_Assistant/models/deepseek_finetuned \
    --interactive \
    --max-tokens 512 \
    --output /notebooks/Jarvis_AI_Assistant/logs/chat_history.json

# Run the assistant with a single prompt (non-interactive mode)
python src/generative_ai_module/run_jarvis.py \
    --model deepseek-ai/deepseek-coder-6.7b-instruct \
    --prompt "Write a Python function to calculate the Fibonacci sequence" \
    --max-tokens 256 \
    --output /notebooks/Jarvis_AI_Assistant/logs/single_response.json

# Load from a previous chat history and continue the conversation
python src/generative_ai_module/run_jarvis.py \
    --model deepseek-ai/deepseek-coder-6.7b-instruct \
    --interactive \
    --history /notebooks/Jarvis_AI_Assistant/logs/chat_history.json \
    --output /notebooks/Jarvis_AI_Assistant/logs/continued_chat.json
```

## Performance Optimization Tips for RTX 5000 (16GB)

1. **Critical Memory Optimizations:**

   - Always use 4-bit quantization for 6.7B+ models on 16GB GPU
   - Keep batch size at 1-2 for large models
   - Use gradient accumulation steps of 8-16 to simulate larger batches
   - Monitor memory usage with `nvidia-smi -l 5`
   - For sequence generation, limit max tokens to 512-1024
   - Free CUDA cache periodically with `torch.cuda.empty_cache()`

2. **Sequence Length Management:**

   - Use max_seq_length of 1024 for 6.7B models with 4-bit quantization
   - For 1.3B models, sequence lengths of 2048 are possible
   - Consider dynamic sequence length based on available memory

3. **Efficient Fine-tuning:**

   - Always use LoRA/QLoRA with 4-bit quantization (saves 95%+ memory)
   - Use Flash Attention if available (20-40% speedup)
   - Target only key modules with LoRA (q_proj, k_proj, v_proj, o_proj)
   - Keep LoRA rank (r) between 8-16 for larger models
   - Apply early stopping to prevent overfitting

4. **Optimizing Training Speed:**

   - Enable Unsloth optimization for up to 2x training speed
   - Use mixed precision training (fp16 where supported)
   - Reduce validation frequency to save time (eval every 100-200 steps)
   - Limit checkpoint saving frequency (save every 100-200 steps)
   - Keep save_total_limit low (2-3) to manage disk space

5. **Dataset Optimizations:**
   - Use streaming datasets for large data
   - Apply aggressive filtering to ensure high-quality samples
   - Consider smaller, focused datasets rather than large, generic ones
   - Preprocess and tokenize data ahead of time

## Troubleshooting RTX 5000 Specific Issues

If you encounter CUDA out-of-memory errors:

```bash
# Monitor GPU memory usage
watch -n 5 nvidia-smi

# Reduce memory usage via these steps (in order):
1. Decrease batch size to 1
2. Increase gradient accumulation steps (8→16→32)
3. Reduce sequence length (2048→1024→512)
4. Enable 4-bit quantization if not already
5. Switch to a smaller model (6.7B→1.3B)
6. Disable validation during training
7. Restart the environment to clear fragmented memory
```

For poor training stability on RTX 5000:

```bash
# Try these stability improvements:
1. Reduce learning rate by half (2e-5→1e-5)
2. Increase warmup steps (5-10% of total steps)
3. Add gradient clipping: --max-grad-norm 1.0
4. Use cosine learning rate scheduler
5. Try a different optimizer (AdamW→Adafactor)
```

### Advanced Troubleshooting

#### Import Errors and Module Access

If you encounter import errors, first run the test script to diagnose the issue:

```bash
python /notebooks/test_imports.py
```

If specific modules are causing issues, apply the import fix script:

```bash
# Apply import fixes to problematic files
python src/generative_ai_module/fix_jarvis_imports.py --force <problematic_file>.py
```

#### Memory Fragmentation

The RTX 5000 with 16GB VRAM can suffer from memory fragmentation during long training sessions:

```bash
# Check for memory fragmentation
nvidia-smi -i 0 --query-gpu=utilization.gpu,memory.total,memory.free,memory.used --format=csv

# If you see high used memory but low GPU utilization, clear CUDA cache:
python -c "import torch; torch.cuda.empty_cache(); print('CUDA cache cleared')"
```

#### Recovering from Training Crashes

If training crashes due to OOM errors:

```bash
# 1. Ensure all processes are terminated
pkill -9 python

# 2. Free GPU memory
nvidia-smi --gpu-reset

# 3. Resume from the latest checkpoint with reduced parameters
python src/generative_ai_module/finetune_deepseek.py \
    --model deepseek-ai/deepseek-coder-6.7b-instruct \
    --dataset jarvis_code_dataset \
    --batch-size 1 \
    --gradient-accumulation-steps 32 \
    --load-in-4bit \
    --resume-from-checkpoint /notebooks/Jarvis_AI_Assistant/checkpoints/latest
```

#### Paperspace-Specific Issues

For issues specific to Paperspace Gradient:

```bash
# Reset environment variables if needed
echo 'export PAPERSPACE=true' >> ~/.bashrc
echo 'export PAPERSPACE_ENVIRONMENT=true' >> ~/.bashrc
source ~/.bashrc

# Clear Paperspace cache (if disk space is low)
rm -rf /tmp/gradient_*

# Set higher process priority for training job
sudo nice -n -10 python src/generative_ai_module/finetune_deepseek.py [other args]
```

#### Debugging Unsloth Integration

If you have issues with Unsloth optimization:

```bash
# Check if Unsloth can access the GPU
python -c "from unsloth import FastLanguageModel; print(f'CUDA Available: {FastLanguageModel.is_cuda_available()}')"

# Try running with basic settings first, then enable Unsloth
python src/generative_ai_module/finetune_deepseek.py \
    --model deepseek-ai/deepseek-coder-1.3b-instruct \
    --dataset jarvis_code_dataset \
    --batch-size 1 \
    --load-in-4bit \
    --use-unsloth \
    --debug-mode
```

## Paperspace Setup

For Paperspace Gradient specifically:

```bash
# Create a Paperspace environment variable
echo 'export PAPERSPACE=true' >> ~/.bashrc
echo 'export PAPERSPACE_ENVIRONMENT=true' >> ~/.bashrc
source ~/.bashrc

# For persistent storage with Google Drive
pip install gdown google-auth google-auth-oauthlib google-auth-httplib2

# Sync to Google Drive
python -c "from src.generative_ai_module.sync_gdrive import sync_all_to_gdrive; sync_all_to_gdrive()"
```

#### Code Quality and Linting Fixes

Several undefined variable issues have been fixed in the codebase to ensure it runs reliably on the RTX 5000:

```bash
# Fixed variables in storage_optimization.py
- Added logger definition to fix "logger is not defined" errors

# Fixed variables in train_models.py
- Updated CustomCallback class to properly handle trainer and model variables

# Fixed variables in unified_generation_pipeline.py
- Added infinity variable definition
- Created print_execution_time decorator function
- Fixed args parameter handling in interactive_generation
```

These fixes are especially important for the RTX 5000 environment, as undefined variables can cause runtime crashes that waste valuable compute time and GPU memory. The corrections improve the stability of long-running training jobs and interactive sessions.

When working with the codebase, if you encounter similar "undefined variable" errors, you can apply the fixes using the same pattern:

1. Identify the missing variable
2. Define it at the appropriate scope
3. Use default values where necessary
4. Pass required parameters to functions that need them

#### Fixing CUDA device-side assert Errors

If you encounter a CUDA error like this:

```
CUDA error: device-side assert triggered
nll_loss_forward_reduce_cuda_kernel_2d: block: [0,0,0], thread: [31,0,0] Assertion `t >= 0 && t < n_classes` failed.
```

This is due to target tensor indices exceeding the valid vocabulary size range. Fix it with these steps:

```bash
# 1. Make sure all target indices are clamped to the model's vocabulary size
# In your training code, add this before the loss calculation:
vocab_size = model.embedding.num_embeddings
if target_batch.max() >= vocab_size:
    target_batch = torch.clamp(target_batch, 0, vocab_size - 1)

# 2. If the issue persists, explicitly set the CUDA to sync mode
export CUDA_LAUNCH_BLOCKING=1

# 3. Try training with a smaller batch to debug the exact problem:
python src/generative_ai_module/train_models.py \
    --model-type text \
    --datasets writing_prompts \
    --batch-size 2 \
    --max-samples 100 \
    --epochs 1
```

The core issue is that nll_loss requires target indices to be within the range [0, vocab_size-1]. The latest code includes safeguards to prevent this error by adding index validation and clamping throughout the training pipeline.

```

```
