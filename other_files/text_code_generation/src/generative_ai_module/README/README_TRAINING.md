# Jarvis AI Assistant Training Guide

This guide provides detailed instructions on how to train the various models in the Jarvis AI Assistant system with the optimized parameters.

## Training Requirements

Before starting training, ensure you have:

1. Installed all required dependencies in `requirements.txt`
2. Downloaded and processed the necessary datasets
3. At least 8GB of RAM (16GB+ recommended)
4. GPU acceleration if possible (CUDA or MPS for Apple Silicon)

## Quick Start Commands

### Preprocessing Datasets

Before training, you need to preprocess the datasets:

```bash
# Preprocess all supported datasets
python -m src.generative_ai_module.unified_generation_pipeline --preprocess --datasets all

# Preprocess a specific dataset
python -m src.generative_ai_module.unified_generation_pipeline --preprocess --datasets pile
```

### Training Text Generation Models

Training with the optimized parameters (batch_size=16, gradient_accumulation_steps=4, sequence_length=2048):

```bash
# Train on all datasets
python -m src.generative_ai_module.unified_generation_pipeline --train --datasets all --epochs 50 --use-scheduler

# Train on a specific dataset
python -m src.generative_ai_module.unified_generation_pipeline --train --datasets pile --epochs 50 --use-scheduler
```

### Fine-tuning DeepSeek Code Model

Fine-tune the DeepSeek code model with optimized parameters (batch_size=2, gradient_accumulation_steps=8, sequence_length=2048):

```bash
# Fine-tune the code model
python -m src.generative_ai_module.finetune_deepseek --epochs 30 --batch-size 2 --sequence-length 2048
```

## Advanced Training Options

### Controlling GPU Usage

```bash
# Force GPU usage
python -m src.generative_ai_module.unified_generation_pipeline --train --datasets pile --force-gpu

# Disable GPU (use CPU only)
python -m src.generative_ai_module.unified_generation_pipeline --train --datasets pile --no-gpu
```

### Learning Rate and Optimization

```bash
# Customize learning rate
python -m src.generative_ai_module.unified_generation_pipeline --train --datasets pile --learning-rate 0.001

# Enable gradient clipping
python -m src.generative_ai_module.unified_generation_pipeline --train --datasets pile --clip-value 1.0

# Use learning rate scheduler
python -m src.generative_ai_module.unified_generation_pipeline --train --datasets pile --use-scheduler
```

### Validation and Checkpointing

```bash
# Set validation split ratio
python -m src.generative_ai_module.unified_generation_pipeline --train --datasets pile --validation-split 0.15

# Load from checkpoint and continue training
python -m src.generative_ai_module.unified_generation_pipeline --train --datasets pile --model-path models/pile/checkpoint-123.pt
```

## Monitoring Training

During training, progress will be logged and visualized:

1. **Console Output**: Real-time training metrics
2. **Checkpoint Logs**: Detailed logs saved in the checkpoints directory
3. **Visualizations**: Training curves and metrics saved in the visualizations directory

To view training progress and results:

```bash
# Open the visualizations directory
open visualizations/

# Check training logs
cat checkpoints/pile_checkpoints/pile_training_log.txt
```

## Viewing Results and Generating Text

After training, you can generate text using your trained model:

```bash
# Generate text using the latest model
python -m src.generative_ai_module.unified_generation_pipeline --generate --model-path models/pile/pile_model_latest.pt --prompt "Once upon a time" --max-length 500
```

## Troubleshooting

### Memory Issues

If you encounter memory errors:

1. Reduce batch size (e.g., from 16 to 8)
2. Increase gradient accumulation steps (e.g., from 4 to 8)
3. Reduce sequence length if necessary (e.g., from 2048 to 1024)

```bash
# Example with reduced memory usage
python -m src.generative_ai_module.unified_generation_pipeline --train --datasets pile --batch-size 8 --gradient-accumulation-steps 8 --sequence-length 1024
```

### Training Instability

If training becomes unstable:

1. Reduce learning rate (e.g., from 0.002 to 0.0005)
2. Enable gradient clipping with a lower value (e.g., --clip-value 0.5)
3. Use a fixed random seed for reproducibility (e.g., --seed 42)

## Performance Comparison

To compare the performance of models trained with different parameters:

```bash
# View the training report
open visualizations/training_report_*.html
```

The report includes comparisons of:

- Loss curves
- Accuracy metrics
- Perplexity values
- Training times
- Hardware utilization

## Best Practices

1. **Start Small**: Begin with a small subset of data to verify your training pipeline
2. **Checkpoint Frequently**: Always use checkpointing to avoid losing progress
3. **Monitor Resources**: Keep an eye on GPU/CPU usage and memory consumption
4. **Visualize Results**: Always create visualizations to better understand model performance
5. **Iterative Improvement**: Use results from one training run to inform the next
