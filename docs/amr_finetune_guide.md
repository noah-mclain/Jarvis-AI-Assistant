# Fine-tuning Stable Diffusion with DreamBooth for Personal Images

This guide explains how to fine-tune the Stable Diffusion model on your personal images using DreamBooth technique.

## Overview

DreamBooth is a technique that allows you to personalize text-to-image diffusion models with your own subjects. By training on just a few images of a subject, the model learns to generate new images of that subject in different contexts while preserving its key features.

## Setup for "Amr" Fine-tuning

We've set up a fine-tuning process specifically for your personal images located at:
`data/dreambooth_dataset/Person/amr raw`

### Configuration Details

- **Base Model**: `checkpoints/v1-5-pruned-emaonly.ckpt`
- **Subject Identifier**: "Amr"
- **Class Name**: "person"
- **Training Parameters**:
  - Learning rate: 2e-7
  - Epochs: 50
  - Batch size: 1
  - Gradient accumulation steps: 8
  - Image size: 256 (reduced for memory efficiency)
  - Mixed precision: Enabled

## How to Run the Fine-tuning

We've created a script that handles the entire fine-tuning process. To run it:

```bash
python scripts/run_amr_finetune.py
```

This script will:

1. Set up the proper DreamBooth configuration
2. Process your personal images
3. Fine-tune the model
4. Save the resulting model to `checkpoints/finetuned_amr`

## Using the Fine-tuned Model

After fine-tuning completes, you can generate images with your personalized model using prompts like:

- "a photo of Amr person in Paris"
- "a painting of Amr person as a superhero"
- "Amr person wearing a suit"

The model should maintain your likeness while placing you in these new contexts.

## Troubleshooting

If you encounter memory issues during training:

1. Reduce the image size further (e.g., to 128)
2. Reduce batch size or increase gradient accumulation steps
3. Make sure mixed precision is enabled

## Additional Resources

For more information on fine-tuning, check the main fine-tuning guide at `docs/fine_tuning_guide.md`.
