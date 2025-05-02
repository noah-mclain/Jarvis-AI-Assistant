# Fine-Tuning Guide for Stable Diffusion

This guide explains how to fine-tune the Stable Diffusion model in this project using the provided scripts.

## Overview

Fine-tuning allows you to customize the Stable Diffusion model to better generate images matching your specific style, concept, or subject. This project supports several fine-tuning approaches:

1. **Custom Fine-Tuning**: General fine-tuning on a dataset of images with corresponding prompts
2. **DreamBooth**: Fine-tuning to learn a specific subject or concept with a unique identifier
3. **Textual Inversion**: Learning a new concept while primarily updating the text encoder

## Requirements

- A GPU with at least 8GB VRAM (16GB+ recommended)
- A collection of training images (10-20 minimum for good results)
- Optional: text prompts corresponding to your training images

## Quick Start

The easiest way to fine-tune is using the `finetune_example.py` script:

```bash
python scripts/finetune_example.py --mode custom --image_dir path/to/images --prompt_file path/to/prompts.txt --concept_name "style_name" --output_dir checkpoints/my_finetuned_model
```

## Fine-Tuning Modes

### Custom Fine-Tuning

Use this mode when you have a collection of images and want the model to learn their general style or content:

```bash
python scripts/finetune_example.py \
  --mode custom \
  --image_dir path/to/images \
  --prompt_file path/to/prompts.txt \
  --output_dir checkpoints/custom_finetuned \
  --learning_rate 1e-6 \
  --epochs 1 \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --mixed_precision
```

### DreamBooth Fine-Tuning

Use this mode to teach the model a specific subject with a unique identifier:

```bash
python scripts/finetune_example.py \
  --mode dreambooth \
  --image_dir path/to/subject_images \
  --class_name "dog" \
  --identifier "sks" \
  --output_dir checkpoints/dreambooth_finetuned \
  --learning_rate 5e-7 \
  --epochs 1 \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --mixed_precision
```

After training, you can generate images of your subject using the prompt "a photo of sks dog" (replacing "dog" with your class name).

### Textual Inversion

Use this mode to learn a new concept or style while primarily updating the text encoder:

```bash
python scripts/finetune_example.py \
  --mode textual_inversion \
  --image_dir path/to/style_images \
  --concept_name "watercolor_style" \
  --output_dir checkpoints/textual_inversion_finetuned \
  --learning_rate 1e-5 \
  --epochs 1 \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --mixed_precision
```

## Advanced Usage

For more control over the fine-tuning process, you can use the `finetune_model.py` script directly:

```bash
python scripts/finetune_model.py \
  --base_model checkpoints/v1-5-pruned-emaonly.ckpt \
  --output_dir checkpoints/advanced_finetuned \
  --dataset custom \
  --custom_dataset_path path/to/images \
  --prompt_file path/to/prompts.txt \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-6 \
  --epochs 1 \
  --save_steps 100 \
  --mixed_precision \
  --train_text_encoder \
  --concept_name "my_concept"
```

## Memory Optimization

Fine-tuning Stable Diffusion requires significant GPU memory. The scripts include several memory optimization techniques:

1. **Gradient Accumulation**: Updates weights after accumulating gradients from multiple batches
2. **Mixed Precision Training**: Uses lower precision (FP16) where possible
3. **Selective Layer Training**: Option to train only specific components (UNet, text encoder, etc.)

If you encounter out-of-memory errors, try:

- Reducing batch size to 1
- Increasing gradient accumulation steps
- Enabling mixed precision training
- Using the `--train_unet_only` option

## Using Your Fine-Tuned Model

After fine-tuning, you can use your model with the existing generation scripts:

```bash
python scripts/test_memory_efficient.py
```

Just make sure to update the model path in the script to point to your fine-tuned checkpoint.

## Tips for Better Results

1. **Image Quality**: Use high-quality, consistent images for training
2. **Training Duration**: More epochs generally give better results, but watch for overfitting
3. **Learning Rate**: Start with a small learning rate (1e-6 to 1e-7) to avoid destroying pre-trained knowledge
4. **Prompt Engineering**: For DreamBooth, choose a unique identifier that doesn't conflict with existing concepts
5. **Regularization**: For longer training, consider using regularization images to prevent overfitting

## Troubleshooting

- **Out of Memory Errors**: Reduce batch size, increase gradient accumulation steps, or enable mixed precision
- **Poor Quality Results**: Try more training images, longer training, or adjusting the learning rate
- **Slow Training**: Consider using a more powerful GPU or reducing the model size
