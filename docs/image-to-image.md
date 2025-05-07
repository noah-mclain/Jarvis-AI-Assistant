# Image Generation Script

This script allows you to generate images using Stable Diffusion with either text prompts (text-to-image) or by providing an input image (image-to-image).

## Features

- Text-to-image generation with customizable parameters
- Image-to-image generation with adjustable strength
- Support for negative prompts
- Configurable sampling steps and CFG scale
- Seed control for reproducible results

## Usage

### Basic Text-to-Image Generation

```bash
python scripts/generate_image.py --prompt "a beautiful landscape with mountains and a lake"
```

### Image-to-Image Generation

```bash
python scripts/generate_image.py --prompt "a beautiful landscape with mountains and a lake" --input_image path/to/your/image.jpg --strength 0.7
```

### Advanced Options

```bash
python scripts/generate_image.py \
  --prompt "a beautiful landscape with mountains and a lake" \
  --negative_prompt "blurry, low quality" \
  --cfg_scale 8.0 \
  --steps 40 \
  --seed 42 \
  --output_dir "outputs/landscapes"
```

## Parameters

| Parameter           | Description                                       | Default                                   |
| ------------------- | ------------------------------------------------- | ----------------------------------------- |
| `--model_path`      | Path to the model checkpoint                      | `../checkpoints/v1-5-pruned-emaonly.ckpt` |
| `--prompt`          | Text prompt for image generation                  | (Required)                                |
| `--negative_prompt` | Negative prompt to guide what should not appear   | `""`                                      |
| `--input_image`     | Path to input image for image-to-image generation | `None`                                    |
| `--strength`        | Strength for image-to-image generation (0-1)      | `0.8`                                     |
| `--output_dir`      | Directory to save generated images                | `../outputs`                              |
| `--cfg_scale`       | Classifier-free guidance scale                    | `7.5`                                     |
| `--steps`           | Number of inference steps                         | `50`                                      |
| `--seed`            | Random seed for generation                        | `None` (random)                           |
| `--device`          | Device to use (cuda:0, cuda:1, cpu)               | Auto-select                               |
| `--sampler`         | Sampler to use for diffusion                      | `ddpm`                                    |

## Examples

### Portrait Generation

```bash
python scripts/generate_image.py --prompt "portrait of a person with blue eyes, detailed, realistic" --steps 40 --cfg_scale 7.5
```

### Style Transfer

```bash
python scripts/generate_image.py --prompt "oil painting in the style of Van Gogh" --input_image path/to/photo.jpg --strength 0.9
```

## Notes

- Higher `strength` values (closer to 1.0) will make the output differ more from the input image
- Lower `strength` values (closer to 0.0) will preserve more of the input image
- The `--seed` parameter allows you to reproduce the same image when using identical parameters
- Generated images are saved with a timestamp and seed value in the filename
