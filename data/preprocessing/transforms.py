"""Image transformation pipeline."""

import torchvision.transforms as T
from configs.preprocessing import config

def get_transform_pipeline():
    """Create the image transformation pipeline."""
    return T.Compose([
        T.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])