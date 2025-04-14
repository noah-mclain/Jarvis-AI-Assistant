"""Configuration for data preprocessing."""

# Image processing
IMAGE_SIZE = 256
MAX_SAMPLES = 1000
BATCH_SIZE = 32
MAX_WORKERS = 8
TIMEOUT = 3
EXPECTED_VALID_RATIO = 0.6

# Model configurations
CLIP_MODEL = "openai/clip-vit-base-patch32"

# Normalization parameters
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]