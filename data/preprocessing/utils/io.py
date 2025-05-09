"""I/O utilities for image handling."""

import io
import requests
from PIL import Image
from configs.preprocessing import config

def fetch_image(url, timeout=config.TIMEOUT):
    """Fetch image from URL with timeout."""
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert('RGB')
    except Exception:
        return None

def save_processed_data(data, filepath):
    """Save processed data to disk."""
    torch.save(data, filepath)