"""Image processing utilities."""

from PIL import Image, ImageFile
import warnings
from ..utils.io import fetch_image
from ..transforms import get_transform_pipeline

# Enable loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=UserWarning)

class ImageProcessor:
    def __init__(self):
        self.transform = get_transform_pipeline()
    
    def process_image(self, image_data):
        """Process a single image."""
        if image_data is None:
            return None
            
        try:
            return self.transform(image_data)
        except Exception:
            return None