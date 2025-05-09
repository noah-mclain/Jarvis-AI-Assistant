"""CLIP feature extraction."""

import torch
from transformers import CLIPProcessor, CLIPModel
from configs.preprocessing import config

class CLIPFeatureExtractor:
    def __init__(self):
        self.model = CLIPModel.from_pretrained(config.CLIP_MODEL)
        self.processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL)
        
    def extract_features(self, image, caption):
        """Extract CLIP features from image and text."""
        with torch.no_grad():
            inputs = self.processor(
                text=[caption],
                images=image,
                return_tensors="pt",
                padding=True
            )
            outputs = self.model(**inputs)
            return {
                'image_features': outputs.image_embeds[0].cpu(),
                'text_features': outputs.text_embeds[0].cpu()
            }