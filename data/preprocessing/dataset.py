"""LAION dataset loading and handling."""

from datasets import load_dataset
import torch
from torch.utils.data import Dataset

def load_laion_dataset(split="train", streaming=True):
    """Load the LAION-400M dataset in streaming mode."""
    return load_dataset("laion/laion400m", split=split, streaming=streaming)

class ProcessedImageDataset(Dataset):
    """Dataset for processed images with CLIP features."""
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]