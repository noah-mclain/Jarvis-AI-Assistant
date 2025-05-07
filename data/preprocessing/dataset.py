"""LAION dataset loading and handling."""

from datasets import load_dataset
import torch
from torch.utils.data import Dataset

def load_laion_dataset(dataset_name="laion/laion400m", split="train", streaming=True):
    """Load a LAION dataset in streaming mode.
    
    Args:
        dataset_name (str): The name of the LAION dataset to load.
            Options include:
            - "laion/laion400m": The LAION-400M dataset
            - "laion/laion2B-en-aesthetic": The English subset of LAION-2B with aesthetic scores
            - "laion/laion2B-en": The English subset of LAION-2B
            - "laion/laion2B-multi": The multilingual LAION-2B dataset
        split (str): The dataset split to load (default: "train")
        streaming (bool): Whether to stream the dataset (default: True)
            Streaming is recommended for large datasets like LAION-2B
    
    Returns:
        A dataset object that can be iterated over
    """
    return load_dataset(dataset_name, split=split, streaming=streaming)

class ProcessedImageDataset(Dataset):
    """Dataset for processed images with CLIP features."""
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]