import torch
import gc
from .model_converter import load_from_standard_weights

from src.models.clip import CLIP
from src.models.encoder import VAE_Encoder
from src.models.decoder import VAE_Decoder
from src.models.diffusion import Diffusion

def load_model_component(component, state_dict, device):
    """Load a model component and move it to the specified device."""
    # Move to device first, then load state dict to avoid duplicate memory usage
    component.to(device)
    component.load_state_dict(state_dict, strict=True)
    return component

def preload_models_from_standard_weights(ckpt_path, device):
    """Memory efficient model loading function."""
    # First load the model to CPU
    print("Loading model weights to CPU first...")
    state_dict = load_from_standard_weights(ckpt_path, "cpu")
    
    # Initialize models on CPU first
    encoder = VAE_Encoder()
    decoder = VAE_Decoder()
    diffusion = Diffusion()
    clip = CLIP()
    
    # Load each component and move to GPU
    models = {}
    
    # Load CLIP first as it's needed for text processing
    print("Loading CLIP model...")
    models['clip'] = load_model_component(clip, state_dict['clip'], device)
    del state_dict['clip']
    gc.collect()
    if "cuda" in device:
        torch.cuda.empty_cache()
    
    # Load encoder
    print("Loading encoder...")
    models['encoder'] = load_model_component(encoder, state_dict['encoder'], device)
    del state_dict['encoder']
    gc.collect()
    if "cuda" in device:
        torch.cuda.empty_cache()
    
    # Load decoder
    print("Loading decoder...")
    models['decoder'] = load_model_component(decoder, state_dict['decoder'], device)
    del state_dict['decoder']
    gc.collect()
    if "cuda" in device:
        torch.cuda.empty_cache()
    
    # Load diffusion model last as it's the largest
    print("Loading diffusion model...")
    # For diffusion model, we'll load it in chunks to save memory
    diffusion_model = diffusion
    diffusion_state_dict = state_dict['diffusion']
    
    # Load diffusion model in chunks
    print("Loading diffusion model in chunks...")
    for key, value in diffusion_state_dict.items():
        if key not in diffusion_model.state_dict():
            continue
        # Load each parameter individually
        diffusion_model.state_dict()[key].copy_(value.to(device))
        # Clear memory after each parameter
        if "cuda" in device and key.endswith(('.weight', '.bias')):
            torch.cuda.empty_cache()
    
    # Move the model to device
    diffusion_model.to(device)
    models['diffusion'] = diffusion_model
    
    del state_dict['diffusion']
    gc.collect()
    if "cuda" in device:
        torch.cuda.empty_cache()
    
    print("All models loaded successfully!")
    return models 