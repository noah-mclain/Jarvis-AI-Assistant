# Fixed Issues in Training Script

## Issues Fixed

1. **Attention Mask Dimension Mismatch**
   - Problem: The attention mask had shape `[6, 1, 2048, 2048]` but was being incorrectly reshaped to 2D `[6, 2048]`, causing a size mismatch error.
   - Fix: Updated the attention mask handling to properly handle 4D masks and create compatible masks when needed.

2. **Input IDs Data Type Issue**
   - Problem: `input_ids` were being converted to `torch.float16`, but embeddings require integer indices (`torch.long`).
   - Fix: Added explicit checks and conversions to ensure `input_ids` and `labels` always remain as `torch.long` throughout the training process.

## Key Changes Made

### 1. Fixed Attention Mask Handling in `compute_loss`

```python
# Handle attention mask properly - DO NOT reshape 4D masks to 2D
if "attention_mask" in inputs:
    if inputs["attention_mask"].dim() == 4:
        # Keep 4D mask as-is, just ensure it's on the right device
        if inputs["attention_mask"].device != device:
            inputs["attention_mask"] = inputs["attention_mask"].to(device)
        logger.info(f"Using 4D attention mask with shape: {inputs['attention_mask'].shape}")
    elif inputs["attention_mask"].dim() == 2:
        # Convert 2D mask to 4D causal mask
        batch_size, seq_length = inputs["attention_mask"].shape
        
        # First, expand to [batch_size, 1, 1, seq_length]
        attention_mask_4d = inputs["attention_mask"].unsqueeze(1).unsqueeze(2)
        
        # Then, expand to [batch_size, 1, seq_length, seq_length]
        attention_mask_4d = attention_mask_4d.expand(-1, 1, seq_length, -1)
        
        # Create a causal mask
        causal_mask = torch.triu(
            torch.ones((seq_length, seq_length), device=device, dtype=torch.bool),
            diagonal=1
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]
        causal_mask = causal_mask.expand(batch_size, 1, seq_length, seq_length)
        
        # Combine with the attention mask
        combined_mask = causal_mask | ~attention_mask_4d.bool()
        
        # Convert to the model's dtype if available
        model_dtype = getattr(model, "dtype", None)
        if model_dtype is not None:
            combined_mask = combined_mask.to(dtype=model_dtype)
        
        # Replace the attention mask
        inputs["attention_mask"] = ~combined_mask
        logger.info(f"Converted 2D attention mask to 4D: {inputs['attention_mask'].shape}")
```

### 2. Ensured `input_ids` Remain as `torch.long` in `device_aware_forward`

```python
# CRITICAL FIX: Do not convert input_ids to float16/bfloat16
if k == "input_ids":
    # Ensure input_ids are always torch.long
    if v.dtype != torch.long:
        logger.warning(f"Converting input_ids from {v.dtype} to torch.long")
        v = v.to(dtype=torch.long)
elif k == "labels":
    # Ensure labels are always torch.long
    if v.dtype != torch.long:
        logger.warning(f"Converting labels from {v.dtype} to torch.long")
        v = v.to(dtype=torch.long)
# For other tensors, convert to model's dtype if needed
elif model_dtype is not None and v.dtype != model_dtype:
    logger.info(f"Converting {k} from {v.dtype} to {model_dtype}")
    v = v.to(dtype=model_dtype)
```

### 3. Fixed `input_ids` Handling in `device_aware_prepare_inputs`

```python
def device_aware_prepare_inputs(input_ids, **kwargs):
    """Ensure all inputs are on the correct device and have the correct dtype"""
    device = model.device

    # CRITICAL FIX: Ensure input_ids remain as torch.long
    if input_ids.dtype != torch.long:
        logger.warning(f"Converting input_ids from {input_ids.dtype} to torch.long in prepare_inputs_for_generation")
        input_ids = input_ids.to(dtype=torch.long)
    
    # Ensure input_ids is on the correct device
    if input_ids.device != device:
        logger.info(f"Moving input_ids from {input_ids.device} to {device}")
        input_ids = input_ids.to(device)
```

### 4. Fixed Direct Forward Call

```python
# CRITICAL FIX: Ensure input_ids remain as torch.long
if input_ids is not None and input_ids.dtype != torch.long:
    logger.warning(f"Converting input_ids from {input_ids.dtype} to torch.long in direct forward call")
    input_ids = input_ids.to(dtype=torch.long)

# Ensure labels are torch.long if present
if labels is not None and labels.dtype != torch.long:
    logger.warning(f"Converting labels from {labels.dtype} to torch.long in direct forward call")
    labels = labels.to(dtype=torch.long)
```

### 5. Fixed Simplified Loss Computation

```python
# CRITICAL FIX: Ensure input_ids remain as torch.long
if input_ids.dtype != torch.long:
    logger.warning(f"Simplified computation - input_ids have incorrect dtype: {input_ids.dtype}. Converting to torch.long")
    input_ids = input_ids.to(dtype=torch.long)

# Forward pass with correct dtype
outputs = model(input_ids=input_ids)
```

## Testing

These changes should resolve the two main issues:

1. The attention mask dimension mismatch error: `shape '[6, 2048]' is invalid for input of size 25165824`
2. The input_ids data type error: `Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.cuda.HalfTensor instead`

To test these changes, run the training script with the `--model-type code` option:

```bash
./setup/train_jarvis.sh --model-type code --gpu-type A6000 --vram 50
```

The training should now proceed without the previous errors.
