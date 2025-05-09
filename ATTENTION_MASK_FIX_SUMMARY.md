# Attention Mask and Input IDs Fix Summary

## Issues Fixed

1. **Attention Mask Dimension Mismatch**
   - Problem: The attention mask had shape `[6, 1, 2048, 2048]` but was being reshaped to 2D `[6, 2048]`, causing a size mismatch error.
   - Fix: Updated the `patched_unmask_unattended` function to properly handle 4D attention masks and create compatible masks when needed.

2. **Input IDs Data Type Issue**
   - Problem: `input_ids` were being converted to `torch.float16`, but embeddings require integer indices (`torch.long`).
   - Fix: Added explicit checks and conversions to ensure `input_ids` and `labels` always remain as `torch.long` throughout the training process.

3. **Unmasked Value Parameter Handling**
   - Problem: The `unmasked_value` parameter in `_unmask_unattended` wasn't being properly handled.
   - Fix: Added proper handling for different types of `unmasked_value` parameters (float, int, tensor, boolean).

## Files Modified

1. **setup/comprehensive_attention_mask_fix.py**
   - Enhanced `fix_dtype_mismatch()` to ensure `input_ids` and `labels` remain as `torch.long`
   - Improved `patched_unmask_unattended()` to properly handle the `unmasked_value` parameter

2. **src/generative_ai_module/unified_deepseek_training.py**
   - Updated `AttentionMaskFixTrainer.compute_loss()` to ensure `input_ids` and `labels` remain as `torch.long`
   - Enhanced `SimpleDataCollator.__call__()` to ensure tensors are created with the correct dtype
   - Fixed fallback tensor creation to ensure correct dtypes
   - Updated simplified loss computation to maintain correct dtypes

## Key Changes

### 1. Ensuring input_ids remain as long integers

```python
# CRITICAL FIX: Ensure input_ids remain as long integers
if arg_name == "input_ids":
    if arg_value.dtype != torch.long:
        logger.warning(f"Input IDs have incorrect dtype: {arg_value.dtype}. Converting to torch.long")
        kwargs[arg_name] = arg_value.to(dtype=torch.long)
        logger.info(f"Fixed input_ids dtype: {kwargs[arg_name].dtype}")
```

### 2. Proper handling of unmasked_value parameter

```python
# Convert mask to the expected type based on unmasked_value
if unmasked_value is not True:
    # If unmasked_value is a float or tensor, convert mask to that dtype and multiply
    if isinstance(unmasked_value, (float, int)) or (isinstance(unmasked_value, torch.Tensor) and unmasked_value.dtype.is_floating_point):
        # Convert mask to the same dtype as attention_mask or to float32 if needed
        mask_dtype = attention_mask.dtype if attention_mask.dtype.is_floating_point else torch.float32
        mask = mask.to(dtype=mask_dtype) * unmasked_value
        logger.info(f"Applied unmasked_value {unmasked_value} to mask, resulting dtype: {mask.dtype}")
    else:
        # For boolean or other types, just use the value directly
        mask = mask * unmasked_value
        logger.info(f"Applied unmasked_value {unmasked_value} to mask without dtype conversion")
```

### 3. Handling attention mask dimension mismatches

```python
# If attention mask is 4D with wrong shape
if attention_mask.dim() == 4:
    batch_size, head_dim, seq_len1, seq_len2 = attention_mask.shape
    if seq_len1 != seq_len2:
        logger.warning(f"Fixing incorrect 4D attention mask shape: {attention_mask.shape}")
        # Create a proper 4D attention mask
        # Create a causal mask
        causal_mask = torch.triu(
            torch.ones((seq_len2, seq_len2), device=device, dtype=torch.bool),
            diagonal=1
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]
        causal_mask = causal_mask.expand(batch_size, head_dim, seq_len2, seq_len2)

        # Convert to the correct dtype
        if model_dtype is not None:
            causal_mask = causal_mask.to(dtype=model_dtype)

        # Replace the attention mask
        attention_mask = ~causal_mask
```

## Testing

To test these changes, run the training script with the `--model-type code` option:

```bash
./setup/train_jarvis.sh --model-type code --gpu-type A6000 --vram 50
```

The fixes should prevent the following errors:
- "The size of tensor a (6) must match the size of tensor b (2048) at non-singleton dimension 2"
- Dtype mismatch errors between input_ids and embeddings
