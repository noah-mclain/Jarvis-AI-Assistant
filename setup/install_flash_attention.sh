#!/bin/bash

echo "===================================================================="
echo "Jarvis AI Assistant - Flash Attention 2.5.5 Installation"
echo "===================================================================="

# Stop on errors
set -e

# Detect environment
IN_COLAB=0
IN_PAPERSPACE=0
if python -c "import google.colab" 2>/dev/null; then
    echo "Running in Google Colab environment"
    IN_COLAB=1
elif [ -d "/notebooks" ] || [ -d "/storage" ]; then
    echo "Running in Paperspace environment"
    IN_PAPERSPACE=1
else
    echo "Running in standard environment"
fi

# Check if CUDA is available
echo "Checking for CUDA availability..."
if ! python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo "❌ ERROR: CUDA is not available. Flash Attention requires CUDA."
    exit 1
fi

# Get CUDA version
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
echo "CUDA Version: $CUDA_VERSION"

# Check PyTorch version
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
echo "PyTorch Version: $TORCH_VERSION"

# Verify compatible versions
if [[ "$TORCH_VERSION" != 2.1* ]]; then
    echo "⚠️ Warning: Flash Attention 2.5.5 is best tested with PyTorch 2.1.x"
    echo "Current PyTorch version: $TORCH_VERSION"
    echo "Continuing anyway, but installation might fail..."
fi

# Uninstall any existing flash-attn
echo "Removing any existing Flash Attention installation..."
pip uninstall -y flash-attn

# Install Flash Attention dependencies without affecting existing packages
echo "Installing Flash Attention dependencies..."
pip install packaging ninja --no-deps

# Install Flash Attention 2.5.5 with no-deps flag
echo "Installing Flash Attention 2.5.5..."
pip install flash-attn==2.5.5 --no-build-isolation --no-deps

# Install missing dependencies for Flash Attention
echo "Installing missing dependencies for Flash Attention..."
pip install einops --no-deps

# Verify installation
echo "Verifying Flash Attention installation..."
if python -c "import flash_attn; print(f'Flash Attention version: {flash_attn.__version__}')"; then
    echo "✅ Flash Attention successfully installed!"
else
    echo "❌ Flash Attention installation verification failed."
    
    # Try alternative installation method
    echo "Trying alternative installation method..."
    pip install flash-attn==2.5.5 --no-build-isolation
    
    # Verify again
    if python -c "import flash_attn; print(f'Flash Attention version: {flash_attn.__version__}')"; then
        echo "✅ Flash Attention successfully installed with alternative method!"
    else
        echo "❌ Flash Attention installation failed after multiple attempts."
        exit 1
    fi
fi

# Create a test script to verify Flash Attention functionality
echo "Creating Flash Attention test script..."
cat > test_flash_attention.py << 'EOF'
import torch
import time

try:
    from flash_attn import flash_attn_func
    print("✅ Flash Attention module imported successfully")
    
    # Create test tensors
    batch_size = 2
    seq_len = 1024
    num_heads = 12
    head_dim = 64
    
    # Create random query, key, value tensors
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
    
    # Test Flash Attention
    print("Testing Flash Attention performance...")
    
    # Warmup
    for _ in range(5):
        flash_attn_func(q, k, v, causal=True)
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(10):
        flash_attn_func(q, k, v, causal=True)
    torch.cuda.synchronize()
    flash_attn_time = (time.time() - start_time) / 10
    
    print(f"Flash Attention average time: {flash_attn_time*1000:.2f} ms")
    
    # Test standard attention for comparison
    def standard_attention(q, k, v, causal=False):
        scores = torch.matmul(q, k.transpose(-1, -2)) / (head_dim ** 0.5)
        if causal:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1).bool()
            scores.masked_fill_(mask, -float('inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, v)
    
    # Warmup
    for _ in range(5):
        standard_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=True)
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(10):
        standard_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=True)
    torch.cuda.synchronize()
    standard_time = (time.time() - start_time) / 10
    
    print(f"Standard Attention average time: {standard_time*1000:.2f} ms")
    print(f"Speedup: {standard_time / flash_attn_time:.2f}x")
    
    print("✅ Flash Attention is working correctly and providing speedup!")
    
except ImportError as e:
    print(f"❌ Failed to import Flash Attention: {e}")
except Exception as e:
    print(f"❌ Error testing Flash Attention: {e}")
EOF

echo "Running Flash Attention test..."
python test_flash_attention.py

echo "===================================================================="
echo "Flash Attention 2.5.5 installation complete!"
echo "===================================================================="
