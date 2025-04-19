#!/usr/bin/env python3
"""
MPS GPU Test Script

This script tests if PyTorch can properly use the Apple Silicon GPU via MPS.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time

def test_mps():
    # Check for MPS support
    print("Checking for MPS support...")
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            print("✅ MPS is available and built!")
            device = torch.device("mps")
            print(f"Using device: {device}")
        else:
            print("❌ MPS is not available. Using CPU instead.")
            device = torch.device("cpu")
            print(f"Using device: {device}")
    else:
        print("❌ This version of PyTorch does not support MPS. Using CPU instead.")
        device = torch.device("cpu")
        print(f"Using device: {device}")

    # Create a simple model
    print("\nCreating a simple neural network...")
    model = nn.Sequential(
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Linear(200, 10)
    ).to(device)
    print(f"Model created on {device}")

    # Create some dummy data
    print("\nCreating dummy data...")
    batch_size = 64
    X = torch.randn(batch_size, 100, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)
    print(f"Data created on {device}")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train for a few epochs
    print("\nTraining model for 5 epochs...")
    start_time = time.time()

    for epoch in range(5):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/5, Loss: {loss.item():.4f}")

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.2f} seconds")

    # Test tensor operations
    print("\nTesting various tensor operations on MPS...")
    
    # Test 1: Matrix multiplication
    start = time.time()
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)
    c = torch.matmul(a, b)
    print(f"Matrix multiplication: {time.time() - start:.4f} seconds")
    
    # Test 2: Convolution
    start = time.time()
    conv = nn.Conv2d(3, 64, kernel_size=3, padding=1).to(device)
    img = torch.randn(8, 3, 224, 224, device=device)
    out = conv(img)
    print(f"Convolution: {time.time() - start:.4f} seconds")
    
    # Test 3: Element-wise operations
    start = time.time()
    x = torch.randn(10000, 10000, device=device)
    y = torch.sigmoid(x)
    print(f"Element-wise sigmoid: {time.time() - start:.4f} seconds")
    
    print("\n✅ All tests completed successfully!")
    print(f"Device used: {device}")

if __name__ == "__main__":
    test_mps() 