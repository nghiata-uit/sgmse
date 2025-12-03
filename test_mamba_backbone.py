#!/usr/bin/env python3
"""
Test script for Mamba backbone in SGMSE

This script tests the forward pass of the ncsnpp_mamba backbone.
"""

import torch
from sgmse.backbones import BackboneRegistry


def test_mamba_backbone():
    """Test the Mamba backbone with a forward pass"""
    print("Testing Mamba backbone...")
    
    # Get the Mamba backbone
    model = BackboneRegistry.get_by_name("ncsnpp_mamba")()
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    # Complex spectrogram with 2 channels (x and y)
    batch_size = 2
    channels = 2
    height = 256
    width = 256
    
    # Create complex-valued input
    x_real = torch.randn(batch_size, channels, height, width).to(device)
    x_imag = torch.randn(batch_size, channels, height, width).to(device)
    x = torch.complex(x_real, x_imag)
    
    # Create time conditioning
    time_cond = torch.randn(batch_size).to(device)
    
    # Forward pass
    print(f"Input shape: {x.shape}")
    print(f"Time conditioning shape: {time_cond.shape}")
    
    with torch.no_grad():
        output = model(x, time_cond)
    
    print(f"Output shape: {output.shape}")
    print(f"✅ Mamba backbone test passed!")
    
    # Test backward pass
    print("\nTesting backward pass...")
    model.train()
    
    # Enable gradients
    x = torch.complex(
        torch.randn(batch_size, channels, height, width, requires_grad=True).to(device),
        torch.randn(batch_size, channels, height, width, requires_grad=True).to(device)
    )
    time_cond = torch.randn(batch_size, requires_grad=True).to(device)
    
    output = model(x, time_cond)
    loss = output.abs().mean()
    loss.backward()
    
    print(f"Loss: {loss.item():.6f}")
    print(f"✅ Backward pass test passed!")
    
    return True


if __name__ == "__main__":
    try:
        test_mamba_backbone()
        print("\n" + "="*50)
        print("All tests passed successfully!")
        print("="*50)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
