# Mamba-3 Integration for SGMSE

This document describes the integration of Mamba-3 architecture into SGMSE (Score-based Generative Model for Speech Enhancement).

## Overview

The Mamba-3 integration replaces the CNN-based layers in the NCSN++ backbone with Mamba blocks, which are more efficient for processing temporal-frequency representations.

## Changes Made

### 1. New Files

#### `sgmse/backbones/mamba_layers.py`
This file contains all the Mamba components:

- **`create_mamba_unit()`**: Factory function to create single-direction Mamba units
- **`BidirectionalMamba`**: Processes sequences in both forward and backward directions
- **`TSMambaBlock`**: Temporal-Frequency Mamba block that processes both time and frequency dimensions
- **`MambaBottleneck`**: Replaces the 3-module bottleneck (ResBlock + Attention + ResBlock)
- **`MambaResBlock`**: Replaces ResNet blocks in encoder/decoder
- **`MambaConv`**: Replaces the initial conv3x3 layer

#### `sgmse/backbones/ncsnpp_mamba.py`
This is a modified version of `ncsnpp.py` that uses Mamba layers instead of CNN layers:

- Registered as `"ncsnpp_mamba"` in the BackboneRegistry
- Uses `MambaConv` for the initial convolution
- Uses `MambaResBlock` for all encoder/decoder blocks
- Uses `MambaBottleneck` for the U-Net bottleneck
- Removes all attention blocks (Mamba handles long-range dependencies)

#### `test_mamba_backbone.py`
A test script to verify the Mamba backbone works correctly:
- Tests forward pass with dummy data
- Tests backward pass and gradient flow
- Can run on CPU or CUDA

### 2. Modified Files

#### `sgmse/backbones/__init__.py`
- Added import for `NCSNppMamba`
- Added to `__all__` list

#### `requirements.txt`
Added Mamba dependencies:
- `mamba-ssm>=1.0.0`
- `causal-conv1d>=1.0.0`
- `triton>=2.0.0`

## Installation

1. Install the dependencies:
```bash
pip install -r requirements.txt
```

Note: The Mamba dependencies require CUDA for optimal performance.

## Usage

### Training with Mamba Backbone

Use the `--backbone ncsnpp_mamba` argument when training:

```bash
python train.py --backbone ncsnpp_mamba --other-args ...
```

### Using Pre-trained Models

If you have a pre-trained SGMSE model and want to use the Mamba backbone:

```python
from sgmse.backbones import BackboneRegistry

# Load Mamba backbone
model = BackboneRegistry.get_by_name("ncsnpp_mamba")(
    nf=128,
    ch_mult=(1, 1, 2, 2, 2, 2, 2),
    num_res_blocks=2,
    # ... other parameters
)
```

### Testing the Implementation

Run the test script to verify everything works:

```bash
python test_mamba_backbone.py
```

Expected output:
```
Testing Mamba backbone...
Using device: cuda
Input shape: torch.Size([2, 2, 256, 256])
Time conditioning shape: torch.Size([2])
Output shape: torch.Size([2, 1, 256, 256])
✅ Mamba backbone test passed!

Testing backward pass...
Loss: 0.xxxxxx
✅ Backward pass test passed!

==================================================
All tests passed successfully!
==================================================
```

## Architecture Details

### TSMambaBlock

The `TSMambaBlock` is the core component that processes spectrograms along both temporal and frequency dimensions:

1. **Temporal Processing**: Applies bidirectional Mamba along the time axis
2. **Frequency Processing**: Applies bidirectional Mamba along the frequency axis

Input/Output shape: `(B, C, T, F)` where:
- B: Batch size
- C: Channels
- T: Time frames
- F: Frequency bins

### MambaBottleneck

Replaces the bottleneck in the U-Net architecture:
- **Original**: ResBlock → Attention → ResBlock (3 modules)
- **Mamba**: Single TSMambaBlock (1 module)

This simplification is possible because Mamba blocks inherently capture long-range dependencies.

### MambaResBlock

Replaces ResNet blocks throughout the encoder and decoder:
- Handles channel projection when input/output channels differ
- Supports time embedding injection
- Maintains residual connections

## Key Differences from Original NCSN++

1. **No Attention Blocks**: Mamba blocks replace both ResNet and attention blocks
2. **Bidirectional Processing**: All Mamba blocks process sequences bidirectionally
3. **Temporal-Frequency Processing**: Explicitly processes both time and frequency dimensions
4. **Simplified Architecture**: Fewer modules overall due to Mamba's efficiency

## Performance Considerations

- **Memory**: Mamba blocks may use more memory than CNNs for the same feature dimensions
- **Speed**: Mamba is generally faster for longer sequences
- **CUDA Required**: For best performance, use CUDA-enabled GPU
- **Gradient Flow**: Mamba provides better gradient flow for long-range dependencies

## Troubleshooting

### Import Errors

If you get import errors for `mamba_ssm`:
```bash
pip install mamba-ssm causal-conv1d triton
```

### CUDA Errors

If you encounter CUDA errors:
1. Ensure you have a compatible CUDA version (11.7+)
2. Check that `triton` is properly installed
3. Try running on CPU first for testing

### Shape Mismatches

If you encounter shape mismatches:
- Ensure input spectrograms are complex-valued with shape `(B, 2, H, W)`
- Check that `time_cond` has shape `(B,)`

## References

- Original SGMSE: https://github.com/sp-uhh/sgmse
- Mamba-SEUNet: https://github.com/nghiata-uit/mamba-seunet
- Mamba SSM: https://github.com/state-spaces/mamba

## Citation

If you use this Mamba integration, please cite both SGMSE and Mamba papers.
