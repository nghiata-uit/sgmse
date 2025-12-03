# coding=utf-8
# Mamba-based layers for SGMSE
# Adapted from mamba-seunet repository

from functools import partial
import torch
from torch import nn
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.models.mixer_seq_simple import _init_weights, Block
from mamba_ssm.ops.triton.layer_norm import RMSNorm


def create_mamba_unit(dimension, layer_index=0, fused_add_norm=False, residual_in_fp32=False):
    """
    Create a Single-direction Mamba unit with:
    - state_size = 16
    - conv_kernel_size = 4
    - expansion_ratio = 2
    
    Args:
        dimension: Input/output dimension
        layer_index: Layer index for the Mamba block
        fused_add_norm: Whether to use fused add norm
        residual_in_fp32: Whether to use fp32 for residual
        
    Returns:
        A Mamba Block instance
    """
    state_size = 16
    conv_kernel_size = 4
    expansion_ratio = 2

    mixer_cls = partial(
        Mamba,
        layer_idx=layer_index,
        d_state=state_size,
        d_conv=conv_kernel_size,
        expand=expansion_ratio,
    )

    block = Block(
        dimension,
        mixer_cls,
        nn.Identity,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_index
    return block


class BidirectionalMamba(nn.Module):
    """
    Bidirectional Mamba block with forward and backward paths
    
    Input:  Tensor of shape (B, L, C)
    Output: Tensor of shape (B, L, C)
    """
    def __init__(self, input_channels):
        super(BidirectionalMamba, self).__init__()
        self.forward_units  = nn.ModuleList([create_mamba_unit(input_channels)])
        self.backward_units = nn.ModuleList([create_mamba_unit(input_channels)])
        self.norm = RMSNorm(input_channels, eps=1e-5)
        self.concat_projection = nn.Linear(input_channels * 2, input_channels)
        self.apply(partial(_init_weights, n_layer=1))

    def forward(self, x):
        # Forward direction
        forward_out, _ = self.forward_units[0](x)
        y_forward = self.norm(forward_out) + x
        
        # Backward direction
        backward_in = torch.flip(x, [1])
        backward_out, _ = self.backward_units[0](backward_in)
        y_backward = torch.flip(self.norm(backward_out) + backward_in, [1])
        
        # Concatenate and project
        y = torch.cat([y_forward, y_backward], dim=-1)
        y = self.concat_projection(y)
        return y


class TSMambaBlock(nn.Module):
    """
    Temporal–Frequency Mamba block
    Applies Bidirectional Mamba along:
        1. Temporal dimension
        2. Frequency dimension
    
    Input:  Tensor of shape (B, C, T, F)
    Output: Tensor of shape (B, C, T, F)
    """
    def __init__(self, hidden_feature):
        super(TSMambaBlock, self).__init__()
        self.hidden_feature = hidden_feature
        self.time_mamba = BidirectionalMamba(input_channels=self.hidden_feature)
        self.freq_mamba = BidirectionalMamba(input_channels=self.hidden_feature)

    def forward(self, x):
        b, c, t, f = x.size()
        
        # Temporal path
        x = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        x = self.time_mamba(x) + x
        
        # Frequency path
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        x = self.freq_mamba(x) + x
        
        return x.view(b, t, f, c).permute(0, 3, 1, 2)


class MambaBottleneck(nn.Module):
    """
    Mamba-based bottleneck to replace ResBlock + Attention + ResBlock
    
    This replaces the three bottleneck modules in NCSNPP architecture
    """
    def __init__(self, channels, temb_dim=None):
        super().__init__()
        self.mamba_block = TSMambaBlock(channels)
        if temb_dim is not None:
            self.temb_proj = nn.Linear(temb_dim, channels)
        else:
            self.temb_proj = None
    
    def forward(self, x, temb=None):
        if self.temb_proj is not None and temb is not None:
            temb_proj = self.temb_proj(temb)[:, :, None, None]
            x = x + temb_proj
        return self.mamba_block(x)


class MambaResBlock(nn.Module):
    """
    Mamba-based residual block to replace ResnetBlock
    
    Handles channel dimension changes and time embedding
    """
    def __init__(self, in_ch, out_ch, temb_dim=None):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        
        # Channel projection if input/output channels differ
        if in_ch != out_ch:
            self.channel_proj = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.channel_proj = None
            
        self.mamba = TSMambaBlock(out_ch)
        
        # Time embedding projection
        if temb_dim is not None:
            self.temb_proj = nn.Linear(temb_dim, out_ch)
        else:
            self.temb_proj = None
    
    def forward(self, x, temb=None):
        # Project channels if needed
        if self.channel_proj is not None:
            x = self.channel_proj(x)
        
        identity = x
        
        # Add time embedding if provided
        if self.temb_proj is not None and temb is not None:
            temb_proj = self.temb_proj(temb)[:, :, None, None]
            x = x + temb_proj
        
        # Apply Mamba block
        x = self.mamba(x)
        
        # Residual connection
        return x + identity


class MambaConv(nn.Module):
    """
    Mamba-based conv to replace initial conv3x3
    
    Projects input channels to output channels and applies Mamba processing
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 1)
        self.mamba = TSMambaBlock(out_ch)
    
    def forward(self, x):
        x = self.proj(x)
        return self.mamba(x)
