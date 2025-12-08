"""
Encoder components for TinyEfficientViT.

This module contains the encoder architecture:
- TinyEfficientVitBlock: Single transformer block with local features and attention
- TinyEfficientVitStage: Stage with optional downsampling and transformer blocks
- TinyEfficientVitEncoder: Complete multi-stage encoder producing multi-scale features
"""

import torch
import torch.nn as nn

from .layers import TinyConvNorm, TinyPatchEmbedding, TinyMLP
from .attention import TinyLocalWindowAttention


class VitBlock(nn.Module):
    """
    Single EfficientViT block:
    - Depthwise conv for local features
    - Window attention for global features
    - MLP for channel mixing
    """

    def __init__(
        self,
        dim,
        num_heads=1,
        key_dim=4,
        attn_ratio=2,
        window_size=7,
        mlp_ratio=2,
    ):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.dw_conv = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            padding=1,
            groups=dim,
        )
        self.norm2 = nn.BatchNorm2d(dim)
        self.attn = TinyLocalWindowAttention(
            dim=dim,
            num_heads=num_heads,
            key_dim=key_dim,
            attn_ratio=attn_ratio,
            window_size=window_size,
        )
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = TinyMLP(dim, expansion_ratio=mlp_ratio)

    def forward(self, x):
        x = x + self.dw_conv(self.norm1(x))
        x = x + self.attn(self.norm2(x))

        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_flat = x_flat + self.mlp(self.norm3(x_flat))
        x = x_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return x


class TinyEfficientVitStage(nn.Module):
    """
    Single stage of TinyEfficientViT.
    Optional downsampling followed by transformer blocks.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        depth=1,
        num_heads=1,
        key_dim=4,
        attn_ratio=2,
        window_size=7,
        mlp_ratio=2,
        downsample=True,
    ):
        super().__init__()
        self.downsample = None
        if downsample:
            self.downsample = nn.Sequential(
                TinyConvNorm(
                    in_dim,
                    out_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.GELU(),
            )
        elif in_dim != out_dim:
            self.downsample = nn.Sequential(
                TinyConvNorm(
                    in_dim,
                    out_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                nn.GELU(),
            )

        self.blocks = nn.ModuleList(
            [
                VitBlock(
                    dim=out_dim,
                    num_heads=num_heads,
                    key_dim=key_dim,
                    attn_ratio=attn_ratio,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        for block in self.blocks:
            x = block(x)
        return x


class TinyEfficientVitEncoder(nn.Module):
    """
    Complete TinyEfficientViT encoder with 3 stages.
    Produces multi-scale features for segmentation decoder.
    """

    def __init__(
        self,
        in_channels=1,
        embed_dims=(8, 16, 24),
        depths=(1, 1, 1),
        num_heads=(1, 1, 2),
        key_dims=(4, 4, 4),
        attn_ratios=(2, 2, 2),
        window_sizes=(7, 7, 7),
        mlp_ratios=(2, 2, 2),
    ):
        super().__init__()
        self.patch_embed = TinyPatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dims[0],
        )

        self.stage1 = TinyEfficientVitStage(
            in_dim=embed_dims[0],
            out_dim=embed_dims[0],
            depth=depths[0],
            num_heads=num_heads[0],
            key_dim=key_dims[0],
            attn_ratio=attn_ratios[0],
            window_size=window_sizes[0],
            mlp_ratio=mlp_ratios[0],
            downsample=False,
        )

        self.stage2 = TinyEfficientVitStage(
            in_dim=embed_dims[0],
            out_dim=embed_dims[1],
            depth=depths[1],
            num_heads=num_heads[1],
            key_dim=key_dims[1],
            attn_ratio=attn_ratios[1],
            window_size=window_sizes[1],
            mlp_ratio=mlp_ratios[1],
            downsample=True,
        )

        self.stage3 = TinyEfficientVitStage(
            in_dim=embed_dims[1],
            out_dim=embed_dims[2],
            depth=depths[2],
            num_heads=num_heads[2],
            key_dim=key_dims[2],
            attn_ratio=attn_ratios[2],
            window_size=window_sizes[2],
            mlp_ratio=mlp_ratios[2],
            downsample=True,
        )

    def forward(self, x):
        x = self.patch_embed(x)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        return f1, f2, f3
