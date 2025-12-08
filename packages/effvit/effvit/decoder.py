"""
Segmentation decoder for TinyEfficientViT.

This module contains the decoder architecture:
- TinySegmentationDecoder: FPN-style decoder with skip connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationDecoder(nn.Module):
    """
    Lightweight FPN-style decoder with skip connections.
    Progressively upsamples features to input resolution.
    """

    def __init__(
        self,
        encoder_dims=(8, 16, 24),
        decoder_dim=16,
        num_classes=2,
    ):
        super().__init__()
        self.lateral3 = nn.Conv2d(
            encoder_dims[2],
            decoder_dim,
            kernel_size=1,
        )
        self.lateral2 = nn.Conv2d(
            encoder_dims[1],
            decoder_dim,
            kernel_size=1,
        )
        self.lateral1 = nn.Conv2d(
            encoder_dims[0],
            decoder_dim,
            kernel_size=1,
        )

        self.smooth3 = nn.Sequential(
            nn.Conv2d(
                decoder_dim,
                decoder_dim,
                kernel_size=3,
                padding=1,
                groups=decoder_dim,
            ),
            nn.BatchNorm2d(decoder_dim),
            nn.GELU(),
        )
        self.smooth2 = nn.Sequential(
            nn.Conv2d(
                decoder_dim,
                decoder_dim,
                kernel_size=3,
                padding=1,
                groups=decoder_dim,
            ),
            nn.BatchNorm2d(decoder_dim),
            nn.GELU(),
        )
        self.smooth1 = nn.Sequential(
            nn.Conv2d(
                decoder_dim,
                decoder_dim,
                kernel_size=3,
                padding=1,
                groups=decoder_dim,
            ),
            nn.BatchNorm2d(decoder_dim),
            nn.GELU(),
        )

        self.head = nn.Conv2d(
            decoder_dim,
            num_classes,
            kernel_size=1,
        )

    def forward(self, f1, f2, f3, target_size):
        p3 = self.lateral3(f3)
        p3 = self.smooth3(p3)

        p2 = self.lateral2(f2) + F.interpolate(
            p3,
            size=f2.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        p2 = self.smooth2(p2)

        p1 = self.lateral1(f1) + F.interpolate(
            p2,
            size=f1.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        p1 = self.smooth1(p1)

        out = self.head(p1)
        out = F.interpolate(
            out,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        return out
