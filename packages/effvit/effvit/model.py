"""
Complete TinyEfficientViT model for semantic segmentation.

This module contains the main model class that combines encoder and decoder.
"""

import torch
import torch.nn as nn

from .encoder import TinyEfficientVitEncoder
from .decoder import SegmentationDecoder


class TinyEfficientViTSeg(nn.Module):
    """
    Complete TinyEfficientViT for semantic segmentation.
    Combines encoder and decoder with <10k parameters.
    """

    def __init__(
        self,
        in_channels=1,
        num_classes=2,
        embed_dims=(8, 16, 24),
        depths=(1, 1, 1),
        num_heads=(1, 1, 2),
        key_dims=(4, 4, 4),
        attn_ratios=(2, 2, 2),
        window_sizes=(7, 7, 7),
        mlp_ratios=(2, 2, 2),
        decoder_dim=16,
    ):
        super().__init__()
        self.encoder = TinyEfficientVitEncoder(
            in_channels=in_channels,
            embed_dims=embed_dims,
            depths=depths,
            num_heads=num_heads,
            key_dims=key_dims,
            attn_ratios=attn_ratios,
            window_sizes=window_sizes,
            mlp_ratios=mlp_ratios,
        )
        self.decoder = SegmentationDecoder(
            encoder_dims=embed_dims,
            decoder_dim=decoder_dim,
            num_classes=num_classes,
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        target_size = (x.shape[2], x.shape[3])
        f1, f2, f3 = self.encoder(x)
        out = self.decoder(f1, f2, f3, target_size)
        return out
