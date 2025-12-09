"""
DSA Segmentation Model for Pupil Detection.

Complete architecture combining:
1. Convolutional patch embedding for efficient downsampling
2. Multi-stage DSA encoder with progressive feature extraction
3. FPN-style decoder with skip connections for precise localization

Key observations incorporated:
- Pupil segmentation requires intense pixel localization
- The pupil is only found within the eye region (spatial constraint)
- OpenEDS provides additional data beyond just pupil masks

Model is kept small and efficient for real-time inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

try:
    from .sparse_attention import DSABlock, DSAStage
except ImportError:
    from sparse_attention import DSABlock, DSAStage


class DSAPatchEmbedding(nn.Module):
    """
    Convolutional patch embedding for DSA encoder.

    Uses overlapping convolutions to:
    1. Reduce spatial resolution (4x downsampling)
    2. Extract initial features
    3. Embed to initial dimension

    For pupil segmentation, we use small kernels and
    overlapping strides to preserve boundary information.

    Args:
        in_channels: Input image channels (1 for grayscale)
        embed_dim: Output embedding dimension
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 16):
        super().__init__()

        mid_dim = max(embed_dim // 2, 4)

        # First conv: 2x downsample
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(mid_dim),
            nn.GELU(),
        )

        # Second conv: 2x downsample (total 4x)
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_dim, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image (B, C, H, W)

        Returns:
            Embedded patches (B, D, H/4, W/4)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DSAEncoder(nn.Module):
    """
    Multi-stage DSA encoder for hierarchical feature extraction.

    Architecture:
    - Patch embedding (4x downsample)
    - Stage 1: Initial features at 1/4 resolution
    - Stage 2: Higher-level features at 1/8 resolution
    - Stage 3: Global context at 1/16 resolution

    Each stage uses DSA blocks for efficient attention.
    Returns multi-scale features for decoder.

    Args:
        in_channels: Input image channels
        embed_dims: Embedding dimensions for each stage
        depths: Number of DSA blocks per stage
        num_heads: Attention heads per stage
        top_k: Tokens for sparse attention per stage
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dims: Tuple[int, ...] = (16, 32, 48),
        depths: Tuple[int, ...] = (1, 1, 1),
        num_heads: Tuple[int, ...] = (2, 2, 4),
        top_k: Tuple[int, ...] = (64, 32, 16),
    ):
        super().__init__()

        # Patch embedding (4x downsample)
        self.patch_embed = DSAPatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dims[0],
        )

        # Stage 1: no downsample (already at 1/4)
        self.stage1 = DSAStage(
            in_dim=embed_dims[0],
            out_dim=embed_dims[0],
            depth=depths[0],
            num_heads=num_heads[0],
            top_k=top_k[0],
            downsample=False,
        )

        # Stage 2: 2x downsample (1/8 resolution)
        self.stage2 = DSAStage(
            in_dim=embed_dims[0],
            out_dim=embed_dims[1],
            depth=depths[1],
            num_heads=num_heads[1],
            top_k=top_k[1],
            downsample=True,
        )

        # Stage 3: 2x downsample (1/16 resolution)
        self.stage3 = DSAStage(
            in_dim=embed_dims[1],
            out_dim=embed_dims[2],
            depth=depths[2],
            num_heads=num_heads[2],
            top_k=top_k[2],
            downsample=True,
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract multi-scale features.

        Args:
            x: Input image (B, C, H, W)

        Returns:
            f1: Stage 1 features (B, D1, H/4, W/4)
            f2: Stage 2 features (B, D2, H/8, W/8)
            f3: Stage 3 features (B, D3, H/16, W/16)
        """
        x = self.patch_embed(x)  # (B, D1, H/4, W/4)
        f1 = self.stage1(x)       # (B, D1, H/4, W/4)
        f2 = self.stage2(f1)      # (B, D2, H/8, W/8)
        f3 = self.stage3(f2)      # (B, D3, H/16, W/16)

        return f1, f2, f3


class DSADecoder(nn.Module):
    """
    FPN-style decoder with skip connections.

    Progressively upsamples features while incorporating
    skip connections from encoder for precise localization.

    For pupil segmentation:
    - Skip connections preserve boundary details
    - Progressive upsampling avoids checkerboard artifacts
    - Depthwise convs for efficient smoothing

    Args:
        encoder_dims: Dimensions from encoder stages
        decoder_dim: Unified decoder dimension
        num_classes: Number of segmentation classes
    """

    def __init__(
        self,
        encoder_dims: Tuple[int, ...] = (16, 32, 48),
        decoder_dim: int = 24,
        num_classes: int = 2,
    ):
        super().__init__()

        # Lateral connections (1x1 conv to match dimensions)
        self.lateral3 = nn.Conv2d(encoder_dims[2], decoder_dim, kernel_size=1)
        self.lateral2 = nn.Conv2d(encoder_dims[1], decoder_dim, kernel_size=1)
        self.lateral1 = nn.Conv2d(encoder_dims[0], decoder_dim, kernel_size=1)

        # Smoothing layers (depthwise + pointwise)
        self.smooth3 = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1, groups=decoder_dim),
            nn.BatchNorm2d(decoder_dim),
            nn.GELU(),
        )
        self.smooth2 = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1, groups=decoder_dim),
            nn.BatchNorm2d(decoder_dim),
            nn.GELU(),
        )
        self.smooth1 = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1, groups=decoder_dim),
            nn.BatchNorm2d(decoder_dim),
            nn.GELU(),
        )

        # Segmentation head
        self.head = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    def forward(
        self,
        f1: torch.Tensor,
        f2: torch.Tensor,
        f3: torch.Tensor,
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Decode features to segmentation map.

        Args:
            f1: Stage 1 features (B, D1, H/4, W/4)
            f2: Stage 2 features (B, D2, H/8, W/8)
            f3: Stage 3 features (B, D3, H/16, W/16)
            target_size: Output spatial size (H, W)

        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        # Process deepest features
        p3 = self.lateral3(f3)
        p3 = self.smooth3(p3)

        # Upsample and merge with f2
        p2 = self.lateral2(f2) + F.interpolate(
            p3, size=f2.shape[2:], mode='bilinear', align_corners=False
        )
        p2 = self.smooth2(p2)

        # Upsample and merge with f1
        p1 = self.lateral1(f1) + F.interpolate(
            p2, size=f1.shape[2:], mode='bilinear', align_corners=False
        )
        p1 = self.smooth1(p1)

        # Segmentation head
        out = self.head(p1)

        # Upsample to target size
        out = F.interpolate(
            out, size=target_size, mode='bilinear', align_corners=False
        )

        return out


class DSASegmentationModel(nn.Module):
    """
    Complete DSA Segmentation Model for Pupil Detection.

    Combines:
    - DSA Encoder: Efficient hierarchical feature extraction
    - FPN Decoder: Precise spatial localization

    Designed for pupil segmentation with:
    - Small model size (<50k parameters)
    - Efficient sparse attention for speed
    - Multi-scale features for accuracy

    Args:
        in_channels: Input image channels (1 for grayscale)
        num_classes: Number of segmentation classes (2: bg + pupil)
        embed_dims: Encoder stage dimensions
        depths: DSA blocks per stage
        num_heads: Attention heads per stage
        top_k: Tokens for sparse attention
        decoder_dim: Unified decoder dimension
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        embed_dims: Tuple[int, ...] = (16, 32, 48),
        depths: Tuple[int, ...] = (1, 1, 1),
        num_heads: Tuple[int, ...] = (2, 2, 4),
        top_k: Tuple[int, ...] = (64, 32, 16),
        decoder_dim: int = 24,
    ):
        super().__init__()

        self.encoder = DSAEncoder(
            in_channels=in_channels,
            embed_dims=embed_dims,
            depths=depths,
            num_heads=num_heads,
            top_k=top_k,
        )

        self.decoder = DSADecoder(
            encoder_dims=embed_dims,
            decoder_dim=decoder_dim,
            num_classes=num_classes,
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input image (B, C, H, W)

        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        target_size = (x.shape[2], x.shape[3])

        # Encode
        f1, f2, f3 = self.encoder(x)

        # Decode
        out = self.decoder(f1, f2, f3, target_size)

        return out

    def get_indexer_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get aggregate indexer alignment loss for training.

        This helps train the Lightning Indexers to predict
        which tokens will receive high attention.

        Args:
            x: Input image (B, C, H, W)

        Returns:
            Aggregate indexer loss
        """
        total_loss = 0.0
        count = 0

        # Get features from encoder
        f0 = self.encoder.patch_embed(x)

        # Stage 1
        for block in self.encoder.stage1.blocks:
            B, C, H, W = f0.shape
            x_flat = f0.permute(0, 2, 3, 1).reshape(B, H * W, C)
            loss = block.attn.get_indexer_loss(x_flat, H, W)
            total_loss += loss
            count += 1
            f0 = block(f0, H, W)

        # Continue through stages...
        f1 = f0
        if self.encoder.stage2.downsample is not None:
            f1 = self.encoder.stage2.downsample(f1)

        for block in self.encoder.stage2.blocks:
            B, C, H, W = f1.shape
            x_flat = f1.permute(0, 2, 3, 1).reshape(B, H * W, C)
            loss = block.attn.get_indexer_loss(x_flat, H, W)
            total_loss += loss
            count += 1
            f1 = block(f1, H, W)

        f2 = f1
        if self.encoder.stage3.downsample is not None:
            f2 = self.encoder.stage3.downsample(f2)

        for block in self.encoder.stage3.blocks:
            B, C, H, W = f2.shape
            x_flat = f2.permute(0, 2, 3, 1).reshape(B, H * W, C)
            loss = block.attn.get_indexer_loss(x_flat, H, W)
            total_loss += loss
            count += 1
            f2 = block(f2, H, W)

        return total_loss / max(count, 1)


class CombinedLoss(nn.Module):
    """
    Combined loss for pupil segmentation.

    Combines:
    1. Weighted Cross-Entropy: Standard classification loss
    2. Dice Loss: Overlap-based loss for class imbalance
    3. Surface Loss: Boundary-aware loss using distance maps
    4. Indexer Loss: Optional alignment loss for DSA

    Args:
        epsilon: Small constant for numerical stability
        indexer_weight: Weight for indexer alignment loss
    """

    def __init__(self, epsilon: float = 1e-5, indexer_weight: float = 0.1):
        super().__init__()
        self.epsilon = epsilon
        self.indexer_weight = indexer_weight
        self.nll = nn.NLLLoss(reduction='none')

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        spatial_weights: torch.Tensor,
        dist_map: torch.Tensor,
        alpha: float,
        indexer_loss: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            logits: Model output (B, C, H, W)
            target: Ground truth labels (B, H, W)
            spatial_weights: Edge-aware weights (B, H, W)
            dist_map: Distance map for surface loss (B, 2, H, W)
            alpha: Balance between dice and surface loss
            indexer_loss: Optional indexer alignment loss

        Returns:
            total_loss: Combined loss
            ce_loss: Cross-entropy component
            dice_loss: Dice loss component
            surface_loss: Surface loss component
        """
        # Softmax probabilities
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)

        # Weighted cross-entropy loss
        ce_loss = self.nll(log_probs, target)
        weighted_ce = (ce_loss * (1.0 + spatial_weights)).mean()

        # Dice loss with class weights
        target_onehot = F.one_hot(target, num_classes=2).permute(0, 3, 1, 2).float()
        probs_flat = probs.flatten(start_dim=2)
        target_flat = target_onehot.flatten(start_dim=2)

        intersection = (probs_flat * target_flat).sum(dim=2)
        cardinality = (probs_flat + target_flat).sum(dim=2)

        # Inverse frequency weighting
        class_weights = 1.0 / (target_flat.sum(dim=2) ** 2).clamp(min=self.epsilon)

        dice = (
            2.0 * (class_weights * intersection).sum(dim=1)
            / (class_weights * cardinality).sum(dim=1)
        )
        dice_loss = (1.0 - dice.clamp(min=self.epsilon)).mean()

        # Surface (boundary) loss
        surface_loss = (
            (probs.flatten(start_dim=2) * dist_map.flatten(start_dim=2))
            .mean(dim=2)
            .mean(dim=1)
            .mean()
        )

        # Combine losses
        total_loss = weighted_ce + alpha * dice_loss + (1.0 - alpha) * surface_loss

        # Add indexer loss if provided
        if indexer_loss is not None:
            total_loss = total_loss + self.indexer_weight * indexer_loss

        return total_loss, weighted_ce, dice_loss, surface_loss


# Convenience function to create model variants
def create_dsa_tiny(in_channels: int = 1, num_classes: int = 2) -> DSASegmentationModel:
    """Create tiny DSA model (~20k params)."""
    return DSASegmentationModel(
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dims=(12, 24, 36),
        depths=(1, 1, 1),
        num_heads=(2, 2, 3),
        top_k=(48, 24, 12),
        decoder_dim=18,
    )


def create_dsa_small(in_channels: int = 1, num_classes: int = 2) -> DSASegmentationModel:
    """Create small DSA model (~40k params)."""
    return DSASegmentationModel(
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dims=(16, 32, 48),
        depths=(1, 1, 1),
        num_heads=(2, 2, 4),
        top_k=(64, 32, 16),
        decoder_dim=24,
    )


def create_dsa_base(in_channels: int = 1, num_classes: int = 2) -> DSASegmentationModel:
    """Create base DSA model (~80k params)."""
    return DSASegmentationModel(
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dims=(24, 48, 72),
        depths=(1, 2, 1),
        num_heads=(2, 4, 6),
        top_k=(96, 48, 24),
        decoder_dim=36,
    )
