"""
DeepSeek Sparse Attention (DSA) for Pupil Segmentation.

This module implements a lightweight pupil segmentation model based on
DeepSeek Sparse Attention mechanisms. The architecture is optimized for:

1. Intense pixel localization (pupil segmentation is a fine-grained task)
2. Spatial awareness (pupil is only found within the eye region)
3. Efficiency (small model size for real-time inference)

Key components:
- LightningIndexer: Efficient token selection via learned scoring
- DeepSeekSparseAttention: Top-k sparse attention mechanism
- DSAPatchEmbedding: Convolutional patch embedding
- DSASegmentationModel: Complete segmentation architecture

Reference: DeepSeek-V3.2-Exp (https://arxiv.org/abs/2502.11089)
"""

from .model import (
    DSASegmentationModel,
    CombinedLoss,
)
from .lightning_indexer import LightningIndexer
from .sparse_attention import DeepSeekSparseAttention

__all__ = [
    "DSASegmentationModel",
    "CombinedLoss",
    "LightningIndexer",
    "DeepSeekSparseAttention",
]

__version__ = "1.0.0"
