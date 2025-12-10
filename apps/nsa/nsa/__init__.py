"""
NSA (Native Sparse Attention) for Pupil Segmentation.

This module implements a Native Sparse Attention mechanism adapted from
DeepSeek's NSA paper for efficient pupil segmentation in eye images.

Key components:
- Token Compression: Coarse-grained global context
- Token Selection: Fine-grained important region focus
- Sliding Window: Local context for precise boundaries
- Gated Aggregation: Learned combination of all attention paths

Adapted for 2D vision tasks (segmentation) from the original 1D NLP formulation.
"""

from .model import (
    NSAPupilSeg,
    NSABlock,
    SpatialNSA,
    TokenCompression,
    TokenSelection,
    SlidingWindowAttention,
    CombinedLoss,
    create_nsa_pupil_seg,
)

__all__ = [
    "NSAPupilSeg",
    "NSABlock",
    "SpatialNSA",
    "TokenCompression",
    "TokenSelection",
    "SlidingWindowAttention",
    "CombinedLoss",
    "create_nsa_pupil_seg",
]
