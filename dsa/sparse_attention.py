"""
DeepSeek Sparse Attention (DSA) Module.

Implements fine-grained sparse attention with top-k token selection,
powered by the Lightning Indexer for efficient token scoring.

From the DeepSeek-V3.2-Exp paper (Equation 2):

    u_t = Attn(h_t, {c_s | I_{t,s} in Top-k(I_{t,:})})

Where:
- u_t is the attention output for query token t
- c_s are the key-value entries for selected tokens
- Top-k selects tokens with highest index scores

Key benefits:
- Reduces attention complexity from O(L^2) to O(L*k)
- Lightning indexer has minimal overhead (small heads, FP8 compatible)
- Maintains accuracy by selecting the most relevant tokens

For pupil segmentation:
- k is set relatively small since pupil is localized
- Spatial locality is leveraged for efficient selection
- Multi-scale attention captures both fine details and context
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    from .lightning_indexer import LightningIndexer, SpatialLightningIndexer
except ImportError:
    from lightning_indexer import LightningIndexer, SpatialLightningIndexer


class DeepSeekSparseAttention(nn.Module):
    """
    DeepSeek Sparse Attention with Lightning Indexer.

    This module implements the core DSA mechanism:
    1. Lightning Indexer computes index scores for all token pairs
    2. Top-k selection identifies most relevant key-value pairs
    3. Standard attention is computed only on selected pairs

    Optimized for pupil segmentation:
    - Uses MQA (Multi-Query Attention) for efficiency
    - Small top-k value since pupil is spatially localized
    - Spatial-aware indexer leverages 2D image structure

    Args:
        dim: Input embedding dimension
        num_heads: Number of attention heads
        key_dim: Dimension per attention head
        value_dim: Dimension of values (can differ from key_dim)
        top_k: Number of tokens to select per query
        indexer_heads: Number of Lightning Indexer heads
        indexer_dim: Dimension of indexer keys/queries
        dropout: Attention dropout probability
        use_spatial_indexer: Use spatial-aware indexer for images
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        key_dim: int = 16,
        value_dim: int = None,
        top_k: int = 64,
        indexer_heads: int = 2,
        indexer_dim: int = 8,
        dropout: float = 0.0,
        use_spatial_indexer: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim or key_dim
        self.top_k = top_k
        self.scale = key_dim ** -0.5

        # Lightning Indexer for token selection
        if use_spatial_indexer:
            self.indexer = SpatialLightningIndexer(
                dim=dim,
                num_heads=indexer_heads,
                key_dim=indexer_dim,
            )
        else:
            self.indexer = LightningIndexer(
                dim=dim,
                num_heads=indexer_heads,
                key_dim=indexer_dim,
            )

        # Main attention projections (MQA style - shared K,V across heads)
        # Query: per-head projections
        self.q_proj = nn.Linear(dim, num_heads * key_dim, bias=False)
        # Key/Value: shared across heads (MQA for efficiency)
        self.k_proj = nn.Linear(dim, key_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.value_dim, bias=False)
        # Output projection
        self.out_proj = nn.Linear(num_heads * self.value_dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        # For tracking indexer loss during training
        self.indexer_loss = None

        self._init_weights()

    def _init_weights(self):
        """Initialize attention weights."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def _select_top_k(
        self,
        index_scores: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select top-k key-value pairs based on index scores.

        Args:
            index_scores: Scores from Lightning Indexer (B, N_q, N_k)
            keys: All keys (B, N_k, key_dim)
            values: All values (B, N_k, value_dim)

        Returns:
            selected_keys: (B, N_q, k, key_dim)
            selected_values: (B, N_q, k, value_dim)
            selected_indices: (B, N_q, k) indices of selected tokens
        """
        B, N_q, N_k = index_scores.shape

        # Clamp top_k to not exceed N_k
        k = min(self.top_k, N_k)

        # Get top-k indices for each query
        _, top_indices = torch.topk(index_scores, k, dim=-1)  # (B, N_q, k)

        # Gather selected keys and values
        # Expand indices for gathering
        batch_indices = torch.arange(B, device=keys.device)[:, None, None].expand(-1, N_q, k)
        key_indices = top_indices.unsqueeze(-1).expand(-1, -1, -1, self.key_dim)
        value_indices = top_indices.unsqueeze(-1).expand(-1, -1, -1, self.value_dim)

        # Use advanced indexing for efficient gathering
        selected_keys = torch.gather(
            keys.unsqueeze(1).expand(-1, N_q, -1, -1),
            2,
            key_indices
        )  # (B, N_q, k, key_dim)

        selected_values = torch.gather(
            values.unsqueeze(1).expand(-1, N_q, -1, -1),
            2,
            value_indices
        )  # (B, N_q, k, value_dim)

        return selected_keys, selected_values, top_indices

    def forward(
        self,
        x: torch.Tensor,
        height: int = None,
        width: int = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with sparse attention.

        Args:
            x: Input tensor (B, N, D) where N = H * W for images
            height: Feature map height (for spatial indexer)
            width: Feature map width (for spatial indexer)
            return_attention: Whether to return attention weights

        Returns:
            output: Attention output (B, N, D)
            attention_weights: Optional sparse attention weights
        """
        B, N, D = x.shape

        # Compute Lightning Index scores
        index_scores = self.indexer(x, x, height, width)  # (B, N, N)

        # Project to queries, keys, values
        q = self.q_proj(x)  # (B, N, num_heads * key_dim)
        k = self.k_proj(x)  # (B, N, key_dim) - shared across heads
        v = self.v_proj(x)  # (B, N, value_dim) - shared across heads

        # Reshape queries for multi-head attention
        q = q.view(B, N, self.num_heads, self.key_dim).transpose(1, 2)  # (B, H, N, key_dim)

        # Select top-k keys and values based on index scores
        selected_k, selected_v, selected_idx = self._select_top_k(index_scores, k, v)
        # selected_k: (B, N, k, key_dim)
        # selected_v: (B, N, k, value_dim)

        # Compute attention with selected keys/values
        # Expand selected keys for all heads
        selected_k = selected_k.unsqueeze(1).expand(-1, self.num_heads, -1, -1, -1)
        selected_v = selected_v.unsqueeze(1).expand(-1, self.num_heads, -1, -1, -1)
        # (B, H, N, k, dim)

        # Compute attention scores: (B, H, N, k)
        q = q.unsqueeze(-2)  # (B, H, N, 1, key_dim)
        attn_scores = (q * selected_k).sum(dim=-1) * self.scale  # (B, H, N, k)

        # Softmax over selected tokens
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Compute output: (B, H, N, value_dim)
        attn_probs = attn_probs.unsqueeze(-1)  # (B, H, N, k, 1)
        output = (attn_probs * selected_v).sum(dim=-2)  # (B, H, N, value_dim)

        # Concatenate heads and project
        output = output.transpose(1, 2).contiguous().view(B, N, -1)  # (B, N, H * value_dim)
        output = self.out_proj(output)  # (B, N, D)

        if return_attention:
            # Return sparse attention map (indices + weights)
            return output, (selected_idx, attn_probs.squeeze(-1))

        return output, None

    def get_indexer_loss(
        self,
        x: torch.Tensor,
        height: int = None,
        width: int = None,
    ) -> torch.Tensor:
        """
        Compute indexer alignment loss for training.

        This loss aligns the Lightning Indexer output with the actual
        attention distribution, enabling the indexer to predict which
        tokens will receive high attention.

        Args:
            x: Input tensor (B, N, D)
            height: Feature map height
            width: Feature map width

        Returns:
            KL-divergence loss for indexer alignment
        """
        B, N, D = x.shape

        # Compute index scores
        index_scores = self.indexer(x, x, height, width)

        # Compute full attention (for alignment target)
        q = self.q_proj(x).view(B, N, self.num_heads, self.key_dim)
        k = self.k_proj(x)

        # Full attention scores
        q = q.permute(0, 2, 1, 3)  # (B, H, N, key_dim)
        k = k.unsqueeze(1)  # (B, 1, N, key_dim)
        full_attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        full_attn = F.softmax(full_attn, dim=-1)

        # L1-normalize attention across heads
        target = full_attn.sum(dim=1)  # (B, N, N)
        target = target / (target.sum(dim=-1, keepdim=True) + 1e-8)

        # For sparse training, only compute loss on selected tokens
        k = min(self.top_k, N)
        _, top_indices = torch.topk(index_scores, k, dim=-1)

        # Create selection mask
        selected_mask = torch.zeros_like(index_scores, dtype=torch.bool)
        selected_mask.scatter_(-1, top_indices, True)

        # Compute KL loss
        loss = self.indexer.compute_alignment_loss(
            index_scores, target, selected_mask
        )

        return loss


class DSABlock(nn.Module):
    """
    DeepSeek Sparse Attention Block.

    Combines sparse attention with feed-forward network in a
    transformer-style block with residual connections.

    For pupil segmentation:
    - Depthwise conv for local feature extraction
    - Sparse attention for efficient global context
    - MLP for channel mixing

    Args:
        dim: Embedding dimension
        num_heads: Number of attention heads
        top_k: Number of tokens for sparse attention
        mlp_ratio: MLP hidden dimension ratio
        dropout: Dropout probability
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        top_k: int = 64,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        # Depthwise conv for local features
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.dwconv_bn = nn.BatchNorm2d(dim)

        # Sparse attention
        self.attn = DeepSeekSparseAttention(
            dim=dim,
            num_heads=num_heads,
            top_k=top_k,
            dropout=dropout,
            use_spatial_indexer=True,
        )

        # MLP
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """
        Forward pass through DSA block.

        Args:
            x: Input tensor (B, C, H, W)
            height: Feature map height
            width: Feature map width

        Returns:
            Output tensor (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Local features via depthwise conv
        x = x + F.gelu(self.dwconv_bn(self.dwconv(x)))

        # Reshape for attention: (B, C, H, W) -> (B, H*W, C)
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Sparse attention with residual
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, height=H, width=W)
        x_flat = x_flat + attn_out

        # MLP with residual
        x_flat = x_flat + self.mlp(self.norm2(x_flat))

        # Reshape back: (B, H*W, C) -> (B, C, H, W)
        x = x_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return x


class DSAStage(nn.Module):
    """
    Stage of DSA blocks with optional downsampling.

    Args:
        in_dim: Input dimension
        out_dim: Output dimension
        depth: Number of DSA blocks
        num_heads: Attention heads per block
        top_k: Tokens per sparse attention
        downsample: Whether to downsample spatially
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        depth: int = 1,
        num_heads: int = 4,
        top_k: int = 64,
        downsample: bool = True,
    ):
        super().__init__()

        # Optional downsampling
        self.downsample = None
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.GELU(),
            )
        elif in_dim != out_dim:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 1),
                nn.BatchNorm2d(out_dim),
                nn.GELU(),
            )

        # DSA blocks
        self.blocks = nn.ModuleList([
            DSABlock(
                dim=out_dim,
                num_heads=num_heads,
                top_k=top_k,
            )
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through stage.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Output tensor (B, C', H', W')
        """
        if self.downsample is not None:
            x = self.downsample(x)

        B, C, H, W = x.shape

        for block in self.blocks:
            x = block(x, H, W)

        return x
