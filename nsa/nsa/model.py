"""
Native Sparse Attention (NSA) Model for Pupil Segmentation.

Implementation based on DeepSeek's NSA paper:
"Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention"

Adapted for 2D vision/segmentation tasks with domain-specific optimizations for
pupil segmentation where:
- Intense pixel localization is required
- The pupil is only found on the eye (spatial locality)
- OpenEDS provides multi-class data beyond pupil

Architecture:
- Encoder with NSA blocks for hierarchical feature extraction
- Decoder with skip connections for precise segmentation
- NSA combines: Compression (global), Selection (important), Sliding Window (local)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Core Building Blocks
# =============================================================================


class ConvBNReLU(nn.Module):
    """Convolution + BatchNorm + Activation block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        bias: bool = False,
        activation: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(
            out_channels
        )
        self.act = (
            nn.GELU()
            if activation
            else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        return self.act(
            self.bn(self.conv(x))
        )


class PatchEmbedding(nn.Module):
    """
    Embed image patches into tokens for attention processing.
    Uses strided convolutions to reduce spatial resolution.
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 32,
        patch_size: int = 4,
    ):
        super().__init__()
        self.patch_size = patch_size
        mid_dim = embed_dim // 2

        # Two-stage downsampling for smoother feature transition
        self.conv1 = ConvBNReLU(
            in_channels,
            mid_dim,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv2 = ConvBNReLU(
            mid_dim,
            embed_dim,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Input image (B, C, H, W)
        Returns:
            Embedded patches (B, embed_dim, H//4, W//4)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# =============================================================================
# Token Compression Module
# =============================================================================


class TokenCompression(nn.Module):
    """
    Compress spatial blocks into single tokens for coarse-grained attention.

    From NSA paper Eq. 7:
    K_cmp = {φ(k_{id+1:id+l}) | 0 ≤ i ≤ ⌊(t-l)/d⌋}

    Adapted for 2D: compress spatial blocks into representative tokens.
    """

    def __init__(
        self,
        dim: int,
        block_size: int = 4,
        stride: int = 2,
    ):
        super().__init__()
        self.block_size = block_size
        self.stride = stride

        # Learnable compression MLP with position encoding
        self.compress_k = nn.Sequential(
            nn.Linear(
                dim
                * block_size
                * block_size,
                dim * 2,
            ),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.compress_v = nn.Sequential(
            nn.Linear(
                dim
                * block_size
                * block_size,
                dim * 2,
            ),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

        # Intra-block position encoding
        self.pos_embed = nn.Parameter(
            torch.randn(
                1,
                block_size * block_size,
                dim,
            )
            * 0.02
        )

    def forward(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        spatial_size: tuple[int, int],
    ) -> tuple[
        torch.Tensor, torch.Tensor
    ]:
        """
        Compress keys and values into block-level representations.

        Args:
            k: Keys (B, N, dim) where N = H * W
            v: Values (B, N, dim)
            spatial_size: (H, W) tuple for non-square inputs
        Returns:
            k_cmp: Compressed keys (B, N_cmp, dim)
            v_cmp: Compressed values (B, N_cmp, dim)
        """
        B, N, dim = k.shape

        # Use provided spatial dimensions for non-square inputs
        H, W = spatial_size
        bs = self.block_size
        stride = self.stride

        # Calculate number of blocks
        n_blocks_h = (
            H - bs
        ) // stride + 1
        n_blocks_w = (
            W - bs
        ) // stride + 1

        # Extract overlapping blocks using unfold
        # Use reshape instead of view for non-contiguous tensors
        k_2d = (
            k.reshape(B, H, W, dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )  # (B, dim, H, W)
        v_2d = (
            v.reshape(B, H, W, dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        # Unfold to get blocks: (B, dim*bs*bs, n_blocks)
        k_blocks = F.unfold(
            k_2d,
            kernel_size=bs,
            stride=stride,
        )
        v_blocks = F.unfold(
            v_2d,
            kernel_size=bs,
            stride=stride,
        )

        # Reshape for compression: (B, n_blocks, dim*bs*bs)
        n_blocks = k_blocks.shape[2]
        k_blocks = k_blocks.permute(
            0, 2, 1
        ).contiguous()
        v_blocks = v_blocks.permute(
            0, 2, 1
        ).contiguous()

        # Add position encoding before compression
        # Reshape blocks to add position encoding: (B, n_blocks, bs*bs, dim)
        k_blocks_reshaped = (
            k_blocks.reshape(
                B,
                n_blocks,
                bs * bs,
                dim,
            )
        )
        k_blocks_reshaped = (
            k_blocks_reshaped
            + self.pos_embed.unsqueeze(
                0
            )
        )
        k_blocks_pos = (
            k_blocks_reshaped.reshape(
                B,
                n_blocks,
                bs * bs * dim,
            )
        )

        # Compress to single tokens
        k_cmp = self.compress_k(
            k_blocks_pos
        )
        v_cmp = self.compress_v(
            v_blocks
        )

        return k_cmp, v_cmp


# =============================================================================
# Token Selection Module
# =============================================================================


class TokenSelection(nn.Module):
    """
    Select important token blocks based on attention scores.

    From NSA paper Eq. 8-12:
    - Compute importance from compressed attention scores
    - Select top-n blocks for fine-grained attention

    For pupil segmentation: identifies the most relevant spatial regions.
    """

    def __init__(
        self,
        dim: int,
        block_size: int = 4,
        num_select: int = 4,
    ):
        super().__init__()
        self.block_size = block_size
        self.num_select = num_select
        self.dim = dim

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_scores_cmp: torch.Tensor,
        spatial_size: tuple[int, int],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Select important blocks based on compressed attention scores.

        Args:
            q: Queries (B, H, N, dim)
            k: Keys (B, N, dim)
            v: Values (B, N, dim)
            attn_scores_cmp: Attention from compression (B, H, N, N_cmp)
            spatial_size: (height, width) of feature map
        Returns:
            k_slc: Selected keys
            v_slc: Selected values
            indices: Selected block indices
        """
        B, num_heads, N, N_cmp = (
            attn_scores_cmp.shape
        )
        H, W = spatial_size
        bs = self.block_size

        # Sum attention across heads for shared selection (GQA-style)
        importance = (
            attn_scores_cmp.sum(dim=1)
        )  # (B, N, N_cmp)

        # Average importance across queries to get block scores
        block_importance = (
            importance.mean(dim=1)
        )  # (B, N_cmp)

        # Select top-n blocks
        num_select = min(
            self.num_select, N_cmp
        )
        _, indices = torch.topk(
            block_importance,
            num_select,
            dim=-1,
        )  # (B, num_select)

        # Map compressed indices back to original token blocks
        # This is simplified - in practice would need proper index mapping
        # For now, use the indices to gather from original k, v

        # Reshape k, v to blocks
        n_blocks_h = (H - bs) // bs + 1
        n_blocks_w = (W - bs) // bs + 1

        # Gather selected blocks
        k_2d = (
            k.reshape(B, H, W, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        v_2d = (
            v.reshape(B, H, W, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        # Use unfold to extract all blocks
        k_blocks = F.unfold(
            k_2d,
            kernel_size=bs,
            stride=bs,
        )  # (B, dim*bs*bs, n_blocks)
        v_blocks = F.unfold(
            v_2d,
            kernel_size=bs,
            stride=bs,
        )

        n_blocks = k_blocks.shape[2]
        k_blocks = (
            k_blocks.permute(0, 2, 1)
            .contiguous()
            .reshape(
                B, n_blocks, bs * bs, -1
            )
        )
        v_blocks = (
            v_blocks.permute(0, 2, 1)
            .contiguous()
            .reshape(
                B, n_blocks, bs * bs, -1
            )
        )

        # Clamp indices to valid range
        indices = indices.clamp(
            0, n_blocks - 1
        )

        # Gather selected blocks
        indices_expanded = (
            indices.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(
                -1,
                -1,
                bs * bs,
                k.shape[-1],
            )
        )
        k_slc = torch.gather(
            k_blocks,
            1,
            indices_expanded,
        )  # (B, num_select, bs*bs, dim)
        v_slc = torch.gather(
            v_blocks,
            1,
            indices_expanded,
        )

        # Flatten selected blocks
        k_slc = k_slc.view(
            B, num_select * bs * bs, -1
        )
        v_slc = v_slc.view(
            B, num_select * bs * bs, -1
        )

        return k_slc, v_slc, indices


# =============================================================================
# Sliding Window Attention
# =============================================================================


class SlidingWindowAttention(nn.Module):
    """
    Local sliding window attention for fine-grained local context.

    From NSA paper Section 3.3.3:
    Maintains recent tokens in a window for local pattern recognition.

    For pupil segmentation: critical for precise boundary delineation.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 2,
        window_size: int = 7,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(
            dim, dim * 3, bias=qkv_bias
        )
        self.proj = nn.Linear(dim, dim)

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size - 1)
                * (2 * window_size - 1),
                num_heads,
            )
        )
        nn.init.trunc_normal_(
            self.relative_position_bias_table,
            std=0.02,
        )

        # Create position index
        coords_h = torch.arange(
            window_size
        )
        coords_w = torch.arange(
            window_size
        )
        coords = torch.stack(
            torch.meshgrid(
                coords_h,
                coords_w,
                indexing="ij",
            )
        )
        coords_flatten = coords.flatten(
            1
        )
        relative_coords = (
            coords_flatten[:, :, None]
            - coords_flatten[:, None, :]
        )
        relative_coords = (
            relative_coords.permute(
                1, 2, 0
            ).contiguous()
        )
        relative_coords[:, :, 0] += (
            window_size - 1
        )
        relative_coords[:, :, 1] += (
            window_size - 1
        )
        relative_coords[:, :, 0] *= (
            2 * window_size - 1
        )
        relative_position_index = (
            relative_coords.sum(-1)
        )
        self.register_buffer(
            "relative_position_index",
            relative_position_index,
        )

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply sliding window attention.

        Args:
            x: Input features (B, C, H, W)
        Returns:
            Output features (B, C, H, W)
        """
        B, C, H, W = x.shape
        ws = self.window_size

        # Pad to multiple of window size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, (0, pad_w, 0, pad_h)
            )

        _, _, Hp, Wp = x.shape

        # Reshape to windows: (B*num_windows, ws*ws, C)
        x = x.view(
            B,
            C,
            Hp // ws,
            ws,
            Wp // ws,
            ws,
        )
        x = x.permute(
            0, 2, 4, 3, 5, 1
        ).contiguous()
        x = x.view(-1, ws * ws, C)

        # Compute QKV
        B_win = x.shape[0]
        qkv = self.qkv(x).reshape(
            B_win,
            ws * ws,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (
            q @ k.transpose(-2, -1)
        ) * self.scale

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(
                -1
            )
        ].view(
            ws * ws, ws * ws, -1
        )
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()
        attn = (
            attn
            + relative_position_bias.unsqueeze(
                0
            )
        )

        attn = attn.softmax(dim=-1)
        x = (
            (attn @ v)
            .transpose(1, 2)
            .reshape(B_win, ws * ws, C)
        )
        x = self.proj(x)

        # Reshape back
        num_windows_h = Hp // ws
        num_windows_w = Wp // ws
        x = x.view(
            B,
            num_windows_h,
            num_windows_w,
            ws,
            ws,
            C,
        )
        x = x.permute(
            0, 5, 1, 3, 2, 4
        ).contiguous()
        x = x.view(B, C, Hp, Wp)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]

        return x


# =============================================================================
# Native Sparse Attention (NSA) - Core Module
# =============================================================================


class SpatialNSA(nn.Module):
    """
    Native Sparse Attention adapted for 2D spatial features.

    Combines three attention paths (NSA paper Eq. 5):
    o* = Σ g_c · Attn(q, K̃_c, Ṽ_c) for c ∈ {cmp, slc, win}

    Components:
    1. Compressed Attention: Global coarse-grained context
    2. Selected Attention: Fine-grained important regions
    3. Sliding Window: Local context for precise boundaries
    4. Gated Aggregation: Learned combination
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 2,
        compress_block_size: int = 4,
        compress_stride: int = 2,
        select_block_size: int = 4,
        num_select: int = 4,
        window_size: int = 7,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        # Separate QKV for each branch (prevents shortcut learning)
        self.qkv_cmp = nn.Linear(
            dim, dim * 3, bias=qkv_bias
        )
        self.qkv_slc = nn.Linear(
            dim, dim * 3, bias=qkv_bias
        )

        # Token compression module
        self.compression = TokenCompression(
            dim=dim,
            block_size=compress_block_size,
            stride=compress_stride,
        )

        # Token selection module
        self.selection = TokenSelection(
            dim=dim,
            block_size=select_block_size,
            num_select=num_select,
        )

        # Sliding window attention
        self.window_attn = (
            SlidingWindowAttention(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                qkv_bias=qkv_bias,
            )
        )

        # Output projections
        self.proj_cmp = nn.Linear(
            dim, dim
        )
        self.proj_slc = nn.Linear(
            dim, dim
        )

        # Gating mechanism (NSA paper Eq. 5)
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 3),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply Native Sparse Attention.

        Args:
            x: Input features (B, C, H, W)
        Returns:
            Output features (B, C, H, W)
        """
        B, C, H, W = x.shape
        N = H * W

        # Reshape to sequence
        x_seq = x.flatten(2).transpose(
            1, 2
        )  # (B, N, C)

        # =================================================================
        # Branch 1: Compressed Attention (Global Coarse-Grained)
        # =================================================================
        qkv_cmp = self.qkv_cmp(x_seq)
        qkv_cmp = qkv_cmp.reshape(
            B,
            N,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv_cmp = qkv_cmp.permute(
            2, 0, 3, 1, 4
        )
        q_cmp, k_cmp_raw, v_cmp_raw = (
            qkv_cmp[0],
            qkv_cmp[1],
            qkv_cmp[2],
        )

        # Reshape k, v for compression
        k_for_cmp = k_cmp_raw.transpose(
            1, 2
        ).reshape(B, N, C)
        v_for_cmp = v_cmp_raw.transpose(
            1, 2
        ).reshape(B, N, C)

        # Compress tokens
        k_cmp, v_cmp = self.compression(
            k_for_cmp, v_for_cmp, (H, W)
        )
        N_cmp = k_cmp.shape[1]

        # Reshape for multi-head attention
        k_cmp = k_cmp.view(
            B,
            N_cmp,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        v_cmp = v_cmp.view(
            B,
            N_cmp,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        # Compute compressed attention
        attn_cmp = (
            q_cmp
            @ k_cmp.transpose(-2, -1)
        ) * self.scale
        attn_cmp_softmax = (
            attn_cmp.softmax(dim=-1)
        )
        o_cmp = attn_cmp_softmax @ v_cmp
        o_cmp = o_cmp.transpose(
            1, 2
        ).reshape(B, N, C)
        o_cmp = self.proj_cmp(o_cmp)

        # =================================================================
        # Branch 2: Selected Attention (Fine-Grained Important)
        # =================================================================
        qkv_slc = self.qkv_slc(x_seq)
        qkv_slc = qkv_slc.reshape(
            B,
            N,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv_slc = qkv_slc.permute(
            2, 0, 3, 1, 4
        )
        q_slc, k_slc_raw, v_slc_raw = (
            qkv_slc[0],
            qkv_slc[1],
            qkv_slc[2],
        )

        k_for_slc = k_slc_raw.transpose(
            1, 2
        ).reshape(B, N, C)
        v_for_slc = v_slc_raw.transpose(
            1, 2
        ).reshape(B, N, C)

        # Select important blocks based on compressed attention scores
        k_slc, v_slc, _ = (
            self.selection(
                q_slc,
                k_for_slc,
                v_for_slc,
                attn_cmp_softmax,
                (H, W),
            )
        )

        N_slc = k_slc.shape[1]
        k_slc = k_slc.view(
            B,
            N_slc,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        v_slc = v_slc.view(
            B,
            N_slc,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        # Compute selected attention
        attn_slc = (
            q_slc
            @ k_slc.transpose(-2, -1)
        ) * self.scale
        attn_slc = attn_slc.softmax(
            dim=-1
        )
        o_slc = attn_slc @ v_slc
        o_slc = o_slc.transpose(
            1, 2
        ).reshape(B, N, C)
        o_slc = self.proj_slc(o_slc)

        # =================================================================
        # Branch 3: Sliding Window Attention (Local Context)
        # =================================================================
        o_win = self.window_attn(x)
        o_win = o_win.flatten(
            2
        ).transpose(
            1, 2
        )  # (B, N, C)

        # =================================================================
        # Gated Aggregation
        # =================================================================
        # Compute per-token gates
        gates = self.gate(
            x_seq
        )  # (B, N, 3)
        g_cmp = gates[:, :, 0:1]
        g_slc = gates[:, :, 1:2]
        g_win = gates[:, :, 2:3]

        # Weighted combination
        out = (
            g_cmp * o_cmp
            + g_slc * o_slc
            + g_win * o_win
        )

        # Reshape back to spatial
        out = out.transpose(1, 2).view(
            B, C, H, W
        )

        return out


# =============================================================================
# NSA Block (Attention + FFN)
# =============================================================================


class NSABlock(nn.Module):
    """
    Complete NSA block with attention, normalization, and FFN.

    Structure:
    - Depthwise conv for local features (like EfficientViT)
    - Native Sparse Attention for global/selective features
    - FFN for channel mixing
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 2,
        mlp_ratio: float = 2.0,
        compress_block_size: int = 4,
        compress_stride: int = 2,
        select_block_size: int = 4,
        num_select: int = 4,
        window_size: int = 7,
    ):
        super().__init__()

        # Local feature extraction (depthwise conv)
        self.norm1 = nn.BatchNorm2d(dim)
        self.dw_conv = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            padding=1,
            groups=dim,
        )

        # NSA attention
        self.norm2 = nn.BatchNorm2d(dim)
        self.nsa = SpatialNSA(
            dim=dim,
            num_heads=num_heads,
            compress_block_size=compress_block_size,
            compress_stride=compress_stride,
            select_block_size=select_block_size,
            num_select=num_select,
            window_size=window_size,
        )

        # FFN
        self.norm3 = nn.LayerNorm(dim)
        hidden_dim = int(
            dim * mlp_ratio
        )
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Input features (B, C, H, W)
        Returns:
            Output features (B, C, H, W)
        """
        # Local features
        x = x + self.dw_conv(
            self.norm1(x)
        )

        # NSA attention
        x = x + self.nsa(self.norm2(x))

        # FFN
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(
            1, 2
        )  # (B, N, C)
        x_flat = x_flat + self.ffn(
            self.norm3(x_flat)
        )
        x = x_flat.transpose(1, 2).view(
            B, C, H, W
        )

        return x


# =============================================================================
# NSA Stage (Multiple Blocks + Optional Downsampling)
# =============================================================================


class NSAStage(nn.Module):
    """
    Stage containing multiple NSA blocks with optional downsampling.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        depth: int = 1,
        num_heads: int = 2,
        mlp_ratio: float = 2.0,
        compress_block_size: int = 4,
        compress_stride: int = 2,
        select_block_size: int = 4,
        num_select: int = 4,
        window_size: int = 7,
        downsample: bool = True,
    ):
        super().__init__()

        # Downsampling
        self.downsample = None
        if downsample:
            self.downsample = (
                nn.Sequential(
                    ConvBNReLU(
                        in_dim,
                        out_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                )
            )
        elif in_dim != out_dim:
            self.downsample = (
                ConvBNReLU(
                    in_dim,
                    out_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        # NSA blocks
        self.blocks = nn.ModuleList(
            [
                NSABlock(
                    dim=out_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    compress_block_size=compress_block_size,
                    compress_stride=compress_stride,
                    select_block_size=select_block_size,
                    num_select=num_select,
                    window_size=window_size,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        if self.downsample is not None:
            x = self.downsample(x)
        for block in self.blocks:
            x = block(x)
        return x


# =============================================================================
# NSA Encoder
# =============================================================================


class NSAEncoder(nn.Module):
    """
    NSA-based encoder for hierarchical feature extraction.
    Produces multi-scale features for segmentation decoder.
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dims: tuple = (
            32,
            64,
            96,
        ),
        depths: tuple = (1, 1, 1),
        num_heads: tuple = (2, 2, 4),
        mlp_ratios: tuple = (2, 2, 2),
        compress_block_sizes: tuple = (
            4,
            4,
            4,
        ),
        compress_strides: tuple = (
            2,
            2,
            2,
        ),
        select_block_sizes: tuple = (
            4,
            4,
            4,
        ),
        num_selects: tuple = (4, 4, 4),
        window_sizes: tuple = (7, 7, 7),
    ):
        super().__init__()

        # Patch embedding
        self.patch_embed = (
            PatchEmbedding(
                in_channels=in_channels,
                embed_dim=embed_dims[0],
            )
        )

        # Stage 1: No downsampling (already done in patch embed)
        self.stage1 = NSAStage(
            in_dim=embed_dims[0],
            out_dim=embed_dims[0],
            depth=depths[0],
            num_heads=num_heads[0],
            mlp_ratio=mlp_ratios[0],
            compress_block_size=compress_block_sizes[
                0
            ],
            compress_stride=compress_strides[
                0
            ],
            select_block_size=select_block_sizes[
                0
            ],
            num_select=num_selects[0],
            window_size=window_sizes[0],
            downsample=False,
        )

        # Stage 2: Downsample 2x
        self.stage2 = NSAStage(
            in_dim=embed_dims[0],
            out_dim=embed_dims[1],
            depth=depths[1],
            num_heads=num_heads[1],
            mlp_ratio=mlp_ratios[1],
            compress_block_size=compress_block_sizes[
                1
            ],
            compress_stride=compress_strides[
                1
            ],
            select_block_size=select_block_sizes[
                1
            ],
            num_select=num_selects[1],
            window_size=window_sizes[1],
            downsample=True,
        )

        # Stage 3: Downsample 2x
        self.stage3 = NSAStage(
            in_dim=embed_dims[1],
            out_dim=embed_dims[2],
            depth=depths[2],
            num_heads=num_heads[2],
            mlp_ratio=mlp_ratios[2],
            compress_block_size=compress_block_sizes[
                2
            ],
            compress_stride=compress_strides[
                2
            ],
            select_block_size=select_block_sizes[
                2
            ],
            num_select=num_selects[2],
            window_size=window_sizes[2],
            downsample=True,
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple:
        """
        Args:
            x: Input image (B, C, H, W)
        Returns:
            Multi-scale features (f1, f2, f3)
        """
        x = self.patch_embed(x)
        f1 = self.stage1(
            x
        )  # 1/4 resolution
        f2 = self.stage2(
            f1
        )  # 1/8 resolution
        f3 = self.stage3(
            f2
        )  # 1/16 resolution
        return f1, f2, f3


# =============================================================================
# Segmentation Decoder
# =============================================================================


class SegmentationDecoder(nn.Module):
    """
    FPN-style decoder with skip connections for precise segmentation.
    Progressively upsamples features to input resolution.
    """

    def __init__(
        self,
        encoder_dims: tuple = (
            32,
            64,
            96,
        ),
        decoder_dim: int = 32,
        num_classes: int = 2,
    ):
        super().__init__()

        # Lateral connections
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

        # Smoothing convolutions
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

        # Segmentation head
        self.head = nn.Conv2d(
            decoder_dim,
            num_classes,
            kernel_size=1,
        )

    def forward(
        self,
        f1: torch.Tensor,
        f2: torch.Tensor,
        f3: torch.Tensor,
        target_size: tuple,
    ) -> torch.Tensor:
        """
        Args:
            f1, f2, f3: Multi-scale encoder features
            target_size: (H, W) of output
        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        # Top-down path with lateral connections
        p3 = self.lateral3(f3)
        p3 = self.smooth3(p3)

        p2 = self.lateral2(
            f2
        ) + F.interpolate(
            p3,
            size=f2.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        p2 = self.smooth2(p2)

        p1 = self.lateral1(
            f1
        ) + F.interpolate(
            p2,
            size=f1.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        p1 = self.smooth1(p1)

        # Segmentation output
        out = self.head(p1)
        out = F.interpolate(
            out,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        return out


# =============================================================================
# Complete NSA Pupil Segmentation Model
# =============================================================================


class NSAPupilSeg(nn.Module):
    """
    Native Sparse Attention model for Pupil Segmentation.

    Architecture:
    - NSA Encoder: Hierarchical feature extraction with sparse attention
    - FPN Decoder: Multi-scale feature fusion for precise segmentation

    Key NSA components for pupil segmentation:
    - Compression: Captures global eye context (is this an eye? rough pupil location)
    - Selection: Focuses on pupil region with fine-grained attention
    - Sliding Window: Precise local boundaries for pixel-accurate segmentation
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        embed_dims: tuple = (
            32,
            64,
            96,
        ),
        depths: tuple = (1, 1, 1),
        num_heads: tuple = (2, 2, 4),
        mlp_ratios: tuple = (2, 2, 2),
        compress_block_sizes: tuple = (
            4,
            4,
            4,
        ),
        compress_strides: tuple = (
            2,
            2,
            2,
        ),
        select_block_sizes: tuple = (
            4,
            4,
            4,
        ),
        num_selects: tuple = (4, 4, 4),
        window_sizes: tuple = (7, 7, 7),
        decoder_dim: int = 32,
    ):
        super().__init__()

        self.encoder = NSAEncoder(
            in_channels=in_channels,
            embed_dims=embed_dims,
            depths=depths,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            compress_block_sizes=compress_block_sizes,
            compress_strides=compress_strides,
            select_block_sizes=select_block_sizes,
            num_selects=num_selects,
            window_sizes=window_sizes,
        )

        self.decoder = (
            SegmentationDecoder(
                encoder_dims=embed_dims,
                decoder_dim=decoder_dim,
                num_classes=num_classes,
            )
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if m.bias is not None:
                    nn.init.zeros_(
                        m.bias
                    )
            elif isinstance(
                m, nn.BatchNorm2d
            ):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(
                m, nn.Linear
            ):
                nn.init.trunc_normal_(
                    m.weight, std=0.02
                )
                if m.bias is not None:
                    nn.init.zeros_(
                        m.bias
                    )
            elif isinstance(
                m, nn.LayerNorm
            ):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Input image (B, C, H, W)
        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        target_size = (
            x.shape[2],
            x.shape[3],
        )
        f1, f2, f3 = self.encoder(x)
        out = self.decoder(
            f1, f2, f3, target_size
        )
        return out


# =============================================================================
# Loss Function (same as src/ for compatibility)
# =============================================================================


def focal_surface_loss(
    probs: torch.Tensor,
    dist_map: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Surface loss with focal weighting for hard boundary pixels.

    Args:
        probs: Predicted probabilities (B, C, H, W)
        dist_map: Distance transform (B, 2, H, W)
        gamma: Focal weighting exponent

    Returns:
        Focal-weighted surface loss scalar
    """
    focal_weight = (1 - probs) ** gamma
    return (
        (focal_weight * probs * dist_map)
        .flatten(start_dim=2)
        .mean(dim=2)
        .mean(dim=1)
        .mean()
    )


def boundary_dice_loss(
    probs: torch.Tensor,
    target: torch.Tensor,
    kernel_size: int = 3,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    """Dice loss computed only on boundary pixels.

    Args:
        probs: Predicted probabilities (B, C, H, W)
        target: Ground truth labels (B, H, W)
        kernel_size: Size of kernel for boundary extraction
        epsilon: Small constant for numerical stability

    Returns:
        Boundary dice loss scalar
    """
    # Extract boundary via morphological gradient
    target_float = target.float().unsqueeze(1)
    padding = kernel_size // 2
    dilated = F.max_pool2d(
        target_float,
        kernel_size,
        stride=1,
        padding=padding,
    )
    eroded = -F.max_pool2d(
        -target_float,
        kernel_size,
        stride=1,
        padding=padding,
    )
    boundary = (dilated - eroded).squeeze(1)  # (B, H, W)

    # Compute Dice only on boundary pixels
    probs_pupil = probs[:, 1]  # pupil class probabilities (B, H, W)
    probs_boundary = probs_pupil * boundary
    target_boundary = target.float() * boundary

    intersection = (
        probs_boundary * target_boundary
    ).sum(dim=(1, 2))
    union = probs_boundary.sum(
        dim=(1, 2)
    ) + target_boundary.sum(dim=(1, 2))

    dice = (
        2.0 * intersection + epsilon
    ) / (union + epsilon)
    return (1.0 - dice).mean()


class CombinedLoss(nn.Module):
    """
    Combined loss for pupil segmentation:
    - Weighted Cross Entropy: Handles class imbalance
    - Dice Loss: Better for small regions like pupils
    - Focal Surface Loss: Boundary-aware optimization with focal weighting
    - Boundary Dice Loss: Explicit optimization for edge pixels
    """

    def __init__(
        self,
        epsilon: float = 1e-5,
        focal_gamma: float = 2.0,
        boundary_weight: float = 0.3,
        boundary_kernel_size: int = 3,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.focal_gamma = focal_gamma
        self.boundary_weight = boundary_weight
        self.boundary_kernel_size = boundary_kernel_size
        self.nll = nn.NLLLoss(
            reduction="none"
        )

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        spatial_weights: torch.Tensor,
        dist_map: torch.Tensor,
        alpha: float,
        eye_weight: torch.Tensor = None,
    ) -> tuple:
        """
        Args:
            logits: Model output (B, C, H, W)
            target: Ground truth (B, H, W)
            spatial_weights: Spatial weighting map (B, H, W)
            dist_map: Distance map for surface loss (B, 2, H, W)
            alpha: Balance between dice and surface loss
            eye_weight: Soft distance weighting from eye region (B, H, W)
        Returns:
            (total_loss, ce_loss, dice_loss, surface_loss, boundary_loss)
        """
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(
            logits, dim=1
        )

        # Weighted Cross Entropy
        ce_loss = self.nll(
            log_probs, target
        )
        # Apply spatial weights and optional eye weight
        weight_factor = 1.0 + spatial_weights
        if eye_weight is not None:
            weight_factor = weight_factor * eye_weight
        weighted_ce = (
            ce_loss * weight_factor
        ).mean()

        # Dice Loss
        target_onehot = (
            F.one_hot(
                target, num_classes=2
            )
            .permute(0, 3, 1, 2)
            .float()
        )
        probs_flat = probs.flatten(
            start_dim=2
        )
        target_flat = (
            target_onehot.flatten(
                start_dim=2
            )
        )

        intersection = (
            probs_flat * target_flat
        ).sum(dim=2)
        cardinality = (
            probs_flat + target_flat
        ).sum(dim=2)
        class_weights = 1.0 / (
            target_flat.sum(dim=2) ** 2
        ).clamp(min=self.epsilon)

        dice = (
            2.0
            * (
                class_weights
                * intersection
            ).sum(dim=1)
            / (
                class_weights
                * cardinality
            ).sum(dim=1)
        )
        dice_loss = (
            1.0
            - dice.clamp(
                min=self.epsilon
            )
        ).mean()

        # Focal Surface Loss (replaces standard surface loss)
        surface_loss = focal_surface_loss(
            probs,
            dist_map,
            gamma=self.focal_gamma,
        )

        # Boundary Dice Loss
        bdice_loss = boundary_dice_loss(
            probs,
            target,
            kernel_size=self.boundary_kernel_size,
            epsilon=self.epsilon,
        )

        # Total loss with updated weighting
        # Use max(1 - alpha, 0.2) for surface loss weight
        surface_weight = max(1.0 - alpha, 0.2)
        total_loss = (
            weighted_ce
            + alpha * dice_loss
            + surface_weight * surface_loss
            + self.boundary_weight * bdice_loss
        )

        return (
            total_loss,
            weighted_ce,
            dice_loss,
            surface_loss,
            bdice_loss,
        )


# =============================================================================
# Factory function for easy model creation
# =============================================================================


def create_nsa_pupil_seg(
    size: str = "small",
    in_channels: int = 1,
    num_classes: int = 2,
) -> NSAPupilSeg:
    """
    Create NSA Pupil Segmentation model with predefined configurations.

    Args:
        size: Model size ('pico', 'nano', 'tiny', 'small', 'medium')
        in_channels: Number of input channels
        num_classes: Number of output classes
    Returns:
        Configured NSAPupilSeg model
    """
    configs = {
        "pico": {
            "embed_dims": (4, 4, 4),
            "depths": (1, 1, 1),
            "num_heads": (1, 1, 1),
            "mlp_ratios": (
                1.0,
                1.0,
                1.0,
            ),
            "compress_block_sizes": (
                4,
                4,
                4,
            ),
            "compress_strides": (
                4,
                4,
                4,
            ),
            "select_block_sizes": (
                4,
                4,
                4,
            ),
            "num_selects": (1, 1, 1),
            "window_sizes": (3, 3, 3),
            "decoder_dim": 4,
        },
        "nano": {
            "embed_dims": (4, 8, 12),
            "depths": (1, 1, 1),
            "num_heads": (1, 1, 1),
            "mlp_ratios": (
                1.0,
                1.0,
                1.0,
            ),
            "compress_block_sizes": (
                4,
                4,
                4,
            ),
            "compress_strides": (
                4,
                4,
                4,
            ),
            "select_block_sizes": (
                4,
                4,
                4,
            ),
            "num_selects": (1, 1, 1),
            "window_sizes": (3, 3, 3),
            "decoder_dim": 4,
        },
        "tiny": {
            "embed_dims": (8, 12, 16),
            "depths": (1, 1, 1),
            "num_heads": (1, 1, 1),
            "mlp_ratios": (
                1.5,
                1.5,
                1.5,
            ),
            "compress_block_sizes": (
                4,
                4,
                4,
            ),
            "compress_strides": (
                4,
                4,
                4,
            ),
            "select_block_sizes": (
                4,
                4,
                4,
            ),
            "num_selects": (1, 1, 1),
            "window_sizes": (3, 3, 3),
            "decoder_dim": 8,
        },
        "small": {
            "embed_dims": (12, 24, 32),
            "depths": (1, 1, 1),
            "num_heads": (1, 1, 2),
            "mlp_ratios": (
                1.5,
                1.5,
                1.5,
            ),
            "compress_block_sizes": (
                4,
                4,
                4,
            ),
            "compress_strides": (
                4,
                4,
                4,
            ),
            "select_block_sizes": (
                4,
                4,
                4,
            ),
            "num_selects": (1, 1, 1),
            "window_sizes": (3, 3, 3),
            "decoder_dim": 12,
        },
        "medium": {
            "embed_dims": (16, 32, 48),
            "depths": (1, 1, 1),
            "num_heads": (1, 2, 2),
            "mlp_ratios": (
                1.5,
                1.5,
                1.5,
            ),
            "compress_block_sizes": (
                4,
                4,
                4,
            ),
            "compress_strides": (
                3,
                3,
                3,
            ),
            "select_block_sizes": (
                4,
                4,
                4,
            ),
            "num_selects": (2, 2, 2),
            "window_sizes": (3, 3, 3),
            "decoder_dim": 16,
        },
    }

    if size not in configs:
        raise ValueError(
            f"Unknown size: {size}. Choose from {list(configs.keys())}"
        )

    return NSAPupilSeg(
        in_channels=in_channels,
        num_classes=num_classes,
        **configs[size],
    )


# =============================================================================
# Testing / Verification
# =============================================================================


if __name__ == "__main__":
    # Test model creation and forward pass
    print(
        "Testing NSA Pupil Segmentation Model"
    )
    print("=" * 60)

    # Create models of different sizes
    for size in [
        "pico",
        "nano",
        "tiny",
        "small",
        "medium",
    ]:
        model = create_nsa_pupil_seg(
            size=size
        )

        # Count parameters
        n_params = sum(
            p.numel()
            for p in model.parameters()
        )

        # Test forward pass
        x = torch.randn(
            2, 1, 400, 640
        )  # OpenEDS image size

        model.eval()
        with torch.no_grad():
            out = model(x)

        print(
            f"\n{size.upper()} Model:"
        )
        print(
            f"  Parameters: {n_params:,}"
        )
        print(
            f"  Input shape: {x.shape}"
        )
        print(
            f"  Output shape: {out.shape}"
        )

    print("\n" + "=" * 60)
    print("All tests passed!")
