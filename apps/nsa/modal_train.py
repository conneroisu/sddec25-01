"""
Modal training script for NSA (Native Sparse Attention) Pupil Segmentation.

This script runs the NSA model training on Modal's cloud infrastructure with:
- GPU acceleration (L4)
- Dataset caching via Modal Volumes
- MLflow tracking via Databricks
- ONNX model export

Usage:
    modal run modal_train.py
"""

import modal
from os import environ

cuda_version = "12.8.1"  # should be no greater than host CUDA version
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

train_image = (
    modal.Image.from_registry(
        f"nvidia/cuda:{tag}",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install(
        "libgl1",
        "libglib2.0-0",
        "unzip",
        "wget",
        "git",
    )
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
        "matplotlib>=3.5.0",
        "huggingface_hub>=0.21.0",
        "datasets>=4.4.1",
        "pyarrow>=14.0.0",
        "mlflow[databricks]>=3.5.0",
        "kornia>=0.8.0",
        "pyyaml>=6.0.0",
    )
)

app = modal.App("NSA-PupilSeg")

dataset_volume = modal.Volume.from_name(
    "sddec25-01-dataset-cache",
    create_if_missing=True,
)
VOLUME_PATH = "/data/sddec25-01"


@app.function(
    gpu="L4",
    cpu=16.0,
    memory=32768,
    image=train_image,
    timeout=3600 * 16,
    volumes={
        VOLUME_PATH: dataset_volume
    },
    secrets=[
        modal.Secret.from_name(
            "databricks-nsa"
        ),
    ],
)
def train(
    model_size: str = "small",
    batch_size: int = 8,
    epochs: int = 30,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    num_workers: int = 8,
    seed: int = 42,
):
    """
    Train NSA Pupil Segmentation model on Modal.

    Args:
        model_size: Model size ('pico', 'nano', 'tiny', 'small', 'medium')
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Initial learning rate
        weight_decay: AdamW weight decay
        num_workers: DataLoader workers
        seed: Random seed for reproducibility
    """
    import os
    import math
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import (
        Dataset,
        DataLoader,
    )
    from tqdm import tqdm
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from datasets import (
        load_dataset,
        load_from_disk,
    )
    import mlflow
    import kornia.augmentation as K
    from kornia.augmentation import (
        AugmentationSequential,
    )

    IMAGE_HEIGHT = 400
    IMAGE_WIDTH = 640
    HF_DATASET_REPO = (
        "Conner/sddec25-01"
    )

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
        """Embed image patches into tokens for attention processing."""

        def __init__(
            self,
            in_channels: int = 1,
            embed_dim: int = 32,
            patch_size: int = 4,
        ):
            super().__init__()
            self.patch_size = patch_size
            mid_dim = embed_dim // 2

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
            x = self.conv1(x)
            x = self.conv2(x)
            return x

    class TokenCompression(nn.Module):
        """Compress spatial blocks into single tokens for coarse-grained attention."""

        def __init__(
            self,
            dim: int,
            block_size: int = 4,
            stride: int = 2,
        ):
            super().__init__()
            self.block_size = block_size
            self.stride = stride

            self.compress_k = (
                nn.Sequential(
                    nn.Linear(
                        dim
                        * block_size
                        * block_size,
                        dim * 2,
                    ),
                    nn.GELU(),
                    nn.Linear(
                        dim * 2, dim
                    ),
                )
            )
            self.compress_v = (
                nn.Sequential(
                    nn.Linear(
                        dim
                        * block_size
                        * block_size,
                        dim * 2,
                    ),
                    nn.GELU(),
                    nn.Linear(
                        dim * 2, dim
                    ),
                )
            )
            self.pos_embed = (
                nn.Parameter(
                    torch.randn(
                        1,
                        block_size
                        * block_size,
                        dim,
                    )
                    * 0.02
                )
            )

        def forward(
            self,
            k: torch.Tensor,
            v: torch.Tensor,
            spatial_size: tuple[
                int, int
            ],
        ) -> tuple[
            torch.Tensor, torch.Tensor
        ]:
            B, N, dim = k.shape
            H, W = spatial_size
            bs = self.block_size
            stride = self.stride

            n_blocks_h = (
                H - bs
            ) // stride + 1
            n_blocks_w = (
                W - bs
            ) // stride + 1

            k_2d = (
                k.reshape(B, H, W, dim)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            v_2d = (
                v.reshape(B, H, W, dim)
                .permute(0, 3, 1, 2)
                .contiguous()
            )

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

            n_blocks = k_blocks.shape[2]
            k_blocks = k_blocks.permute(
                0, 2, 1
            ).contiguous()
            v_blocks = v_blocks.permute(
                0, 2, 1
            ).contiguous()

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
            k_blocks_pos = k_blocks_reshaped.reshape(
                B,
                n_blocks,
                bs * bs * dim,
            )

            k_cmp = self.compress_k(
                k_blocks_pos
            )
            v_cmp = self.compress_v(
                v_blocks
            )

            return k_cmp, v_cmp

    class TokenSelection(nn.Module):
        """Select important token blocks based on attention scores."""

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
            spatial_size: tuple[
                int, int
            ],
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]:
            B, num_heads, N, N_cmp = (
                attn_scores_cmp.shape
            )
            H, W = spatial_size
            bs = self.block_size

            importance = (
                attn_scores_cmp.sum(
                    dim=1
                )
            )
            block_importance = (
                importance.mean(dim=1)
            )

            num_select = min(
                self.num_select, N_cmp
            )
            _, indices = torch.topk(
                block_importance,
                num_select,
                dim=-1,
            )

            n_blocks_h = (
                H - bs
            ) // bs + 1
            n_blocks_w = (
                W - bs
            ) // bs + 1

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

            k_blocks = F.unfold(
                k_2d,
                kernel_size=bs,
                stride=bs,
            )
            v_blocks = F.unfold(
                v_2d,
                kernel_size=bs,
                stride=bs,
            )

            n_blocks = k_blocks.shape[2]
            k_blocks = (
                k_blocks.permute(
                    0, 2, 1
                )
                .contiguous()
                .reshape(
                    B,
                    n_blocks,
                    bs * bs,
                    -1,
                )
            )
            v_blocks = (
                v_blocks.permute(
                    0, 2, 1
                )
                .contiguous()
                .reshape(
                    B,
                    n_blocks,
                    bs * bs,
                    -1,
                )
            )

            indices = indices.clamp(
                0, n_blocks - 1
            )

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
            )
            v_slc = torch.gather(
                v_blocks,
                1,
                indices_expanded,
            )

            k_slc = k_slc.view(
                B,
                num_select * bs * bs,
                -1,
            )
            v_slc = v_slc.view(
                B,
                num_select * bs * bs,
                -1,
            )

            return k_slc, v_slc, indices

    class SlidingWindowAttention(
        nn.Module
    ):
        """Local sliding window attention for fine-grained local context."""

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
            self.window_size = (
                window_size
            )
            self.head_dim = (
                dim // num_heads
            )
            self.scale = (
                self.head_dim**-0.5
            )

            self.qkv = nn.Linear(
                dim,
                dim * 3,
                bias=qkv_bias,
            )
            self.proj = nn.Linear(
                dim, dim
            )

            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (
                        2 * window_size
                        - 1
                    )
                    * (
                        2 * window_size
                        - 1
                    ),
                    num_heads,
                )
            )
            nn.init.trunc_normal_(
                self.relative_position_bias_table,
                std=0.02,
            )

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
            coords_flatten = (
                coords.flatten(1)
            )
            relative_coords = (
                coords_flatten[
                    :, :, None
                ]
                - coords_flatten[
                    :, None, :
                ]
            )
            relative_coords = (
                relative_coords.permute(
                    1, 2, 0
                ).contiguous()
            )
            relative_coords[
                :, :, 0
            ] += (window_size - 1)
            relative_coords[
                :, :, 1
            ] += (window_size - 1)
            relative_coords[
                :, :, 0
            ] *= (2 * window_size - 1)
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
            B, C, H, W = x.shape
            ws = self.window_size

            pad_h = (ws - H % ws) % ws
            pad_w = (ws - W % ws) % ws
            if pad_h > 0 or pad_w > 0:
                x = F.pad(
                    x,
                    (
                        0,
                        pad_w,
                        0,
                        pad_h,
                    ),
                )

            _, _, Hp, Wp = x.shape

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

            B_win = x.shape[0]
            qkv = self.qkv(x).reshape(
                B_win,
                ws * ws,
                3,
                self.num_heads,
                self.head_dim,
            )
            qkv = qkv.permute(
                2, 0, 3, 1, 4
            )
            q, k, v = (
                qkv[0],
                qkv[1],
                qkv[2],
            )

            attn = (
                q @ k.transpose(-2, -1)
            ) * self.scale

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
                .reshape(
                    B_win, ws * ws, C
                )
            )
            x = self.proj(x)

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

            if pad_h > 0 or pad_w > 0:
                x = x[:, :, :H, :W]

            return x

    class SpatialNSA(nn.Module):
        """Native Sparse Attention adapted for 2D spatial features."""

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
            self.head_dim = (
                dim // num_heads
            )
            self.scale = (
                self.head_dim**-0.5
            )

            self.qkv_cmp = nn.Linear(
                dim,
                dim * 3,
                bias=qkv_bias,
            )
            self.qkv_slc = nn.Linear(
                dim,
                dim * 3,
                bias=qkv_bias,
            )

            self.compression = TokenCompression(
                dim=dim,
                block_size=compress_block_size,
                stride=compress_stride,
            )
            self.selection = TokenSelection(
                dim=dim,
                block_size=select_block_size,
                num_select=num_select,
            )
            self.window_attn = SlidingWindowAttention(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                qkv_bias=qkv_bias,
            )

            self.proj_cmp = nn.Linear(
                dim, dim
            )
            self.proj_slc = nn.Linear(
                dim, dim
            )

            self.gate = nn.Sequential(
                nn.Linear(
                    dim, dim // 4
                ),
                nn.GELU(),
                nn.Linear(dim // 4, 3),
                nn.Sigmoid(),
            )

        def forward(
            self, x: torch.Tensor
        ) -> torch.Tensor:
            B, C, H, W = x.shape
            N = H * W

            x_seq = x.flatten(
                2
            ).transpose(1, 2)

            # Branch 1: Compressed Attention
            qkv_cmp = self.qkv_cmp(
                x_seq
            )
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
            (
                q_cmp,
                k_cmp_raw,
                v_cmp_raw,
            ) = (
                qkv_cmp[0],
                qkv_cmp[1],
                qkv_cmp[2],
            )

            k_for_cmp = (
                k_cmp_raw.transpose(
                    1, 2
                ).reshape(B, N, C)
            )
            v_for_cmp = (
                v_cmp_raw.transpose(
                    1, 2
                ).reshape(B, N, C)
            )

            k_cmp, v_cmp = (
                self.compression(
                    k_for_cmp,
                    v_for_cmp,
                    (H, W),
                )
            )
            N_cmp = k_cmp.shape[1]

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

            attn_cmp = (
                q_cmp
                @ k_cmp.transpose(
                    -2, -1
                )
            ) * self.scale
            attn_cmp_softmax = (
                attn_cmp.softmax(dim=-1)
            )
            o_cmp = (
                attn_cmp_softmax @ v_cmp
            )
            o_cmp = o_cmp.transpose(
                1, 2
            ).reshape(B, N, C)
            o_cmp = self.proj_cmp(o_cmp)

            # Branch 2: Selected Attention
            qkv_slc = self.qkv_slc(
                x_seq
            )
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
            (
                q_slc,
                k_slc_raw,
                v_slc_raw,
            ) = (
                qkv_slc[0],
                qkv_slc[1],
                qkv_slc[2],
            )

            k_for_slc = (
                k_slc_raw.transpose(
                    1, 2
                ).reshape(B, N, C)
            )
            v_for_slc = (
                v_slc_raw.transpose(
                    1, 2
                ).reshape(B, N, C)
            )

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

            attn_slc = (
                q_slc
                @ k_slc.transpose(
                    -2, -1
                )
            ) * self.scale
            attn_slc = attn_slc.softmax(
                dim=-1
            )
            o_slc = attn_slc @ v_slc
            o_slc = o_slc.transpose(
                1, 2
            ).reshape(B, N, C)
            o_slc = self.proj_slc(o_slc)

            # Branch 3: Sliding Window Attention
            o_win = self.window_attn(x)
            o_win = o_win.flatten(
                2
            ).transpose(1, 2)

            # Gated Aggregation
            gates = self.gate(x_seq)
            g_cmp = gates[:, :, 0:1]
            g_slc = gates[:, :, 1:2]
            g_win = gates[:, :, 2:3]

            out = (
                g_cmp * o_cmp
                + g_slc * o_slc
                + g_win * o_win
            )
            out = out.transpose(
                1, 2
            ).view(B, C, H, W)

            return out

    class NSABlock(nn.Module):
        """Complete NSA block with attention, normalization, and FFN."""

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

            self.norm1 = nn.BatchNorm2d(
                dim
            )
            self.dw_conv = nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                padding=1,
                groups=dim,
            )

            self.norm2 = nn.BatchNorm2d(
                dim
            )
            self.nsa = SpatialNSA(
                dim=dim,
                num_heads=num_heads,
                compress_block_size=compress_block_size,
                compress_stride=compress_stride,
                select_block_size=select_block_size,
                num_select=num_select,
                window_size=window_size,
            )

            self.norm3 = nn.LayerNorm(
                dim
            )
            hidden_dim = int(
                dim * mlp_ratio
            )
            self.ffn = nn.Sequential(
                nn.Linear(
                    dim, hidden_dim
                ),
                nn.GELU(),
                nn.Linear(
                    hidden_dim, dim
                ),
            )

        def forward(
            self, x: torch.Tensor
        ) -> torch.Tensor:
            x = x + self.dw_conv(
                self.norm1(x)
            )
            x = x + self.nsa(
                self.norm2(x)
            )

            B, C, H, W = x.shape
            x_flat = x.flatten(
                2
            ).transpose(1, 2)
            x_flat = x_flat + self.ffn(
                self.norm3(x_flat)
            )
            x = x_flat.transpose(
                1, 2
            ).view(B, C, H, W)

            return x

    class NSAStage(nn.Module):
        """Stage containing multiple NSA blocks with optional downsampling."""

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

            self.downsample = None
            if downsample:
                self.downsample = nn.Sequential(
                    ConvBNReLU(
                        in_dim,
                        out_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
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
                    for _ in range(
                        depth
                    )
                ]
            )

        def forward(
            self, x: torch.Tensor
        ) -> torch.Tensor:
            if (
                self.downsample
                is not None
            ):
                x = self.downsample(x)
            for block in self.blocks:
                x = block(x)
            return x

    class NSAEncoder(nn.Module):
        """NSA-based encoder for hierarchical feature extraction."""

        def __init__(
            self,
            in_channels: int = 1,
            embed_dims: tuple = (
                32,
                64,
                96,
            ),
            depths: tuple = (1, 1, 1),
            num_heads: tuple = (
                2,
                2,
                4,
            ),
            mlp_ratios: tuple = (
                2,
                2,
                2,
            ),
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
            num_selects: tuple = (
                4,
                4,
                4,
            ),
            window_sizes: tuple = (
                7,
                7,
                7,
            ),
        ):
            super().__init__()

            self.patch_embed = PatchEmbedding(
                in_channels=in_channels,
                embed_dim=embed_dims[0],
            )

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
                num_select=num_selects[
                    0
                ],
                window_size=window_sizes[
                    0
                ],
                downsample=False,
            )

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
                num_select=num_selects[
                    1
                ],
                window_size=window_sizes[
                    1
                ],
                downsample=True,
            )

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
                num_select=num_selects[
                    2
                ],
                window_size=window_sizes[
                    2
                ],
                downsample=True,
            )

        def forward(
            self, x: torch.Tensor
        ) -> tuple:
            x = self.patch_embed(x)
            f1 = self.stage1(x)
            f2 = self.stage2(f1)
            f3 = self.stage3(f2)
            return f1, f2, f3

    class SegmentationDecoder(
        nn.Module
    ):
        """FPN-style decoder with skip connections for precise segmentation."""

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
                nn.BatchNorm2d(
                    decoder_dim
                ),
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
                nn.BatchNorm2d(
                    decoder_dim
                ),
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
                nn.BatchNorm2d(
                    decoder_dim
                ),
                nn.GELU(),
            )

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

            out = self.head(p1)
            out = F.interpolate(
                out,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

            return out

    class NSAPupilSeg(nn.Module):
        """Native Sparse Attention model for Pupil Segmentation."""

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
            num_heads: tuple = (
                2,
                2,
                4,
            ),
            mlp_ratios: tuple = (
                2,
                2,
                2,
            ),
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
            num_selects: tuple = (
                4,
                4,
                4,
            ),
            window_sizes: tuple = (
                7,
                7,
                7,
            ),
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

            self.decoder = SegmentationDecoder(
                encoder_dims=embed_dims,
                decoder_dim=decoder_dim,
                num_classes=num_classes,
            )

            self._initialize_weights()

        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(
                    m, nn.Conv2d
                ):
                    nn.init.kaiming_normal_(
                        m.weight,
                        mode="fan_out",
                        nonlinearity="relu",
                    )
                    if (
                        m.bias
                        is not None
                    ):
                        nn.init.zeros_(
                            m.bias
                        )
                elif isinstance(
                    m, nn.BatchNorm2d
                ):
                    nn.init.ones_(
                        m.weight
                    )
                    nn.init.zeros_(
                        m.bias
                    )
                elif isinstance(
                    m, nn.Linear
                ):
                    nn.init.trunc_normal_(
                        m.weight,
                        std=0.02,
                    )
                    if (
                        m.bias
                        is not None
                    ):
                        nn.init.zeros_(
                            m.bias
                        )
                elif isinstance(
                    m, nn.LayerNorm
                ):
                    nn.init.ones_(
                        m.weight
                    )
                    nn.init.zeros_(
                        m.bias
                    )

        def forward(
            self, x: torch.Tensor
        ) -> torch.Tensor:
            target_size = (
                x.shape[2],
                x.shape[3],
            )
            f1, f2, f3 = self.encoder(x)
            out = self.decoder(
                f1, f2, f3, target_size
            )
            return out

    # Model configurations
    MODEL_CONFIGS = {
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

    def create_nsa_pupil_seg(
        size: str = "small",
        in_channels: int = 1,
        num_classes: int = 2,
    ):
        if size not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown size: {size}. Choose from {list(MODEL_CONFIGS.keys())}"
            )
        return NSAPupilSeg(
            in_channels=in_channels,
            num_classes=num_classes,
            **MODEL_CONFIGS[size],
        )

    # Loss Functions
    def focal_surface_loss(
        probs: torch.Tensor,
        dist_map: torch.Tensor,
        gamma: float = 2.0,
    ) -> torch.Tensor:
        focal_weight = (
            1 - probs
        ) ** gamma
        return (
            (
                focal_weight
                * probs
                * dist_map
            )
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
        target_float = (
            target.float().unsqueeze(1)
        )
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
        boundary = (
            dilated - eroded
        ).squeeze(1)

        probs_pupil = probs[:, 1]
        probs_boundary = (
            probs_pupil * boundary
        )
        target_boundary = (
            target.float() * boundary
        )

        intersection = (
            probs_boundary
            * target_boundary
        ).sum(dim=(1, 2))
        union = probs_boundary.sum(
            dim=(1, 2)
        ) + target_boundary.sum(
            dim=(1, 2)
        )

        dice = (
            2.0 * intersection + epsilon
        ) / (union + epsilon)
        return (1.0 - dice).mean()

    class CombinedLoss(nn.Module):
        def __init__(
            self,
            epsilon: float = 1e-5,
            focal_gamma: float = 2.0,
            boundary_weight: float = 0.3,
            boundary_kernel_size: int = 3,
        ):
            super().__init__()
            self.epsilon = epsilon
            self.focal_gamma = (
                focal_gamma
            )
            self.boundary_weight = (
                boundary_weight
            )
            self.boundary_kernel_size = (
                boundary_kernel_size
            )
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
            probs = F.softmax(
                logits, dim=1
            )
            log_probs = F.log_softmax(
                logits, dim=1
            )

            ce_loss = self.nll(
                log_probs, target
            )
            weight_factor = (
                1.0 + spatial_weights
            )
            if eye_weight is not None:
                weight_factor = (
                    weight_factor
                    * eye_weight
                )
            weighted_ce = (
                ce_loss * weight_factor
            ).mean()

            target_onehot = (
                F.one_hot(
                    target,
                    num_classes=2,
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
                target_flat.sum(dim=2)
                ** 2
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

            surface_loss = focal_surface_loss(
                probs,
                dist_map,
                gamma=self.focal_gamma,
            )
            bdice_loss = boundary_dice_loss(
                probs,
                target,
                kernel_size=self.boundary_kernel_size,
                epsilon=self.epsilon,
            )

            surface_weight = max(
                1.0 - alpha, 0.2
            )
            total_loss = (
                weighted_ce
                + alpha * dice_loss
                + surface_weight
                * surface_loss
                + self.boundary_weight
                * bdice_loss
            )

            return (
                total_loss,
                weighted_ce,
                dice_loss,
                surface_loss,
                bdice_loss,
            )

    # GPU Augmentation
    class GPUAugmentation(nn.Module):
        def __init__(
            self, training: bool = True
        ):
            super().__init__()
            self.training_mode = (
                training
            )

            if training:
                self.geometric = AugmentationSequential(
                    K.RandomHorizontalFlip(
                        p=0.5
                    ),
                    K.RandomRotation(
                        degrees=10,
                        p=0.3,
                    ),
                    K.RandomAffine(
                        degrees=0,
                        translate=(
                            0.05,
                            0.05,
                        ),
                        scale=(
                            0.95,
                            1.05,
                        ),
                        p=0.2,
                    ),
                    data_keys=[
                        "input",
                        "mask",
                    ],
                    same_on_batch=False,
                )
                self.intensity = nn.Sequential(
                    K.RandomBrightness(
                        brightness=(
                            0.9,
                            1.1,
                        ),
                        p=0.3,
                    ),
                    K.RandomContrast(
                        contrast=(
                            0.9,
                            1.1,
                        ),
                        p=0.3,
                    ),
                    K.RandomGaussianNoise(
                        mean=0.0,
                        std=0.05,
                        p=0.2,
                    ),
                    K.RandomGaussianBlur(
                        kernel_size=(
                            3,
                            3,
                        ),
                        sigma=(
                            0.1,
                            0.5,
                        ),
                        p=0.1,
                    ),
                )
            else:
                self.geometric = None
                self.intensity = None

        def forward(
            self,
            image,
            label,
            spatial_weights,
            dist_map,
            eye_mask,
            eye_weight,
        ):
            if (
                not self.training_mode
                or self.geometric
                is None
            ):
                return (
                    image,
                    label,
                    spatial_weights,
                    dist_map,
                    eye_mask,
                    eye_weight,
                )

            B, _, H, W = image.shape

            label_float = (
                label.float().unsqueeze(
                    1
                )
            )
            spatial_weights_4d = spatial_weights.unsqueeze(
                1
            )
            eye_mask_float = eye_mask.float().unsqueeze(
                1
            )
            eye_weight_4d = (
                eye_weight.unsqueeze(1)
            )

            all_masks = torch.cat(
                [
                    label_float,
                    spatial_weights_4d,
                    eye_mask_float,
                    eye_weight_4d,
                    dist_map,
                ],
                dim=1,
            )

            image_aug, masks_aug = (
                self.geometric(
                    image, all_masks
                )
            )

            label_aug = (
                masks_aug[:, 0:1, :, :]
                .squeeze(1)
                .round()
                .long()
            )
            spatial_weights_aug = (
                masks_aug[
                    :, 1:2, :, :
                ].squeeze(1)
            )
            eye_mask_aug = (
                masks_aug[:, 2:3, :, :]
                .squeeze(1)
                .round()
                .long()
            )
            eye_weight_aug = masks_aug[
                :, 3:4, :, :
            ].squeeze(1)
            dist_map_aug = masks_aug[
                :, 4:6, :, :
            ]

            image_aug = self.intensity(
                image_aug
            )

            return (
                image_aug,
                label_aug,
                spatial_weights_aug,
                dist_map_aug,
                eye_mask_aug,
                eye_weight_aug,
            )

    # Dataset
    class IrisDataset(Dataset):
        def __init__(self, hf_dataset):
            self.dataset = hf_dataset
            self.normalize_mean = 0.5
            self.normalize_std = 0.5

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            sample = self.dataset[idx]

            image = np.array(
                sample["image"],
                dtype=np.uint8,
            ).reshape(
                IMAGE_HEIGHT,
                IMAGE_WIDTH,
            )
            label = np.array(
                sample["label"],
                dtype=np.uint8,
            ).reshape(
                IMAGE_HEIGHT,
                IMAGE_WIDTH,
            )
            spatial_weights = np.array(
                sample[
                    "spatial_weights"
                ],
                dtype=np.float32,
            ).reshape(
                IMAGE_HEIGHT,
                IMAGE_WIDTH,
            )
            dist_map = np.array(
                sample["dist_map"],
                dtype=np.float32,
            ).reshape(
                2,
                IMAGE_HEIGHT,
                IMAGE_WIDTH,
            )
            eye_mask = np.array(
                sample["eye_mask"],
                dtype=np.uint8,
            ).reshape(
                IMAGE_HEIGHT,
                IMAGE_WIDTH,
            )
            eye_weight = np.array(
                sample["eye_weight"],
                dtype=np.float32,
            ).reshape(
                IMAGE_HEIGHT,
                IMAGE_WIDTH,
            )

            image = (
                image.astype(np.float32)
                / 255.0
            )
            image = (
                image
                - self.normalize_mean
            ) / self.normalize_std
            img_tensor = (
                torch.from_numpy(
                    image
                ).unsqueeze(0)
            )

            label_tensor = (
                torch.from_numpy(
                    label.astype(
                        np.int64
                    )
                )
            )
            spatial_weights_tensor = (
                torch.from_numpy(
                    spatial_weights
                )
            )
            dist_map_tensor = (
                torch.from_numpy(
                    dist_map
                )
            )
            eye_mask_tensor = (
                torch.from_numpy(
                    eye_mask.astype(
                        np.int64
                    )
                )
            )
            eye_weight_tensor = (
                torch.from_numpy(
                    eye_weight
                )
            )

            return (
                img_tensor,
                label_tensor,
                spatial_weights_tensor,
                dist_map_tensor,
                eye_mask_tensor,
                eye_weight_tensor,
            )

    # Helper Functions
    def compute_iou_tensors(
        predictions,
        targets,
        num_classes=2,
    ):
        intersection = torch.zeros(
            num_classes,
            device=predictions.device,
        )
        union = torch.zeros(
            num_classes,
            device=predictions.device,
        )
        for c in range(num_classes):
            pred_c = predictions == c
            target_c = targets == c
            intersection[c] = (
                torch.logical_and(
                    pred_c, target_c
                )
                .sum()
                .float()
            )
            union[c] = (
                torch.logical_or(
                    pred_c, target_c
                )
                .sum()
                .float()
            )
        return intersection, union

    def finalize_iou(
        total_intersection, total_union
    ):
        iou_per_class = (
            (
                total_intersection
                / total_union.clamp(
                    min=1
                )
            )
            .cpu()
            .numpy()
        )
        return (
            float(
                np.mean(iou_per_class)
            ),
            iou_per_class.tolist(),
        )

    def get_predictions(output):
        bs, _, h, w = output.size()
        _, indices = output.max(1)
        indices = indices.view(bs, h, w)
        return indices

    def get_nparams(model):
        return sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

    def save_model_checkpoint(
        model, output_path
    ):
        export_model = model
        if hasattr(model, "_orig_mod"):
            export_model = (
                model._orig_mod
            )
        export_model.eval()
        torch.save(
            export_model.state_dict(),
            output_path,
        )
        if not os.path.exists(
            output_path
        ):
            raise RuntimeError(
                f"Model save failed - file not created at {output_path}"
            )
        print(
            f"Model saved: {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.2f} MB)"
        )

    def create_training_plots(
        train_metrics,
        valid_metrics,
        save_dir="plots",
    ):
        os.makedirs(
            save_dir, exist_ok=True
        )

        # Loss curves
        fig, ax = plt.subplots(
            figsize=(10, 6)
        )
        epochs = range(
            1,
            len(train_metrics["loss"])
            + 1,
        )
        ax.plot(
            epochs,
            train_metrics["loss"],
            "b-",
            label="Train Loss",
            linewidth=2,
        )
        ax.plot(
            epochs,
            valid_metrics["loss"],
            "r-",
            label="Valid Loss",
            linewidth=2,
        )
        ax.set_xlabel(
            "Epoch", fontsize=12
        )
        ax.set_ylabel(
            "Loss", fontsize=12
        )
        ax.set_title(
            "Training and Validation Loss",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        loss_plot_path = os.path.join(
            save_dir, "loss_curves.png"
        )
        plt.savefig(
            loss_plot_path,
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        # IoU curves
        fig, ax = plt.subplots(
            figsize=(10, 6)
        )
        ax.plot(
            epochs,
            train_metrics["iou"],
            "b-",
            label="Train mIoU",
            linewidth=2,
        )
        ax.plot(
            epochs,
            valid_metrics["iou"],
            "r-",
            label="Valid mIoU",
            linewidth=2,
        )
        ax.set_xlabel(
            "Epoch", fontsize=12
        )
        ax.set_ylabel(
            "mIoU", fontsize=12
        )
        ax.set_title(
            "Training and Validation mIoU",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        iou_plot_path = os.path.join(
            save_dir, "iou_curves.png"
        )
        plt.savefig(
            iou_plot_path,
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        # Learning rate
        lr_plot_path = None
        if (
            "lr" in train_metrics
            and len(train_metrics["lr"])
            > 0
        ):
            fig, ax = plt.subplots(
                figsize=(10, 6)
            )
            ax.plot(
                epochs,
                train_metrics["lr"],
                "g-",
                linewidth=2,
            )
            ax.set_xlabel(
                "Epoch", fontsize=12
            )
            ax.set_ylabel(
                "Learning Rate",
                fontsize=12,
            )
            ax.set_title(
                "Learning Rate Schedule",
                fontsize=14,
                fontweight="bold",
            )
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            lr_plot_path = os.path.join(
                save_dir,
                "learning_rate.png",
            )
            plt.savefig(
                lr_plot_path,
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

        # Loss components
        components_plot_path = None
        if all(
            k in train_metrics
            for k in [
                "ce_loss",
                "dice_loss",
                "surface_loss",
                "boundary_loss",
            ]
        ):
            fig, axes = plt.subplots(
                2, 2, figsize=(14, 10)
            )

            axes[0, 0].plot(
                epochs,
                train_metrics[
                    "ce_loss"
                ],
                "b-",
                label="Train",
                linewidth=2,
            )
            axes[0, 0].plot(
                epochs,
                valid_metrics[
                    "ce_loss"
                ],
                "r-",
                label="Valid",
                linewidth=2,
            )
            axes[0, 0].set_title(
                "Cross Entropy Loss",
                fontweight="bold",
            )
            axes[0, 0].set_xlabel(
                "Epoch"
            )
            axes[0, 0].set_ylabel(
                "CE Loss"
            )
            axes[0, 0].legend()
            axes[0, 0].grid(
                True, alpha=0.3
            )

            axes[0, 1].plot(
                epochs,
                train_metrics[
                    "dice_loss"
                ],
                "b-",
                label="Train",
                linewidth=2,
            )
            axes[0, 1].plot(
                epochs,
                valid_metrics[
                    "dice_loss"
                ],
                "r-",
                label="Valid",
                linewidth=2,
            )
            axes[0, 1].set_title(
                "Dice Loss",
                fontweight="bold",
            )
            axes[0, 1].set_xlabel(
                "Epoch"
            )
            axes[0, 1].set_ylabel(
                "Dice Loss"
            )
            axes[0, 1].legend()
            axes[0, 1].grid(
                True, alpha=0.3
            )

            axes[1, 0].plot(
                epochs,
                train_metrics[
                    "surface_loss"
                ],
                "b-",
                label="Train",
                linewidth=2,
            )
            axes[1, 0].plot(
                epochs,
                valid_metrics[
                    "surface_loss"
                ],
                "r-",
                label="Valid",
                linewidth=2,
            )
            axes[1, 0].set_title(
                "Surface Loss",
                fontweight="bold",
            )
            axes[1, 0].set_xlabel(
                "Epoch"
            )
            axes[1, 0].set_ylabel(
                "Surface Loss"
            )
            axes[1, 0].legend()
            axes[1, 0].grid(
                True, alpha=0.3
            )

            axes[1, 1].plot(
                epochs,
                train_metrics[
                    "boundary_loss"
                ],
                "b-",
                label="Train",
                linewidth=2,
            )
            axes[1, 1].plot(
                epochs,
                valid_metrics[
                    "boundary_loss"
                ],
                "r-",
                label="Valid",
                linewidth=2,
            )
            axes[1, 1].set_title(
                "Boundary Dice Loss",
                fontweight="bold",
            )
            axes[1, 1].set_xlabel(
                "Epoch"
            )
            axes[1, 1].set_ylabel(
                "Boundary Loss"
            )
            axes[1, 1].legend()
            axes[1, 1].grid(
                True, alpha=0.3
            )

            plt.tight_layout()
            components_plot_path = os.path.join(
                save_dir,
                "loss_components.png",
            )
            plt.savefig(
                components_plot_path,
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

        return {
            "loss_curves": loss_plot_path,
            "iou_curves": iou_plot_path,
            "learning_rate": lr_plot_path,
            "loss_components": components_plot_path,
        }

    def create_prediction_visualization(
        model,
        dataloader,
        device,
        num_samples=4,
        save_path="predictions.png",
    ):
        model.eval()
        samples_collected = 0
        images_to_plot = []
        labels_to_plot = []
        preds_to_plot = []

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        with torch.no_grad():
            for (
                img,
                labels,
                _,
                _,
                _,
                _,
            ) in dataloader:
                if (
                    samples_collected
                    >= num_samples
                ):
                    break
                single_img = img[
                    0:1
                ].to(device)
                single_target = (
                    labels[0:1]
                    .to(device)
                    .long()
                )
                output = model(
                    single_img
                )
                predictions = (
                    get_predictions(
                        output
                    )
                )
                images_to_plot.append(
                    img[0]
                    .cpu()
                    .squeeze()
                    .numpy()
                )
                labels_to_plot.append(
                    single_target[0]
                    .cpu()
                    .numpy()
                )
                preds_to_plot.append(
                    predictions[0]
                    .cpu()
                    .numpy()
                )
                del (
                    single_img,
                    single_target,
                    output,
                    predictions,
                )
                if (
                    torch.cuda.is_available()
                ):
                    torch.cuda.empty_cache()
                samples_collected += 1

        fig, axes = plt.subplots(
            num_samples,
            3,
            figsize=(
                12,
                4 * num_samples,
            ),
        )
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            axes[i, 0].imshow(
                images_to_plot[i],
                cmap="gray",
            )
            axes[i, 0].set_title(
                "Input Image",
                fontweight="bold",
            )
            axes[i, 0].axis("off")
            axes[i, 1].imshow(
                labels_to_plot[i],
                cmap="jet",
                vmin=0,
                vmax=1,
            )
            axes[i, 1].set_title(
                "Ground Truth",
                fontweight="bold",
            )
            axes[i, 1].axis("off")
            axes[i, 2].imshow(
                preds_to_plot[i],
                cmap="jet",
                vmin=0,
                vmax=1,
            )
            axes[i, 2].set_title(
                "Prediction",
                fontweight="bold",
            )
            axes[i, 2].axis("off")

        plt.tight_layout()
        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
        return save_path

    # Main Training Loop
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)

    DATASET_CACHE_PATH = (
        f"{VOLUME_PATH}/dataset"
    )
    CACHE_MARKER_FILE = (
        f"{VOLUME_PATH}/.cache_complete"
    )
    cache_exists = os.path.exists(
        CACHE_MARKER_FILE
    )

    if cache_exists:
        print(
            f"Found cached dataset at: {DATASET_CACHE_PATH}"
        )
        print(
            "Loading from volume cache (fast)..."
        )
        try:
            hf_dataset = load_from_disk(
                DATASET_CACHE_PATH
            )
            print("Loaded from cache!")
        except Exception as e:
            print(
                f"Cache corrupted, re-downloading: {e}"
            )
            import shutil

            shutil.rmtree(
                DATASET_CACHE_PATH,
                ignore_errors=True,
            )
            if os.path.exists(
                CACHE_MARKER_FILE
            ):
                os.remove(
                    CACHE_MARKER_FILE
                )
            cache_exists = False

    if not cache_exists:
        print(
            f"Downloading from HuggingFace: {HF_DATASET_REPO}"
        )
        print(
            "First run takes ~20 min, subsequent runs will be fast."
        )
        hf_dataset = load_dataset(
            HF_DATASET_REPO
        )
        os.makedirs(
            VOLUME_PATH, exist_ok=True
        )
        hf_dataset.save_to_disk(
            DATASET_CACHE_PATH
        )
        with open(
            CACHE_MARKER_FILE, "w"
        ) as f:
            f.write(
                f"Cached from {HF_DATASET_REPO}\n"
            )
        dataset_volume.commit()
        print(
            "Dataset cached to volume!"
        )

    print(
        f"Train samples: {len(hf_dataset['train'])}"
    )
    print(
        f"Validation samples: {len(hf_dataset['validation'])}"
    )

    # Initialize model
    print("\n" + "=" * 80)
    print(
        f"Initializing NSAPupilSeg model (size={model_size})..."
    )
    print("=" * 80)

    model = create_nsa_pupil_seg(
        size=model_size,
        in_channels=1,
        num_classes=2,
    ).to(device)
    nparams = get_nparams(model)
    print(
        f"Model parameters: {nparams:,}"
    )

    # Initialize augmentation
    train_augment = GPUAugmentation(
        training=True
    ).to(device)
    val_augment = GPUAugmentation(
        training=False
    ).to(device)

    # Setup training
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=learning_rate * 0.01,
    )

    use_amp = torch.cuda.is_available()
    scaler = (
        torch.amp.GradScaler("cuda")
        if use_amp
        else None
    )

    if use_amp:
        print(
            "Mixed precision training (AMP) enabled"
        )

    # Create datasets
    train_dataset = IrisDataset(
        hf_dataset["train"]
    )
    valid_dataset = IrisDataset(
        hf_dataset["validation"]
    )

    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers
        > 0,
        prefetch_factor=(
            2
            if num_workers > 0
            else None
        ),
        drop_last=True,
    )
    validloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers
        > 0,
        prefetch_factor=(
            2
            if num_workers > 0
            else None
        ),
    )

    # Alpha schedule
    alpha_schedule = [
        max(
            1.0 - i / min(125, epochs),
            0.2,
        )
        for i in range(epochs)
    ]

    # Training metrics
    train_metrics = {
        "loss": [],
        "iou": [],
        "ce_loss": [],
        "dice_loss": [],
        "surface_loss": [],
        "boundary_loss": [],
        "alpha": [],
        "lr": [],
    }
    valid_metrics = {
        "loss": [],
        "iou": [],
        "ce_loss": [],
        "dice_loss": [],
        "surface_loss": [],
        "boundary_loss": [],
    }

    best_valid_iou = 0.0
    best_epoch = 0

    mlflow.set_tracking_uri(
        environ["MLFLOW_TRACKING_URI"]
    )
    mlflow.set_experiment(
        experiment_id=environ["MLFLOW_EXPERIMENT_ID"]
    )
    print(
        f"\nMLflow configured with experiment ID: {environ['MLFLOW_EXPERIMENT_ID']}"
    )

    print("\n" + "=" * 80)
    print("Starting NSA Training")
    print("=" * 80)
    print(f"  Model Size: {model_size}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Weight Decay: {weight_decay}")
    print(f"  Parameters: {nparams:,}")
    print(f"  AMP: {use_amp}")
    print("=" * 80)

    with mlflow.start_run(
        run_name=f"nsa-{model_size}-training"
    ) as run:
        mlflow.set_tags(
            {
                "model_type": "NSAPupilSeg",
                "model_size": model_size,
                "task": "semantic_segmentation",
                "dataset": "SDDEC25-01",
                "framework": "PyTorch",
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingLR",
            }
        )
        mlflow.log_params(
            {
                "model_size": model_size,
                "batch_size": batch_size,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "seed": seed,
                "num_workers": num_workers,
                "model_params": nparams,
                "use_amp": use_amp,
                "train_samples": len(
                    train_dataset
                ),
                "valid_samples": len(
                    valid_dataset
                ),
                "augmentation": "kornia_gpu",
            }
        )

        print(
            f"MLflow run started: {run.info.run_id}"
        )

        for epoch in range(epochs):
            alpha = alpha_schedule[
                epoch
            ]

            # Training phase
            model.train()
            train_augment.train()
            train_loss_sum = (
                torch.tensor(
                    0.0, device=device
                )
            )
            train_ce_sum = torch.tensor(
                0.0, device=device
            )
            train_dice_sum = (
                torch.tensor(
                    0.0, device=device
                )
            )
            train_surface_sum = (
                torch.tensor(
                    0.0, device=device
                )
            )
            train_boundary_sum = (
                torch.tensor(
                    0.0, device=device
                )
            )
            train_intersection = (
                torch.zeros(
                    2, device=device
                )
            )
            train_union = torch.zeros(
                2, device=device
            )

            pbar = tqdm(
                trainloader,
                desc=f"Epoch {epoch+1}/{epochs} [Train]",
            )
            for (
                images,
                labels,
                spatial_weights,
                dist_maps,
                eye_masks,
                eye_weights,
            ) in pbar:
                images = images.to(
                    device,
                    non_blocking=True,
                )
                labels = labels.to(
                    device,
                    non_blocking=True,
                )
                spatial_weights = spatial_weights.to(
                    device,
                    non_blocking=True,
                )
                dist_maps = dist_maps.to(
                    device,
                    non_blocking=True,
                )
                eye_masks = eye_masks.to(
                    device,
                    non_blocking=True,
                )
                eye_weights = eye_weights.to(
                    device,
                    non_blocking=True,
                )

                # Apply augmentation
                (
                    images,
                    labels,
                    spatial_weights,
                    dist_maps,
                    eye_masks,
                    eye_weights,
                ) = train_augment(
                    images,
                    labels,
                    spatial_weights,
                    dist_maps,
                    eye_masks,
                    eye_weights,
                )

                optimizer.zero_grad()
                with torch.amp.autocast(
                    "cuda",
                    enabled=use_amp,
                ):
                    outputs = model(
                        images
                    )
                    (
                        loss,
                        ce_loss,
                        dice_loss,
                        surface_loss,
                        boundary_loss,
                    ) = criterion(
                        outputs,
                        labels,
                        spatial_weights,
                        dist_maps,
                        alpha,
                        eye_weights,
                    )

                scaler.scale(
                    loss
                ).backward()
                scaler.unscale_(
                    optimizer
                )
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )
                scaler.step(optimizer)
                scaler.update()

                train_loss_sum += (
                    loss.detach()
                )
                train_ce_sum += (
                    ce_loss.detach()
                )
                train_dice_sum += (
                    dice_loss.detach()
                )
                train_surface_sum += (
                    surface_loss.detach()
                )
                train_boundary_sum += (
                    boundary_loss.detach()
                )

                preds = get_predictions(
                    outputs
                )
                inter, uni = (
                    compute_iou_tensors(
                        preds, labels
                    )
                )
                train_intersection += (
                    inter
                )
                train_union += uni

                pbar.set_postfix(
                    {
                        "alpha": f"{alpha:.3f}"
                    }
                )

            n_train_batches = len(
                trainloader
            )

            # Validation phase
            model.eval()
            val_augment.eval()
            valid_loss_sum = (
                torch.tensor(
                    0.0, device=device
                )
            )
            valid_ce_sum = torch.tensor(
                0.0, device=device
            )
            valid_dice_sum = (
                torch.tensor(
                    0.0, device=device
                )
            )
            valid_surface_sum = (
                torch.tensor(
                    0.0, device=device
                )
            )
            valid_boundary_sum = (
                torch.tensor(
                    0.0, device=device
                )
            )
            valid_intersection = (
                torch.zeros(
                    2, device=device
                )
            )
            valid_union = torch.zeros(
                2, device=device
            )

            with torch.no_grad():
                pbar = tqdm(
                    validloader,
                    desc=f"Epoch {epoch+1}/{epochs} [Valid]",
                )
                for (
                    images,
                    labels,
                    spatial_weights,
                    dist_maps,
                    eye_masks,
                    eye_weights,
                ) in pbar:
                    images = images.to(
                        device,
                        non_blocking=True,
                    )
                    labels = labels.to(
                        device,
                        non_blocking=True,
                    )
                    spatial_weights = spatial_weights.to(
                        device,
                        non_blocking=True,
                    )
                    dist_maps = dist_maps.to(
                        device,
                        non_blocking=True,
                    )
                    eye_masks = eye_masks.to(
                        device,
                        non_blocking=True,
                    )
                    eye_weights = eye_weights.to(
                        device,
                        non_blocking=True,
                    )

                    (
                        images,
                        labels,
                        spatial_weights,
                        dist_maps,
                        eye_masks,
                        eye_weights,
                    ) = val_augment(
                        images,
                        labels,
                        spatial_weights,
                        dist_maps,
                        eye_masks,
                        eye_weights,
                    )

                    with torch.amp.autocast(
                        "cuda"
                    ):
                        outputs = model(
                            images
                        )
                        (
                            loss,
                            ce_loss,
                            dice_loss,
                            surface_loss,
                            boundary_loss,
                        ) = criterion(
                            outputs,
                            labels,
                            spatial_weights,
                            dist_maps,
                            alpha,
                            eye_weights,
                        )

                    valid_loss_sum += (
                        loss.detach()
                    )
                    valid_ce_sum += (
                        ce_loss.detach()
                    )
                    valid_dice_sum += (
                        dice_loss.detach()
                    )
                    valid_surface_sum += (
                        surface_loss.detach()
                    )
                    valid_boundary_sum += (
                        boundary_loss.detach()
                    )

                    preds = (
                        get_predictions(
                            outputs
                        )
                    )
                    inter, uni = (
                        compute_iou_tensors(
                            preds,
                            labels,
                        )
                    )
                    valid_intersection += (
                        inter
                    )
                    valid_union += uni

            n_valid_batches = len(
                validloader
            )

            # Compute metrics
            train_loss_val = (
                train_loss_sum
                / n_train_batches
            ).item()
            train_ce_val = (
                train_ce_sum
                / n_train_batches
            ).item()
            train_dice_val = (
                train_dice_sum
                / n_train_batches
            ).item()
            train_surface_val = (
                train_surface_sum
                / n_train_batches
            ).item()
            train_boundary_val = (
                train_boundary_sum
                / n_train_batches
            ).item()
            train_iou, _ = finalize_iou(
                train_intersection,
                train_union,
            )

            valid_loss_val = (
                valid_loss_sum
                / n_valid_batches
            ).item()
            valid_ce_val = (
                valid_ce_sum
                / n_valid_batches
            ).item()
            valid_dice_val = (
                valid_dice_sum
                / n_valid_batches
            ).item()
            valid_surface_val = (
                valid_surface_sum
                / n_valid_batches
            ).item()
            valid_boundary_val = (
                valid_boundary_sum
                / n_valid_batches
            ).item()
            valid_iou, _ = finalize_iou(
                valid_intersection,
                valid_union,
            )

            scheduler.step()
            current_lr = (
                optimizer.param_groups[
                    0
                ]["lr"]
            )

            # Store metrics
            train_metrics[
                "loss"
            ].append(train_loss_val)
            train_metrics["iou"].append(
                train_iou
            )
            train_metrics[
                "ce_loss"
            ].append(train_ce_val)
            train_metrics[
                "dice_loss"
            ].append(train_dice_val)
            train_metrics[
                "surface_loss"
            ].append(train_surface_val)
            train_metrics[
                "boundary_loss"
            ].append(train_boundary_val)
            train_metrics[
                "alpha"
            ].append(alpha)
            train_metrics["lr"].append(
                current_lr
            )

            valid_metrics[
                "loss"
            ].append(valid_loss_val)
            valid_metrics["iou"].append(
                valid_iou
            )
            valid_metrics[
                "ce_loss"
            ].append(valid_ce_val)
            valid_metrics[
                "dice_loss"
            ].append(valid_dice_val)
            valid_metrics[
                "surface_loss"
            ].append(valid_surface_val)
            valid_metrics[
                "boundary_loss"
            ].append(valid_boundary_val)

            # Log to MLflow
            mlflow.log_metrics(
                {
                    "train_loss": train_loss_val,
                    "train_iou": train_iou,
                    "train_ce_loss": train_ce_val,
                    "train_dice_loss": train_dice_val,
                    "train_surface_loss": train_surface_val,
                    "train_boundary_loss": train_boundary_val,
                    "valid_loss": valid_loss_val,
                    "valid_iou": valid_iou,
                    "valid_ce_loss": valid_ce_val,
                    "valid_dice_loss": valid_dice_val,
                    "valid_surface_loss": valid_surface_val,
                    "valid_boundary_loss": valid_boundary_val,
                    "learning_rate": current_lr,
                    "alpha": alpha,
                },
                step=epoch,
            )

            # Print progress
            print(
                f"\nEpoch {epoch+1}/{epochs}"
            )
            print(
                f"  Train Loss: {train_loss_val:.4f} | Valid Loss: {valid_loss_val:.4f}"
            )
            print(
                f"  Train IoU:  {train_iou:.4f} | Valid IoU:  {valid_iou:.4f}"
            )
            print(
                f"  CE: {train_ce_val:.4f}/{valid_ce_val:.4f} | Dice: {train_dice_val:.4f}/{valid_dice_val:.4f}"
            )
            print(
                f"  Surface: {train_surface_val:.4f}/{valid_surface_val:.4f} | Boundary: {train_boundary_val:.4f}/{valid_boundary_val:.4f}"
            )
            print(
                f"  LR: {current_lr:.6f} | Alpha: {alpha:.4f}"
            )

            # Save best model
            if (
                valid_iou
                > best_valid_iou
            ):
                best_valid_iou = (
                    valid_iou
                )
                best_epoch = epoch + 1
                best_model_path = f"best_nsa_{model_size}_model.pth"
                save_model_checkpoint(
                    model,
                    best_model_path,
                )
                mlflow.log_artifact(
                    best_model_path
                )
                mlflow.log_metric(
                    "best_valid_iou",
                    best_valid_iou,
                    step=epoch,
                )
                print(
                    f"  >> New best model! Valid mIoU: {best_valid_iou:.4f}"
                )

            # Periodic visualizations
            if (
                epoch + 1
            ) % 5 == 0 or epoch == 0:
                print(
                    "Generating sample predictions..."
                )
                pred_vis_path = f"predictions_epoch_{epoch+1}.png"
                create_prediction_visualization(
                    model,
                    validloader,
                    device,
                    num_samples=4,
                    save_path=pred_vis_path,
                )
                mlflow.log_artifact(
                    pred_vis_path
                )
                print(
                    f"Sample predictions logged: {pred_vis_path}"
                )

            if (
                (epoch + 1) % 10 == 0
                or epoch == epochs - 1
            ):
                print(
                    "Generating training curves..."
                )
                plot_paths = create_training_plots(
                    train_metrics,
                    valid_metrics,
                    save_dir="plots",
                )
                for (
                    plot_path
                ) in (
                    plot_paths.values()
                ):
                    if (
                        plot_path
                        is not None
                    ):
                        mlflow.log_artifact(
                            plot_path
                        )
                print(
                    "Training curves logged to MLflow"
                )

            # Periodic checkpoints
            if (
                (epoch + 1) % 10 == 0
                or epoch == epochs - 1
            ):
                checkpoint_path = f"nsa_{model_size}_epoch_{epoch+1}.pth"
                save_model_checkpoint(
                    model,
                    checkpoint_path,
                )
                mlflow.log_artifact(
                    checkpoint_path
                )
                print(
                    f"Checkpoint saved: {checkpoint_path}"
                )

        # Final metrics
        mlflow.log_metrics(
            {
                "final_train_loss": train_loss_val,
                "final_train_iou": train_iou,
                "final_valid_loss": valid_loss_val,
                "final_valid_iou": valid_iou,
                "best_valid_iou": best_valid_iou,
                "best_epoch": best_epoch,
            }
        )

        print("\n" + "=" * 80)
        print("NSA Training Complete!")
        print(
            f"Final validation mIoU: {valid_iou:.4f}"
        )
        print(
            f"Best validation mIoU: {best_valid_iou:.4f} (epoch {best_epoch})"
        )
        print(
            f"Final train mIoU: {train_iou:.4f}"
        )
        print(
            f"MLflow run ID: {run.info.run_id}"
        )
        print("=" * 80)

    return {
        "best_valid_iou": best_valid_iou,
        "best_epoch": best_epoch,
        "final_valid_iou": valid_iou,
        "model_params": nparams,
    }


@app.local_entrypoint()
def main(
    model_size: str = "small",
    batch_size: int = 8,
    epochs: int = 30,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    num_workers: int = 8,
    seed: int = 42,
):
    """
    Local entrypoint for Modal training.

    Usage:
        modal run modal_train.py
        modal run modal_train.py --model-size tiny --epochs 10
        modal run modal_train.py --model-size medium --batch-size 4 --epochs 50
    """
    print(
        f"Starting NSA training on Modal..."
    )
    print(f"  Model size: {model_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(
        f"  Learning rate: {learning_rate}"
    )

    result = train.remote(
        model_size=model_size,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_workers=num_workers,
        seed=seed,
    )

    print("\nTraining complete!")
    print(f"Results: {result}")
    return result
