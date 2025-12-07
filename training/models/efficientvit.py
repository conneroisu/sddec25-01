"""
TinyEfficientViT model for semantic segmentation.

This module contains only the model architecture with minimal dependencies (torch only).
For training code, see train_local.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyConvNorm(nn.Module):
    """Convolution + BatchNorm layer (parameter-efficient)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=1,
        bias=False,
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
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class TinyPatchEmbedding(nn.Module):
    """
    Lightweight patch embedding with 2 conv layers and stride 4.
    Reduces spatial resolution by 4x while embedding to initial dim.
    """

    def __init__(self, in_channels=1, embed_dim=8):
        super().__init__()
        mid_dim = embed_dim // 2 if embed_dim >= 4 else 2
        self.conv1 = TinyConvNorm(
            in_channels,
            mid_dim,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.act1 = nn.GELU()
        self.conv2 = TinyConvNorm(
            mid_dim,
            embed_dim,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.act2 = nn.GELU()

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        return x


class TinyCascadedGroupAttention(nn.Module):
    """
    Tiny version of Cascaded Group Attention.
    Uses minimal heads and key dimensions for efficiency.
    """

    def __init__(
        self,
        dim,
        num_heads=1,
        key_dim=4,
        attn_ratio=2,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.scale = key_dim**-0.5
        self.d = int(attn_ratio * key_dim)
        self.attn_ratio = attn_ratio

        qkv_dim = (num_heads * key_dim * 2) + (num_heads * self.d)
        self.qkv = nn.Linear(dim, qkv_dim)
        self.proj = nn.Linear(num_heads * self.d, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)

        q_total = self.num_heads * self.key_dim
        k_total = self.num_heads * self.key_dim
        v_total = self.num_heads * self.d

        q = (
            qkv[:, :, :q_total]
            .reshape(B, N, self.num_heads, self.key_dim)
            .permute(0, 2, 1, 3)
        )
        k = (
            qkv[:, :, q_total : q_total + k_total]
            .reshape(B, N, self.num_heads, self.key_dim)
            .permute(0, 2, 1, 3)
        )
        v = (
            qkv[:, :, q_total + k_total :]
            .reshape(B, N, self.num_heads, self.d)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.num_heads * self.d)
        x = self.proj(x)
        return x


class TinyLocalWindowAttention(nn.Module):
    """
    Local window attention wrapper.
    Partitions input into windows and applies attention within each window.
    """

    def __init__(
        self,
        dim,
        num_heads=1,
        key_dim=4,
        attn_ratio=2,
        window_size=7,
    ):
        super().__init__()
        self.window_size = window_size
        self.attn = TinyCascadedGroupAttention(
            dim=dim,
            num_heads=num_heads,
            key_dim=key_dim,
            attn_ratio=attn_ratio,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.window_size

        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, Hp, Wp = x.shape

        x = x.view(B, C, Hp // ws, ws, Wp // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.view(B * (Hp // ws) * (Wp // ws), ws * ws, C)

        x = self.attn(x)

        x = x.view(B, Hp // ws, Wp // ws, ws, ws, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, C, Hp, Wp)

        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]

        return x


class TinyMLP(nn.Module):
    """Tiny MLP with expansion ratio."""

    def __init__(self, dim, expansion_ratio=2):
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TinyEfficientVitBlock(nn.Module):
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
                TinyEfficientVitBlock(
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


class TinySegmentationDecoder(nn.Module):
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
        self.decoder = TinySegmentationDecoder(
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
