#!/usr/bin/env python3
"""
Local training script for TinyEfficientViT semantic segmentation.

This script provides the same functionality as train_efficientvit.py but runs
locally without Modal cloud infrastructure. It supports:
- HuggingFace dataset caching (uses HF_DATASETS_CACHE env var or ~/.cache/huggingface)
- Environment variable configuration for MLflow
- Device detection (CUDA/MPS/CPU) with user override
- Configurable hyperparameters via command-line arguments

Usage:
    python train_efficientvit_local.py --epochs 10 --batch-size 32
    python train_efficientvit_local.py --device cuda --output-dir ./checkpoints
    python train_efficientvit_local.py --resume                 # Auto-detect best model
    python train_efficientvit_local.py --resume path/to/model.pt  # Specific checkpoint
    python train_efficientvit_local.py --help
"""

import argparse
import os
import math
import random
import platform

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset

import mlflow


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
    Combines encoder and decoder with <60k parameters.
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


class CombinedLoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(CombinedLoss, self).__init__()
        self.epsilon = epsilon
        self.nll = nn.NLLLoss(reduction="none")

    def forward(
        self,
        logits,
        target,
        spatial_weights,
        dist_map,
        alpha,
    ):
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        ce_loss = self.nll(log_probs, target)
        weighted_ce = (ce_loss * (1.0 + spatial_weights)).mean()
        target_onehot = F.one_hot(target, num_classes=2).permute(0, 3, 1, 2).float()
        probs_flat = probs.flatten(start_dim=2)
        target_flat = target_onehot.flatten(start_dim=2)
        intersection = (probs_flat * target_flat).sum(dim=2)
        cardinality = (probs_flat + target_flat).sum(dim=2)
        class_weights = 1.0 / (target_flat.sum(dim=2) ** 2).clamp(min=self.epsilon)
        dice = (
            2.0
            * (class_weights * intersection).sum(dim=1)
            / (class_weights * cardinality).sum(dim=1)
        )
        dice_loss = (1.0 - dice.clamp(min=self.epsilon)).mean()
        surface_loss = (
            (probs.flatten(start_dim=2) * dist_map.flatten(start_dim=2))
            .mean(dim=2)
            .mean(dim=1)
            .mean()
        )
        total_loss = weighted_ce + alpha * dice_loss + (1.0 - alpha) * surface_loss
        return (
            total_loss,
            weighted_ce,
            dice_loss,
            surface_loss,
        )


def compute_iou_tensors(predictions, targets, num_classes=2):
    """Compute IoU tensors fully on GPU without Python loops."""
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()

    classes = torch.arange(num_classes, device=predictions.device).view(-1, 1)
    pred_mask = pred_flat.unsqueeze(0) == classes  # [num_classes, N]
    target_mask = target_flat.unsqueeze(0) == classes  # [num_classes, N]

    intersection = (pred_mask & target_mask).sum(dim=1).float()
    union = (pred_mask | target_mask).sum(dim=1).float()

    return intersection, union


def get_predictions(output):
    bs, _, h, w = output.size()
    _, indices = output.max(1)
    indices = indices.view(bs, h, w)
    return indices


def get_nparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def total_metric(nparams, miou):
    S = nparams * 4.0 / (1024 * 1024)
    total = min(1, 1.0 / S) + miou
    return total * 0.5


def create_training_plots(train_metrics, valid_metrics, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_metrics["loss"]) + 1)
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
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(
        "Training and Validation Loss",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    loss_plot_path = os.path.join(save_dir, "loss_curves.png")
    plt.savefig(loss_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
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
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("mIoU", fontsize=12)
    ax.set_title(
        "Training and Validation mIoU",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    iou_plot_path = os.path.join(save_dir, "iou_curves.png")
    plt.savefig(iou_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    lr_plot_path = None
    if "lr" in train_metrics and len(train_metrics["lr"]) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, train_metrics["lr"], "g-", linewidth=2)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Learning Rate", fontsize=12)
        ax.set_title(
            "Learning Rate Schedule",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        lr_plot_path = os.path.join(save_dir, "learning_rate.png")
        plt.savefig(lr_plot_path, dpi=150, bbox_inches="tight")
        plt.close()

    components_plot_path = None
    if all(k in train_metrics for k in ["ce_loss", "dice_loss", "surface_loss"]):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes[0, 0].plot(
            epochs,
            train_metrics["ce_loss"],
            "b-",
            label="Train",
            linewidth=2,
        )
        axes[0, 0].plot(
            epochs,
            valid_metrics["ce_loss"],
            "r-",
            label="Valid",
            linewidth=2,
        )
        axes[0, 0].set_title("Cross Entropy Loss", fontweight="bold")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("CE Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(
            epochs,
            train_metrics["dice_loss"],
            "b-",
            label="Train",
            linewidth=2,
        )
        axes[0, 1].plot(
            epochs,
            valid_metrics["dice_loss"],
            "r-",
            label="Valid",
            linewidth=2,
        )
        axes[0, 1].set_title("Dice Loss", fontweight="bold")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Dice Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(
            epochs,
            train_metrics["surface_loss"],
            "b-",
            label="Train",
            linewidth=2,
        )
        axes[1, 0].plot(
            epochs,
            valid_metrics["surface_loss"],
            "r-",
            label="Valid",
            linewidth=2,
        )
        axes[1, 0].set_title("Surface Loss", fontweight="bold")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Surface Loss")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(
            epochs,
            train_metrics["alpha"],
            "purple",
            linewidth=2,
        )
        axes[1, 1].set_title(
            "Alpha Weight (Dice vs Surface)",
            fontweight="bold",
        )
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Alpha")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        components_plot_path = os.path.join(
            save_dir,
            "loss_components.png",
        )
        plt.savefig(components_plot_path, dpi=150, bbox_inches="tight")
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
        for img, labels, _, _, _ in dataloader:
            if samples_collected >= num_samples:
                break
            single_img = img[0:1].to(device, memory_format=torch.channels_last)
            single_target = labels[0:1].to(device).long()
            output = model(single_img)
            predictions = get_predictions(output)
            images_to_plot.append(img[0].cpu().squeeze().numpy())
            labels_to_plot.append(single_target[0].cpu().numpy())
            preds_to_plot.append(predictions[0].cpu().numpy())
            del single_img, single_target, output, predictions
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            samples_collected += 1

    fig, axes = plt.subplots(
        num_samples,
        3,
        figsize=(12, 4 * num_samples),
    )
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    for i in range(num_samples):
        axes[i, 0].imshow(images_to_plot[i], cmap="gray")
        axes[i, 0].set_title("Input Image", fontweight="bold")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(labels_to_plot[i], cmap="jet", vmin=0, vmax=1)
        axes[i, 1].set_title("Ground Truth", fontweight="bold")
        axes[i, 1].axis("off")
        axes[i, 2].imshow(preds_to_plot[i], cmap="jet", vmin=0, vmax=1)
        axes[i, 2].set_title("Prediction", fontweight="bold")
        axes[i, 2].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def save_model_checkpoint(model, output_path):
    """Save model checkpoint as PyTorch .pt file."""
    save_model = model
    if hasattr(model, "_orig_mod"):
        save_model = model._orig_mod

    save_model = save_model.to(memory_format=torch.contiguous_format)
    save_model.eval()

    torch.save(save_model.state_dict(), output_path)

    if not os.path.exists(output_path):
        raise RuntimeError(f"Model save failed - file not created at {output_path}")
    print(
        f"Model saved: {output_path} "
        f"({os.path.getsize(output_path) / 1024 / 1024:.2f} MB)"
    )


def load_model_checkpoint(model, checkpoint_path, device):
    """Load model checkpoint from PyTorch .pt file.

    Args:
        model: The model to load weights into
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model onto

    Returns:
        The model with loaded weights
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print(
        f"Checkpoint loaded successfully "
        f"({os.path.getsize(checkpoint_path) / 1024 / 1024:.2f} MB)"
    )
    return model


def resolve_resume_path(resume_arg, output_dir):
    """Resolve the checkpoint path for resuming training.

    Args:
        resume_arg: The --resume argument value ("auto", a path, or None)
        output_dir: The output directory for checkpoints

    Returns:
        Path to checkpoint file, or None if no checkpoint should be loaded
    """
    if resume_arg is None:
        return None

    if resume_arg == "auto":
        # Auto-detect best model in output directory
        default_path = os.path.join(output_dir, "best_efficientvit_model.pt")
        if os.path.exists(default_path):
            print(f"Auto-detected checkpoint: {default_path}")
            return default_path
        else:
            print(f"No checkpoint found at {default_path}, starting fresh")
            return None
    else:
        # User provided explicit path
        if os.path.exists(resume_arg):
            return resume_arg
        else:
            raise FileNotFoundError(f"Specified checkpoint not found: {resume_arg}")


class RandomHorizontalFlip(object):
    def __call__(self, img, label):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), label.transpose(
                Image.FLIP_LEFT_RIGHT
            )
        return img, label


class Gaussian_blur(object):
    def __call__(self, img):
        sigma_value = np.random.randint(2, 7)
        return Image.fromarray(cv2.GaussianBlur(img, (7, 7), sigma_value))


class Line_augment(object):
    def __call__(self, base):
        yc, xc = (0.3 + 0.4 * np.random.rand()) * np.array(base.shape)
        aug_base = np.copy(base)
        num_lines = np.random.randint(1, 10)
        for _ in np.arange(0, num_lines):
            theta = np.pi * np.random.rand()
            x1 = xc - 50 * np.random.rand() * (1 if np.random.rand() < 0.5 else -1)
            y1 = (x1 - xc) * np.tan(theta) + yc
            x2 = xc - (150 * np.random.rand() + 50) * (
                1 if np.random.rand() < 0.5 else -1
            )
            y2 = (x2 - xc) * np.tan(theta) + yc
            aug_base = cv2.line(
                aug_base,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (255, 255, 255),
                4,
            )
        aug_base = aug_base.astype(np.uint8)
        return Image.fromarray(aug_base)


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int64)).long()


HF_DATASET_REPO = "Conner/openeds-precomputed"
IMAGE_HEIGHT = 400
IMAGE_WIDTH = 640


class IrisDataset(Dataset):
    def __init__(self, hf_dataset, split="train", transform=None):
        self.transform = transform
        self.split = split
        self.clahe = cv2.createCLAHE(
            clipLimit=1.5,
            tileGridSize=(8, 8),
        )
        self.gamma_table = 255.0 * (np.linspace(0, 1, 256) ** 0.8)
        self.dataset = hf_dataset
        # Detect if dataset has preprocessed column for conditional preprocessing
        self.has_preprocessed_column = "preprocessed" in hf_dataset.column_names

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = np.array(sample["image"], dtype=np.uint8).reshape(
            IMAGE_HEIGHT, IMAGE_WIDTH
        )
        label = np.array(sample["label"], dtype=np.uint8).reshape(
            IMAGE_HEIGHT, IMAGE_WIDTH
        )
        spatial_weights = np.array(sample["spatial_weights"], dtype=np.float32).reshape(
            IMAGE_HEIGHT, IMAGE_WIDTH
        )
        dist_map = np.array(sample["dist_map"], dtype=np.float32).reshape(
            2, IMAGE_HEIGHT, IMAGE_WIDTH
        )
        filename = sample["filename"]

        # Check if sample is already preprocessed (gamma + CLAHE applied)
        is_preprocessed = (
            self.has_preprocessed_column and sample.get("preprocessed", False)
        )

        if is_preprocessed:
            # Image already has gamma + CLAHE applied, use directly
            pilimg = image
        else:
            # Apply deterministic gamma correction
            pilimg = cv2.LUT(image, self.gamma_table)

        # Stochastic augmentations (applied regardless of preprocessing status)
        if self.transform is not None and self.split == "train":
            if random.random() < 0.2:
                pilimg = Line_augment()(np.array(pilimg))
            if random.random() < 0.2:
                pilimg = Gaussian_blur()(np.array(pilimg))

        if is_preprocessed:
            # Already preprocessed, just ensure correct dtype
            img = np.array(np.uint8(pilimg))
        else:
            # Apply CLAHE for non-preprocessed data
            img = self.clahe.apply(np.array(np.uint8(pilimg)))
        img = Image.fromarray(img)
        label_pil = Image.fromarray(label)

        if self.transform is not None:
            if self.split == "train":
                img, label_pil = RandomHorizontalFlip()(img, label_pil)
                if np.array(label_pil)[0, 0] != label[0, 0]:
                    spatial_weights = np.fliplr(spatial_weights).copy()
                    dist_map = np.flip(dist_map, axis=2).copy()
            img = self.transform(img)

        label_tensor = MaskToTensor()(label_pil)
        return img, label_tensor, filename, spatial_weights, dist_map


def get_device(device_override: str | None = None) -> torch.device:
    """
    Detect the best available device or use user override.

    Args:
        device_override: User-specified device ('cuda', 'mps', 'cpu', or None for auto)

    Returns:
        torch.device: The device to use for training
    """
    if device_override:
        device_str = device_override.lower()
        if device_str == "cuda" and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available, falling back to CPU")
            return torch.device("cpu")
        if device_str == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            print("WARNING: MPS requested but not available, falling back to CPU")
            return torch.device("cpu")
        return torch.device(device_str)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_dataset_with_cache():
    """
    Load the OpenEDS dataset using HuggingFace's built-in caching.

    HuggingFace caches datasets automatically at ~/.cache/huggingface/datasets/.
    Override with the HF_DATASETS_CACHE environment variable if needed.

    Returns:
        HuggingFace DatasetDict with train and validation splits
    """
    print(f"Loading dataset from HuggingFace: {HF_DATASET_REPO}")
    print("(First run downloads data; subsequent runs use cache)")

    hf_dataset = load_dataset(HF_DATASET_REPO)

    return hf_dataset


def setup_mlflow():
    """
    Configure MLflow from environment variables.

    Environment variables:
        MLFLOW_TRACKING_URI: MLflow tracking server URI
        MLFLOW_EXPERIMENT_ID: Experiment ID to use

    Raises:
        RuntimeError: If required environment variables are not set
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    experiment_id = os.environ.get("MLFLOW_EXPERIMENT_ID")

    if not tracking_uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI environment variable is not set. "
            "Please set it to your MLflow tracking server URI."
        )

    if not experiment_id:
        raise RuntimeError(
            "MLFLOW_EXPERIMENT_ID environment variable is not set. "
            "Please set it to your MLflow experiment ID."
        )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_id=experiment_id)
    print(f"MLflow configured with experiment ID: {experiment_id}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train TinyEfficientViT for semantic segmentation (local version)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training and validation",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        type=float,
        default=1e-3,
        dest="learning_rate",
        help="Initial learning rate",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints",
        help="Directory for saving model checkpoints",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="./plots",
        help="Directory for saving training plots",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Device to use for training (auto-detect if not specified)",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile() for debugging",
    )
    parser.add_argument(
        "--resume",
        type=str,
        nargs="?",
        const="auto",
        default=None,
        help="Resume from checkpoint. Use without value to auto-detect best model, or provide path",
    )

    return parser.parse_args()


def train(args):
    """Main training function."""
    device = get_device(args.device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        _ = torch.manual_seed(42)

    hf_dataset = load_dataset_with_cache()
    print(f"Train samples: {len(hf_dataset['train'])}")
    print(f"Validation samples: {len(hf_dataset['validation'])}")

    print("\n" + "=" * 80)
    print("Initializing TinyEfficientViT model")
    print("=" * 80)

    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    NUM_WORKERS = args.num_workers

    model = TinyEfficientViTSeg(
        in_channels=1,
        num_classes=2,
        embed_dims=(16, 32, 64),
        depths=(1, 1, 1),
        num_heads=(1, 1, 2),
        key_dims=(4, 4, 4),
        attn_ratios=(2, 2, 2),
        window_sizes=(7, 7, 7),
        mlp_ratios=(2, 2, 2),
        decoder_dim=32,
    ).to(device)

    # Load checkpoint if resuming
    resume_path = resolve_resume_path(args.resume, args.output_dir)
    if resume_path:
        model = load_model_checkpoint(model, resume_path, device)

    nparams = get_nparams(model)
    print(f"N parameters: {nparams:,}")
    if nparams >= 60000:
        print(
            f"WARNING: Model has {nparams} parameters, "
            f"exceeds 60k limit by {nparams - 60000}"
        )
    else:
        print(f"Model is within 60k parameter budget: {nparams} < 60000")

    use_compile = device.type == "cuda" and not args.no_compile
    if use_compile:
        print("Converting model to channels_last memory format...")
        model = model.to(memory_format=torch.channels_last)
        print("Compiling model with torch.compile(mode='max-autotune')...")
        try:
            model = torch.compile(model, mode="max-autotune")
        except Exception as e:
            print(f"WARNING: torch.compile() failed: {e}")
            print("Falling back to eager mode (no compilation)")
            use_compile = False

    print(f"\nVerifying forward pass with batch_size={BATCH_SIZE}...")
    with torch.no_grad():
        memory_format = (
            torch.channels_last if device.type == "cuda" else torch.contiguous_format
        )
        test_input = torch.randn(BATCH_SIZE, 1, IMAGE_HEIGHT, IMAGE_WIDTH).to(
            device, memory_format=memory_format
        )
        try:
            with torch.amp.autocast(device.type):
                test_output = model(test_input)
        except Exception as e:
            if use_compile:
                print(f"WARNING: Compiled model forward pass failed: {e}")
                print("Falling back to eager mode (no compilation)")
                model = TinyEfficientViTSeg(
                    in_channels=1,
                    num_classes=2,
                    embed_dims=(16, 32, 64),
                    depths=(1, 1, 1),
                    num_heads=(1, 1, 2),
                    key_dims=(4, 4, 4),
                    attn_ratios=(2, 2, 2),
                    window_sizes=(7, 7, 7),
                    mlp_ratios=(2, 2, 2),
                    decoder_dim=32,
                ).to(device)
                # Reload checkpoint for fallback model
                if resume_path:
                    model = load_model_checkpoint(model, resume_path, device)
                use_compile = False
                with torch.amp.autocast(device.type):
                    test_output = model(test_input)
            else:
                raise
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {test_output.shape}")
        assert test_output.shape == (BATCH_SIZE, 2, IMAGE_HEIGHT, IMAGE_WIDTH), (
            f"Output shape mismatch: expected ({BATCH_SIZE}, 2, {IMAGE_HEIGHT}, "
            f"{IMAGE_WIDTH}), got {test_output.shape}"
        )
        print("Forward pass verification: PASSED")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Use CosineAnnealingLR instead of ReduceLROnPlateau to avoid per-epoch CPU metric transfer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=LEARNING_RATE * 0.01,  # Minimum LR is 1% of initial
    )
    criterion = CombinedLoss()
    scaler = torch.amp.GradScaler(device.type)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    train_dataset = IrisDataset(
        hf_dataset["train"],
        split="train",
        transform=transform,
    )
    valid_dataset = IrisDataset(
        hf_dataset["validation"],
        split="validation",
        transform=transform,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")

    print(f"\n{'='*80}")
    print("Training Configuration:")
    print(f"{'='*80}")
    print(f"  Model: TinyEfficientViT")
    print(f"  Parameters: {nparams:,}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Num Workers: {NUM_WORKERS}")
    print(f"  Output Directory: {args.output_dir}")
    print(f"  Resume From: {resume_path if resume_path else 'None (training from scratch)'}")
    print(f"{'='*80}")

    trainloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
        drop_last=True,
    )
    validloader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
    )

    alpha_np = np.zeros(EPOCHS)
    alpha_np[0 : min(125, EPOCHS)] = 1 - np.arange(1, min(125, EPOCHS) + 1) / min(
        125, EPOCHS
    )
    if EPOCHS > 125:
        alpha_np[125:] = 1
    alpha = torch.tensor(alpha_np, dtype=torch.float32, device=device)

    # Pre-allocate GPU tensors for all epoch metrics (eliminates per-epoch CPU transfers)
    # Training: loss, ce, dice, surface, miou, bg_iou, pupil_iou, alpha (8 metrics)
    # Validation: loss, ce, dice, surface, miou, bg_iou, pupil_iou (7 metrics)
    all_train_metrics_gpu = torch.zeros(EPOCHS, 8, device=device)
    all_valid_metrics_gpu = torch.zeros(EPOCHS, 7, device=device)
    # Pre-allocate learning rate storage (stored as Python list since it's CPU-side)
    all_lr_values = []
    # Best model tracking on GPU (no CPU transfer for comparison)
    best_valid_iou_gpu = torch.tensor(0.0, device=device)
    best_epoch_gpu = torch.tensor(0, device=device, dtype=torch.long)

    # Pre-allocate per-epoch accumulators (reused each epoch, no allocation overhead)
    train_loss_accum = torch.tensor(0.0, device=device)
    train_ce_accum = torch.tensor(0.0, device=device)
    train_dice_accum = torch.tensor(0.0, device=device)
    train_surface_accum = torch.tensor(0.0, device=device)
    train_intersection_accum = torch.zeros(2, device=device)
    train_union_accum = torch.zeros(2, device=device)
    valid_loss_accum = torch.tensor(0.0, device=device)
    valid_ce_accum = torch.tensor(0.0, device=device)
    valid_dice_accum = torch.tensor(0.0, device=device)
    valid_surface_accum = torch.tensor(0.0, device=device)
    valid_intersection_accum = torch.zeros(2, device=device)
    valid_union_accum = torch.zeros(2, device=device)

    system_info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    if device.type == "cuda":
        system_info["gpu_name"] = torch.cuda.get_device_name(0)
        system_info["gpu_memory_gb"] = (
            torch.cuda.get_device_properties(0).total_memory / 1e9
        )
        system_info["cuda_version"] = torch.version.cuda

    dataset_stats = {
        "train_samples": len(train_dataset),
        "valid_samples": len(valid_dataset),
        "image_size": f"{IMAGE_WIDTH}x{IMAGE_HEIGHT}",
        "num_classes": 2,
        "class_names": "background,pupil",
    }

    model_details = {
        "architecture": "TinyEfficientViT",
        "input_channels": 1,
        "output_channels": 2,
        "embed_dims": "(16, 32, 64)",
        "depths": "(1, 1, 1)",
        "num_heads": "(1, 1, 2)",
        "key_dims": "(4, 4, 4)",
        "decoder_dim": 32,
        "window_sizes": "(7, 7, 7)",
    }

    augmentation_settings = {
        "horizontal_flip": 0.5,
        "line_augment": 0.2,
        "gaussian_blur": 0.2,
        "clahe": True,
        "gamma_correction": 0.8,
    }

    setup_mlflow()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("Starting training")
    print("=" * 80)

    # Metrics stored on GPU during training, transferred to CPU only at end
    # (train_metrics and valid_metrics dicts populated after training completes)

    with mlflow.start_run(run_name="tiny-efficientvit-training") as run:
        mlflow.set_tags(
            {
                "model_type": "TinyEfficientViT",
                "task": "semantic_segmentation",
                "dataset": "OpenEDS",
                "framework": "PyTorch",
                "optimizer": "Adam",
                "scheduler": "CosineAnnealingLR",
                "version": "v1.0-local",
            }
        )
        mlflow.log_params(
            {
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "model_params": nparams,
                "num_workers": NUM_WORKERS,
                "scheduler_T_max": EPOCHS,
                "scheduler_eta_min": LEARNING_RATE * 0.01,
            }
        )
        mlflow.log_params({f"system_{k}": v for k, v in system_info.items()})
        mlflow.log_params({f"dataset_{k}": v for k, v in dataset_stats.items()})
        mlflow.log_params({f"model_{k}": v for k, v in model_details.items()})
        mlflow.log_params(
            {f"augmentation_{k}": v for k, v in augmentation_settings.items()}
        )
        print(f"MLflow run started: {run.info.run_id}")

        for epoch in range(EPOCHS):
            _ = model.train()
            # Zero pre-allocated accumulators (in-place, no allocation)
            train_loss_accum.zero_()
            train_ce_accum.zero_()
            train_dice_accum.zero_()
            train_surface_accum.zero_()
            train_intersection_accum.zero_()
            train_union_accum.zero_()
            train_batch_count = 0

            pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
            for batchdata in pbar:
                img, labels, _, spatialWeights, maxDist = batchdata
                memory_format = (
                    torch.channels_last
                    if device.type == "cuda"
                    else torch.contiguous_format
                )
                data = img.to(device, non_blocking=True, memory_format=memory_format)
                target = labels.to(device, non_blocking=True).long()
                spatial_weights_gpu = spatialWeights.to(
                    device, non_blocking=True
                ).float()
                dist_map_gpu = maxDist.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device.type):
                    output = model(data)
                    (
                        total_loss,
                        ce_loss,
                        dice_loss,
                        surface_loss,
                    ) = criterion(
                        output,
                        target,
                        spatial_weights_gpu,
                        dist_map_gpu,
                        alpha[epoch],
                    )
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Accumulate on GPU (no CPU transfer)
                train_loss_accum += total_loss.detach()
                train_ce_accum += ce_loss.detach()
                train_dice_accum += dice_loss.detach()
                train_surface_accum += surface_loss.detach()
                train_batch_count += 1

                predict = get_predictions(output)
                batch_intersection, batch_union = compute_iou_tensors(predict, target)
                train_intersection_accum += batch_intersection
                train_union_accum += batch_union

            # Compute training metrics on GPU and store in pre-allocated tensor
            iou_per_class_train = train_intersection_accum / train_union_accum.clamp(min=1)
            miou_train_gpu = iou_per_class_train.mean()
            all_train_metrics_gpu[epoch, 0] = train_loss_accum / train_batch_count
            all_train_metrics_gpu[epoch, 1] = train_ce_accum / train_batch_count
            all_train_metrics_gpu[epoch, 2] = train_dice_accum / train_batch_count
            all_train_metrics_gpu[epoch, 3] = train_surface_accum / train_batch_count
            all_train_metrics_gpu[epoch, 4] = miou_train_gpu
            all_train_metrics_gpu[epoch, 5] = iou_per_class_train[0]  # background IoU
            all_train_metrics_gpu[epoch, 6] = iou_per_class_train[1]  # pupil IoU
            all_train_metrics_gpu[epoch, 7] = alpha[epoch]
            # Store LR (this is a CPU-side value from optimizer, unavoidable)
            all_lr_values.append(optimizer.param_groups[0]["lr"])

            _ = model.eval()
            # Zero pre-allocated accumulators (in-place, no allocation)
            valid_loss_accum.zero_()
            valid_ce_accum.zero_()
            valid_dice_accum.zero_()
            valid_surface_accum.zero_()
            valid_intersection_accum.zero_()
            valid_union_accum.zero_()
            valid_batch_count = 0

            with torch.no_grad():
                for batchdata in validloader:
                    img, labels, _, spatialWeights, maxDist = batchdata
                    memory_format = (
                        torch.channels_last
                        if device.type == "cuda"
                        else torch.contiguous_format
                    )
                    data = img.to(
                        device,
                        non_blocking=True,
                        memory_format=memory_format,
                    )
                    target = labels.to(device, non_blocking=True).long()
                    spatial_weights_gpu = spatialWeights.to(
                        device, non_blocking=True
                    ).float()
                    dist_map_gpu = maxDist.to(device, non_blocking=True)

                    with torch.amp.autocast(device.type):
                        output = model(data)
                        (
                            total_loss,
                            ce_loss,
                            dice_loss,
                            surface_loss,
                        ) = criterion(
                            output,
                            target,
                            spatial_weights_gpu,
                            dist_map_gpu,
                            alpha[epoch],
                        )

                    # Accumulate on GPU (no CPU transfer)
                    valid_loss_accum += total_loss.detach()
                    valid_ce_accum += ce_loss.detach()
                    valid_dice_accum += dice_loss.detach()
                    valid_surface_accum += surface_loss.detach()
                    valid_batch_count += 1

                    predict = get_predictions(output)
                    batch_intersection, batch_union = compute_iou_tensors(
                        predict, target
                    )
                    valid_intersection_accum += batch_intersection
                    valid_union_accum += batch_union

            # Compute validation metrics on GPU and store in pre-allocated tensor
            iou_per_class_valid = valid_intersection_accum / valid_union_accum.clamp(min=1)
            miou_valid_gpu = iou_per_class_valid.mean()
            all_valid_metrics_gpu[epoch, 0] = valid_loss_accum / valid_batch_count
            all_valid_metrics_gpu[epoch, 1] = valid_ce_accum / valid_batch_count
            all_valid_metrics_gpu[epoch, 2] = valid_dice_accum / valid_batch_count
            all_valid_metrics_gpu[epoch, 3] = valid_surface_accum / valid_batch_count
            all_valid_metrics_gpu[epoch, 4] = miou_valid_gpu
            all_valid_metrics_gpu[epoch, 5] = iou_per_class_valid[0]  # background IoU
            all_valid_metrics_gpu[epoch, 6] = iou_per_class_valid[1]  # pupil IoU

            # Update learning rate with CosineAnnealingLR (no CPU metric needed)
            scheduler.step()

            # Best model comparison entirely on GPU (no CPU transfer)
            if miou_valid_gpu > best_valid_iou_gpu:
                best_valid_iou_gpu = miou_valid_gpu.clone()
                best_epoch_gpu.fill_(epoch + 1)
                best_model_path = os.path.join(
                    args.output_dir, "best_efficientvit_model.pt"
                )
                save_model_checkpoint(model, best_model_path)
                print(f"Epoch {epoch+1}: New best model saved!")

            # Periodic checkpoint saving (no metric transfer needed)
            if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
                checkpoint_path = os.path.join(
                    args.output_dir, f"efficientvit_model_epoch_{epoch+1}.pt"
                )
                save_model_checkpoint(model, checkpoint_path)

            # Minimal progress indicator (no metric transfer)
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} completed")

        # === END OF TRAINING: Single GPU-to-CPU transfer for all metrics ===
        print("\n" + "=" * 80)
        print("Transferring metrics from GPU to CPU...")

        # Single transfer of all training metrics
        train_metrics_cpu = all_train_metrics_gpu.cpu().numpy()
        valid_metrics_cpu = all_valid_metrics_gpu.cpu().numpy()
        best_valid_iou = best_valid_iou_gpu.cpu().item()
        best_epoch = int(best_epoch_gpu.cpu().item())

        # Build metrics dictionaries for plotting and logging
        train_metrics = {
            "loss": train_metrics_cpu[:, 0].tolist(),
            "ce_loss": train_metrics_cpu[:, 1].tolist(),
            "dice_loss": train_metrics_cpu[:, 2].tolist(),
            "surface_loss": train_metrics_cpu[:, 3].tolist(),
            "iou": train_metrics_cpu[:, 4].tolist(),
            "background_iou": train_metrics_cpu[:, 5].tolist(),
            "pupil_iou": train_metrics_cpu[:, 6].tolist(),
            "alpha": train_metrics_cpu[:, 7].tolist(),
            "lr": all_lr_values,
        }
        valid_metrics = {
            "loss": valid_metrics_cpu[:, 0].tolist(),
            "ce_loss": valid_metrics_cpu[:, 1].tolist(),
            "dice_loss": valid_metrics_cpu[:, 2].tolist(),
            "surface_loss": valid_metrics_cpu[:, 3].tolist(),
            "iou": valid_metrics_cpu[:, 4].tolist(),
            "background_iou": valid_metrics_cpu[:, 5].tolist(),
            "pupil_iou": valid_metrics_cpu[:, 6].tolist(),
        }

        # Final metrics
        loss_train = train_metrics["loss"][-1]
        miou_train = train_metrics["iou"][-1]
        loss_valid = valid_metrics["loss"][-1]
        miou_valid = valid_metrics["iou"][-1]

        # Generate training curves (using transferred metrics)
        print("Generating training curves...")
        plot_paths = create_training_plots(
            train_metrics,
            valid_metrics,
            save_dir=args.plots_dir,
        )

        # Generate final prediction visualization
        print("Generating sample predictions...")
        pred_vis_path = os.path.join(args.plots_dir, "predictions_final.png")
        create_prediction_visualization(
            model,
            validloader,
            device,
            num_samples=4,
            save_path=pred_vis_path,
        )

        # MLflow logging (all at end, using CPU metrics)
        # Log all epoch metrics
        for ep in range(EPOCHS):
            mlflow.log_metrics(
                {
                    "train_loss": train_metrics["loss"][ep],
                    "train_iou": train_metrics["iou"][ep],
                    "train_ce_loss": train_metrics["ce_loss"][ep],
                    "train_dice_loss": train_metrics["dice_loss"][ep],
                    "train_surface_loss": train_metrics["surface_loss"][ep],
                    "train_background_iou": train_metrics["background_iou"][ep],
                    "train_pupil_iou": train_metrics["pupil_iou"][ep],
                    "valid_loss": valid_metrics["loss"][ep],
                    "valid_iou": valid_metrics["iou"][ep],
                    "valid_ce_loss": valid_metrics["ce_loss"][ep],
                    "valid_dice_loss": valid_metrics["dice_loss"][ep],
                    "valid_surface_loss": valid_metrics["surface_loss"][ep],
                    "valid_background_iou": valid_metrics["background_iou"][ep],
                    "valid_pupil_iou": valid_metrics["pupil_iou"][ep],
                    "learning_rate": train_metrics["lr"][ep],
                    "alpha": train_metrics["alpha"][ep],
                },
                step=ep,
            )
        # Log final metrics
        mlflow.log_metrics(
            {
                "final_train_loss": loss_train,
                "final_train_iou": miou_train,
                "final_valid_loss": loss_valid,
                "final_valid_iou": miou_valid,
                "best_valid_iou": best_valid_iou,
                "best_epoch": best_epoch,
            }
        )
        # Log artifacts
        for plot_path in plot_paths.values():
            if plot_path is not None:
                mlflow.log_artifact(plot_path)
        mlflow.log_artifact(pred_vis_path)
        mlflow.log_artifact(best_model_path)

        print("\n" + "=" * 80)
        print("Training completed!")
        print(f"Final validation mIoU: {miou_valid:.4f}")
        print(f"Best validation mIoU: {best_valid_iou:.4f} (epoch {best_epoch})")
        print(f"Final train mIoU: {miou_train:.4f}")
        print(f"MLflow run ID: {run.info.run_id}")
        print(f"Checkpoints saved to: {args.output_dir}")
        print(f"Plots saved to: {args.plots_dir}")
        print("=" * 80)


if __name__ == "__main__":
    args = parse_args()
    train(args)
