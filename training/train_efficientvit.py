import modal
from os import environ

train_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.10",
    )
    .entrypoint([])
    .apt_install(
        "libgl1",
        "libglib2.0-0",
        "unzip",
        "wget",
        "git",
    )
    .uv_pip_install(
        "torch==2.8.0",
        "numpy>=1.21.0",
        "torchvision>=0.22.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.3.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
        "mlflow[databricks]>=3.5.0",
        "requests>=2.28.1",
        "matplotlib>=3.5.0",
        "datasets>=4.4.1",
        "huggingface_hub>=0.16.0",
        "pyarrow>=14.0.0",
        "onnx>=1.15.0",
        "onnxruntime>=1.17.0",
        "pyyaml>=6.0.0",
    )
)
app = modal.App("EFFICIENTVIT")
dataset_volume = modal.Volume.from_name(
    "openeds-dataset-cache",
    create_if_missing=True,
)
VOLUME_PATH = "/data/openeds"


@app.function(
    gpu="T4",
    cpu=16.0,
    memory=32768,
    image=train_image,
    timeout=3600 * 16,
    volumes={VOLUME_PATH: dataset_volume},
    secrets=[
        modal.Secret.from_name("databricks-shallow"),
    ],
)
def train():
    import os
    import math
    import random
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import (
        Dataset,
        DataLoader,
    )
    from torchvision import (
        transforms,
    )
    from PIL import Image
    import cv2
    from tqdm import tqdm
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from datasets import (
        load_dataset,
        load_from_disk,
    )
    import mlflow

    # =========================================================================
    # TinyEfficientViT Model Components
    # =========================================================================

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

            q = qkv[:, :, :q_total].reshape(
                B, N, self.num_heads, self.key_dim
            ).permute(0, 2, 1, 3)
            k = qkv[:, :, q_total : q_total + k_total].reshape(
                B, N, self.num_heads, self.key_dim
            ).permute(0, 2, 1, 3)
            v = qkv[:, :, q_total + k_total :].reshape(
                B, N, self.num_heads, self.d
            ).permute(0, 2, 1, 3)

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

            self.blocks = nn.ModuleList([
                TinyEfficientVitBlock(
                    dim=out_dim,
                    num_heads=num_heads,
                    key_dim=key_dim,
                    attn_ratio=attn_ratio,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ])

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

    # =========================================================================
    # Loss Functions and Metrics (from train.py)
    # =========================================================================

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
            target_onehot = (
                F.one_hot(target, num_classes=2)
                .permute(0, 3, 1, 2)
                .float()
            )
            probs_flat = probs.flatten(start_dim=2)
            target_flat = target_onehot.flatten(start_dim=2)
            intersection = (probs_flat * target_flat).sum(dim=2)
            cardinality = (probs_flat + target_flat).sum(dim=2)
            class_weights = 1.0 / (target_flat.sum(dim=2) ** 2).clamp(
                min=self.epsilon
            )
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
            total_loss = (
                weighted_ce + alpha * dice_loss + (1.0 - alpha) * surface_loss
            )
            return (
                total_loss,
                weighted_ce,
                dice_loss,
                surface_loss,
            )

    def compute_iou_tensors(predictions, targets, num_classes=2):
        intersection = torch.zeros(num_classes, device=predictions.device)
        union = torch.zeros(num_classes, device=predictions.device)
        for c in range(num_classes):
            pred_c = predictions == c
            target_c = targets == c
            intersection[c] = (
                torch.logical_and(pred_c, target_c).sum().float()
            )
            union[c] = (
                torch.logical_or(pred_c, target_c).sum().float()
            )
        return intersection, union

    def finalize_iou(total_intersection, total_union):
        iou_per_class = (
            (total_intersection / total_union.clamp(min=1)).cpu().numpy()
        )
        return float(np.mean(iou_per_class)), iou_per_class.tolist()

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

    # =========================================================================
    # Visualization Utilities (from train.py)
    # =========================================================================

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
        if all(
            k in train_metrics
            for k in ["ce_loss", "dice_loss", "surface_loss"]
        ):
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
                single_img = img[0:1].to(device)
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

    def export_model_to_onnx(
        model,
        output_path,
        input_shape=(1, 1, 400, 640),
    ):
        export_model = model
        if hasattr(model, "_orig_mod"):
            export_model = model._orig_mod
        export_model.eval()
        dummy_input = torch.randn(input_shape).to(
            next(export_model.parameters()).device
        )
        torch.onnx.export(
            export_model,
            dummy_input,
            output_path,
            opset_version=14,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
            do_constant_folding=True,
        )
        if not os.path.exists(output_path):
            raise RuntimeError(
                f"ONNX export failed - file not created at {output_path}"
            )
        print(
            f"Model exported to ONNX: {output_path} "
            f"({os.path.getsize(output_path) / 1024 / 1024:.2f} MB)"
        )

    # =========================================================================
    # Data Augmentation Classes (from train.py)
    # =========================================================================

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
            return Image.fromarray(
                cv2.GaussianBlur(img, (7, 7), sigma_value)
            )

    class Line_augment(object):
        def __call__(self, base):
            yc, xc = (0.3 + 0.4 * np.random.rand(1)) * np.array(base.shape)
            aug_base = np.copy(base)
            num_lines = np.random.randint(1, 10)
            for _ in np.arange(0, num_lines):
                theta = np.pi * np.random.rand(1)
                x1 = xc - 50 * np.random.rand(1) * (
                    1 if np.random.rand(1) < 0.5 else -1
                )
                y1 = (x1 - xc) * np.tan(theta) + yc
                x2 = xc - (150 * np.random.rand(1) + 50) * (
                    1 if np.random.rand(1) < 0.5 else -1
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

    # =========================================================================
    # Dataset (from train.py)
    # =========================================================================

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
            spatial_weights = np.array(
                sample["spatial_weights"], dtype=np.float32
            ).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
            dist_map = np.array(
                sample["dist_map"], dtype=np.float32
            ).reshape(2, IMAGE_HEIGHT, IMAGE_WIDTH)
            filename = sample["filename"]

            pilimg = cv2.LUT(image, self.gamma_table)
            if self.transform is not None and self.split == "train":
                if random.random() < 0.2:
                    pilimg = Line_augment()(np.array(pilimg))
                if random.random() < 0.2:
                    pilimg = Gaussian_blur()(np.array(pilimg))
            img = self.clahe.apply(np.array(np.uint8(pilimg)))
            img = Image.fromarray(img)
            label_pil = Image.fromarray(label)

            if self.transform is not None:
                if self.split == "train":
                    img, label_pil = RandomHorizontalFlip()(img, label_pil)
                    if (
                        np.array(label_pil)[0, 0] != label[0, 0]
                    ):
                        spatial_weights = np.fliplr(spatial_weights).copy()
                        dist_map = np.flip(dist_map, axis=2).copy()
                img = self.transform(img)

            label_tensor = MaskToTensor()(label_pil)
            return img, label_tensor, filename, spatial_weights, dist_map

    # =========================================================================
    # Main Training Logic
    # =========================================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.manual_seed(42)
        # Performance optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        _ = torch.manual_seed(42)

    DATASET_CACHE_PATH = f"{VOLUME_PATH}/dataset"
    CACHE_MARKER_FILE = f"{VOLUME_PATH}/.cache_complete"
    cache_exists = os.path.exists(CACHE_MARKER_FILE)
    if cache_exists:
        print(f"Found cached dataset at: {DATASET_CACHE_PATH}")
        print("Loading from volume cache (fast)...")
        try:
            hf_dataset = load_from_disk(DATASET_CACHE_PATH)
            print("Loaded from cache!")
        except Exception as e:
            print(f"Cache corrupted, re-downloading: {e}")
            import shutil
            shutil.rmtree(DATASET_CACHE_PATH, ignore_errors=True)
            if os.path.exists(CACHE_MARKER_FILE):
                os.remove(CACHE_MARKER_FILE)
            cache_exists = False

    if not cache_exists:
        print(f"Downloading from HuggingFace: {HF_DATASET_REPO}")
        print("First run takes ~20 min, subsequent runs will be fast.")
        hf_dataset = load_dataset(HF_DATASET_REPO)
        os.makedirs(VOLUME_PATH, exist_ok=True)
        hf_dataset.save_to_disk(DATASET_CACHE_PATH)
        with open(CACHE_MARKER_FILE, "w") as f:
            f.write(f"Cached from {HF_DATASET_REPO}\n")
        dataset_volume.commit()
        print("Dataset cached to volume!")

    print(f"Train samples: {len(hf_dataset['train'])}")
    print(f"Validation samples: {len(hf_dataset['validation'])}")

    print("\n" + "=" * 80)
    print("Initializing TinyEfficientViT model")
    print("=" * 80)

    BATCH_SIZE = 128
    EPOCHS = 1
    LEARNING_RATE = 1e-2
    NUM_WORKERS = 8

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

    nparams = get_nparams(model)
    print(f"N parameters: {nparams:,}")
    if nparams >= 60000:
        print(
            f"WARNING: Model has {nparams} parameters, "
            f"exceeds 60k limit by {nparams - 60000}"
        )
    else:
        print(f"Model is within 60k parameter budget: {nparams} < 60000")

    # Compile model for faster training (PyTorch 2.x)
    if torch.cuda.is_available():
        print("Compiling model with torch.compile()...")
        model = torch.compile(model, mode="reduce-overhead")

    print("\nVerifying forward pass...")
    with torch.no_grad():
        test_input = torch.randn(1, 1, IMAGE_HEIGHT, IMAGE_WIDTH).to(device)
        test_output = model(test_input)
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {test_output.shape}")
        assert test_output.shape == (1, 2, IMAGE_HEIGHT, IMAGE_WIDTH), (
            f"Output shape mismatch: expected (1, 2, {IMAGE_HEIGHT}, "
            f"{IMAGE_WIDTH}), got {test_output.shape}"
        )
        print("Forward pass verification: PASSED")

    use_amp = torch.cuda.is_available()
    if use_amp:
        print("Mixed precision training (AMP) enabled")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        "min",
        patience=5,
    )
    criterion = CombinedLoss()
    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

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
    print(f"  Mixed Precision (AMP): {use_amp}")
    print(f"{'='*80}")

    trainloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=4,
        drop_last=True,
    )
    validloader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=4,
    )

    alpha = np.zeros(EPOCHS)
    alpha[0 : min(125, EPOCHS)] = 1 - np.arange(
        1, min(125, EPOCHS) + 1
    ) / min(125, EPOCHS)
    if EPOCHS > 125:
        alpha[125:] = 1

    import platform
    system_info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    if torch.cuda.is_available():
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

    mlflow.set_tracking_uri(environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(experiment_id=environ["MLFLOW_EXPERIMENT_ID"])
    print(
        f"\nMLflow configured with Databricks experiment ID: "
        f"{environ['MLFLOW_EXPERIMENT_ID']}"
    )

    print("\n" + "=" * 80)
    print("Starting training")
    print("=" * 80)

    train_metrics = {
        "loss": [],
        "iou": [],
        "ce_loss": [],
        "dice_loss": [],
        "surface_loss": [],
        "alpha": [],
        "lr": [],
        "background_iou": [],
        "pupil_iou": [],
    }
    valid_metrics = {
        "loss": [],
        "iou": [],
        "ce_loss": [],
        "dice_loss": [],
        "surface_loss": [],
        "background_iou": [],
        "pupil_iou": [],
    }
    best_valid_iou = 0.0
    best_epoch = 0

    with mlflow.start_run(run_name="tiny-efficientvit-training") as run:
        mlflow.set_tags({
            "model_type": "TinyEfficientViT",
            "task": "semantic_segmentation",
            "dataset": "OpenEDS",
            "framework": "PyTorch",
            "optimizer": "Adam",
            "scheduler": "ReduceLROnPlateau",
            "version": "v1.0",
        })
        mlflow.log_params({
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "model_params": nparams,
            "num_workers": NUM_WORKERS,
            "scheduler_patience": 5,
            "use_amp": use_amp,
        })
        mlflow.log_params({f"system_{k}": v for k, v in system_info.items()})
        mlflow.log_params({f"dataset_{k}": v for k, v in dataset_stats.items()})
        mlflow.log_params({f"model_{k}": v for k, v in model_details.items()})
        mlflow.log_params(
            {f"augmentation_{k}": v for k, v in augmentation_settings.items()}
        )
        print(f"MLflow run started: {run.info.run_id}")

        for epoch in range(EPOCHS):
            _ = model.train()
            train_loss_sum = torch.tensor(0.0, device=device)
            train_ce_sum = torch.tensor(0.0, device=device)
            train_dice_sum = torch.tensor(0.0, device=device)
            train_surface_sum = torch.tensor(0.0, device=device)
            train_batch_count = 0
            train_intersection = torch.zeros(2, device=device)
            train_union = torch.zeros(2, device=device)

            pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
            for batchdata in pbar:
                img, labels, _, spatialWeights, maxDist = batchdata
                data = img.to(device, non_blocking=True)
                target = labels.to(device, non_blocking=True).long()
                spatial_weights_gpu = spatialWeights.to(
                    device, non_blocking=True
                ).float()
                dist_map_gpu = maxDist.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    with torch.amp.autocast("cuda"):
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
                else:
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
                    total_loss.backward()
                    optimizer.step()

                train_loss_sum += total_loss.detach()
                train_ce_sum += ce_loss.detach()
                train_dice_sum += dice_loss.detach()
                train_surface_sum += surface_loss.detach()
                train_batch_count += 1

                predict = get_predictions(output)
                batch_intersection, batch_union = compute_iou_tensors(
                    predict, target
                )
                train_intersection += batch_intersection
                train_union += batch_union

            miou_train, per_class_ious_train = finalize_iou(
                train_intersection, train_union
            )
            bg_iou_train, pupil_iou_train = (
                per_class_ious_train[0],
                per_class_ious_train[1],
            )
            loss_train: float = (train_loss_sum / train_batch_count).item()
            ce_loss_train: float = (train_ce_sum / train_batch_count).item()
            dice_loss_train: float = (train_dice_sum / train_batch_count).item()
            surface_loss_train: float = (
                train_surface_sum / train_batch_count
            ).item()

            train_metrics["loss"].append(loss_train)
            train_metrics["iou"].append(miou_train)
            train_metrics["ce_loss"].append(ce_loss_train)
            train_metrics["dice_loss"].append(dice_loss_train)
            train_metrics["surface_loss"].append(surface_loss_train)
            train_metrics["alpha"].append(alpha[epoch])
            train_metrics["lr"].append(optimizer.param_groups[0]["lr"])
            train_metrics["background_iou"].append(bg_iou_train)
            train_metrics["pupil_iou"].append(pupil_iou_train)

            _ = model.eval()
            valid_loss_sum = torch.tensor(0.0, device=device)
            valid_ce_sum = torch.tensor(0.0, device=device)
            valid_dice_sum = torch.tensor(0.0, device=device)
            valid_surface_sum = torch.tensor(0.0, device=device)
            valid_batch_count = 0
            valid_intersection = torch.zeros(2, device=device)
            valid_union = torch.zeros(2, device=device)

            with torch.no_grad():
                for batchdata in validloader:
                    img, labels, _, spatialWeights, maxDist = batchdata
                    data = img.to(device, non_blocking=True)
                    target = labels.to(device, non_blocking=True).long()
                    spatial_weights_gpu = spatialWeights.to(
                        device, non_blocking=True
                    ).float()
                    dist_map_gpu = maxDist.to(device, non_blocking=True)

                    if use_amp:
                        with torch.amp.autocast("cuda"):
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
                    else:
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

                    valid_loss_sum += total_loss.detach()
                    valid_ce_sum += ce_loss.detach()
                    valid_dice_sum += dice_loss.detach()
                    valid_surface_sum += surface_loss.detach()
                    valid_batch_count += 1

                    predict = get_predictions(output)
                    batch_intersection, batch_union = compute_iou_tensors(
                        predict, target
                    )
                    valid_intersection += batch_intersection
                    valid_union += batch_union

            miou_valid, per_class_ious_valid = finalize_iou(
                valid_intersection, valid_union
            )
            bg_iou_valid, pupil_iou_valid = (
                per_class_ious_valid[0],
                per_class_ious_valid[1],
            )
            loss_valid: float = (valid_loss_sum / valid_batch_count).item()
            ce_loss_valid: float = (valid_ce_sum / valid_batch_count).item()
            dice_loss_valid: float = (valid_dice_sum / valid_batch_count).item()
            surface_loss_valid: float = (
                valid_surface_sum / valid_batch_count
            ).item()

            valid_metrics["loss"].append(loss_valid)
            valid_metrics["iou"].append(miou_valid)
            valid_metrics["ce_loss"].append(ce_loss_valid)
            valid_metrics["dice_loss"].append(dice_loss_valid)
            valid_metrics["surface_loss"].append(surface_loss_valid)
            valid_metrics["background_iou"].append(bg_iou_valid)
            valid_metrics["pupil_iou"].append(pupil_iou_valid)

            mlflow.log_metrics(
                {
                    "train_loss": loss_train,
                    "train_iou": miou_train,
                    "train_ce_loss": ce_loss_train,
                    "train_dice_loss": dice_loss_train,
                    "train_surface_loss": surface_loss_train,
                    "train_background_iou": bg_iou_train,
                    "train_pupil_iou": pupil_iou_train,
                    "valid_loss": loss_valid,
                    "valid_iou": miou_valid,
                    "valid_ce_loss": ce_loss_valid,
                    "valid_dice_loss": dice_loss_valid,
                    "valid_surface_loss": surface_loss_valid,
                    "valid_background_iou": bg_iou_valid,
                    "valid_pupil_iou": pupil_iou_valid,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "alpha": alpha[epoch],
                },
                step=epoch,
            )

            scheduler.step(loss_valid)

            print(f"\nEpoch {epoch+1}/{EPOCHS}")
            print(
                f"Train Loss: {loss_train:.4f} | Valid Loss: {loss_valid:.4f}"
            )
            print(
                f"Train mIoU: {miou_train:.4f} | Valid mIoU: {miou_valid:.4f}"
            )
            print(
                f"Train BG IoU: {bg_iou_train:.4f} | "
                f"Train Pupil IoU: {pupil_iou_train:.4f}"
            )
            print(
                f"Valid BG IoU: {bg_iou_valid:.4f} | "
                f"Valid Pupil IoU: {pupil_iou_valid:.4f}"
            )
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

            if miou_valid > best_valid_iou:
                best_valid_iou = miou_valid
                best_epoch = epoch + 1
                best_model_path = "best_efficientvit_model.onnx"
                export_model_to_onnx(model, best_model_path)
                mlflow.log_artifact(best_model_path)
                mlflow.log_metric("best_valid_iou", best_valid_iou, step=epoch)
                print(f"New best model! Valid mIoU: {best_valid_iou:.4f}")

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print("Generating sample predictions...")
                pred_vis_path = f"predictions_epoch_{epoch+1}.png"
                create_prediction_visualization(
                    model,
                    validloader,
                    device,
                    num_samples=4,
                    save_path=pred_vis_path,
                )
                mlflow.log_artifact(pred_vis_path)
                print(f"Sample predictions logged: {pred_vis_path}")

            if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
                print("Generating training curves...")
                plot_paths = create_training_plots(
                    train_metrics,
                    valid_metrics,
                    save_dir="plots",
                )
                for plot_path in plot_paths.values():
                    if plot_path is not None:
                        mlflow.log_artifact(plot_path)
                print("Training curves logged to MLflow")

            if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
                checkpoint_path = f"efficientvit_model_epoch_{epoch+1}.onnx"
                export_model_to_onnx(model, checkpoint_path)
                mlflow.log_artifact(checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")

        mlflow.log_metrics({
            "final_train_loss": loss_train,
            "final_train_iou": miou_train,
            "final_valid_loss": loss_valid,
            "final_valid_iou": miou_valid,
            "best_valid_iou": best_valid_iou,
            "best_epoch": best_epoch,
        })

        print("\n" + "=" * 80)
        print("Training completed!")
        print(f"Final validation mIoU: {miou_valid:.4f}")
        print(f"Best validation mIoU: {best_valid_iou:.4f} (epoch {best_epoch})")
        print(f"Final train mIoU: {miou_train:.4f}")
        print(f"MLflow run ID: {run.info.run_id}")
        print("=" * 80)


@app.local_entrypoint()
def main():
    print("Starting TinyEfficientViT training...")
    result = train.remote()
    print("Training complete!")
    return result
