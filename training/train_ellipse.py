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

app = modal.App("ELLIPSE-REGRESSION")

dataset_volume = modal.Volume.from_name(
    "openeds-dataset-cache",
    create_if_missing=True,
)

VOLUME_PATH = "/data/openeds"


@app.function(
    gpu="L4",
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
    # Constants
    # =========================================================================
    HF_DATASET_REPO = "Conner/openeds-precomputed"
    IMAGE_HEIGHT = 400
    IMAGE_WIDTH = 640

    # Normalization factors for ellipse parameters
    # Center is normalized by image dimensions
    # Radii are normalized by max possible radius (half of image diagonal)
    MAX_RADIUS = math.sqrt(IMAGE_WIDTH**2 + IMAGE_HEIGHT**2) / 2

    # =========================================================================
    # Ellipse Parameter Extraction
    # =========================================================================
    def extract_ellipse_params(mask):
        """
        Extract ellipse parameters from a binary mask.
        Returns: (cx, cy, rx, ry, angle) where:
            - cx, cy: center coordinates (pixels)
            - rx, ry: semi-axes lengths (pixels)
            - angle: rotation angle in degrees
        If no valid contour found, returns zeros.
        """
        # Find contours
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if len(contours) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        # Get the largest contour (in case of multiple)
        largest_contour = max(contours, key=cv2.contourArea)

        # Need at least 5 points to fit an ellipse
        if len(largest_contour) < 5:
            # Fall back to moments-based center and approximate radius
            M = cv2.moments(largest_contour)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                area = cv2.contourArea(largest_contour)
                radius = math.sqrt(area / math.pi)
                return cx, cy, radius, radius, 0.0
            return 0.0, 0.0, 0.0, 0.0, 0.0

        try:
            ellipse = cv2.fitEllipse(largest_contour)
            (cx, cy), (width, height), angle = ellipse
            # width and height are full axes lengths, we want semi-axes
            rx = width / 2.0
            ry = height / 2.0
            return cx, cy, rx, ry, angle
        except cv2.error:
            # Fallback if fitEllipse fails
            M = cv2.moments(largest_contour)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                area = cv2.contourArea(largest_contour)
                radius = math.sqrt(area / math.pi)
                return cx, cy, radius, radius, 0.0
            return 0.0, 0.0, 0.0, 0.0, 0.0

    def normalize_ellipse_params(cx, cy, rx, ry):
        """Normalize ellipse parameters to [0, 1] range."""
        cx_norm = cx / IMAGE_WIDTH
        cy_norm = cy / IMAGE_HEIGHT
        rx_norm = rx / MAX_RADIUS
        ry_norm = ry / MAX_RADIUS
        return cx_norm, cy_norm, rx_norm, ry_norm

    def denormalize_ellipse_params(cx_norm, cy_norm, rx_norm, ry_norm):
        """Denormalize ellipse parameters back to pixel values."""
        cx = cx_norm * IMAGE_WIDTH
        cy = cy_norm * IMAGE_HEIGHT
        rx = rx_norm * MAX_RADIUS
        ry = ry_norm * MAX_RADIUS
        return cx, cy, rx, ry

    def render_ellipse_mask(cx, cy, rx, ry, height=IMAGE_HEIGHT, width=IMAGE_WIDTH):
        """Render an ellipse mask from parameters."""
        mask = np.zeros((height, width), dtype=np.uint8)
        if rx > 0 and ry > 0:
            cv2.ellipse(
                mask,
                center=(int(round(cx)), int(round(cy))),
                axes=(int(round(rx)), int(round(ry))),
                angle=0,
                startAngle=0,
                endAngle=360,
                color=1,
                thickness=-1,
            )
        return mask

    # =========================================================================
    # Loss Functions
    # =========================================================================
    class EllipseRegressionLoss(nn.Module):
        """
        Combined loss for ellipse regression:
        - Smooth L1 for center prediction
        - Smooth L1 for radii prediction
        - Optional IoU loss computed by rendering ellipses
        """

        def __init__(self, center_weight=1.0, radius_weight=1.0, iou_weight=0.5):
            super(EllipseRegressionLoss, self).__init__()
            self.center_weight = center_weight
            self.radius_weight = radius_weight
            self.iou_weight = iou_weight
            self.smooth_l1 = nn.SmoothL1Loss(reduction="mean")

        def forward(self, pred, target, compute_iou=True):
            """
            pred: (B, 4) - cx, cy, rx, ry (normalized)
            target: (B, 4) - cx, cy, rx, ry (normalized)
            """
            # Center loss
            center_loss = self.smooth_l1(pred[:, :2], target[:, :2])

            # Radius loss
            radius_loss = self.smooth_l1(pred[:, 2:], target[:, 2:])

            # Total regression loss
            total_loss = (
                self.center_weight * center_loss + self.radius_weight * radius_loss
            )

            # Optional: Differentiable IoU approximation
            # For actual IoU, we'd need to render masks which isn't differentiable
            # Instead, we use a soft IoU approximation based on ellipse overlap
            if self.iou_weight > 0 and compute_iou:
                # Approximate IoU using distance between parameters
                # This encourages predictions to match targets more closely
                param_dist = torch.mean((pred - target) ** 2, dim=1)
                iou_proxy_loss = torch.mean(param_dist)
                total_loss = total_loss + self.iou_weight * iou_proxy_loss

            return total_loss, center_loss, radius_loss

    # =========================================================================
    # Metrics
    # =========================================================================
    def compute_center_error(pred, target):
        """Compute mean center error in pixels."""
        # Denormalize
        pred_cx = pred[:, 0] * IMAGE_WIDTH
        pred_cy = pred[:, 1] * IMAGE_HEIGHT
        target_cx = target[:, 0] * IMAGE_WIDTH
        target_cy = target[:, 1] * IMAGE_HEIGHT

        # Euclidean distance
        dist = torch.sqrt((pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2)
        return dist.mean().item()

    def compute_radius_error(pred, target):
        """Compute mean radius error in pixels."""
        pred_rx = pred[:, 2] * MAX_RADIUS
        pred_ry = pred[:, 3] * MAX_RADIUS
        target_rx = target[:, 2] * MAX_RADIUS
        target_ry = target[:, 3] * MAX_RADIUS

        rx_error = torch.abs(pred_rx - target_rx)
        ry_error = torch.abs(pred_ry - target_ry)
        return ((rx_error + ry_error) / 2).mean().item()

    def compute_iou_from_ellipses(pred, target, device):
        """
        Compute IoU by rendering predicted and target ellipses.
        This is computed on CPU for efficiency.
        """
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        batch_size = pred_np.shape[0]
        ious = []

        for i in range(batch_size):
            # Denormalize
            pred_cx, pred_cy, pred_rx, pred_ry = denormalize_ellipse_params(
                pred_np[i, 0], pred_np[i, 1], pred_np[i, 2], pred_np[i, 3]
            )
            target_cx, target_cy, target_rx, target_ry = denormalize_ellipse_params(
                target_np[i, 0], target_np[i, 1], target_np[i, 2], target_np[i, 3]
            )

            # Render masks
            pred_mask = render_ellipse_mask(pred_cx, pred_cy, pred_rx, pred_ry)
            target_mask = render_ellipse_mask(
                target_cx, target_cy, target_rx, target_ry
            )

            # Compute IoU
            intersection = np.logical_and(pred_mask, target_mask).sum()
            union = np.logical_or(pred_mask, target_mask).sum()

            if union > 0:
                iou = intersection / union
            else:
                iou = 1.0 if intersection == 0 else 0.0

            ious.append(iou)

        return np.mean(ious)

    def compute_iou_with_gt_mask(pred, gt_masks, device):
        """
        Compute IoU between predicted ellipses and ground truth masks.
        gt_masks: (B, H, W) ground truth binary masks
        """
        pred_np = pred.detach().cpu().numpy()
        gt_masks_np = gt_masks.cpu().numpy()

        batch_size = pred_np.shape[0]
        ious_bg = []
        ious_pupil = []

        for i in range(batch_size):
            # Denormalize prediction
            pred_cx, pred_cy, pred_rx, pred_ry = denormalize_ellipse_params(
                pred_np[i, 0], pred_np[i, 1], pred_np[i, 2], pred_np[i, 3]
            )

            # Render predicted mask
            pred_mask = render_ellipse_mask(pred_cx, pred_cy, pred_rx, pred_ry)
            target_mask = gt_masks_np[i]

            # Pupil IoU (class 1)
            pred_pupil = pred_mask == 1
            target_pupil = target_mask == 1
            intersection_pupil = np.logical_and(pred_pupil, target_pupil).sum()
            union_pupil = np.logical_or(pred_pupil, target_pupil).sum()
            iou_pupil = intersection_pupil / max(union_pupil, 1)
            ious_pupil.append(iou_pupil)

            # Background IoU (class 0)
            pred_bg = pred_mask == 0
            target_bg = target_mask == 0
            intersection_bg = np.logical_and(pred_bg, target_bg).sum()
            union_bg = np.logical_or(pred_bg, target_bg).sum()
            iou_bg = intersection_bg / max(union_bg, 1)
            ious_bg.append(iou_bg)

        mean_bg_iou = np.mean(ious_bg)
        mean_pupil_iou = np.mean(ious_pupil)
        mean_iou = (mean_bg_iou + mean_pupil_iou) / 2

        return mean_iou, mean_bg_iou, mean_pupil_iou

    # =========================================================================
    # Model Architecture
    # =========================================================================
    class DownBlock(nn.Module):
        """Encoder block with depthwise separable convolutions."""

        def __init__(
            self,
            input_channels,
            output_channels,
            down_size,
            dropout=False,
            prob=0,
        ):
            super(DownBlock, self).__init__()
            self.depthwise_conv1 = nn.Conv2d(
                input_channels,
                input_channels,
                kernel_size=3,
                padding=1,
                groups=input_channels,
            )
            self.pointwise_conv1 = nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=1,
            )
            self.conv21 = nn.Conv2d(
                input_channels + output_channels,
                output_channels,
                kernel_size=1,
                padding=0,
            )
            self.depthwise_conv22 = nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                groups=output_channels,
            )
            self.pointwise_conv22 = nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=1,
            )
            self.conv31 = nn.Conv2d(
                input_channels + 2 * output_channels,
                output_channels,
                kernel_size=1,
                padding=0,
            )
            self.depthwise_conv32 = nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                groups=output_channels,
            )
            self.pointwise_conv32 = nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=1,
            )
            self.max_pool = nn.AvgPool2d(kernel_size=down_size) if down_size else None
            self.relu = nn.LeakyReLU()
            self.down_size = down_size
            self.dropout = dropout
            self.dropout1 = nn.Dropout(p=prob)
            self.dropout2 = nn.Dropout(p=prob)
            self.dropout3 = nn.Dropout(p=prob)
            self.bn = torch.nn.BatchNorm2d(num_features=output_channels)

        def forward(self, x):
            if self.max_pool is not None:
                x = self.max_pool(x)

            if self.dropout:
                x1 = self.relu(
                    self.dropout1(self.pointwise_conv1(self.depthwise_conv1(x)))
                )
                x21 = torch.cat((x, x1), dim=1)
                x22 = self.relu(
                    self.dropout2(
                        self.pointwise_conv22(self.depthwise_conv22(self.conv21(x21)))
                    )
                )
                x31 = torch.cat((x21, x22), dim=1)
                out = self.relu(
                    self.dropout3(
                        self.pointwise_conv32(self.depthwise_conv32(self.conv31(x31)))
                    )
                )
            else:
                x1 = self.relu(self.pointwise_conv1(self.depthwise_conv1(x)))
                x21 = torch.cat((x, x1), dim=1)
                x22 = self.relu(
                    self.pointwise_conv22(self.depthwise_conv22(self.conv21(x21)))
                )
                x31 = torch.cat((x21, x22), dim=1)
                out = self.relu(
                    self.pointwise_conv32(self.depthwise_conv32(self.conv31(x31)))
                )

            return self.bn(out)

    class EllipseRegressionNet(nn.Module):
        """
        Lightweight CNN for ellipse parameter regression.
        Predicts: (cx, cy, rx, ry) - center and semi-axes of pupil ellipse.
        
        Architecture:
        - 4 encoder blocks with progressive downsampling
        - Global average pooling
        - FC layers for regression
        
        This is significantly lighter than a full U-Net since we don't need
        a decoder - we just need to predict 4 scalar values.
        """

        def __init__(
            self,
            in_channels=1,
            channel_size=32,
            dropout=False,
            prob=0,
        ):
            super(EllipseRegressionNet, self).__init__()

            # Encoder blocks
            # 640x400 -> 640x400
            self.down_block1 = DownBlock(
                input_channels=in_channels,
                output_channels=channel_size,
                down_size=None,
                dropout=dropout,
                prob=prob,
            )
            # 640x400 -> 320x200
            self.down_block2 = DownBlock(
                input_channels=channel_size,
                output_channels=channel_size,
                down_size=(2, 2),
                dropout=dropout,
                prob=prob,
            )
            # 320x200 -> 160x100
            self.down_block3 = DownBlock(
                input_channels=channel_size,
                output_channels=channel_size * 2,
                down_size=(2, 2),
                dropout=dropout,
                prob=prob,
            )
            # 160x100 -> 80x50
            self.down_block4 = DownBlock(
                input_channels=channel_size * 2,
                output_channels=channel_size * 2,
                down_size=(2, 2),
                dropout=dropout,
                prob=prob,
            )

            # Global average pooling
            self.global_pool = nn.AdaptiveAvgPool2d(1)

            # Regression head
            fc_input_size = channel_size * 2
            self.fc = nn.Sequential(
                nn.Linear(fc_input_size, 128),
                nn.LeakyReLU(),
                nn.Dropout(p=prob) if dropout else nn.Identity(),
                nn.Linear(128, 64),
                nn.LeakyReLU(),
                nn.Dropout(p=prob) if dropout else nn.Identity(),
                nn.Linear(64, 4),  # cx, cy, rx, ry
                nn.Sigmoid(),  # Output in [0, 1] range
            )

            self._initialize_weights()

        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if m.groups == m.in_channels and m.in_channels == m.out_channels:
                        n = m.kernel_size[0] * m.kernel_size[1]
                        m.weight.data.normal_(0, math.sqrt(2.0 / n))
                    elif m.kernel_size == (1, 1):
                        n = m.in_channels
                        m.weight.data.normal_(0, math.sqrt(2.0 / n))
                    else:
                        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                        m.weight.data.normal_(0, math.sqrt(2.0 / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    n = m.weight.size(1)
                    m.weight.data.normal_(0, 0.01)
                    if m.bias is not None:
                        m.bias.data.zero_()

        def forward(self, x):
            # Encoder
            x = self.down_block1(x)
            x = self.down_block2(x)
            x = self.down_block3(x)
            x = self.down_block4(x)

            # Global pooling
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)

            # Regression
            params = self.fc(x)

            return params

    # =========================================================================
    # Utility Functions
    # =========================================================================
    def get_nparams(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def total_metric(nparams, miou):
        S = nparams * 4.0 / (1024 * 1024)
        total = min(1, 1.0 / S) + miou
        return total * 0.5

    # =========================================================================
    # Visualization Functions
    # =========================================================================
    def create_training_plots(train_metrics, valid_metrics, save_dir="plots"):
        os.makedirs(save_dir, exist_ok=True)

        # Loss curves
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(train_metrics["loss"]) + 1)
        ax.plot(epochs, train_metrics["loss"], "b-", label="Train Loss", linewidth=2)
        ax.plot(epochs, valid_metrics["loss"], "r-", label="Valid Loss", linewidth=2)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        loss_plot_path = os.path.join(save_dir, "loss_curves.png")
        plt.savefig(loss_plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        # IoU curves
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, train_metrics["iou"], "b-", label="Train mIoU", linewidth=2)
        ax.plot(epochs, valid_metrics["iou"], "r-", label="Valid mIoU", linewidth=2)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("mIoU", fontsize=12)
        ax.set_title("Training and Validation mIoU", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        iou_plot_path = os.path.join(save_dir, "iou_curves.png")
        plt.savefig(iou_plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        # Learning rate
        if "lr" in train_metrics and len(train_metrics["lr"]) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(epochs, train_metrics["lr"], "g-", linewidth=2)
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("Learning Rate", fontsize=12)
            ax.set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            lr_plot_path = os.path.join(save_dir, "learning_rate.png")
            plt.savefig(lr_plot_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            lr_plot_path = None

        # Error metrics
        if "center_error" in train_metrics:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            axes[0, 0].plot(
                epochs,
                train_metrics["center_error"],
                "b-",
                label="Train",
                linewidth=2,
            )
            axes[0, 0].plot(
                epochs,
                valid_metrics["center_error"],
                "r-",
                label="Valid",
                linewidth=2,
            )
            axes[0, 0].set_title("Center Error (pixels)", fontweight="bold")
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].set_ylabel("Error")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            axes[0, 1].plot(
                epochs,
                train_metrics["radius_error"],
                "b-",
                label="Train",
                linewidth=2,
            )
            axes[0, 1].plot(
                epochs,
                valid_metrics["radius_error"],
                "r-",
                label="Valid",
                linewidth=2,
            )
            axes[0, 1].set_title("Radius Error (pixels)", fontweight="bold")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("Error")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            axes[1, 0].plot(
                epochs,
                train_metrics["center_loss"],
                "b-",
                label="Train",
                linewidth=2,
            )
            axes[1, 0].plot(
                epochs,
                valid_metrics["center_loss"],
                "r-",
                label="Valid",
                linewidth=2,
            )
            axes[1, 0].set_title("Center Loss", fontweight="bold")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Loss")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            axes[1, 1].plot(
                epochs,
                train_metrics["radius_loss"],
                "b-",
                label="Train",
                linewidth=2,
            )
            axes[1, 1].plot(
                epochs,
                valid_metrics["radius_loss"],
                "r-",
                label="Valid",
                linewidth=2,
            )
            axes[1, 1].set_title("Radius Loss", fontweight="bold")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Loss")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            components_plot_path = os.path.join(save_dir, "error_components.png")
            plt.savefig(components_plot_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            components_plot_path = None

        return {
            "loss_curves": loss_plot_path,
            "iou_curves": iou_plot_path,
            "learning_rate": lr_plot_path,
            "error_components": components_plot_path,
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
        params_to_plot = []

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        with torch.no_grad():
            for batch in dataloader:
                if samples_collected >= num_samples:
                    break

                img, labels, ellipse_params, _, _, _ = batch
                single_img = img[0:1].to(device)

                if USE_CHANNELS_LAST:
                    single_img = single_img.to(memory_format=torch.channels_last)

                output = model(single_img)

                # Denormalize predictions
                pred_params = output[0].cpu().numpy()
                pred_cx, pred_cy, pred_rx, pred_ry = denormalize_ellipse_params(
                    pred_params[0], pred_params[1], pred_params[2], pred_params[3]
                )

                # Render predicted mask
                pred_mask = render_ellipse_mask(pred_cx, pred_cy, pred_rx, pred_ry)

                images_to_plot.append(img[0].cpu().squeeze().numpy())
                labels_to_plot.append(labels[0].cpu().numpy())
                preds_to_plot.append(pred_mask)
                params_to_plot.append((pred_cx, pred_cy, pred_rx, pred_ry))

                del single_img, output
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                samples_collected += 1

        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
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
            cx, cy, rx, ry = params_to_plot[i]
            axes[i, 2].set_title(
                f"Prediction\ncx={cx:.1f}, cy={cy:.1f}, rx={rx:.1f}, ry={ry:.1f}",
                fontweight="bold",
            )
            axes[i, 2].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        return save_path

    def export_model_to_onnx(model, output_path, input_shape=(1, 1, 640, 400)):
        export_model = model
        if hasattr(model, "_orig_mod"):
            export_model = model._orig_mod

        export_model.eval()
        dummy_input = torch.randn(input_shape).to(next(export_model.parameters()).device)

        torch.onnx.export(
            export_model,
            dummy_input,
            output_path,
            opset_version=11,
            input_names=["input"],
            output_names=["ellipse_params"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "ellipse_params": {0: "batch_size"},
            },
            do_constant_folding=True,
        )

        if not os.path.exists(output_path):
            raise RuntimeError(f"ONNX export failed - file not created at {output_path}")

        print(
            f"Model exported to ONNX: {output_path} "
            f"({os.path.getsize(output_path) / 1024 / 1024:.2f} MB)"
        )

    # =========================================================================
    # Data Augmentation
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
                cv2.GaussianBlur(np.array(img), (7, 7), sigma_value)
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
    # Dataset
    # =========================================================================
    class IrisDataset(Dataset):
        def __init__(self, hf_dataset, split="train", transform=None):
            self.transform = transform
            self.split = split
            self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
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
            spatial_weights = np.array(sample["spatial_weights"], dtype=np.float32).reshape(
                IMAGE_HEIGHT, IMAGE_WIDTH
            )
            dist_map = np.array(sample["dist_map"], dtype=np.float32).reshape(
                2, IMAGE_HEIGHT, IMAGE_WIDTH
            )
            filename = sample["filename"]

            # Extract ellipse parameters from mask
            cx, cy, rx, ry, _ = extract_ellipse_params(label)
            cx_norm, cy_norm, rx_norm, ry_norm = normalize_ellipse_params(
                cx, cy, rx, ry
            )
            ellipse_params = torch.tensor(
                [cx_norm, cy_norm, rx_norm, ry_norm], dtype=torch.float32
            )

            # Image preprocessing
            pilimg = cv2.LUT(image, self.gamma_table)

            if self.transform is not None and self.split == "train":
                if random.random() < 0.2:
                    pilimg = Line_augment()(np.array(pilimg))
                if random.random() < 0.2:
                    pilimg = Gaussian_blur()(np.array(pilimg))

            img = self.clahe.apply(np.array(np.uint8(pilimg)))
            img = Image.fromarray(img)
            label_pil = Image.fromarray(label)

            flipped = False
            if self.transform is not None:
                if self.split == "train":
                    img, label_pil = RandomHorizontalFlip()(img, label_pil)
                    # Check if flipped
                    if np.array(label_pil)[0, 0] != label[0, 0]:
                        flipped = True
                        spatial_weights = np.fliplr(spatial_weights).copy()
                        dist_map = np.flip(dist_map, axis=2).copy()
                        # Flip ellipse center x coordinate
                        cx_norm = 1.0 - cx_norm
                        ellipse_params = torch.tensor(
                            [cx_norm, cy_norm, rx_norm, ry_norm], dtype=torch.float32
                        )

                img = self.transform(img)

            label_tensor = MaskToTensor()(label_pil)

            return (
                img,
                label_tensor,
                ellipse_params,
                filename,
                spatial_weights,
                dist_map,
            )

    # =========================================================================
    # Main Training Logic
    # =========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.manual_seed(12)
    else:
        _ = torch.manual_seed(12)

    # Load dataset
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
    print("Initializing model and training setup")
    print("=" * 80)

    # Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 1e-3
    NUM_WORKERS = 8
    GRADIENT_ACCUMULATION_STEPS = 1
    EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS

    # Model
    model = EllipseRegressionNet(
        in_channels=1,
        channel_size=32,
        dropout=True,
        prob=0.2,
    ).to(device)

    nparams = get_nparams(model)
    print(f"N parameters: {nparams:,}")

    USE_CHANNELS_LAST = True
    if USE_CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)
        print("Model converted to channels_last memory format")

    if hasattr(torch, "compile"):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model, mode="reduce-overhead")
        print("Model compiled successfully")

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5
    )

    # Loss function
    criterion = EllipseRegressionLoss(
        center_weight=1.0,
        radius_weight=1.0,
        iou_weight=0.5,
    )

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None
    use_amp = torch.cuda.is_available()
    if use_amp:
        print("Mixed precision training (AMP) enabled")

    # Data transforms
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
    print("Active Optimizations:")
    print(f"{'='*80}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Gradient Accumulation Steps: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Effective Batch Size: {EFFECTIVE_BATCH_SIZE}")
    print(f"  Num Workers: {NUM_WORKERS}")
    print(f"  Mixed Precision (AMP): {use_amp}")
    print(f"  torch.compile(): {hasattr(torch, 'compile')}")
    print(f"  Channels Last Format: {USE_CHANNELS_LAST}")
    print(f"{'='*80}")

    trainloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=2,
        drop_last=True,
    )

    validloader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=2,
    )

    # System info for logging
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
        "image_size": "640x400",
        "num_classes": 2,
        "class_names": "background,pupil",
    }

    model_details = {
        "architecture": "EllipseRegressionNet",
        "input_channels": 1,
        "output_params": 4,
        "output_description": "cx,cy,rx,ry",
        "channel_size": 32,
        "dropout": True,
        "dropout_prob": 0.2,
        "down_blocks": 4,
        "bottleneck_size": "80x50",
    }

    augmentation_settings = {
        "horizontal_flip": 0.5,
        "line_augment": 0.2,
        "gaussian_blur": 0.2,
        "clahe": True,
        "gamma_correction": 0.8,
    }

    # MLflow setup
    mlflow.set_tracking_uri(environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(experiment_id=environ["MLFLOW_EXPERIMENT_ID"])
    print(
        f"\nMLflow configured with Databricks experiment ID: {environ['MLFLOW_EXPERIMENT_ID']}"
    )

    print("\n" + "=" * 80)
    print("Starting training")
    print("=" * 80)

    # Metrics storage
    train_metrics = {
        "loss": [],
        "iou": [],
        "center_loss": [],
        "radius_loss": [],
        "center_error": [],
        "radius_error": [],
        "lr": [],
        "background_iou": [],
        "pupil_iou": [],
    }

    valid_metrics = {
        "loss": [],
        "iou": [],
        "center_loss": [],
        "radius_loss": [],
        "center_error": [],
        "radius_error": [],
        "background_iou": [],
        "pupil_iou": [],
    }

    best_valid_iou = 0.0
    best_epoch = 0

    with mlflow.start_run(run_name="ellipse-regression-training") as run:
        mlflow.set_tags(
            {
                "model_type": "EllipseRegressionNet",
                "task": "ellipse_regression",
                "dataset": "OpenEDS",
                "framework": "PyTorch",
                "optimizer": "Adam",
                "scheduler": "ReduceLROnPlateau",
                "version": "v1.0",
            }
        )

        mlflow.log_params(
            {
                "batch_size": BATCH_SIZE,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "effective_batch_size": EFFECTIVE_BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "model_params": nparams,
                "num_workers": NUM_WORKERS,
                "scheduler_patience": 5,
                "use_amp": use_amp,
                "use_channels_last": USE_CHANNELS_LAST,
                "torch_compile": hasattr(torch, "compile"),
                "center_weight": 1.0,
                "radius_weight": 1.0,
                "iou_weight": 0.5,
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
            # Training
            _ = model.train()

            train_loss_sum = torch.tensor(0.0, device=device)
            train_center_loss_sum = torch.tensor(0.0, device=device)
            train_radius_loss_sum = torch.tensor(0.0, device=device)
            train_batch_count = 0

            train_center_errors = []
            train_radius_errors = []
            train_ious = []
            train_bg_ious = []
            train_pupil_ious = []

            pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

            for batchdata in pbar:
                (
                    img,
                    labels,
                    ellipse_params,
                    _,
                    _,
                    _,
                ) = batchdata

                data = img.to(device, non_blocking=True)
                if USE_CHANNELS_LAST:
                    data = data.to(memory_format=torch.channels_last)

                target_params = ellipse_params.to(device, non_blocking=True)
                target_labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                if use_amp:
                    with torch.amp.autocast("cuda"):
                        output = model(data)
                        total_loss, center_loss, radius_loss = criterion(
                            output, target_params
                        )
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(data)
                    total_loss, center_loss, radius_loss = criterion(
                        output, target_params
                    )
                    total_loss.backward()
                    optimizer.step()

                train_loss_sum += total_loss.detach()
                train_center_loss_sum += center_loss.detach()
                train_radius_loss_sum += radius_loss.detach()
                train_batch_count += 1

                # Compute metrics
                train_center_errors.append(compute_center_error(output, target_params))
                train_radius_errors.append(compute_radius_error(output, target_params))

                # Compute IoU with ground truth masks
                miou, bg_iou, pupil_iou = compute_iou_with_gt_mask(
                    output, target_labels, device
                )
                train_ious.append(miou)
                train_bg_ious.append(bg_iou)
                train_pupil_ious.append(pupil_iou)

            # Aggregate training metrics
            loss_train = (train_loss_sum / train_batch_count).item()
            center_loss_train = (train_center_loss_sum / train_batch_count).item()
            radius_loss_train = (train_radius_loss_sum / train_batch_count).item()
            center_error_train = np.mean(train_center_errors)
            radius_error_train = np.mean(train_radius_errors)
            miou_train = np.mean(train_ious)
            bg_iou_train = np.mean(train_bg_ious)
            pupil_iou_train = np.mean(train_pupil_ious)

            train_metrics["loss"].append(loss_train)
            train_metrics["iou"].append(miou_train)
            train_metrics["center_loss"].append(center_loss_train)
            train_metrics["radius_loss"].append(radius_loss_train)
            train_metrics["center_error"].append(center_error_train)
            train_metrics["radius_error"].append(radius_error_train)
            train_metrics["lr"].append(optimizer.param_groups[0]["lr"])
            train_metrics["background_iou"].append(bg_iou_train)
            train_metrics["pupil_iou"].append(pupil_iou_train)

            # Validation
            _ = model.eval()

            valid_loss_sum = torch.tensor(0.0, device=device)
            valid_center_loss_sum = torch.tensor(0.0, device=device)
            valid_radius_loss_sum = torch.tensor(0.0, device=device)
            valid_batch_count = 0

            valid_center_errors = []
            valid_radius_errors = []
            valid_ious = []
            valid_bg_ious = []
            valid_pupil_ious = []

            with torch.no_grad():
                for batchdata in validloader:
                    (
                        img,
                        labels,
                        ellipse_params,
                        _,
                        _,
                        _,
                    ) = batchdata

                    data = img.to(device, non_blocking=True)
                    if USE_CHANNELS_LAST:
                        data = data.to(memory_format=torch.channels_last)

                    target_params = ellipse_params.to(device, non_blocking=True)
                    target_labels = labels.to(device, non_blocking=True)

                    if use_amp:
                        with torch.amp.autocast("cuda"):
                            output = model(data)
                            total_loss, center_loss, radius_loss = criterion(
                                output, target_params, compute_iou=False
                            )
                    else:
                        output = model(data)
                        total_loss, center_loss, radius_loss = criterion(
                            output, target_params, compute_iou=False
                        )

                    valid_loss_sum += total_loss.detach()
                    valid_center_loss_sum += center_loss.detach()
                    valid_radius_loss_sum += radius_loss.detach()
                    valid_batch_count += 1

                    # Compute metrics
                    valid_center_errors.append(
                        compute_center_error(output, target_params)
                    )
                    valid_radius_errors.append(
                        compute_radius_error(output, target_params)
                    )

                    # Compute IoU with ground truth masks
                    miou, bg_iou, pupil_iou = compute_iou_with_gt_mask(
                        output, target_labels, device
                    )
                    valid_ious.append(miou)
                    valid_bg_ious.append(bg_iou)
                    valid_pupil_ious.append(pupil_iou)

            # Aggregate validation metrics
            loss_valid = (valid_loss_sum / valid_batch_count).item()
            center_loss_valid = (valid_center_loss_sum / valid_batch_count).item()
            radius_loss_valid = (valid_radius_loss_sum / valid_batch_count).item()
            center_error_valid = np.mean(valid_center_errors)
            radius_error_valid = np.mean(valid_radius_errors)
            miou_valid = np.mean(valid_ious)
            bg_iou_valid = np.mean(valid_bg_ious)
            pupil_iou_valid = np.mean(valid_pupil_ious)

            valid_metrics["loss"].append(loss_valid)
            valid_metrics["iou"].append(miou_valid)
            valid_metrics["center_loss"].append(center_loss_valid)
            valid_metrics["radius_loss"].append(radius_loss_valid)
            valid_metrics["center_error"].append(center_error_valid)
            valid_metrics["radius_error"].append(radius_error_valid)
            valid_metrics["background_iou"].append(bg_iou_valid)
            valid_metrics["pupil_iou"].append(pupil_iou_valid)

            # Log to MLflow
            mlflow.log_metrics(
                {
                    "train_loss": loss_train,
                    "train_iou": miou_train,
                    "train_center_loss": center_loss_train,
                    "train_radius_loss": radius_loss_train,
                    "train_center_error": center_error_train,
                    "train_radius_error": radius_error_train,
                    "train_background_iou": bg_iou_train,
                    "train_pupil_iou": pupil_iou_train,
                    "valid_loss": loss_valid,
                    "valid_iou": miou_valid,
                    "valid_center_loss": center_loss_valid,
                    "valid_radius_loss": radius_loss_valid,
                    "valid_center_error": center_error_valid,
                    "valid_radius_error": radius_error_valid,
                    "valid_background_iou": bg_iou_valid,
                    "valid_pupil_iou": pupil_iou_valid,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )

            scheduler.step(loss_valid)

            print(f"\nEpoch {epoch+1}/{EPOCHS}")
            print(f"Train Loss: {loss_train:.4f} | Valid Loss: {loss_valid:.4f}")
            print(f"Train mIoU: {miou_train:.4f} | Valid mIoU: {miou_valid:.4f}")
            print(
                f"Train Center Err: {center_error_train:.2f}px | "
                f"Valid Center Err: {center_error_valid:.2f}px"
            )
            print(
                f"Train Radius Err: {radius_error_train:.2f}px | "
                f"Valid Radius Err: {radius_error_valid:.2f}px"
            )
            print(
                f"Train BG IoU: {bg_iou_train:.4f} | Train Pupil IoU: {pupil_iou_train:.4f}"
            )
            print(
                f"Valid BG IoU: {bg_iou_valid:.4f} | Valid Pupil IoU: {pupil_iou_valid:.4f}"
            )
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

            # Save best model
            if miou_valid > best_valid_iou:
                best_valid_iou = miou_valid
                best_epoch = epoch + 1
                best_model_path = "best_ellipse_model.onnx"
                export_model_to_onnx(model, best_model_path)
                mlflow.log_artifact(best_model_path)
                mlflow.log_metric("best_valid_iou", best_valid_iou, step=epoch)
                print(f"New best model! Valid mIoU: {best_valid_iou:.4f}")

            # Visualizations
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
                checkpoint_path = f"ellipse_model_epoch_{epoch+1}.onnx"
                export_model_to_onnx(model, checkpoint_path)
                mlflow.log_artifact(checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")

        # Final metrics
        mlflow.log_metrics(
            {
                "final_train_loss": loss_train,
                "final_train_iou": miou_train,
                "final_valid_loss": loss_valid,
                "final_valid_iou": miou_valid,
                "final_center_error": center_error_valid,
                "final_radius_error": radius_error_valid,
                "best_valid_iou": best_valid_iou,
                "best_epoch": best_epoch,
            }
        )

        print("\n" + "=" * 80)
        print("Training completed!")
        print(f"Final validation mIoU: {miou_valid:.4f}")
        print(f"Best validation mIoU: {best_valid_iou:.4f} (epoch {best_epoch})")
        print(f"Final center error: {center_error_valid:.2f} pixels")
        print(f"Final radius error: {radius_error_valid:.2f} pixels")
        print(f"Final train mIoU: {miou_train:.4f}")
        print(f"MLflow run ID: {run.info.run_id}")
        print("=" * 80)


@app.local_entrypoint()
def main():
    print("Starting ellipse regression training...")
    result = train.remote()
    print("Training complete!")
    return result
