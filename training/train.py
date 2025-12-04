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
app = modal.App("SHALLOWNET")
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
    import yaml

    class CombinedLoss(nn.Module):
        def __init__(self, epsilon=1e-5):
            super(
                CombinedLoss,
                self,
            ).__init__()
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
            log_probs = F.log_softmax(
                logits,
                dim=1,
            )
            ce_loss = self.nll(
                log_probs,
                target,
            )
            weighted_ce = (ce_loss * (1.0 + spatial_weights)).mean()
            target_onehot = (
                F.one_hot(
                    target,
                    num_classes=2,
                )
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
                    pred_c,
                    target_c,
                )
                .sum()
                .float()
            )
            union[c] = (
                torch.logical_or(
                    pred_c,
                    target_c,
                )
                .sum()
                .float()
            )
        return (
            intersection,
            union,
        )

    def finalize_iou(
        total_intersection,
        total_union,
    ):
        iou_per_class = (
            (total_intersection / total_union.clamp(min=1)).cpu().numpy()
        )
        return (
            float(np.mean(iou_per_class)),
            iou_per_class.tolist(),
        )

    def get_predictions(
        output,
    ):
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

    def create_training_plots(
        train_metrics,
        valid_metrics,
        save_dir="plots",
    ):
        os.makedirs(
            save_dir,
            exist_ok=True,
        )
        fig, ax = plt.subplots(
            figsize=(
                10,
                6,
            )
        )
        epochs = range(
            1,
            len(train_metrics["loss"]) + 1,
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
            "Epoch",
            fontsize=12,
        )
        ax.set_ylabel(
            "Loss",
            fontsize=12,
        )
        ax.set_title(
            "Training and Validation Loss",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        loss_plot_path = os.path.join(
            save_dir,
            "loss_curves.png",
        )
        plt.savefig(
            loss_plot_path,
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
        fig, ax = plt.subplots(
            figsize=(
                10,
                6,
            )
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
            "Epoch",
            fontsize=12,
        )
        ax.set_ylabel(
            "mIoU",
            fontsize=12,
        )
        ax.set_title(
            "Training and Validation mIoU",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        iou_plot_path = os.path.join(
            save_dir,
            "iou_curves.png",
        )
        plt.savefig(
            iou_plot_path,
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
        if "lr" in train_metrics and len(train_metrics["lr"]) > 0:
            fig, ax = plt.subplots(
                figsize=(
                    10,
                    6,
                )
            )
            ax.plot(
                epochs,
                train_metrics["lr"],
                "g-",
                linewidth=2,
            )
            ax.set_xlabel(
                "Epoch",
                fontsize=12,
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
            ax.grid(
                True,
                alpha=0.3,
            )
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
        if all(
            k in train_metrics
            for k in [
                "ce_loss",
                "dice_loss",
                "surface_loss",
            ]
        ):
            fig, axes = plt.subplots( 2, 2, figsize=( 14, 10,),)
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
            axes[0, 0].set_title(
                "Cross Entropy Loss",
                fontweight="bold",
            )
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].set_ylabel("CE Loss")
            axes[0, 0].legend()
            axes[0, 0].grid(
                True,
                alpha=0.3,
            )
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
            axes[0, 1].set_title(
                "Dice Loss",
                fontweight="bold",
            )
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("Dice Loss")
            axes[0, 1].legend()
            axes[0, 1].grid(
                True,
                alpha=0.3,
            )
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
            axes[1, 0].set_title(
                "Surface Loss",
                fontweight="bold",
            )
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Surface Loss")
            axes[1, 0].legend()
            axes[1, 0].grid(
                True,
                alpha=0.3,
            )
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
            axes[1, 1].grid(
                True,
                alpha=0.3,
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
            "learning_rate": (lr_plot_path if "lr" in train_metrics else None),
            "loss_components": (
                components_plot_path
                if all(
                    k in train_metrics
                    for k in [
                        "ce_loss",
                        "dice_loss",
                        "surface_loss",
                    ]
                )
                else None
            ),
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
            ) in dataloader:
                if samples_collected >= num_samples:
                    break
                single_img = img[0:1].to(device)
                if USE_CHANNELS_LAST:
                    single_img = single_img.to(
                        memory_format=torch.channels_last
                    )
                single_target = labels[0:1].to(device).long()
                output = model(single_img)
                predictions = get_predictions(output)
                images_to_plot.append(img[0].cpu().squeeze().numpy())
                labels_to_plot.append(single_target[0].cpu().numpy())
                preds_to_plot.append(predictions[0].cpu().numpy())
                del (
                    single_img,
                    single_target,
                    output,
                    predictions,
                )
                if torch.cuda.is_available():
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

    def export_model_to_onnx(
        model,
        output_path,
        input_shape=(
            1,
            1,
            640,
            400,
        ),
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
            opset_version=11,
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
            f"Model exported to ONNX: {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.2f} MB)"
        )

    class DownBlock(nn.Module):
        def __init__(
            self,
            input_channels,
            output_channels,
            down_size,
            dropout=False,
            prob=0,
        ):
            super(
                DownBlock,
                self,
            ).__init__()
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
            self.max_pool = nn.AvgPool2d(kernel_size=down_size)
            self.relu = nn.LeakyReLU()
            self.down_size = down_size
            self.dropout = dropout
            self.dropout1 = nn.Dropout(p=prob)
            self.dropout2 = nn.Dropout(p=prob)
            self.dropout3 = nn.Dropout(p=prob)
            self.bn = torch.nn.BatchNorm2d(num_features=output_channels)

        def forward(self, x):
            if self.down_size is not None:
                x = self.max_pool(x)
            if self.dropout:
                x1 = self.relu(
                    self.dropout1(self.pointwise_conv1(self.depthwise_conv1(x)))
                )
                x21 = torch.cat(
                    (x, x1),
                    dim=1,
                )
                x22 = self.relu(
                    self.dropout2(
                        self.pointwise_conv22(
                            self.depthwise_conv22(self.conv21(x21))
                        )
                    )
                )
                x31 = torch.cat(
                    (
                        x21,
                        x22,
                    ),
                    dim=1,
                )
                out = self.relu(
                    self.dropout3(
                        self.pointwise_conv32(
                            self.depthwise_conv32(self.conv31(x31))
                        )
                    )
                )
            else:
                x1 = self.relu(self.pointwise_conv1(self.depthwise_conv1(x)))
                x21 = torch.cat(
                    (x, x1),
                    dim=1,
                )
                x22 = self.relu(
                    self.pointwise_conv22(
                        self.depthwise_conv22(self.conv21(x21))
                    )
                )
                x31 = torch.cat(
                    (
                        x21,
                        x22,
                    ),
                    dim=1,
                )
                out = self.relu(
                    self.pointwise_conv32(
                        self.depthwise_conv32(self.conv31(x31))
                    )
                )
            return self.bn(out)

    class UpBlockConcat(nn.Module):
        def __init__(
            self,
            skip_channels,
            input_channels,
            output_channels,
            up_stride,
            dropout=False,
            prob=0,
        ):
            super(
                UpBlockConcat,
                self,
            ).__init__()
            self.conv11 = nn.Conv2d(
                skip_channels + input_channels,
                output_channels,
                kernel_size=1,
                padding=0,
            )
            self.depthwise_conv12 = nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                groups=output_channels,
            )
            self.pointwise_conv12 = nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=1,
            )
            self.conv21 = nn.Conv2d(
                skip_channels + input_channels + output_channels,
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
            self.relu = nn.LeakyReLU()
            self.up_stride = up_stride
            self.dropout = dropout
            self.dropout1 = nn.Dropout(p=prob)
            self.dropout2 = nn.Dropout(p=prob)

        def forward(
            self,
            prev_feature_map,
            x,
        ):
            x = nn.functional.interpolate(
                x,
                scale_factor=self.up_stride,
                mode="nearest",
            )
            x = torch.cat(
                (
                    x,
                    prev_feature_map,
                ),
                dim=1,
            )
            if self.dropout:
                x1 = self.relu(
                    self.dropout1(
                        self.pointwise_conv12(
                            self.depthwise_conv12(self.conv11(x))
                        )
                    )
                )
                x21 = torch.cat(
                    (x, x1),
                    dim=1,
                )
                out = self.relu(
                    self.dropout2(
                        self.pointwise_conv22(
                            self.depthwise_conv22(self.conv21(x21))
                        )
                    )
                )
            else:
                x1 = self.relu(
                    self.pointwise_conv12(self.depthwise_conv12(self.conv11(x)))
                )
                x21 = torch.cat(
                    (x, x1),
                    dim=1,
                )
                out = self.relu(
                    self.pointwise_conv22(
                        self.depthwise_conv22(self.conv21(x21))
                    )
                )
            return out

    class ShallowNet(nn.Module):
        def __init__(
            self,
            in_channels=1,
            out_channels=2,
            channel_size=32,
            concat=True,
            dropout=False,
            prob=0,
        ):
            super(
                ShallowNet,
                self,
            ).__init__()
            self.down_block1 = DownBlock(
                input_channels=in_channels,
                output_channels=channel_size,
                down_size=None,
                dropout=dropout,
                prob=prob,
            )
            self.down_block2 = DownBlock(
                input_channels=channel_size,
                output_channels=channel_size,
                down_size=(
                    2,
                    2,
                ),
                dropout=dropout,
                prob=prob,
            )
            self.down_block3 = DownBlock(
                input_channels=channel_size,
                output_channels=channel_size,
                down_size=(
                    2,
                    2,
                ),
                dropout=dropout,
                prob=prob,
            )
            self.down_block4 = DownBlock(
                input_channels=channel_size,
                output_channels=channel_size,
                down_size=(
                    2,
                    2,
                ),
                dropout=dropout,
                prob=prob,
            )
            self.up_block1 = UpBlockConcat(
                skip_channels=channel_size,
                input_channels=channel_size,
                output_channels=channel_size,
                up_stride=(
                    2,
                    2,
                ),
                dropout=dropout,
                prob=prob,
            )
            self.up_block2 = UpBlockConcat(
                skip_channels=channel_size,
                input_channels=channel_size,
                output_channels=channel_size,
                up_stride=(
                    2,
                    2,
                ),
                dropout=dropout,
                prob=prob,
            )
            self.up_block3 = UpBlockConcat(
                skip_channels=channel_size,
                input_channels=channel_size,
                output_channels=channel_size,
                up_stride=(
                    2,
                    2,
                ),
                dropout=dropout,
                prob=prob,
            )
            self.out_conv1 = nn.Conv2d(
                in_channels=channel_size,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
            )
            self.concat = concat
            self.dropout = dropout
            self.dropout1 = nn.Dropout(p=prob)
            self._initialize_weights()

        def _initialize_weights(
            self,
        ):
            for m in self.modules():
                if isinstance(
                    m,
                    nn.Conv2d,
                ):
                    if (
                        m.groups == m.in_channels
                        and m.in_channels == m.out_channels
                    ):
                        n = m.kernel_size[0] * m.kernel_size[1]
                        m.weight.data.normal_(
                            0,
                            math.sqrt(2.0 / n),
                        )
                    elif m.kernel_size == (
                        1,
                        1,
                    ):
                        n = m.in_channels
                        m.weight.data.normal_(
                            0,
                            math.sqrt(2.0 / n),
                        )
                    else:
                        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                        m.weight.data.normal_(
                            0,
                            math.sqrt(2.0 / n),
                        )
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(
                    m,
                    nn.BatchNorm2d,
                ):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(
                    m,
                    nn.Linear,
                ):
                    n = m.weight.size(1)
                    m.weight.data.normal_(
                        0,
                        0.01,
                    )
                    m.bias.data.zero_()

        def forward(self, x):
            x1 = self.down_block1(x)
            x2 = self.down_block2(x1)
            x3 = self.down_block3(x2)
            x4 = self.down_block4(x3)
            x5 = self.up_block1(x3, x4)
            x6 = self.up_block2(x2, x5)
            x7 = self.up_block3(x1, x6)
            if self.dropout:
                out = self.out_conv1(self.dropout1(x7))
            else:
                out = self.out_conv1(x7)
            return out

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
                cv2.GaussianBlur(
                    img,
                    (7, 7),
                    sigma_value,
                )
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
                    (
                        int(x1),
                        int(y1),
                    ),
                    (
                        int(x2),
                        int(y2),
                    ),
                    (
                        255,
                        255,
                        255,
                    ),
                    4,
                )
            aug_base = aug_base.astype(np.uint8)
            return Image.fromarray(aug_base)

    class MaskToTensor(object):
        def __call__(self, img):
            return torch.from_numpy(
                np.array(
                    img,
                    dtype=np.int64,
                )
            ).long()

    HF_DATASET_REPO = "Conner/openeds-precomputed"
    IMAGE_HEIGHT = 400
    IMAGE_WIDTH = 640

    class IrisDataset(Dataset):
        def __init__(
            self,
            hf_dataset,
            split="train",
            transform=None,
        ):
            self.transform = transform
            self.split = split
            self.clahe = cv2.createCLAHE(
                clipLimit=1.5,
                tileGridSize=(
                    8,
                    8,
                ),
            )
            self.gamma_table = 255.0 * (np.linspace(0, 1, 256) ** 0.8)
            self.dataset = hf_dataset

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
                sample["spatial_weights"],
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
            filename = sample["filename"]
            pilimg = cv2.LUT(
                image,
                self.gamma_table,
            )
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
                    (
                        img,
                        label_pil,
                    ) = RandomHorizontalFlip()(
                        img,
                        label_pil,
                    )
                    if (
                        np.array(label_pil)[
                            0,
                            0,
                        ]
                        != label[
                            0,
                            0,
                        ]
                    ):
                        spatial_weights = np.fliplr(spatial_weights).copy()
                        dist_map = np.flip(
                            dist_map,
                            axis=2,
                        ).copy()
                img = self.transform(img)
            label_tensor = MaskToTensor()(label_pil)
            return (
                img,
                label_tensor,
                filename,
                spatial_weights,
                dist_map,
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.manual_seed(12)
    else:
        _ = torch.manual_seed(12)
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

            shutil.rmtree(
                DATASET_CACHE_PATH,
                ignore_errors=True,
            )
            if os.path.exists(CACHE_MARKER_FILE):
                os.remove(CACHE_MARKER_FILE)
            cache_exists = False
    if not cache_exists:
        print(f"Downloading from HuggingFace: {HF_DATASET_REPO}")
        print("First run takes ~20 min, subsequent runs will be fast.")
        hf_dataset = load_dataset(HF_DATASET_REPO)
        os.makedirs(
            VOLUME_PATH,
            exist_ok=True,
        )
        hf_dataset.save_to_disk(DATASET_CACHE_PATH)
        with open(
            CACHE_MARKER_FILE,
            "w",
        ) as f:
            f.write(f"Cached from {HF_DATASET_REPO}\n")
        dataset_volume.commit()
        print("Dataset cached to volume!")
    print(f"Train samples: {len(hf_dataset['train'])}")
    print(f"Validation samples: {len(hf_dataset['validation'])}")
    print("\n" + "=" * 80)
    print("Initializing model and training setup")
    print("=" * 80)
    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 1e-3
    NUM_WORKERS = 8
    GRADIENT_ACCUMULATION_STEPS = 1
    EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    model = ShallowNet(
        in_channels=1,
        out_channels=2,
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
        model = torch.compile(
            model,
            mode="reduce-overhead",
        )
        print("Model compiled successfully")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        "min",
        patience=5,
    )
    criterion = CombinedLoss()
    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None
    use_amp = torch.cuda.is_available()
    if use_amp:
        print("Mixed precision training (AMP) enabled")
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
    alpha = np.zeros(EPOCHS)
    alpha[0 : min(125, EPOCHS)] = 1 - np.arange(
        1,
        min(125, EPOCHS) + 1,
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
        "image_size": "640x400",
        "num_classes": 2,
        "class_names": "background,pupil",
    }
    model_details = {
        "architecture": "ShallowNet",
        "input_channels": 1,
        "output_channels": 2,
        "channel_size": 32,
        "dropout": True,
        "dropout_prob": 0.2,
        "down_blocks": 4,
        "up_blocks": 3,
        "bottleneck_size": "80x50",
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
        f"\nMLflow configured with Databricks experiment ID: {environ['MLFLOW_EXPERIMENT_ID']}"
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
    with mlflow.start_run(run_name="shallow-net-training") as run:
        mlflow.set_tags(
            {
                "model_type": "ShallowNet",
                "task": "semantic_segmentation",
                "dataset": "OpenEDS",
                "framework": "PyTorch",
                "optimizer": "Adam",
                "scheduler": "ReduceLROnPlateau",
                "version": "v1.1",
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
                "torch_compile": hasattr(
                    torch,
                    "compile",
                ),
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
            train_loss_sum = torch.tensor(
                0.0,
                device=device,
            )
            train_ce_sum = torch.tensor(
                0.0,
                device=device,
            )
            train_dice_sum = torch.tensor(
                0.0,
                device=device,
            )
            train_surface_sum = torch.tensor(
                0.0,
                device=device,
            )
            train_batch_count = 0
            train_intersection = torch.zeros(
                2,
                device=device,
            )
            train_union = torch.zeros(
                2,
                device=device,
            )
            pbar = tqdm(
                trainloader,
                desc=f"Epoch {epoch+1}/{EPOCHS}",
            )
            for batchdata in pbar:
                (
                    img,
                    labels,
                    _,
                    spatialWeights,
                    maxDist,
                ) = batchdata
                data = img.to(
                    device,
                    non_blocking=True,
                )
                if USE_CHANNELS_LAST:
                    data = data.to(memory_format=torch.channels_last)
                target = labels.to(
                    device,
                    non_blocking=True,
                ).long()
                spatial_weights_gpu = spatialWeights.to(
                    device,
                    non_blocking=True,
                ).float()
                dist_map_gpu = maxDist.to(
                    device,
                    non_blocking=True,
                )
                if USE_CHANNELS_LAST:
                    dist_map_gpu = dist_map_gpu.to(
                        memory_format=torch.channels_last
                    )
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
                (
                    batch_intersection,
                    batch_union,
                ) = compute_iou_tensors(
                    predict,
                    target,
                )
                train_intersection += batch_intersection
                train_union += batch_union
            (
                miou_train,
                per_class_ious_train,
            ) = finalize_iou(
                train_intersection,
                train_union,
            )
            (
                bg_iou_train,
                pupil_iou_train,
            ) = (
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
            valid_loss_sum = torch.tensor(
                0.0,
                device=device,
            )
            valid_ce_sum = torch.tensor(
                0.0,
                device=device,
            )
            valid_dice_sum = torch.tensor(
                0.0,
                device=device,
            )
            valid_surface_sum = torch.tensor(
                0.0,
                device=device,
            )
            valid_batch_count = 0
            valid_intersection = torch.zeros(
                2,
                device=device,
            )
            valid_union = torch.zeros(
                2,
                device=device,
            )
            with torch.no_grad():
                for batchdata in validloader:
                    (
                        img,
                        labels,
                        _,
                        spatialWeights,
                        maxDist,
                    ) = batchdata
                    data = img.to(
                        device,
                        non_blocking=True,
                    )
                    if USE_CHANNELS_LAST:
                        data = data.to(memory_format=torch.channels_last)
                    target = labels.to(
                        device,
                        non_blocking=True,
                    ).long()
                    spatial_weights_gpu = spatialWeights.to(
                        device,
                        non_blocking=True,
                    ).float()
                    dist_map_gpu = maxDist.to(
                        device,
                        non_blocking=True,
                    )
                    if USE_CHANNELS_LAST:
                        dist_map_gpu = dist_map_gpu.to(
                            memory_format=torch.channels_last
                        )
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
                    (
                        batch_intersection,
                        batch_union,
                    ) = compute_iou_tensors(
                        predict,
                        target,
                    )
                    valid_intersection += batch_intersection
                    valid_union += batch_union
            (
                miou_valid,
                per_class_ious_valid,
            ) = finalize_iou(
                valid_intersection,
                valid_union,
            )
            (
                bg_iou_valid,
                pupil_iou_valid,
            ) = (
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
                f"Train BG IoU: {bg_iou_train:.4f} | Train Pupil IoU: {pupil_iou_train:.4f}"
            )
            print(
                f"Valid BG IoU: {bg_iou_valid:.4f} | Valid Pupil IoU: {pupil_iou_valid:.4f}"
            )
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            if miou_valid > best_valid_iou:
                best_valid_iou = miou_valid
                best_epoch = epoch + 1
                best_model_path = "best_shallow_model.onnx"
                export_model_to_onnx(
                    model,
                    best_model_path,
                )
                mlflow.log_artifact(best_model_path)
                mlflow.log_metric(
                    "best_valid_iou",
                    best_valid_iou,
                    step=epoch,
                )
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
                checkpoint_path = f"shallow_model_epoch_{epoch+1}.onnx"
                export_model_to_onnx(
                    model,
                    checkpoint_path,
                )
                mlflow.log_artifact(checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
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
        print("\n" + "=" * 80)
        print("Training completed!")
        print(f"Final validation mIoU: {miou_valid:.4f}")
        print(
            f"Best validation mIoU: {best_valid_iou:.4f} (epoch {best_epoch})"
        )
        print(f"Final train mIoU: {miou_train:.4f}")
        print(f"MLflow run ID: {run.info.run_id}")
        print("=" * 80)


@app.local_entrypoint()
def main():
    print("Starting training...")
    result = train.remote()
    print("Training complete!")
    return result
