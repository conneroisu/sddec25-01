"""
Local Ellipse Regression Training Script

This script trains an ellipse regression model for pupil detection without requiring
Modal cloud infrastructure. It can run on local GPU workstations.

Usage:
    # Basic usage with defaults
    train-ellipse

    # Custom training configuration
    train-ellipse --epochs 20 --batch-size 16 --lr 0.0005

    # Specify output directory
    train-ellipse --output-dir ./checkpoints

    # Force CPU training
    train-ellipse --device cpu

    # Disable MLflow logging
    train-ellipse --no-mlflow

Environment Variables (for MLflow):
    MLFLOW_TRACKING_URI: MLflow tracking server URI
    MLFLOW_EXPERIMENT_ID: MLflow experiment ID

Example with MLflow:
    export MLFLOW_TRACKING_URI="https://your-mlflow-server"
    export MLFLOW_EXPERIMENT_ID="123456"
    train-ellipse
"""

import argparse
import os
import platform
import random

import cv2
import matplotlib
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ellipse import (
    EllipseRegressionLoss,
    EllipseRegressionNet,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MAX_RADIUS,
    compute_center_error,
    compute_iou_with_gt_mask_gpu,
    compute_radius_error,
    denormalize_ellipse_params,
    get_nparams,
    render_ellipse_mask,
    total_metric,
)


HF_DATASET_REPO = "Conner/sddec25-01"


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
        return Image.fromarray(cv2.GaussianBlur(np.array(img), (7, 7), sigma_value))


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


class IrisDataset(Dataset):
    def __init__(self, hf_dataset, split="train", transform=None):
        self.transform = transform
        self.split = split
        self.dataset = hf_dataset
        self.has_ellipse_params = all(
            k in hf_dataset.column_names for k in ("cx", "cy", "rx", "ry")
        )

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

        # Use precomputed ellipse parameters from the dataset
        cx_norm = sample["cx"]
        cy_norm = sample["cy"]
        rx_norm = sample["rx"]
        ry_norm = sample["ry"]

        ellipse_params = torch.tensor(
            [cx_norm, cy_norm, rx_norm, ry_norm], dtype=torch.float32
        )

        # Image is already preprocessed by precompute.py
        pilimg = image

        # Stochastic augmentations still apply for training
        if self.transform is not None and self.split == "train":
            if random.random() < 0.2:
                pilimg = Line_augment()(np.array(pilimg))
            if random.random() < 0.2:
                pilimg = Gaussian_blur()(np.array(pilimg))

        img = np.array(np.uint8(pilimg))
        img = Image.fromarray(img)
        label_pil = Image.fromarray(label)

        flipped = False
        if self.transform is not None:
            if self.split == "train":
                img, label_pil = RandomHorizontalFlip()(img, label_pil)

                if np.array(label_pil)[0, 0] != label[0, 0]:
                    flipped = True
                    spatial_weights = np.fliplr(spatial_weights).copy()
                    dist_map = np.flip(dist_map, axis=2).copy()

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


def create_training_plots(train_metrics, valid_metrics, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)

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
    use_channels_last,
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

            if use_channels_last:
                single_img = single_img.to(memory_format=torch.channels_last)

            output = model(single_img)

            pred_params = output[0].cpu().numpy()
            pred_cx, pred_cy, pred_rx, pred_ry = denormalize_ellipse_params(
                pred_params[0], pred_params[1], pred_params[2], pred_params[3]
            )

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


def save_model_checkpoint(model, output_path):

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ellipse regression model for pupil detection (local version)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for training",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        type=float,
        default=1e-3,
        dest="learning_rate",
        help="Learning rate",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints",
        help="Directory for model checkpoints and plots",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for training (auto will detect CUDA availability)",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loader workers",
    )

    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging",
    )

    parser.add_argument(
        "--channel-size",
        type=int,
        default=32,
        help="Base channel size for the model",
    )
    parser.add_argument(
        "--dropout-prob",
        type=float,
        default=0.2,
        help="Dropout probability",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 80)
    print("Ellipse Regression Training (Local)")
    print("=" * 80)
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.manual_seed(12)

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        _ = torch.manual_seed(12)

    use_mlflow = not args.no_mlflow
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    mlflow_experiment_id = os.environ.get("MLFLOW_EXPERIMENT_ID")

    if use_mlflow:
        if mlflow_tracking_uri and mlflow_experiment_id:
            try:
                import mlflow

                mlflow.set_tracking_uri(mlflow_tracking_uri)
                mlflow.set_experiment(experiment_id=mlflow_experiment_id)
                print(f"MLflow configured with experiment ID: {mlflow_experiment_id}")
            except ImportError:
                print("WARNING: MLflow not installed. Disabling MLflow logging.")
                use_mlflow = False
        else:
            print("WARNING: MLFLOW_TRACKING_URI and/or MLFLOW_EXPERIMENT_ID not set.")
            print("         MLflow logging is disabled.")
            use_mlflow = False

    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    print(f"Plots directory: {plots_dir}")

    print(f"Loading dataset from HuggingFace: {HF_DATASET_REPO}")
    print("First run downloads data, subsequent runs use HuggingFace cache.")
    hf_dataset = load_dataset(HF_DATASET_REPO)

    print(f"Train samples: {len(hf_dataset['train'])}")
    print(f"Validation samples: {len(hf_dataset['validation'])}")

    print("\n" + "=" * 80)
    print("Initializing model and training setup")
    print("=" * 80)

    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    NUM_WORKERS = args.num_workers
    GRADIENT_ACCUMULATION_STEPS = 1
    EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS

    model = EllipseRegressionNet(
        in_channels=1,
        channel_size=args.channel_size,
        dropout=True,
        prob=args.dropout_prob,
    ).to(device)

    nparams = get_nparams(model)
    print(f"N parameters: {nparams:,}")

    USE_CHANNELS_LAST = device.type == "cuda"
    if USE_CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)
        print("Model converted to channels_last memory format")

    # torch.compile disabled - requires Triton which needs /sbin/ldconfig
    use_torch_compile = False

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    use_amp = device.type == "cuda"
    if use_amp:
        print("Mixed precision training (AMP) enabled")

    print(f"\nVerifying forward pass with batch_size={BATCH_SIZE} (AMP={use_amp})...")
    with torch.no_grad():
        test_input = torch.randn(BATCH_SIZE, 1, IMAGE_HEIGHT, IMAGE_WIDTH).to(device)
        if USE_CHANNELS_LAST:
            test_input = test_input.to(memory_format=torch.channels_last)
        with torch.amp.autocast("cuda", enabled=use_amp):
            test_output = model(test_input)
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {test_output.shape}")
        assert test_output.shape == (
            BATCH_SIZE,
            4,
        ), f"Output shape mismatch: expected ({BATCH_SIZE}, 4), got {test_output.shape}"
        print("Forward pass verification: PASSED")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5)

    criterion = EllipseRegressionLoss(
        center_weight=1.0,
        radius_weight=1.0,
        iou_weight=0.5,
    )

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
    print(f"  torch.compile(): {use_torch_compile}")
    print(f"  Channels Last Format: {USE_CHANNELS_LAST}")
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
        "image_size": "640x400",
        "num_classes": 2,
        "class_names": "background,pupil",
    }

    model_details = {
        "architecture": "EllipseRegressionNet",
        "input_channels": 1,
        "output_params": 4,
        "output_description": "cx,cy,rx,ry",
        "channel_size": args.channel_size,
        "dropout": True,
        "dropout_prob": args.dropout_prob,
        "down_blocks": 4,
        "bottleneck_size": "80x50",
    }

    augmentation_settings = {
        "horizontal_flip": 0.5,
        "line_augment": 0.2,
        "gaussian_blur": 0.2,
    }

    print("\n" + "=" * 80)
    print("Starting training")
    print("=" * 80)

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

    if use_mlflow:
        import mlflow

        mlflow_context = mlflow.start_run(run_name="ellipse-regression-training-local")
    else:
        from contextlib import nullcontext

        mlflow_context = nullcontext()

    with mlflow_context as run:
        if use_mlflow:
            mlflow.set_tags(
                {
                    "model_type": "EllipseRegressionNet",
                    "task": "ellipse_regression",
                    "dataset": "OpenEDS",
                    "framework": "PyTorch",
                    "optimizer": "Adam",
                    "scheduler": "ReduceLROnPlateau",
                    "version": "v1.0-local",
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
                    "torch_compile": use_torch_compile,
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

            _ = model.train()

            # All accumulators on GPU - no CPU transfers during training
            train_loss_sum = torch.tensor(0.0, device=device)
            train_center_loss_sum = torch.tensor(0.0, device=device)
            train_radius_loss_sum = torch.tensor(0.0, device=device)
            train_center_error_sum = torch.tensor(0.0, device=device)
            train_radius_error_sum = torch.tensor(0.0, device=device)
            train_iou_sum = torch.tensor(0.0, device=device)
            train_bg_iou_sum = torch.tensor(0.0, device=device)
            train_pupil_iou_sum = torch.tensor(0.0, device=device)
            train_batch_count = 0

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

                with torch.amp.autocast("cuda"):
                    output = model(data)
                    total_loss, center_loss, radius_loss = criterion(
                        output, target_params
                    )
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Accumulate all metrics on GPU (no .item() calls)
                train_loss_sum += total_loss.detach()
                train_center_loss_sum += center_loss.detach()
                train_radius_loss_sum += radius_loss.detach()
                train_center_error_sum += compute_center_error(output, target_params)
                train_radius_error_sum += compute_radius_error(output, target_params)

                miou, bg_iou, pupil_iou = compute_iou_with_gt_mask_gpu(
                    output, target_labels, device
                )
                train_iou_sum += miou
                train_bg_iou_sum += bg_iou
                train_pupil_iou_sum += pupil_iou
                train_batch_count += 1

            # Single CPU transfer at end of epoch
            loss_train = (train_loss_sum / train_batch_count).item()
            center_loss_train = (train_center_loss_sum / train_batch_count).item()
            radius_loss_train = (train_radius_loss_sum / train_batch_count).item()
            center_error_train = (train_center_error_sum / train_batch_count).item()
            radius_error_train = (train_radius_error_sum / train_batch_count).item()
            miou_train = (train_iou_sum / train_batch_count).item()
            bg_iou_train = (train_bg_iou_sum / train_batch_count).item()
            pupil_iou_train = (train_pupil_iou_sum / train_batch_count).item()

            train_metrics["loss"].append(loss_train)
            train_metrics["iou"].append(miou_train)
            train_metrics["center_loss"].append(center_loss_train)
            train_metrics["radius_loss"].append(radius_loss_train)
            train_metrics["center_error"].append(center_error_train)
            train_metrics["radius_error"].append(radius_error_train)
            train_metrics["lr"].append(optimizer.param_groups[0]["lr"])
            train_metrics["background_iou"].append(bg_iou_train)
            train_metrics["pupil_iou"].append(pupil_iou_train)

            _ = model.eval()

            # All accumulators on GPU - no CPU transfers during validation
            valid_loss_sum = torch.tensor(0.0, device=device)
            valid_center_loss_sum = torch.tensor(0.0, device=device)
            valid_radius_loss_sum = torch.tensor(0.0, device=device)
            valid_center_error_sum = torch.tensor(0.0, device=device)
            valid_radius_error_sum = torch.tensor(0.0, device=device)
            valid_iou_sum = torch.tensor(0.0, device=device)
            valid_bg_iou_sum = torch.tensor(0.0, device=device)
            valid_pupil_iou_sum = torch.tensor(0.0, device=device)
            valid_batch_count = 0

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

                    with torch.amp.autocast("cuda"):
                        output = model(data)
                        total_loss, center_loss, radius_loss = criterion(
                            output, target_params, compute_iou=False
                        )

                    # Accumulate all metrics on GPU (no .item() calls)
                    valid_loss_sum += total_loss.detach()
                    valid_center_loss_sum += center_loss.detach()
                    valid_radius_loss_sum += radius_loss.detach()
                    valid_center_error_sum += compute_center_error(output, target_params)
                    valid_radius_error_sum += compute_radius_error(output, target_params)

                    miou, bg_iou, pupil_iou = compute_iou_with_gt_mask_gpu(
                        output, target_labels, device
                    )
                    valid_iou_sum += miou
                    valid_bg_iou_sum += bg_iou
                    valid_pupil_iou_sum += pupil_iou
                    valid_batch_count += 1

            # Single CPU transfer at end of validation
            loss_valid = (valid_loss_sum / valid_batch_count).item()
            center_loss_valid = (valid_center_loss_sum / valid_batch_count).item()
            radius_loss_valid = (valid_radius_loss_sum / valid_batch_count).item()
            center_error_valid = (valid_center_error_sum / valid_batch_count).item()
            radius_error_valid = (valid_radius_error_sum / valid_batch_count).item()
            miou_valid = (valid_iou_sum / valid_batch_count).item()
            bg_iou_valid = (valid_bg_iou_sum / valid_batch_count).item()
            pupil_iou_valid = (valid_pupil_iou_sum / valid_batch_count).item()

            valid_metrics["loss"].append(loss_valid)
            valid_metrics["iou"].append(miou_valid)
            valid_metrics["center_loss"].append(center_loss_valid)
            valid_metrics["radius_loss"].append(radius_loss_valid)
            valid_metrics["center_error"].append(center_error_valid)
            valid_metrics["radius_error"].append(radius_error_valid)
            valid_metrics["background_iou"].append(bg_iou_valid)
            valid_metrics["pupil_iou"].append(pupil_iou_valid)

            if use_mlflow:
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

            if miou_valid > best_valid_iou:
                best_valid_iou = miou_valid
                best_epoch = epoch + 1
                best_model_path = os.path.join(args.output_dir, "best_ellipse_model.pt")
                save_model_checkpoint(model, best_model_path)
                if use_mlflow:
                    mlflow.log_artifact(best_model_path)
                    mlflow.log_metric("best_valid_iou", best_valid_iou, step=epoch)
                print(f"New best model! Valid mIoU: {best_valid_iou:.4f}")

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print("Generating sample predictions...")
                pred_vis_path = os.path.join(
                    plots_dir, f"predictions_epoch_{epoch+1}.png"
                )
                create_prediction_visualization(
                    model,
                    validloader,
                    device,
                    USE_CHANNELS_LAST,
                    num_samples=4,
                    save_path=pred_vis_path,
                )
                if use_mlflow:
                    mlflow.log_artifact(pred_vis_path)
                print(f"Sample predictions saved: {pred_vis_path}")

            if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
                print("Generating training curves...")
                plot_paths = create_training_plots(
                    train_metrics,
                    valid_metrics,
                    save_dir=plots_dir,
                )
                if use_mlflow:
                    for plot_path in plot_paths.values():
                        if plot_path is not None:
                            mlflow.log_artifact(plot_path)
                print("Training curves saved")

            if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
                checkpoint_path = os.path.join(
                    args.output_dir, f"ellipse_model_epoch_{epoch+1}.pt"
                )
                save_model_checkpoint(model, checkpoint_path)
                if use_mlflow:
                    mlflow.log_artifact(checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")

        if use_mlflow:
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
        print(f"Output directory: {args.output_dir}")
        if use_mlflow:
            print(f"MLflow run ID: {run.info.run_id}")
        print("=" * 80)


if __name__ == "__main__":
    main()
