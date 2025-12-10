#!/usr/bin/env python3
"""
Training script for NSA Pupil Segmentation on OpenEDS dataset.
This script trains the NSAPupilSeg model (Native Sparse Attention) for pupil
segmentation using the precomputed OpenEDS dataset from HuggingFace.
NSA Key Features:
- Token Compression: Global coarse-grained context
- Token Selection: Fine-grained focus on important regions (pupil)
- Sliding Window: Local context for precise boundaries
- Gated Aggregation: Learned combination of attention paths
Dataset: Conner/openeds-precomputed (HuggingFace)
Image size: 640x400 grayscale
Classes: 2 (background=0, pupil=1)
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (
    Dataset,
    DataLoader,
)
from tqdm import tqdm
from datasets import load_dataset
import mlflow
import kornia.augmentation as K
from kornia.augmentation import AugmentationSequential

# MLflow for experiment tracking
try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
# Import NSA model and loss
from nsa import (
    NSAPupilSeg,
    CombinedLoss,
    create_nsa_pupil_seg,
)

# Required MLflow environment variables
MLFLOW_ENV_VARS = [
    "DATABRICKS_TOKEN",
    "DATABRICKS_HOST",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_REGISTRY_URI",
    "MLFLOW_EXPERIMENT_ID",
]


def setup_mlflow():
    """Configure MLflow with Databricks credentials from environment variables.
    Raises:
        EnvironmentError: If any required environment variable is not set.
    """
    missing_vars = [
        var
        for var in MLFLOW_ENV_VARS
        if not os.environ.get(var)
    ]
    if missing_vars:
        raise EnvironmentError(
            f"Missing required MLflow environment variables: {', '.join(missing_vars)}\n"
            f"Please set: {', '.join(MLFLOW_ENV_VARS)}"
        )
    mlflow.set_tracking_uri(
        os.environ[
            "MLFLOW_TRACKING_URI"
        ]
    )
    mlflow.set_experiment(
        experiment_id=os.environ[
            "MLFLOW_EXPERIMENT_ID"
        ]
    )
    print(
        f"MLflow configured with tracking URI: {os.environ['MLFLOW_TRACKING_URI']}"
    )
    print(
        f"MLflow experiment ID: {os.environ['MLFLOW_EXPERIMENT_ID']}"
    )


# Constants
IMAGE_HEIGHT = 400
IMAGE_WIDTH = 640
HF_DATASET_REPO = (
    "Conner/openeds-precomputed"
)


class GPUAugmentation(nn.Module):
    """GPU-native augmentation using Kornia."""

    def __init__(self, training: bool = True):
        super().__init__()
        self.training_mode = training

        if training:
            # Geometric augmentations (applied to image AND all masks)
            self.geometric = AugmentationSequential(
                K.RandomHorizontalFlip(p=0.5),
                K.RandomRotation(degrees=10, p=0.3),
                K.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), p=0.2),
                data_keys=["input", "mask"],
                same_on_batch=False,
            )

            # Intensity augmentations (image only)
            self.intensity = nn.Sequential(
                K.RandomBrightness(brightness=(0.9, 1.1), p=0.3),
                K.RandomContrast(contrast=(0.9, 1.1), p=0.3),
                K.RandomGaussianNoise(mean=0.0, std=0.05, p=0.2),
                K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.5), p=0.1),
            )
        else:
            self.geometric = None
            self.intensity = None

    def forward(self, image, label, spatial_weights, dist_map, eye_mask, eye_weight):
        """
        Apply augmentation to batch on GPU.

        Args:
            image: (B, 1, H, W) - grayscale image
            label: (B, H, W) - segmentation mask (long tensor)
            spatial_weights: (B, H, W) - float weights
            dist_map: (B, 2, H, W) - distance maps
            eye_mask: (B, H, W) - eye region mask
            eye_weight: (B, H, W) - eye region weights

        Returns:
            Augmented tensors with same shapes
        """
        if not self.training_mode or self.geometric is None:
            return image, label, spatial_weights, dist_map, eye_mask, eye_weight

        B, _, H, W = image.shape

        # Stack all masks into a single tensor for consistent geometric transform
        # label needs to be float for interpolation, will convert back
        label_float = label.float().unsqueeze(1)  # (B, 1, H, W)
        spatial_weights_4d = spatial_weights.unsqueeze(1)  # (B, 1, H, W)
        eye_mask_float = eye_mask.float().unsqueeze(1)  # (B, 1, H, W)
        eye_weight_4d = eye_weight.unsqueeze(1)  # (B, 1, H, W)
        # dist_map is already (B, 2, H, W)

        # Concatenate all masks: (B, 6, H, W)
        all_masks = torch.cat([
            label_float, spatial_weights_4d, eye_mask_float, eye_weight_4d, dist_map
        ], dim=1)

        # Apply geometric transforms to image and all masks together
        image_aug, masks_aug = self.geometric(image, all_masks)

        # Split masks back
        label_aug = masks_aug[:, 0:1, :, :].squeeze(1).round().long()  # Back to long
        spatial_weights_aug = masks_aug[:, 1:2, :, :].squeeze(1)
        eye_mask_aug = masks_aug[:, 2:3, :, :].squeeze(1).round().long()
        eye_weight_aug = masks_aug[:, 3:4, :, :].squeeze(1)
        dist_map_aug = masks_aug[:, 4:6, :, :]

        # Apply intensity augmentation (image only)
        image_aug = self.intensity(image_aug)

        return image_aug, label_aug, spatial_weights_aug, dist_map_aug, eye_mask_aug, eye_weight_aug


# Helper Functions
def is_mlflow_configured():
    """Check if MLflow is available and properly configured via environment variables."""
    if not MLFLOW_AVAILABLE:
        return False
    # Check for required environment variables
    required_vars = [
        "DATABRICKS_TOKEN",
        "DATABRICKS_HOST",
        "MLFLOW_TRACKING_URI",
    ]
    for var in required_vars:
        if not os.environ.get(var):
            return False
    return True


def compute_iou_tensors(
    predictions, targets, num_classes=2
):
    """
    Compute per-class intersection and union for IoU calculation.
    """
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


def get_predictions(output):
    """Convert model logits to predicted class labels."""
    bs, _, h, w = output.size()
    _, indices = output.max(1)
    indices = indices.view(bs, h, w)
    return indices


def get_nparams(model):
    """Count total trainable parameters in model."""
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def compute_iou_cpu(
    intersection_vals, union_vals
):
    """Calculate mean IoU from intersection and union values on CPU."""
    iou_per_class = [
        intersection_vals[i]
        / max(union_vals[i], 1.0)
        for i in range(
            len(intersection_vals)
        )
    ]
    return sum(iou_per_class) / len(
        iou_per_class
    )


# Dataset Class
class IrisDataset(Dataset):
    """Dataset class for OpenEDS precomputed dataset."""

    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
        self.normalize_mean = 0.5
        self.normalize_std = 0.5

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # Reshape flat arrays to images
        image = np.array(sample["image"], dtype=np.uint8).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
        label = np.array(sample["label"], dtype=np.uint8).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
        spatial_weights = np.array(sample["spatial_weights"], dtype=np.float32).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
        dist_map = np.array(sample["dist_map"], dtype=np.float32).reshape(2, IMAGE_HEIGHT, IMAGE_WIDTH)
        eye_mask = np.array(sample["eye_mask"], dtype=np.uint8).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
        eye_weight = np.array(sample["eye_weight"], dtype=np.float32).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)

        # Normalize image and convert to tensor
        image = image.astype(np.float32) / 255.0
        image = (image - self.normalize_mean) / self.normalize_std
        img_tensor = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)

        # Convert other arrays to tensors
        label_tensor = torch.from_numpy(label.astype(np.int64))
        spatial_weights_tensor = torch.from_numpy(spatial_weights)
        dist_map_tensor = torch.from_numpy(dist_map)
        eye_mask_tensor = torch.from_numpy(eye_mask.astype(np.int64))
        eye_weight_tensor = torch.from_numpy(eye_weight)

        return (img_tensor, label_tensor, spatial_weights_tensor, dist_map_tensor,
                eye_mask_tensor, eye_weight_tensor)


# Training Function
def train(args):
    """Main training loop."""
    # Setup MLflow before training
    setup_mlflow()
    # Device setup
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
        torch.cuda.manual_seed(
            args.seed
        )
    else:
        torch.manual_seed(args.seed)
    # Load dataset
    print(
        f"\nLoading dataset from {HF_DATASET_REPO}..."
    )
    print(
        "(First run will download ~3GB, subsequent runs use cached data)"
    )
    hf_dataset = load_dataset(
        HF_DATASET_REPO
    )
    num_train_samples = len(
        hf_dataset["train"]
    )
    num_valid_samples = len(
        hf_dataset["validation"]
    )
    print(
        f"Train samples: {num_train_samples}"
    )
    print(
        f"Validation samples: {num_valid_samples}"
    )
    # Create datasets (no CPU augmentation - augmentation happens on GPU)
    train_dataset = IrisDataset(hf_dataset["train"])
    valid_dataset = IrisDataset(hf_dataset["validation"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    # Initialize NSA Model
    print(
        f"\nInitializing NSAPupilSeg model (size={args.model_size})..."
    )
    model = create_nsa_pupil_seg(
        size=args.model_size,
        in_channels=1,
        num_classes=2,
    ).to(device)
    nparams = get_nparams(model)
    print(
        f"Model parameters: {nparams:,}"
    )

    # Initialize GPU augmentation modules
    train_augment = GPUAugmentation(training=True).to(device)
    val_augment = GPUAugmentation(training=False).to(device)

    use_mlflow = is_mlflow_configured()
    mlflow_run = None
    print(
        "\nMLflow is configured. Starting experiment tracking..."
    )
    # Configure MLflow tracking URI (Databricks or custom)
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI"
    )
    if tracking_uri:
        mlflow.set_tracking_uri(
            tracking_uri
        )
        print(
            f"MLflow tracking URI: {tracking_uri}"
        )
    # Configure registry URI if provided
    registry_uri = os.environ.get(
        "MLFLOW_REGISTRY_URI"
    )
    if registry_uri:
        mlflow.set_registry_uri(
            registry_uri
        )
        print(
            f"MLflow registry URI: {registry_uri}"
        )
    # Set experiment if MLFLOW_EXPERIMENT_ID is provided
    experiment_id = os.environ.get(
        "MLFLOW_EXPERIMENT_ID"
    )
    if experiment_id:
        mlflow.set_experiment(
            experiment_id=experiment_id
        )
    # Start MLflow run
    mlflow.end_run()
    mlflow_run = mlflow.start_run()
    # Log all hyperparameters
    mlflow.log_params(
        {
            "model_size": args.model_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "num_workers": args.num_workers,
            "num_parameters": nparams,
            "image_height": IMAGE_HEIGHT,
            "image_width": IMAGE_WIDTH,
            "dataset": HF_DATASET_REPO,
            "num_train_samples": num_train_samples,
            "num_valid_samples": num_valid_samples,
            "device": str(device),
            "augmentation": "kornia_gpu",
        }
    )
    print(
        f"MLflow run started: {mlflow_run.info.run_id}"
    )
    # Loss, Optimizer, Scheduler
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )
    # Mixed precision training
    use_amp = torch.cuda.is_available()
    scaler = (
        torch.amp.GradScaler("cuda")
        if use_amp
        else None
    )
    # Resume from Checkpoint
    start_epoch = 0
    best_iou = 0.0
    if args.resume:
        print(
            f"\nResuming from checkpoint: {args.resume}"
        )
        checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )
        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )
        scheduler.load_state_dict(
            checkpoint[
                "scheduler_state_dict"
            ]
        )
        start_epoch = (
            checkpoint["epoch"] + 1
        )
        best_iou = checkpoint.get(
            "valid_iou", 0.0
        )
        print(
            f"Resumed from epoch {checkpoint['epoch'] + 1}, best IoU: {best_iou:.4f}"
        )
    # Alpha schedule: 1.0 -> 0.2 over 125 epochs (maintain minimum for surface loss)
    # Use Python list (not numpy) to avoid CPU numpy scalar during training
    alpha_schedule = [
        max(1.0 - i / min(125, args.epochs), 0.2)
        for i in range(args.epochs)
    ]
    # Training Loop
    print("\n" + "=" * 80)
    print("Starting NSA Training")
    print("=" * 80)
    os.makedirs(
        args.checkpoint_dir,
        exist_ok=True,
    )
    for epoch in range(
        start_epoch, args.epochs
    ):
        alpha = alpha_schedule[epoch]
        # Training Phase
        model.train()
        train_augment.train()
        train_loss = torch.zeros(1, device=device)
        train_ce_loss = torch.zeros(1, device=device)
        train_dice_loss = torch.zeros(1, device=device)
        train_surface_loss = torch.zeros(1, device=device)
        train_boundary_loss = torch.zeros(1, device=device)
        train_intersection = torch.zeros(2, device=device)
        train_union = torch.zeros(2, device=device)
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs} [Train]",
        )
        for (images, labels, spatial_weights, dist_maps, eye_masks, eye_weights) in pbar:
            # Move to GPU
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            spatial_weights = spatial_weights.to(device, non_blocking=True)
            dist_maps = dist_maps.to(device, non_blocking=True)
            eye_masks = eye_masks.to(device, non_blocking=True)
            eye_weights = eye_weights.to(device, non_blocking=True)

            # Apply GPU augmentation
            images, labels, spatial_weights, dist_maps, eye_masks, eye_weights = train_augment(
                images, labels, spatial_weights, dist_maps, eye_masks, eye_weights
            )

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images)
                (loss, ce_loss, dice_loss, surface_loss, boundary_loss) = criterion(
                    outputs,
                    labels,
                    spatial_weights,
                    dist_maps,
                    alpha,
                    eye_weights,
                )
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )
                optimizer.step()
            train_loss += loss.detach()
            train_ce_loss += ce_loss.detach()
            train_dice_loss += dice_loss.detach()
            train_surface_loss += surface_loss.detach()
            train_boundary_loss += boundary_loss.detach()
            preds = get_predictions(outputs)
            inter, uni = compute_iou_tensors(preds, labels)
            train_intersection += inter
            train_union += uni
            pbar.set_postfix({"alpha": f"{alpha:.3f}"})
        n_train_batches = len(train_loader)
        # Validation Phase
        model.eval()
        val_augment.eval()
        valid_loss = torch.zeros(1, device=device)
        valid_ce_loss = torch.zeros(1, device=device)
        valid_dice_loss = torch.zeros(1, device=device)
        valid_surface_loss = torch.zeros(1, device=device)
        valid_boundary_loss = torch.zeros(1, device=device)
        valid_intersection = torch.zeros(2, device=device)
        valid_union = torch.zeros(2, device=device)
        with torch.no_grad():
            pbar = tqdm(
                valid_loader,
                desc=f"Epoch {epoch+1}/{args.epochs} [Valid]",
            )
            for (images, labels, spatial_weights, dist_maps, eye_masks, eye_weights) in pbar:
                # Move to GPU
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                spatial_weights = spatial_weights.to(device, non_blocking=True)
                dist_maps = dist_maps.to(device, non_blocking=True)
                eye_masks = eye_masks.to(device, non_blocking=True)
                eye_weights = eye_weights.to(device, non_blocking=True)

                # Apply validation augmentation (no-op but keeps consistent interface)
                images, labels, spatial_weights, dist_maps, eye_masks, eye_weights = val_augment(
                    images, labels, spatial_weights, dist_maps, eye_masks, eye_weights
                )

                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(images)
                    (loss, ce_loss, dice_loss, surface_loss, boundary_loss) = criterion(
                        outputs,
                        labels,
                        spatial_weights,
                        dist_maps,
                        alpha,
                        eye_weights,
                    )
                valid_loss += loss.detach()
                valid_ce_loss += ce_loss.detach()
                valid_dice_loss += dice_loss.detach()
                valid_surface_loss += surface_loss.detach()
                valid_boundary_loss += boundary_loss.detach()
                preds = get_predictions(outputs)
                inter, uni = compute_iou_tensors(preds, labels)
                valid_intersection += inter
                valid_union += uni
        # Metrics Calculation
        n_valid_batches = len(valid_loader)
        all_metrics_gpu = torch.cat([
            train_loss / n_train_batches,
            train_ce_loss / n_train_batches,
            train_dice_loss / n_train_batches,
            train_surface_loss / n_train_batches,
            train_boundary_loss / n_train_batches,
            valid_loss / n_valid_batches,
            valid_ce_loss / n_valid_batches,
            valid_dice_loss / n_valid_batches,
            valid_surface_loss / n_valid_batches,
            valid_boundary_loss / n_valid_batches,
            train_intersection,
            train_union,
            valid_intersection,
            valid_union,
        ])
        all_metrics = all_metrics_gpu.tolist()
        (
            train_loss_val,
            train_ce_val,
            train_dice_val,
            train_surface_val,
            train_boundary_val,
            valid_loss_val,
            valid_ce_val,
            valid_dice_val,
            valid_surface_val,
            valid_boundary_val,
            train_inter_0,
            train_inter_1,
            train_union_0,
            train_union_1,
            valid_inter_0,
            valid_inter_1,
            valid_union_0,
            valid_union_1,
        ) = all_metrics
        train_iou = compute_iou_cpu(
            [train_inter_0, train_inter_1],
            [train_union_0, train_union_1],
        )
        valid_iou = compute_iou_cpu(
            [valid_inter_0, valid_inter_1],
            [valid_union_0, valid_union_1],
        )
        # Learning Rate Scheduling
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        # Logging
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss_val:.4f} | Valid Loss: {valid_loss_val:.4f}")
        print(f"  Train IoU:  {train_iou:.4f} | Valid IoU:  {valid_iou:.4f}")
        print(f"  CE Loss:    {train_ce_val:.4f} | {valid_ce_val:.4f}")
        print(f"  Dice Loss:  {train_dice_val:.4f} | {valid_dice_val:.4f}")
        print(f"  Surf Loss:  {train_surface_val:.4f} | {valid_surface_val:.4f}")
        print(f"  Bound Loss: {train_boundary_val:.4f} | {valid_boundary_val:.4f}")
        print(f"  LR: {current_lr:.6f} | Alpha: {alpha:.4f}")
        # MLflow Metrics Logging
        if use_mlflow:
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
        # Save Checkpoints
        if valid_iou > best_iou:
            best_iou = valid_iou
            best_model_path = f"{args.checkpoint_dir}/best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "valid_iou": valid_iou,
                    "valid_loss": valid_loss_val,
                    "model_size": args.model_size,
                },
                best_model_path,
            )
            print(
                f"  >> Saved best model with IoU={best_iou:.4f}"
            )
            # Log best model to MLflow
            mlflow.log_metric(
                "best_iou",
                best_iou,
                step=epoch,
            )
            mlflow.log_artifact(
                best_model_path,
                artifact_path="checkpoints",
            )
        # Periodic checkpoints
        if (epoch + 1) % 5 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "valid_iou": valid_iou,
                    "valid_loss": valid_loss_val,
                    "model_size": args.model_size,
                },
                f"{args.checkpoint_dir}/nsa_{args.model_size}_checkpoint_epoch_{epoch+1}.pth",
            )
            print(
                f"  >> Saved checkpoint at epoch {epoch+1}"
            )
    # Training Complete
    print("\n" + "=" * 80)
    print("NSA Training Complete!")
    print("=" * 80)
    print(
        f"Best validation IoU: {best_iou:.4f}"
    )
    print(
        f"\nCheckpoint saved to: {args.checkpoint_dir}/best_nsa_{args.model_size}_model.pth"
    )
    # MLflow Finalization
    if use_mlflow:
        # Log final best IoU metric
        mlflow.log_metric(
            "final_best_iou", best_iou
        )
        # End the MLflow run
        mlflow.end_run()
        print(
            "\nMLflow run completed successfully."
        )


def main():
    """Parse arguments and start training."""
    parser = argparse.ArgumentParser(
        description="Train NSAPupilSeg (Native Sparse Attention) for pupil segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model configuration
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=[
            "pico",
            "nano",
            "tiny",
            "small",
            "medium",
        ],
        help="Model size configuration",
    )
    # Training hyperparameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for training and validation",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW optimizer",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    # Data loading
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers",
    )
    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints_nsa",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    args = parser.parse_args()
    # Print configuration
    print("=" * 80)
    print(
        "NSAPupilSeg (Native Sparse Attention) Training Configuration"
    )
    print("=" * 80)
    print(
        f"Model size: {args.model_size}"
    )
    print(f"Dataset: {HF_DATASET_REPO}")
    print(
        f"Image size: {IMAGE_WIDTH}x{IMAGE_HEIGHT}"
    )
    print(
        f"Batch size: {args.batch_size}"
    )
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(
        f"Weight decay: {args.weight_decay}"
    )
    print(
        f"Num workers: {args.num_workers}"
    )
    print(
        f"Checkpoint dir: {args.checkpoint_dir}"
    )
    print(
        f"Resume from: {args.resume if args.resume else 'None'}"
    )
    print(f"Random seed: {args.seed}")
    print("Augmentation: Kornia (GPU-native)")
    print("=" * 80)
    # Start training
    try:
        train(args)
    except KeyboardInterrupt:
        print(
            "\n\nTraining interrupted by user!"
        )
        # End MLflow run on interrupt
        if mlflow.active_run():
            mlflow.end_run(
                status="KILLED"
            )
            print(
                "MLflow run ended with status: KILLED"
            )
    except Exception as e:
        print(
            f"\n\nError during training: {e}"
        )
        # End MLflow run on error
        if mlflow.active_run():
            mlflow.end_run(
                status="FAILED"
            )
            print(
                "MLflow run ended with status: FAILED"
            )
        raise


if __name__ == "__main__":
    main()
