#!/usr/bin/env python3
"""
Local training script for TinyEfficientViTSeg on OpenEDS dataset.

This script trains the TinyEfficientViTSeg model for pupil segmentation
using the precomputed OpenEDS dataset from HuggingFace. It runs locally
without any cloud dependencies (Modal, etc.).

Requirements:
- PyTorch with CUDA support (optional but recommended)
- HuggingFace datasets
- model.py in the same directory

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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

# Import model and loss from local model.py
from model import TinyEfficientViTSeg, CombinedLoss

# ============================================================================
# Constants
# ============================================================================

IMAGE_HEIGHT = 400
IMAGE_WIDTH = 640
HF_DATASET_REPO = "Conner/openeds-precomputed"

# ============================================================================
# Helper Functions (from training/train.py)
# ============================================================================


def compute_iou_tensors(predictions, targets, num_classes=2):
    """
    Compute per-class intersection and union for IoU calculation.

    Args:
        predictions: Predicted class labels (B, H, W)
        targets: Ground truth labels (B, H, W)
        num_classes: Number of classes

    Returns:
        intersection: Per-class intersection counts (num_classes,)
        union: Per-class union counts (num_classes,)
    """
    intersection = torch.zeros(num_classes, device=predictions.device)
    union = torch.zeros(num_classes, device=predictions.device)

    for c in range(num_classes):
        pred_c = predictions == c
        target_c = targets == c
        intersection[c] = torch.logical_and(pred_c, target_c).sum().float()
        union[c] = torch.logical_or(pred_c, target_c).sum().float()

    return intersection, union


def get_predictions(output):
    """
    Convert model logits to predicted class labels.

    Args:
        output: Model output logits (B, C, H, W)

    Returns:
        Predicted class labels (B, H, W)
    """
    bs, _, h, w = output.size()
    _, indices = output.max(1)
    indices = indices.view(bs, h, w)
    return indices


def get_nparams(model):
    """
    Count total trainable parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_iou_cpu(intersection_vals, union_vals):
    """
    Calculate mean IoU from intersection and union values on CPU.

    This function operates on CPU values (no GPU sync required).

    Args:
        intersection_vals: List of intersection counts [class0, class1]
        union_vals: List of union counts [class0, class1]

    Returns:
        Mean IoU across all classes
    """
    iou_per_class = [
        intersection_vals[i] / max(union_vals[i], 1.0)
        for i in range(len(intersection_vals))
    ]
    return sum(iou_per_class) / len(iou_per_class)


# ============================================================================
# Dataset Class
# ============================================================================


class MaskToTensor:
    """Convert PIL Image mask to long tensor."""

    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int64))


class IrisDataset(Dataset):
    """
    Dataset class for OpenEDS precomputed dataset.
    Loads raw images with minimal preprocessing (only normalization).
    """

    def __init__(self, hf_dataset, transform=None):
        """
        Initialize dataset.

        Args:
            hf_dataset: HuggingFace dataset split
            transform: Torchvision transforms for images
        """
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get a single sample.

        Returns:
            img: Transformed image tensor (1, H, W)
            label_tensor: Ground truth mask (H, W)
            spatial_weights: Spatial weighting map (H, W)
            dist_map: Distance map for surface loss (2, H, W)
        """
        sample = self.dataset[idx]

        # Reshape flat arrays to images
        image = np.array(sample["image"], dtype=np.uint8).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
        label = np.array(sample["label"], dtype=np.uint8).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
        spatial_weights = torch.from_numpy(
            np.array(sample["spatial_weights"], dtype=np.float32).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
        )
        dist_map = torch.from_numpy(
            np.array(sample["dist_map"], dtype=np.float32).reshape(2, IMAGE_HEIGHT, IMAGE_WIDTH)
        )

        # Convert to PIL for transforms
        img = Image.fromarray(image)
        if self.transform:
            img = self.transform(img)

        # Convert label to tensor
        label_tensor = MaskToTensor()(Image.fromarray(label))

        return img, label_tensor, spatial_weights, dist_map


def train(args):
    """
    Main training loop.

    Args:
        args: Command line arguments
    """
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    print(f"\nLoading dataset from {HF_DATASET_REPO}...")
    print("(First run will download ~3GB, subsequent runs use cached data)")

    hf_dataset = load_dataset(HF_DATASET_REPO)

    print(f"Train samples: {len(hf_dataset['train'])}")
    print(f"Validation samples: {len(hf_dataset['validation'])}")

    # Transforms - only normalization (no preprocessing)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # Normalize grayscale to [-1, 1]
    ])

    train_dataset = IrisDataset(hf_dataset['train'], transform=transform)
    valid_dataset = IrisDataset(hf_dataset['validation'], transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # ========================================================================
    # Initialize Model
    # ========================================================================

    print("\nInitializing TinyEfficientViTSeg model...")

    model = TinyEfficientViTSeg(
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
    ).to(device)

    nparams = get_nparams(model)
    print(f"Model parameters: {nparams:,}")

    # ========================================================================
    # Loss, Optimizer, Scheduler
    # ========================================================================

    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=5,
        factor=0.5,
    )

    # Mixed precision training setup
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # ========================================================================
    # Alpha Scheduling (for loss weighting)
    # ========================================================================

    # Alpha decays from 1 to 0 over 125 epochs
    # Controls balance between dice loss (alpha) and surface loss (1-alpha)
    alpha_schedule = np.zeros(args.epochs)
    alpha_schedule[0:min(125, args.epochs)] = 1 - np.arange(1, min(125, args.epochs) + 1) / min(125, args.epochs)
    if args.epochs > 125:
        alpha_schedule[125:] = 0

    # ========================================================================
    # Training Loop
    # ========================================================================

    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_iou = 0.0

    for epoch in range(args.epochs):
        alpha = alpha_schedule[epoch]

        # ====================================================================
        # Training Phase
        # ====================================================================

        model.train()
        train_loss = torch.zeros(1, device=device)
        train_ce_loss = torch.zeros(1, device=device)
        train_dice_loss = torch.zeros(1, device=device)
        train_surface_loss = torch.zeros(1, device=device)
        train_intersection = torch.zeros(2, device=device)
        train_union = torch.zeros(2, device=device)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")

        for images, labels, spatial_weights, dist_maps in pbar:
            # Move to device (non_blocking for async CPU->GPU transfer)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            spatial_weights = spatial_weights.to(device, non_blocking=True)
            dist_maps = dist_maps.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Forward pass with mixed precision
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(images)
                loss, ce_loss, dice_loss, surface_loss = criterion(
                    outputs, labels, spatial_weights, dist_maps, alpha
                )

            # Backward pass with gradient scaling
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            # Accumulate losses (stay on GPU, no sync)
            train_loss += loss.detach()
            train_ce_loss += ce_loss.detach()
            train_dice_loss += dice_loss.detach()
            train_surface_loss += surface_loss.detach()

            # Compute IoU
            preds = get_predictions(outputs)
            inter, uni = compute_iou_tensors(preds, labels)
            train_intersection += inter
            train_union += uni

            # Update progress bar (no GPU sync)
            pbar.set_postfix({'alpha': f'{alpha:.3f}'})

        # Store batch count for later metrics calculation (no GPU sync yet)
        n_train_batches = len(train_loader)

        # ====================================================================
        # Validation Phase
        # ====================================================================

        model.eval()
        valid_loss = torch.zeros(1, device=device)
        valid_ce_loss = torch.zeros(1, device=device)
        valid_dice_loss = torch.zeros(1, device=device)
        valid_surface_loss = torch.zeros(1, device=device)
        valid_intersection = torch.zeros(2, device=device)
        valid_union = torch.zeros(2, device=device)

        with torch.no_grad():
            pbar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Valid]")

            for images, labels, spatial_weights, dist_maps in pbar:
                # Move to device (non_blocking for async CPU->GPU transfer)
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                spatial_weights = spatial_weights.to(device, non_blocking=True)
                dist_maps = dist_maps.to(device, non_blocking=True)

                # Forward pass
                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = model(images)
                    loss, ce_loss, dice_loss, surface_loss = criterion(
                        outputs, labels, spatial_weights, dist_maps, alpha
                    )

                # Accumulate losses (stay on GPU, no sync)
                valid_loss += loss.detach()
                valid_ce_loss += ce_loss.detach()
                valid_dice_loss += dice_loss.detach()
                valid_surface_loss += surface_loss.detach()

                # Compute IoU
                preds = get_predictions(outputs)
                inter, uni = compute_iou_tensors(preds, labels)
                valid_intersection += inter
                valid_union += uni

                # No GPU sync in progress bar

        # ====================================================================
        # Single Batched GPU->CPU Transfer (THE ONLY SYNC POINT PER EPOCH)
        # ====================================================================

        # Batch ALL epoch metrics into one tensor: 16 values total
        n_valid_batches = len(valid_loader)
        all_metrics_gpu = torch.cat([
            # Train losses (4 values)
            train_loss / n_train_batches,
            train_ce_loss / n_train_batches,
            train_dice_loss / n_train_batches,
            train_surface_loss / n_train_batches,
            # Valid losses (4 values)
            valid_loss / n_valid_batches,
            valid_ce_loss / n_valid_batches,
            valid_dice_loss / n_valid_batches,
            valid_surface_loss / n_valid_batches,
            # Train IoU components (4 values: 2 intersection + 2 union)
            train_intersection,
            train_union,
            # Valid IoU components (4 values: 2 intersection + 2 union)
            valid_intersection,
            valid_union,
        ])

        # Single GPU->CPU transfer for entire epoch
        all_metrics = all_metrics_gpu.tolist()

        # Unpack metrics
        (
            train_loss_val, train_ce_val, train_dice_val, train_surface_val,
            valid_loss_val, valid_ce_val, valid_dice_val, valid_surface_val,
            train_inter_0, train_inter_1,
            train_union_0, train_union_1,
            valid_inter_0, valid_inter_1,
            valid_union_0, valid_union_1,
        ) = all_metrics

        # Compute IoU on CPU (no GPU sync)
        train_iou = compute_iou_cpu(
            [train_inter_0, train_inter_1],
            [train_union_0, train_union_1]
        )
        valid_iou = compute_iou_cpu(
            [valid_inter_0, valid_inter_1],
            [valid_union_0, valid_union_1]
        )

        # ====================================================================
        # Learning Rate Scheduling
        # ====================================================================

        scheduler.step(valid_loss_val)
        current_lr = optimizer.param_groups[0]['lr']

        # ====================================================================
        # Logging
        # ====================================================================

        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss_val:.4f} | Valid Loss: {valid_loss_val:.4f}")
        print(f"  Train IoU:  {train_iou:.4f} | Valid IoU:  {valid_iou:.4f}")
        print(f"  CE Loss:    {train_ce_val:.4f} | {valid_ce_val:.4f}")
        print(f"  Dice Loss:  {train_dice_val:.4f} | {valid_dice_val:.4f}")
        print(f"  Surf Loss:  {train_surface_val:.4f} | {valid_surface_val:.4f}")
        print(f"  LR: {current_lr:.6f} | Alpha: {alpha:.4f}")

        # ====================================================================
        # Save Best Model
        # ====================================================================

        if valid_iou > best_iou:
            best_iou = valid_iou

            # Save PyTorch checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'valid_iou': valid_iou,
                'valid_loss': valid_loss_val,
            }, f"{args.checkpoint_dir}/best_model.pth")

            print(f"  >> Saved best model with IoU={best_iou:.4f}")

        # Save periodic checkpoints
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'valid_iou': valid_iou,
                'valid_loss': valid_loss_val,
            }, f"{args.checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth")
            print(f"  >> Saved checkpoint at epoch {epoch+1}")

    # ========================================================================
    # Training Complete
    # ========================================================================

    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Best validation IoU: {best_iou:.4f}")
    print(f"\nPyTorch checkpoint saved to: {args.checkpoint_dir}/best_model.pth")


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Parse arguments and start training."""
    parser = argparse.ArgumentParser(
        description='Train TinyEfficientViTSeg for pupil segmentation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training and validation')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # Data loading
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of dataloader workers')

    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')

    args = parser.parse_args()

    # Print configuration
    print("="*80)
    print("TinyEfficientViTSeg Training Configuration")
    print("="*80)
    print(f"Dataset: {HF_DATASET_REPO}")
    print(f"Image size: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Num workers: {args.num_workers}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Random seed: {args.seed}")
    print("="*80)

    # Start training
    try:
        train(args)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        raise


if __name__ == "__main__":
    main()
