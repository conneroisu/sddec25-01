#!/usr/bin/env python3
"""
Training script for DSA Segmentation Model on OpenEDS dataset.

This script trains the DeepSeek Sparse Attention (DSA) model for pupil
segmentation using the precomputed OpenEDS dataset from HuggingFace.

Key features:
- Two-stage training (dense warm-up + sparse training) as per DSA paper
- Mixed precision training for efficiency
- Indexer alignment loss for DSA optimization
- Checkpoint resume support

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

from model import DSASegmentationModel, CombinedLoss, create_dsa_tiny, create_dsa_small, create_dsa_base

# Constants
IMAGE_HEIGHT = 400
IMAGE_WIDTH = 640
HF_DATASET_REPO = "Conner/openeds-precomputed"


def compute_iou_tensors(predictions, targets, num_classes=2):
    """Compute per-class intersection and union for IoU calculation."""
    intersection = torch.zeros(num_classes, device=predictions.device)
    union = torch.zeros(num_classes, device=predictions.device)

    for c in range(num_classes):
        pred_c = predictions == c
        target_c = targets == c
        intersection[c] = torch.logical_and(pred_c, target_c).sum().float()
        union[c] = torch.logical_or(pred_c, target_c).sum().float()

    return intersection, union


def get_predictions(output):
    """Convert model logits to predicted class labels."""
    bs, _, h, w = output.size()
    _, indices = output.max(1)
    indices = indices.view(bs, h, w)
    return indices


def get_nparams(model):
    """Count total trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_iou_cpu(intersection_vals, union_vals):
    """Calculate mean IoU from intersection and union values on CPU."""
    iou_per_class = [
        intersection_vals[i] / max(union_vals[i], 1.0)
        for i in range(len(intersection_vals))
    ]
    return sum(iou_per_class) / len(iou_per_class)


class MaskToTensor:
    """Convert PIL Image mask to long tensor."""

    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int64))


class IrisDataset(Dataset):
    """Dataset class for OpenEDS precomputed dataset."""

    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # Reshape flat arrays to images
        image = np.array(sample["image"], dtype=np.uint8).reshape(
            IMAGE_HEIGHT, IMAGE_WIDTH
        )
        label = np.array(sample["label"], dtype=np.uint8).reshape(
            IMAGE_HEIGHT, IMAGE_WIDTH
        )
        spatial_weights = torch.from_numpy(
            np.array(sample["spatial_weights"], dtype=np.float32).reshape(
                IMAGE_HEIGHT, IMAGE_WIDTH
            )
        )
        dist_map = torch.from_numpy(
            np.array(sample["dist_map"], dtype=np.float32).reshape(
                2, IMAGE_HEIGHT, IMAGE_WIDTH
            )
        )

        # Convert to PIL for transforms
        img = Image.fromarray(image)
        if self.transform:
            img = self.transform(img)

        # Convert label to tensor
        label_tensor = MaskToTensor()(Image.fromarray(label))

        return img, label_tensor, spatial_weights, dist_map


def train(args):
    """Main training loop."""
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    print(f"\nLoading dataset from {HF_DATASET_REPO}...")
    hf_dataset = load_dataset(HF_DATASET_REPO)

    print(f"Train samples: {len(hf_dataset['train'])}")
    print(f"Validation samples: {len(hf_dataset['validation'])}")

    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    train_dataset = IrisDataset(hf_dataset["train"], transform=transform)
    valid_dataset = IrisDataset(hf_dataset["validation"], transform=transform)

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

    # Initialize Model
    print(f"\nInitializing DSA model (variant: {args.model_size})...")

    if args.model_size == "tiny":
        model = create_dsa_tiny(in_channels=1, num_classes=2)
    elif args.model_size == "small":
        model = create_dsa_small(in_channels=1, num_classes=2)
    elif args.model_size == "base":
        model = create_dsa_base(in_channels=1, num_classes=2)
    else:
        model = DSASegmentationModel(
            in_channels=1,
            num_classes=2,
            embed_dims=(16, 32, 48),
            depths=(1, 1, 1),
            num_heads=(2, 2, 4),
            top_k=(64, 32, 16),
            decoder_dim=24,
        )

    model = model.to(device)
    nparams = get_nparams(model)
    print(f"Model parameters: {nparams:,}")

    # Loss, Optimizer, Scheduler
    criterion = CombinedLoss(indexer_weight=args.indexer_weight)

    # Separate parameter groups for main model and indexer (per DSA paper)
    main_params = []
    indexer_params = []
    for name, param in model.named_parameters():
        if 'indexer' in name:
            indexer_params.append(param)
        else:
            main_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': main_params, 'lr': args.lr},
        {'params': indexer_params, 'lr': args.indexer_lr},
    ], weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )

    # Mixed precision
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Resume from checkpoint
    start_epoch = 0
    best_iou = 0.0

    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_iou = checkpoint.get("valid_iou", 0.0)
        print(f"Resumed from epoch {checkpoint['epoch'] + 1}, best IoU: {best_iou:.4f}")

    # Alpha scheduling (dice vs surface loss balance)
    alpha_schedule = np.zeros(args.epochs)
    warmup_epochs = min(125, args.epochs)
    alpha_schedule[:warmup_epochs] = 1 - np.arange(1, warmup_epochs + 1) / warmup_epochs
    if args.epochs > warmup_epochs:
        alpha_schedule[warmup_epochs:] = 0

    # Training loop
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        alpha = alpha_schedule[epoch]

        # Determine training mode
        # Dense warm-up: First few epochs train indexer only
        # Sparse training: Rest of training uses sparse attention
        is_warmup = epoch < args.warmup_epochs

        if is_warmup:
            print(f"\n[Epoch {epoch+1}] Dense Warm-up Phase (indexer training)")
            # Freeze main model, only train indexer
            for name, param in model.named_parameters():
                if 'indexer' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        else:
            # Unfreeze all for sparse training
            for param in model.parameters():
                param.requires_grad = True

        # Training phase
        model.train()
        train_loss = torch.zeros(1, device=device)
        train_ce_loss = torch.zeros(1, device=device)
        train_dice_loss = torch.zeros(1, device=device)
        train_surface_loss = torch.zeros(1, device=device)
        train_intersection = torch.zeros(2, device=device)
        train_union = torch.zeros(2, device=device)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")

        for images, labels, spatial_weights, dist_maps in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            spatial_weights = spatial_weights.to(device, non_blocking=True)
            dist_maps = dist_maps.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images)

                # Get indexer loss during training
                indexer_loss = None
                if args.use_indexer_loss and not is_warmup:
                    indexer_loss = model.get_indexer_loss(images)

                loss, ce_loss, dice_loss, surface_loss = criterion(
                    outputs, labels, spatial_weights, dist_maps, alpha, indexer_loss
                )

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.detach()
            train_ce_loss += ce_loss.detach()
            train_dice_loss += dice_loss.detach()
            train_surface_loss += surface_loss.detach()

            preds = get_predictions(outputs)
            inter, uni = compute_iou_tensors(preds, labels)
            train_intersection += inter
            train_union += uni

            pbar.set_postfix({"alpha": f"{alpha:.3f}"})

        n_train_batches = len(train_loader)

        # Validation phase
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
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                spatial_weights = spatial_weights.to(device, non_blocking=True)
                dist_maps = dist_maps.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(images)
                    loss, ce_loss, dice_loss, surface_loss = criterion(
                        outputs, labels, spatial_weights, dist_maps, alpha
                    )

                valid_loss += loss.detach()
                valid_ce_loss += ce_loss.detach()
                valid_dice_loss += dice_loss.detach()
                valid_surface_loss += surface_loss.detach()

                preds = get_predictions(outputs)
                inter, uni = compute_iou_tensors(preds, labels)
                valid_intersection += inter
                valid_union += uni

        # Metrics calculation
        n_valid_batches = len(valid_loader)
        all_metrics_gpu = torch.cat([
            train_loss / n_train_batches,
            train_ce_loss / n_train_batches,
            train_dice_loss / n_train_batches,
            train_surface_loss / n_train_batches,
            valid_loss / n_valid_batches,
            valid_ce_loss / n_valid_batches,
            valid_dice_loss / n_valid_batches,
            valid_surface_loss / n_valid_batches,
            train_intersection,
            train_union,
            valid_intersection,
            valid_union,
        ])

        all_metrics = all_metrics_gpu.tolist()

        (
            train_loss_val, train_ce_val, train_dice_val, train_surface_val,
            valid_loss_val, valid_ce_val, valid_dice_val, valid_surface_val,
            train_inter_0, train_inter_1, train_union_0, train_union_1,
            valid_inter_0, valid_inter_1, valid_union_0, valid_union_1,
        ) = all_metrics

        train_iou = compute_iou_cpu(
            [train_inter_0, train_inter_1], [train_union_0, train_union_1]
        )
        valid_iou = compute_iou_cpu(
            [valid_inter_0, valid_inter_1], [valid_union_0, valid_union_1]
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Logging
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss_val:.4f} | Valid Loss: {valid_loss_val:.4f}")
        print(f"  Train IoU:  {train_iou:.4f} | Valid IoU:  {valid_iou:.4f}")
        print(f"  CE Loss:    {train_ce_val:.4f} | {valid_ce_val:.4f}")
        print(f"  Dice Loss:  {train_dice_val:.4f} | {valid_dice_val:.4f}")
        print(f"  Surf Loss:  {train_surface_val:.4f} | {valid_surface_val:.4f}")
        print(f"  LR: {current_lr:.6f} | Alpha: {alpha:.4f}")

        # Save best model
        if valid_iou > best_iou:
            best_iou = valid_iou

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "valid_iou": valid_iou,
                    "valid_loss": valid_loss_val,
                },
                f"{args.checkpoint_dir}/best_model.pth",
            )

            print(f"  >> Saved best model with IoU={best_iou:.4f}")

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
                },
                f"{args.checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth",
            )
            print(f"  >> Saved checkpoint at epoch {epoch+1}")

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best validation IoU: {best_iou:.4f}")
    print(f"\nCheckpoint saved to: {args.checkpoint_dir}/best_model.pth")


def main():
    """Parse arguments and start training."""
    parser = argparse.ArgumentParser(
        description="Train DSA Segmentation Model for pupil segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=["tiny", "small", "base"],
        help="Model size variant",
    )

    # Training hyperparameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training and validation",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for main model",
    )
    parser.add_argument(
        "--indexer-lr",
        type=float,
        default=1e-3,
        help="Learning rate for indexer (per DSA paper)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # DSA specific
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=2,
        help="Dense warm-up epochs for indexer initialization",
    )
    parser.add_argument(
        "--use-indexer-loss",
        action="store_true",
        default=True,
        help="Use indexer alignment loss during training",
    )
    parser.add_argument(
        "--indexer-weight",
        type=float,
        default=0.1,
        help="Weight for indexer alignment loss",
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
        default="checkpoints_dsa",
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
    print("DSA Segmentation Model Training Configuration")
    print("=" * 80)
    print(f"Model size: {args.model_size}")
    print(f"Dataset: {HF_DATASET_REPO}")
    print(f"Image size: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate (main): {args.lr}")
    print(f"Learning rate (indexer): {args.indexer_lr}")
    print(f"Warmup epochs: {args.warmup_epochs}")
    print(f"Use indexer loss: {args.use_indexer_loss}")
    print(f"Indexer weight: {args.indexer_weight}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print("=" * 80)

    try:
        train(args)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        raise


if __name__ == "__main__":
    main()
