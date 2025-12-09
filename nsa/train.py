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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
import mlflow

# MLflow for experiment tracking
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

<<<<<<< HEAD:nsa/train.py
# Import NSA model and loss
from nsa import NSAPupilSeg, CombinedLoss, create_nsa_pupil_seg

# =============================================================================
=======
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
    missing_vars = [var for var in MLFLOW_ENV_VARS if not os.environ.get(var)]
    if missing_vars:
        raise EnvironmentError(
            f"Missing required MLflow environment variables: {', '.join(missing_vars)}\n"
            f"Please set: {', '.join(MLFLOW_ENV_VARS)}"
        )

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(experiment_id=os.environ["MLFLOW_EXPERIMENT_ID"])
    print(f"MLflow configured with tracking URI: {os.environ['MLFLOW_TRACKING_URI']}")
    print(f"MLflow experiment ID: {os.environ['MLFLOW_EXPERIMENT_ID']}")

>>>>>>> c99cfee (added mlflow to dsa training):dsa/train.py
# Constants
# =============================================================================

IMAGE_HEIGHT = 400
IMAGE_WIDTH = 640
HF_DATASET_REPO = "Conner/openeds-precomputed"

# =============================================================================
# Helper Functions
# =============================================================================


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


def compute_iou_tensors(predictions, targets, num_classes=2):
    """
    Compute per-class intersection and union for IoU calculation.
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


# =============================================================================
# Dataset Class
# =============================================================================


class MaskToTensor:
    """Convert PIL Image mask to long tensor."""

    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int64))


class IrisDataset(Dataset):
    """
    Dataset class for OpenEDS precomputed dataset.
    """

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


# =============================================================================
# Training Function
# =============================================================================


def train(args):
    """Main training loop."""
    # Setup MLflow before training
    setup_mlflow()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    # Load dataset
    print(f"\nLoading dataset from {HF_DATASET_REPO}...")
    print("(First run will download ~3GB, subsequent runs use cached data)")

    hf_dataset = load_dataset(HF_DATASET_REPO)

    num_train_samples = len(hf_dataset['train'])
    num_valid_samples = len(hf_dataset['validation'])
    print(f"Train samples: {num_train_samples}")
    print(f"Validation samples: {num_valid_samples}")

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

    # ==========================================================================
    # Initialize NSA Model
    # ==========================================================================

    print(f"\nInitializing NSAPupilSeg model (size={args.model_size})...")

    model = create_nsa_pupil_seg(
        size=args.model_size,
        in_channels=1,
        num_classes=2,
    ).to(device)

    nparams = get_nparams(model)
    print(f"Model parameters: {nparams:,}")

<<<<<<< HEAD:nsa/train.py
    # ==========================================================================
    # MLflow Setup
    # ==========================================================================

    use_mlflow = is_mlflow_configured()
    mlflow_run = None

    if use_mlflow:
        print("\nMLflow is configured. Starting experiment tracking...")

        # Configure MLflow tracking URI (Databricks or custom)
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            print(f"MLflow tracking URI: {tracking_uri}")

        # Configure registry URI if provided
        registry_uri = os.environ.get("MLFLOW_REGISTRY_URI")
        if registry_uri:
            mlflow.set_registry_uri(registry_uri)
            print(f"MLflow registry URI: {registry_uri}")

        # Set experiment if MLFLOW_EXPERIMENT_ID is provided
        experiment_id = os.environ.get("MLFLOW_EXPERIMENT_ID")
        if experiment_id:
            mlflow.set_experiment(experiment_id=experiment_id)

        # Start MLflow run
        mlflow_run = mlflow.start_run()

        # Log hyperparameters
        mlflow.log_params({
            "model_size": args.model_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "nparams": nparams,
        })
        print(f"MLflow run started: {mlflow_run.info.run_id}")
    else:
        print("\nMLflow not configured. Training will proceed without experiment tracking.")

    # ==========================================================================
=======
    # Start MLflow run and log parameters
    mlflow.start_run()
    print(f"MLflow run started: {mlflow.active_run().info.run_id}")

    # Log all hyperparameters
    mlflow.log_params({
        "model_size": args.model_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "indexer_lr": args.indexer_lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "warmup_epochs": args.warmup_epochs,
        "use_indexer_loss": args.use_indexer_loss,
        "indexer_weight": args.indexer_weight,
        "num_workers": args.num_workers,
        "num_parameters": nparams,
        "image_height": IMAGE_HEIGHT,
        "image_width": IMAGE_WIDTH,
        "dataset": HF_DATASET_REPO,
        "num_train_samples": num_train_samples,
        "num_valid_samples": num_valid_samples,
        "device": str(device),
    })

>>>>>>> c99cfee (added mlflow to dsa training):dsa/train.py
    # Loss, Optimizer, Scheduler
    # ==========================================================================

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
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # ==========================================================================
    # Resume from Checkpoint
    # ==========================================================================

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

    # ==========================================================================
    # Alpha Scheduling
    # ==========================================================================

    alpha_schedule = np.zeros(args.epochs)
    alpha_schedule[0 : min(125, args.epochs)] = 1 - np.arange(
        1, min(125, args.epochs) + 1
    ) / min(125, args.epochs)
    if args.epochs > 125:
        alpha_schedule[125:] = 0

    # ==========================================================================
    # Training Loop
    # ==========================================================================

    print("\n" + "=" * 80)
    print("Starting NSA Training")
    print("=" * 80)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        alpha = alpha_schedule[epoch]

        # ======================================================================
        # Training Phase
        # ======================================================================

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
                loss, ce_loss, dice_loss, surface_loss = criterion(
                    outputs, labels, spatial_weights, dist_maps, alpha
                )

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

        # ======================================================================
        # Validation Phase
        # ======================================================================

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

        # ======================================================================
        # Metrics Calculation
        # ======================================================================

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

        # ======================================================================
        # Learning Rate Scheduling
        # ======================================================================

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # ======================================================================
        # Logging
        # ======================================================================

        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss_val:.4f} | Valid Loss: {valid_loss_val:.4f}")
        print(f"  Train IoU:  {train_iou:.4f} | Valid IoU:  {valid_iou:.4f}")
        print(f"  CE Loss:    {train_ce_val:.4f} | {valid_ce_val:.4f}")
        print(f"  Dice Loss:  {train_dice_val:.4f} | {valid_dice_val:.4f}")
        print(f"  Surf Loss:  {train_surface_val:.4f} | {valid_surface_val:.4f}")
        print(f"  LR: {current_lr:.6f} | Alpha: {alpha:.4f}")

<<<<<<< HEAD:nsa/train.py
        # ======================================================================
        # MLflow Metrics Logging
        # ======================================================================

        if use_mlflow:
            mlflow.log_metrics(
                {
                    "train_loss": train_loss_val,
                    "train_iou": train_iou,
                    "train_ce_loss": train_ce_val,
                    "train_dice_loss": train_dice_val,
                    "train_surface_loss": train_surface_val,
                    "valid_loss": valid_loss_val,
                    "valid_iou": valid_iou,
                    "valid_ce_loss": valid_ce_val,
                    "valid_dice_loss": valid_dice_val,
                    "valid_surface_loss": valid_surface_val,
                    "learning_rate": current_lr,
                    "alpha": alpha,
                },
                step=epoch,
            )

        # ======================================================================
        # Save Checkpoints
        # ======================================================================

=======
        # Log metrics to MLflow
        mlflow.log_metrics({
            "train_loss": train_loss_val,
            "train_iou": train_iou,
            "train_ce_loss": train_ce_val,
            "train_dice_loss": train_dice_val,
            "train_surface_loss": train_surface_val,
            "valid_loss": valid_loss_val,
            "valid_iou": valid_iou,
            "valid_ce_loss": valid_ce_val,
            "valid_dice_loss": valid_dice_val,
            "valid_surface_loss": valid_surface_val,
            "learning_rate": current_lr,
            "alpha": alpha,
            "is_warmup_phase": 1 if is_warmup else 0,
        }, step=epoch)

        # Save best model
>>>>>>> c99cfee (added mlflow to dsa training):dsa/train.py
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
<<<<<<< HEAD:nsa/train.py
                f"{args.checkpoint_dir}/best_nsa_{args.model_size}_model.pth",
=======
                best_model_path,
>>>>>>> c99cfee (added mlflow to dsa training):dsa/train.py
            )

            print(f"  >> Saved best model with IoU={best_iou:.4f}")

<<<<<<< HEAD:nsa/train.py
=======
            # Log best model to MLflow
            mlflow.log_metric("best_iou", best_iou, step=epoch)
            mlflow.log_artifact(best_model_path, artifact_path="checkpoints")

        # Periodic checkpoints
>>>>>>> c99cfee (added mlflow to dsa training):dsa/train.py
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
            print(f"  >> Saved checkpoint at epoch {epoch+1}")

    # ==========================================================================
    # Training Complete
    # ==========================================================================

    print("\n" + "=" * 80)
    print("NSA Training Complete!")
    print("=" * 80)
    print(f"Best validation IoU: {best_iou:.4f}")
    print(f"\nCheckpoint saved to: {args.checkpoint_dir}/best_nsa_{args.model_size}_model.pth")

    # ==========================================================================
    # MLflow Finalization
    # ==========================================================================

    if use_mlflow:
        # Log final best IoU metric
        mlflow.log_metric("best_iou", best_iou)
        # End the MLflow run
        mlflow.end_run()
        print(f"\nMLflow run completed successfully.")


# =============================================================================
# Main Entry Point
# =============================================================================

    # End MLflow run on successful completion
    mlflow.log_metric("final_best_iou", best_iou)
    mlflow.end_run()
    print("MLflow run completed successfully.")


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
        choices=["pico", "nano", "tiny", "small", "medium"],
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
        default=1e-3,
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
    print("NSAPupilSeg (Native Sparse Attention) Training Configuration")
    print("=" * 80)
    print(f"Model size: {args.model_size}")
    print(f"Dataset: {HF_DATASET_REPO}")
    print(f"Image size: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Num workers: {args.num_workers}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Resume from: {args.resume if args.resume else 'None'}")
    print(f"Random seed: {args.seed}")
    print("=" * 80)

    # Start training
    try:
        train(args)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        # End MLflow run on interrupt
        if mlflow.active_run():
            mlflow.end_run(status="KILLED")
            print("MLflow run ended with status: KILLED")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        # End MLflow run on error
        if mlflow.active_run():
            mlflow.end_run(status="FAILED")
            print("MLflow run ended with status: FAILED")
        raise


if __name__ == "__main__":
    main()
