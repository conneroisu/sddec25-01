#!/usr/bin/env python3
"""
Demo script for NSA Pupil Segmentation.

This script demonstrates the NSAPupilSeg model on sample images,
showing the Native Sparse Attention mechanism in action for pupil detection.
"""

import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from nsa import create_nsa_pupil_seg


def load_model(
    checkpoint_path: str,
    model_size: str = "small",
    device: str = "cuda",
):
    """Load trained NSA model from checkpoint."""
    model = create_nsa_pupil_seg(
        size=model_size,
        in_channels=1,
        num_classes=2,
    )

    if checkpoint_path:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )
        print(
            f"Loaded model from {checkpoint_path}"
        )
        print(
            f"Best IoU: {checkpoint.get('valid_iou', 'N/A'):.4f}"
        )

    model = model.to(device)
    model.eval()

    return model


def preprocess_image(
    image_path: str,
    target_size: tuple = (400, 640),
):
    """Load and preprocess an image for inference."""
    img = Image.open(
        image_path
    ).convert("L")

    # Resize if needed
    if img.size != (
        target_size[1],
        target_size[0],
    ):
        img = img.resize(
            (
                target_size[1],
                target_size[0],
            ),
            Image.BILINEAR,
        )

    # Transform
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5], [0.5]
            ),
        ]
    )

    img_tensor = transform(
        img
    ).unsqueeze(0)

    return img_tensor, np.array(img)


def predict(
    model,
    img_tensor,
    device: str = "cuda",
):
    """Run inference on an image."""
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        output = model(img_tensor)
        pred = (
            output.argmax(dim=1)
            .squeeze()
            .cpu()
            .numpy()
        )

    return pred


def visualize_results(
    original: np.ndarray,
    prediction: np.ndarray,
    save_path: str = None,
    show: bool = True,
):
    """Visualize original image and segmentation overlay."""
    fig, axes = plt.subplots(
        1, 3, figsize=(15, 5)
    )

    # Original image
    axes[0].imshow(
        original, cmap="gray"
    )
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Prediction mask
    axes[1].imshow(
        prediction, cmap="viridis"
    )
    axes[1].set_title(
        "Pupil Segmentation"
    )
    axes[1].axis("off")

    # Overlay
    overlay = (
        np.stack(
            [original] * 3, axis=-1
        ).astype(np.float32)
        / 255.0
    )
    pupil_mask = prediction == 1
    overlay[pupil_mask, 0] = 0.2  # R
    overlay[pupil_mask, 1] = 0.8  # G
    overlay[pupil_mask, 2] = 0.2  # B

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight",
        )
        print(
            f"Saved visualization to {save_path}"
        )

    if show:
        plt.show()

    plt.close()


def demo_synthetic():
    """Demo with synthetic eye-like image."""
    print("\n" + "=" * 60)
    print(
        "NSA Pupil Segmentation Demo (Synthetic Image)"
    )
    print("=" * 60)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # Create model
    print(
        "\nCreating NSAPupilSeg model..."
    )
    model = create_nsa_pupil_seg(
        size="small"
    ).to(device)
    model.eval()

    n_params = sum(
        p.numel()
        for p in model.parameters()
    )
    print(
        f"Model parameters: {n_params:,}"
    )

    # Create synthetic eye-like image
    print(
        "\nGenerating synthetic eye image..."
    )
    H, W = 400, 640

    # Create grayscale background (sclera)
    img = (
        np.ones(
            (H, W), dtype=np.float32
        )
        * 200
    )

    # Add iris (darker circle)
    cy, cx = H // 2, W // 2
    iris_radius = 100
    Y, X = np.ogrid[:H, :W]
    iris_mask = (X - cx) ** 2 + (
        Y - cy
    ) ** 2 <= iris_radius**2
    img[iris_mask] = 80

    # Add pupil (even darker circle)
    pupil_radius = 40
    pupil_mask = (X - cx) ** 2 + (
        Y - cy
    ) ** 2 <= pupil_radius**2
    img[pupil_mask] = 20

    # Add some noise
    img += np.random.randn(H, W) * 10
    img = np.clip(img, 0, 255).astype(
        np.uint8
    )

    # Preprocess
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5], [0.5]
            ),
        ]
    )
    img_tensor = (
        transform(Image.fromarray(img))
        .unsqueeze(0)
        .to(device)
    )

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        output = model(img_tensor)
        pred = (
            output.argmax(dim=1)
            .squeeze()
            .cpu()
            .numpy()
        )

    # Calculate IoU with ground truth
    gt = pupil_mask.astype(np.int64)
    intersection = np.logical_and(
        pred == 1, gt == 1
    ).sum()
    union = np.logical_or(
        pred == 1, gt == 1
    ).sum()
    iou = intersection / max(union, 1)

    print(f"\nResults:")
    print(
        f"  Predicted pupil pixels: {(pred == 1).sum():,}"
    )
    print(
        f"  Ground truth pupil pixels: {gt.sum():,}"
    )
    print(f"  IoU: {iou:.4f}")

    # Visualize
    visualize_results(
        img,
        pred,
        save_path="nsa_demo_synthetic.png",
        show=True,
    )


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(
        description="Demo NSAPupilSeg model",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image (if not provided, uses synthetic)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=[
            "tiny",
            "small",
            "medium",
        ],
        help="Model size",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for visualization",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display visualization",
    )

    args = parser.parse_args()

    if args.image is None:
        # Run synthetic demo
        demo_synthetic()
    else:
        # Run on provided image
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        print(f"Device: {device}")

        model = load_model(
            args.checkpoint,
            args.model_size,
            device,
        )
        img_tensor, original = (
            preprocess_image(args.image)
        )
        prediction = predict(
            model, img_tensor, device
        )

        output_path = (
            args.output
            or f"nsa_output_{args.image.split('/')[-1]}"
        )
        visualize_results(
            original,
            prediction,
            save_path=output_path,
            show=not args.no_show,
        )


if __name__ == "__main__":
    main()
