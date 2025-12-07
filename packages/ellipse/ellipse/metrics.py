"""
Ellipse Metrics and Helper Functions

This module provides helper functions for ellipse parameter manipulation,
mask rendering, and IoU computation for ellipse regression models.
"""

import math

import cv2
import numpy as np
import torch

# Image dimensions
IMAGE_HEIGHT = 400
IMAGE_WIDTH = 640

# Maximum radius (half diagonal of the image)
MAX_RADIUS = math.sqrt(IMAGE_WIDTH**2 + IMAGE_HEIGHT**2) / 2


def extract_ellipse_params(mask):
    """Extract ellipse parameters from a binary mask using OpenCV.

    Args:
        mask: Binary mask as numpy array

    Returns:
        Tuple of (cx, cy, rx, ry, angle) in pixel coordinates.
        Returns (0, 0, 0, 0, 0) if no valid contour is found.
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if len(contours) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    largest_contour = max(contours, key=cv2.contourArea)

    if len(largest_contour) < 5:
        # Not enough points to fit ellipse, use moments
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

        rx = width / 2.0
        ry = height / 2.0
        return cx, cy, rx, ry, angle
    except cv2.error:
        # Fallback to moments if fitEllipse fails
        M = cv2.moments(largest_contour)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            area = cv2.contourArea(largest_contour)
            radius = math.sqrt(area / math.pi)
            return cx, cy, radius, radius, 0.0
        return 0.0, 0.0, 0.0, 0.0, 0.0


def normalize_ellipse_params(cx, cy, rx, ry):
    """Normalize ellipse parameters to [0, 1] range.

    Args:
        cx: Center x coordinate in pixels
        cy: Center y coordinate in pixels
        rx: Radius in x direction in pixels
        ry: Radius in y direction in pixels

    Returns:
        Tuple of (cx_norm, cy_norm, rx_norm, ry_norm) all in [0, 1]
    """
    cx_norm = cx / IMAGE_WIDTH
    cy_norm = cy / IMAGE_HEIGHT
    rx_norm = rx / MAX_RADIUS
    ry_norm = ry / MAX_RADIUS
    return cx_norm, cy_norm, rx_norm, ry_norm


def denormalize_ellipse_params(cx_norm, cy_norm, rx_norm, ry_norm):
    """Convert normalized ellipse parameters back to pixel values.

    Args:
        cx_norm: Normalized center x coordinate [0, 1]
        cy_norm: Normalized center y coordinate [0, 1]
        rx_norm: Normalized radius x [0, 1]
        ry_norm: Normalized radius y [0, 1]

    Returns:
        Tuple of (cx, cy, rx, ry) in pixel coordinates
    """
    cx = cx_norm * IMAGE_WIDTH
    cy = cy_norm * IMAGE_HEIGHT
    rx = rx_norm * MAX_RADIUS
    ry = ry_norm * MAX_RADIUS
    return cx, cy, rx, ry


def render_ellipse_mask(cx, cy, rx, ry, height=IMAGE_HEIGHT, width=IMAGE_WIDTH):
    """Render an ellipse mask on CPU using OpenCV.

    Args:
        cx: Center x coordinate in pixels
        cy: Center y coordinate in pixels
        rx: Radius in x direction in pixels
        ry: Radius in y direction in pixels
        height: Output mask height (default: IMAGE_HEIGHT)
        width: Output mask width (default: IMAGE_WIDTH)

    Returns:
        Binary mask as numpy array of shape (height, width) with values 0 or 1
    """
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


def render_ellipse_mask_gpu(pred_params, height=IMAGE_HEIGHT, width=IMAGE_WIDTH):
    """Render ellipse masks entirely on GPU using vectorized operations.

    Args:
        pred_params: Tensor of shape (batch, 4) with normalized (cx, cy, rx, ry)
        height: Image height
        width: Image width

    Returns:
        Tensor of shape (batch, height, width) with ellipse masks (1 inside, 0 outside)
    """
    batch_size = pred_params.shape[0]
    device = pred_params.device

    # Denormalize parameters
    cx = pred_params[:, 0] * width  # (batch,)
    cy = pred_params[:, 1] * height  # (batch,)
    rx = pred_params[:, 2] * MAX_RADIUS  # (batch,)
    ry = pred_params[:, 3] * MAX_RADIUS  # (batch,)

    # Create coordinate grids
    y_coords = torch.arange(height, device=device, dtype=pred_params.dtype)
    x_coords = torch.arange(width, device=device, dtype=pred_params.dtype)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")  # (H, W)

    # Expand for batch: (batch, H, W)
    xx = xx.unsqueeze(0).expand(batch_size, -1, -1)
    yy = yy.unsqueeze(0).expand(batch_size, -1, -1)

    # Expand params for broadcasting: (batch, 1, 1)
    cx = cx.view(batch_size, 1, 1)
    cy = cy.view(batch_size, 1, 1)
    rx = rx.view(batch_size, 1, 1).clamp(min=1e-6)  # Avoid division by zero
    ry = ry.view(batch_size, 1, 1).clamp(min=1e-6)

    # Ellipse equation: ((x-cx)/rx)^2 + ((y-cy)/ry)^2 <= 1
    ellipse_eq = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    masks = (ellipse_eq <= 1.0).float()  # (batch, H, W)

    return masks


def compute_iou_from_ellipses(pred, target, device):
    """Compute IoU between predicted and target ellipse parameters.

    This function renders both ellipses as masks and computes IoU.
    Operates on CPU for compatibility.

    Args:
        pred: Predicted parameters tensor of shape (batch, 4)
        target: Target parameters tensor of shape (batch, 4)
        device: Device (used for compatibility, computation is on CPU)

    Returns:
        Mean IoU as a float
    """
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    batch_size = pred_np.shape[0]
    ious = []

    for i in range(batch_size):
        pred_cx, pred_cy, pred_rx, pred_ry = denormalize_ellipse_params(
            pred_np[i, 0], pred_np[i, 1], pred_np[i, 2], pred_np[i, 3]
        )
        target_cx, target_cy, target_rx, target_ry = denormalize_ellipse_params(
            target_np[i, 0], target_np[i, 1], target_np[i, 2], target_np[i, 3]
        )

        pred_mask = render_ellipse_mask(pred_cx, pred_cy, pred_rx, pred_ry)
        target_mask = render_ellipse_mask(target_cx, target_cy, target_rx, target_ry)

        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()

        if union > 0:
            iou = intersection / union
        else:
            iou = 1.0 if intersection == 0 else 0.0

        ious.append(iou)

    return np.mean(ious)


def compute_iou_with_gt_mask_gpu(pred, gt_masks, device):
    """Compute IoU metrics entirely on GPU.

    Args:
        pred: Predicted ellipse parameters (batch, 4) - normalized
        gt_masks: Ground truth masks (batch, H, W) - integer labels
        device: Device for computation

    Returns:
        Tuple of (mean_iou, bg_iou, pupil_iou) as GPU tensors
    """
    # Render predicted ellipse masks on GPU
    pred_masks = render_ellipse_mask_gpu(pred)  # (batch, H, W)

    # Convert gt_masks to float for operations
    gt_masks_float = gt_masks.float()

    # Pupil IoU (class 1)
    pred_pupil = pred_masks  # Already 1 inside ellipse, 0 outside
    target_pupil = (gt_masks_float == 1).float()

    intersection_pupil = (pred_pupil * target_pupil).sum(dim=(1, 2))  # (batch,)
    union_pupil = ((pred_pupil + target_pupil) > 0).float().sum(dim=(1, 2))  # (batch,)
    iou_pupil = intersection_pupil / union_pupil.clamp(min=1.0)  # (batch,)

    # Background IoU (class 0)
    pred_bg = 1.0 - pred_masks
    target_bg = (gt_masks_float == 0).float()

    intersection_bg = (pred_bg * target_bg).sum(dim=(1, 2))  # (batch,)
    union_bg = ((pred_bg + target_bg) > 0).float().sum(dim=(1, 2))  # (batch,)
    iou_bg = intersection_bg / union_bg.clamp(min=1.0)  # (batch,)

    # Mean across batch
    mean_pupil_iou = iou_pupil.mean()
    mean_bg_iou = iou_bg.mean()
    mean_iou = (mean_bg_iou + mean_pupil_iou) / 2

    return mean_iou, mean_bg_iou, mean_pupil_iou


def compute_center_error(pred, target):
    """Compute center error in pixels.

    Args:
        pred: Predicted parameters tensor of shape (batch, 4)
        target: Target parameters tensor of shape (batch, 4)

    Returns:
        Mean center error in pixels as a GPU tensor
    """
    pred_cx = pred[:, 0] * IMAGE_WIDTH
    pred_cy = pred[:, 1] * IMAGE_HEIGHT
    target_cx = target[:, 0] * IMAGE_WIDTH
    target_cy = target[:, 1] * IMAGE_HEIGHT

    dist = torch.sqrt((pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2)
    return dist.mean()


def compute_radius_error(pred, target):
    """Compute radius error in pixels.

    Args:
        pred: Predicted parameters tensor of shape (batch, 4)
        target: Target parameters tensor of shape (batch, 4)

    Returns:
        Mean radius error in pixels as a GPU tensor
    """
    pred_rx = pred[:, 2] * MAX_RADIUS
    pred_ry = pred[:, 3] * MAX_RADIUS
    target_rx = target[:, 2] * MAX_RADIUS
    target_ry = target[:, 3] * MAX_RADIUS

    rx_error = torch.abs(pred_rx - target_rx)
    ry_error = torch.abs(pred_ry - target_ry)
    return ((rx_error + ry_error) / 2).mean()


def get_nparams(model):
    """Get number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def total_metric(nparams, miou):
    """Compute total metric score combining model size and IoU.

    This metric rewards both small model size and high IoU performance.

    Args:
        nparams: Number of model parameters
        miou: Mean IoU score

    Returns:
        Total metric score in [0, 1]
    """
    S = nparams * 4.0 / (1024 * 1024)  # Size in MB (4 bytes per float32)
    total = min(1, 1.0 / S) + miou
    return total * 0.5
