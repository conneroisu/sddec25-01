"""
Ellipse Regression Package

This package provides neural network components for ellipse regression
used in pupil detection tasks.

Exports:
    - EllipseRegressionNet: CNN model for ellipse parameter prediction
    - DownBlock: Depthwise separable convolution block
    - EllipseRegressionLoss: Combined loss function
    - Helper functions for ellipse manipulation and metrics
    - Constants for image dimensions
"""

from ellipse.loss import EllipseRegressionLoss
from ellipse.metrics import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MAX_RADIUS,
    compute_center_error,
    compute_iou_from_ellipses,
    compute_iou_with_gt_mask_gpu,
    compute_radius_error,
    denormalize_ellipse_params,
    extract_ellipse_params,
    get_nparams,
    normalize_ellipse_params,
    render_ellipse_mask,
    render_ellipse_mask_gpu,
    total_metric,
)
from ellipse.model import DownBlock, EllipseRegressionNet

__all__ = [
    # Model classes
    "EllipseRegressionNet",
    "DownBlock",
    # Loss class
    "EllipseRegressionLoss",
    # Helper functions
    "extract_ellipse_params",
    "normalize_ellipse_params",
    "denormalize_ellipse_params",
    "render_ellipse_mask",
    "render_ellipse_mask_gpu",
    "compute_iou_from_ellipses",
    "compute_iou_with_gt_mask_gpu",
    "compute_center_error",
    "compute_radius_error",
    "get_nparams",
    "total_metric",
    # Constants
    "IMAGE_HEIGHT",
    "IMAGE_WIDTH",
    "MAX_RADIUS",
]
