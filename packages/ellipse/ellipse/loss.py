"""
Ellipse Regression Loss

This module contains the loss function for ellipse regression training,
combining center loss, radius loss, and an IoU proxy loss.
"""

import torch
import torch.nn as nn


class EllipseRegressionLoss(nn.Module):
    """Combined loss function for ellipse regression.

    This loss combines:
    - Center loss: Smooth L1 loss on (cx, cy) parameters
    - Radius loss: Smooth L1 loss on (rx, ry) parameters
    - IoU proxy loss: Mean squared parameter distance as IoU approximation

    Args:
        center_weight: Weight for center loss (default: 1.0)
        radius_weight: Weight for radius loss (default: 1.0)
        iou_weight: Weight for IoU proxy loss (default: 0.5)
    """

    def __init__(self, center_weight=1.0, radius_weight=1.0, iou_weight=0.5):
        super(EllipseRegressionLoss, self).__init__()
        self.center_weight = center_weight
        self.radius_weight = radius_weight
        self.iou_weight = iou_weight
        self.smooth_l1 = nn.SmoothL1Loss(reduction="mean")

    def forward(self, pred, target, compute_iou=True):
        """Compute the combined loss.

        Args:
            pred: Predicted parameters tensor of shape (batch, 4)
                  [cx_norm, cy_norm, rx_norm, ry_norm]
            target: Target parameters tensor of shape (batch, 4)
            compute_iou: Whether to include IoU proxy loss (default: True)

        Returns:
            Tuple of (total_loss, center_loss, radius_loss)
        """
        # Center loss (cx, cy)
        center_loss = self.smooth_l1(pred[:, :2], target[:, :2])

        # Radius loss (rx, ry)
        radius_loss = self.smooth_l1(pred[:, 2:], target[:, 2:])

        total_loss = self.center_weight * center_loss + self.radius_weight * radius_loss

        if self.iou_weight > 0 and compute_iou:
            # IoU proxy loss: penalize parameter distance as approximation of IoU
            param_dist = torch.mean((pred - target) ** 2, dim=1)
            iou_proxy_loss = torch.mean(param_dist)
            total_loss = total_loss + self.iou_weight * iou_proxy_loss

        return total_loss, center_loss, radius_loss
