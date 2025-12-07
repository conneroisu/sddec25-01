"""
Ellipse Regression Model

This module contains the neural network architecture for ellipse regression,
which outputs normalized ellipse parameters (cx, cy, rx, ry) for pupil detection.
"""

import math

import torch
import torch.nn as nn


class DownBlock(nn.Module):
    """Depthwise separable convolution block with optional dropout.

    This block performs dense connections with depthwise separable convolutions
    for efficient feature extraction with reduced parameter count.

    Args:
        input_channels: Number of input channels
        output_channels: Number of output channels
        down_size: Tuple for average pooling size, or None to skip pooling
        dropout: Whether to apply dropout
        prob: Dropout probability
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        down_size,
        dropout=False,
        prob=0,
    ):
        super(DownBlock, self).__init__()
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
        self.max_pool = nn.AvgPool2d(kernel_size=down_size) if down_size else None
        self.relu = nn.LeakyReLU()
        self.down_size = down_size
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)
        self.dropout3 = nn.Dropout(p=prob)
        self.bn = torch.nn.BatchNorm2d(num_features=output_channels)

    def forward(self, x):
        if self.max_pool is not None:
            x = self.max_pool(x)

        if self.dropout:
            x1 = self.relu(self.dropout1(self.pointwise_conv1(self.depthwise_conv1(x))))
            x21 = torch.cat((x, x1), dim=1)
            x22 = self.relu(
                self.dropout2(
                    self.pointwise_conv22(self.depthwise_conv22(self.conv21(x21)))
                )
            )
            x31 = torch.cat((x21, x22), dim=1)
            out = self.relu(
                self.dropout3(
                    self.pointwise_conv32(self.depthwise_conv32(self.conv31(x31)))
                )
            )
        else:
            x1 = self.relu(self.pointwise_conv1(self.depthwise_conv1(x)))
            x21 = torch.cat((x, x1), dim=1)
            x22 = self.relu(
                self.pointwise_conv22(self.depthwise_conv22(self.conv21(x21)))
            )
            x31 = torch.cat((x21, x22), dim=1)
            out = self.relu(
                self.pointwise_conv32(self.depthwise_conv32(self.conv31(x31)))
            )

        return self.bn(out)


class EllipseRegressionNet(nn.Module):
    """CNN for ellipse regression that outputs 4 normalized ellipse parameters.

    The network predicts normalized (cx, cy, rx, ry) parameters where:
    - cx, cy: center coordinates normalized to [0, 1] relative to image dimensions
    - rx, ry: radii normalized to [0, 1] relative to max radius

    Args:
        in_channels: Number of input channels (default: 1 for grayscale)
        channel_size: Base channel size for the network (default: 32)
        dropout: Whether to use dropout (default: False)
        prob: Dropout probability (default: 0)
    """

    def __init__(
        self,
        in_channels=1,
        channel_size=32,
        dropout=False,
        prob=0,
    ):
        super(EllipseRegressionNet, self).__init__()

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
            down_size=(2, 2),
            dropout=dropout,
            prob=prob,
        )

        self.down_block3 = DownBlock(
            input_channels=channel_size,
            output_channels=channel_size * 2,
            down_size=(2, 2),
            dropout=dropout,
            prob=prob,
        )

        self.down_block4 = DownBlock(
            input_channels=channel_size * 2,
            output_channels=channel_size * 2,
            down_size=(2, 2),
            dropout=dropout,
            prob=prob,
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        fc_input_size = channel_size * 2
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, 128),
            nn.LeakyReLU(),
            nn.Dropout(p=prob) if dropout else nn.Identity(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Dropout(p=prob) if dropout else nn.Identity(),
            nn.Linear(64, 4),
            nn.Sigmoid(),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using He initialization for conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.groups == m.in_channels and m.in_channels == m.out_channels:
                    # Depthwise convolution
                    n = m.kernel_size[0] * m.kernel_size[1]
                    m.weight.data.normal_(0, math.sqrt(2.0 / n))
                elif m.kernel_size == (1, 1):
                    # Pointwise convolution
                    n = m.in_channels
                    m.weight.data.normal_(0, math.sqrt(2.0 / n))
                else:
                    # Standard convolution
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Tensor of shape (batch, 4) with normalized ellipse parameters
            [cx_norm, cy_norm, rx_norm, ry_norm]
        """
        x = self.down_block1(x)
        x = self.down_block2(x)
        x = self.down_block3(x)
        x = self.down_block4(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        params = self.fc(x)

        return params
