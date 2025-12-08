import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(
            CombinedLoss,
            self,
        ).__init__()
        self.epsilon = epsilon
        self.nll = nn.NLLLoss(reduction="none")

    def forward(
        self,
        logits,
        target,
        spatial_weights,
        dist_map,
        alpha,
    ):
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(
            logits,
            dim=1,
        )
        ce_loss = self.nll(
            log_probs,
            target,
        )
        weighted_ce = (ce_loss * (1.0 + spatial_weights)).mean()
        target_onehot = (
            F.one_hot(
                target,
                num_classes=2,
            )
            .permute(0, 3, 1, 2)
            .float()
        )
        probs_flat = probs.flatten(start_dim=2)
        target_flat = target_onehot.flatten(start_dim=2)
        intersection = (probs_flat * target_flat).sum(dim=2)
        cardinality = (probs_flat + target_flat).sum(dim=2)
        class_weights = 1.0 / (target_flat.sum(dim=2) ** 2).clamp(min=self.epsilon)
        dice = (
            2.0
            * (class_weights * intersection).sum(dim=1)
            / (class_weights * cardinality).sum(dim=1)
        )
        dice_loss = (1.0 - dice.clamp(min=self.epsilon)).mean()
        surface_loss = (
            (probs.flatten(start_dim=2) * dist_map.flatten(start_dim=2))
            .mean(dim=2)
            .mean(dim=1)
            .mean()
        )
        total_loss = weighted_ce + alpha * dice_loss + (1.0 - alpha) * surface_loss
        return (
            total_loss,
            weighted_ce,
            dice_loss,
            surface_loss,
        )

class DownBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        down_size,
        dropout=False,
        prob=0,
    ):
        super(
            DownBlock,
            self,
        ).__init__()
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
        self.max_pool = nn.AvgPool2d(kernel_size=down_size)
        self.relu = nn.LeakyReLU()
        self.down_size = down_size
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)
        self.dropout3 = nn.Dropout(p=prob)
        self.bn = torch.nn.BatchNorm2d(num_features=output_channels)

    def forward(self, x):
        if self.down_size is not None:
            x = self.max_pool(x)
        if self.dropout:
            x1 = self.relu(
                self.dropout1(self.pointwise_conv1(self.depthwise_conv1(x)))
            )
            x21 = torch.cat(
                (x, x1),
                dim=1,
            )
            x22 = self.relu(
                self.dropout2(
                    self.pointwise_conv22(self.depthwise_conv22(self.conv21(x21)))
                )
            )
            x31 = torch.cat(
                (
                    x21,
                    x22,
                ),
                dim=1,
            )
            out = self.relu(
                self.dropout3(
                    self.pointwise_conv32(self.depthwise_conv32(self.conv31(x31)))
                )
            )
        else:
            x1 = self.relu(self.pointwise_conv1(self.depthwise_conv1(x)))
            x21 = torch.cat(
                (x, x1),
                dim=1,
            )
            x22 = self.relu(
                self.pointwise_conv22(self.depthwise_conv22(self.conv21(x21)))
            )
            x31 = torch.cat(
                (
                    x21,
                    x22,
                ),
                dim=1,
            )
            out = self.relu(
                self.pointwise_conv32(self.depthwise_conv32(self.conv31(x31)))
            )
        return self.bn(out)

class UpBlockConcat(nn.Module):
    def __init__(
        self,
        skip_channels,
        input_channels,
        output_channels,
        up_stride,
        dropout=False,
        prob=0,
    ):
        super(
            UpBlockConcat,
            self,
        ).__init__()
        self.conv11 = nn.Conv2d(
            skip_channels + input_channels,
            output_channels,
            kernel_size=1,
            padding=0,
        )
        self.depthwise_conv12 = nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            groups=output_channels,
        )
        self.pointwise_conv12 = nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size=1,
        )
        self.conv21 = nn.Conv2d(
            skip_channels + input_channels + output_channels,
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
        self.relu = nn.LeakyReLU()
        self.up_stride = up_stride
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)

    def forward(
        self,
        prev_feature_map,
        x,
    ):
        x = nn.functional.interpolate(
            x,
            scale_factor=self.up_stride,
            mode="nearest",
        )
        x = torch.cat(
            (
                x,
                prev_feature_map,
            ),
            dim=1,
        )
        if self.dropout:
            x1 = self.relu(
                self.dropout1(
                    self.pointwise_conv12(self.depthwise_conv12(self.conv11(x)))
                )
            )
            x21 = torch.cat(
                (x, x1),
                dim=1,
            )
            out = self.relu(
                self.dropout2(
                    self.pointwise_conv22(self.depthwise_conv22(self.conv21(x21)))
                )
            )
        else:
            x1 = self.relu(
                self.pointwise_conv12(self.depthwise_conv12(self.conv11(x)))
            )
            x21 = torch.cat(
                (x, x1),
                dim=1,
            )
            out = self.relu(
                self.pointwise_conv22(self.depthwise_conv22(self.conv21(x21)))
            )
        return out

class ShallowNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=2,
        channel_size=32,
        concat=True,
        dropout=False,
        prob=0,
    ):
        super(
            ShallowNet,
            self,
        ).__init__()
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
            down_size=(
                2,
                2,
            ),
            dropout=dropout,
            prob=prob,
        )
        self.down_block3 = DownBlock(
            input_channels=channel_size,
            output_channels=channel_size,
            down_size=(
                2,
                2,
            ),
            dropout=dropout,
            prob=prob,
        )
        self.down_block4 = DownBlock(
            input_channels=channel_size,
            output_channels=channel_size,
            down_size=(
                2,
                2,
            ),
            dropout=dropout,
            prob=prob,
        )
        self.up_block1 = UpBlockConcat(
            skip_channels=channel_size,
            input_channels=channel_size,
            output_channels=channel_size,
            up_stride=(
                2,
                2,
            ),
            dropout=dropout,
            prob=prob,
        )
        self.up_block2 = UpBlockConcat(
            skip_channels=channel_size,
            input_channels=channel_size,
            output_channels=channel_size,
            up_stride=(
                2,
                2,
            ),
            dropout=dropout,
            prob=prob,
        )
        self.up_block3 = UpBlockConcat(
            skip_channels=channel_size,
            input_channels=channel_size,
            output_channels=channel_size,
            up_stride=(
                2,
                2,
            ),
            dropout=dropout,
            prob=prob,
        )
        self.out_conv1 = nn.Conv2d(
            in_channels=channel_size,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
        )
        self.concat = concat
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self._initialize_weights()

    def _initialize_weights(
        self,
    ):
        for m in self.modules():
            if isinstance(
                m,
                nn.Conv2d,
            ):
                if m.groups == m.in_channels and m.in_channels == m.out_channels:
                    n = m.kernel_size[0] * m.kernel_size[1]
                    m.weight.data.normal_(
                        0,
                        math.sqrt(2.0 / n),
                    )
                elif m.kernel_size == (
                    1,
                    1,
                ):
                    n = m.in_channels
                    m.weight.data.normal_(
                        0,
                        math.sqrt(2.0 / n),
                    )
                else:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(
                        0,
                        math.sqrt(2.0 / n),
                    )
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(
                m,
                nn.BatchNorm2d,
            ):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(
                m,
                nn.Linear,
            ):
                n = m.weight.size(1)
                m.weight.data.normal_(
                    0,
                    0.01,
                )
                m.bias.data.zero_()

    def forward(self, x):
        x1 = self.down_block1(x)
        x2 = self.down_block2(x1)
        x3 = self.down_block3(x2)
        x4 = self.down_block4(x3)
        x5 = self.up_block1(x3, x4)
        x6 = self.up_block2(x2, x5)
        x7 = self.up_block3(x1, x6)
        if self.dropout:
            out = self.out_conv1(self.dropout1(x7))
        else:
            out = self.out_conv1(x7)
        return out
