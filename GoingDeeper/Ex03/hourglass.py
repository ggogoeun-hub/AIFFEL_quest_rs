"""Stacked Hourglass Network for Human Pose Estimation

Newell et al., 2016 - "Stacked Hourglass Networks for Human Pose Estimation"
"""

import torch
import torch.nn as nn


class BottleneckBlock(nn.Module):
    """Residual Bottleneck Block (1x1 → 3x3 → 1x1)"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 2

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)

        self.relu = nn.ReLU(inplace=True)

        self.skip = None
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.skip is not None:
            residual = self.skip(residual)

        return out + residual


class HourglassModule(nn.Module):
    """재귀적 Hourglass Module

    depth가 1이 될 때까지 재귀적으로 인코더-디코더를 구성합니다.
    """

    def __init__(self, depth, num_channels):
        super().__init__()
        self.depth = depth

        self.upper = nn.Sequential(
            BottleneckBlock(num_channels, num_channels),
            BottleneckBlock(num_channels, num_channels),
            BottleneckBlock(num_channels, num_channels),
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.lower_before = nn.Sequential(
            BottleneckBlock(num_channels, num_channels),
            BottleneckBlock(num_channels, num_channels),
            BottleneckBlock(num_channels, num_channels),
        )

        if depth > 1:
            self.inner = HourglassModule(depth - 1, num_channels)
        else:
            self.inner = nn.Sequential(
                BottleneckBlock(num_channels, num_channels),
                BottleneckBlock(num_channels, num_channels),
                BottleneckBlock(num_channels, num_channels),
            )

        self.lower_after = nn.Sequential(
            BottleneckBlock(num_channels, num_channels),
            BottleneckBlock(num_channels, num_channels),
            BottleneckBlock(num_channels, num_channels),
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        upper = self.upper(x)

        lower = self.pool(x)
        lower = self.lower_before(lower)
        lower = self.inner(lower)
        lower = self.lower_after(lower)
        lower = self.upsample(lower)

        return upper + lower


class LinearLayer(nn.Module):
    """Intermediate output을 위한 Linear Layer"""

    def __init__(self, num_channels):
        super().__init__()
        self.block = BottleneckBlock(num_channels, num_channels)
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.block(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class StackedHourglassNetwork(nn.Module):
    """Stacked Hourglass Network for Human Pose Estimation

    Args:
        num_stacks: hourglass 모듈 수 (기본 2)
        num_channels: 내부 채널 수 (기본 256)
        num_joints: 키포인트 수 (기본 16)
        depth: hourglass 재귀 깊이 (기본 4)
    """

    def __init__(self, num_stacks=2, num_channels=256, num_joints=16, depth=4):
        super().__init__()
        self.num_stacks = num_stacks

        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            BottleneckBlock(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BottleneckBlock(128, 128),
            BottleneckBlock(128, num_channels),
        )

        self.hourglass = nn.ModuleList([
            HourglassModule(depth, num_channels) for _ in range(num_stacks)
        ])

        self.linear = nn.ModuleList([
            LinearLayer(num_channels) for _ in range(num_stacks)
        ])

        self.pred = nn.ModuleList([
            nn.Conv2d(num_channels, num_joints, kernel_size=1) for _ in range(num_stacks)
        ])

        self.merge_features = nn.ModuleList([
            nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=False)
            for _ in range(num_stacks - 1)
        ])
        self.merge_preds = nn.ModuleList([
            nn.Conv2d(num_joints, num_channels, kernel_size=1, bias=False)
            for _ in range(num_stacks - 1)
        ])

    def forward(self, x):
        x = self.initial(x)
        outputs = []

        for i in range(self.num_stacks):
            hg_out = self.hourglass[i](x)
            features = self.linear[i](hg_out)
            preds = self.pred[i](features)
            outputs.append(preds)

            if i < self.num_stacks - 1:
                x = x + self.merge_features[i](features) + self.merge_preds[i](preds)

        return outputs
