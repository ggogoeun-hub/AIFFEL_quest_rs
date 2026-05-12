"""SimpleBaseline for Human Pose Estimation

Xiao et al., 2018 - "Simple Baselines for Human Pose Estimation and Tracking"

ResNet backbone (pre-trained) + 3 Deconvolution layers + 1x1 Conv
"""

import torch
import torch.nn as nn
import torchvision.models as models


class SimpleBaselineModel(nn.Module):
    """Simple Baseline for Human Pose Estimation

    Args:
        backbone: backbone 종류 ('resnet50', 'resnet101')
        num_joints: 키포인트 수
        pretrained: ImageNet pre-trained 가중치 사용 여부
    """

    def __init__(self, backbone='resnet50', num_joints=16, pretrained=True):
        super().__init__()

        if backbone == 'resnet50':
            resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            backbone_out_channels = 2048
        elif backbone == 'resnet101':
            resnet = models.resnet101(weights='IMAGENET1K_V1' if pretrained else None)
            backbone_out_channels = 2048
        else:
            raise ValueError(f"지원하지 않는 backbone: {backbone}")

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        deconv_channels = 256
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(backbone_out_channels, deconv_channels,
                              kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(deconv_channels),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(deconv_channels, deconv_channels,
                              kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(deconv_channels),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(deconv_channels, deconv_channels,
                              kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(deconv_channels),
            nn.ReLU(inplace=True),
        )

        self.final_layer = nn.Conv2d(deconv_channels, num_joints, kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x
