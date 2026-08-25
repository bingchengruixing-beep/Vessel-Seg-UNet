"""U-Net 解码器配合 ImageNet 预训练 ResNet 编码器。"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet34_Weights, ResNet50_Weights, resnet34, resnet50


class DecoderBlock(nn.Module):
    """上采样、跳跃连接和双卷积模块。"""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
            x = torch.cat([x, skip], dim=1)
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return self.conv(x)


class UNetResNet(nn.Module):
    """成员B的 ImageNet 预训练 ResNet34/50 U-Net。"""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        encoder_name: str = "resnet34",
        pretrained: bool = True,
        temporal_input: bool = False,
    ):
        super().__init__()
        if encoder_name == "resnet34":
            builder = resnet34
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        elif encoder_name == "resnet50":
            builder = resnet50
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        else:
            raise ValueError(f"Unknown encoder: {encoder_name}")
        try:
            encoder = builder(weights=weights)
        except (OSError, RuntimeError) as exc:
            if not pretrained:
                raise
            warnings.warn(f"ImageNet 权重加载失败，将使用随机初始化: {exc}")
            encoder = builder(weights=None)

        if temporal_input and in_channels == 3:
            with torch.no_grad():
                temporal_weight = encoder.conv1.weight.data.mean(dim=1, keepdim=True) / 3.0
                encoder.conv1.weight.copy_(temporal_weight.repeat(1, 3, 1, 1))
        elif in_channels != 3:
            old_weight = encoder.conv1.weight.data
            encoder.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                if in_channels == 1:
                    encoder.conv1.weight.copy_(old_weight.mean(dim=1, keepdim=True))
                else:
                    encoder.conv1.weight.copy_(old_weight[:, :in_channels])

        self.conv1, self.bn1, self.relu = encoder.conv1, encoder.bn1, encoder.relu
        self.maxpool = encoder.maxpool
        self.layer1, self.layer2 = encoder.layer1, encoder.layer2
        self.layer3, self.layer4 = encoder.layer3, encoder.layer4
        encoder_output_channels = 512 if encoder_name == "resnet34" else 2048
        self.dec4 = DecoderBlock(encoder_output_channels, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 32)
        self.dec0 = DecoderBlock(32, 0, 16)
        self.head = nn.Conv2d(16, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f0 = self.relu(self.bn1(self.conv1(x)))
        f1 = self.maxpool(f0)
        f2 = self.layer1(f1)
        f3 = self.layer2(f2)
        f4 = self.layer3(f3)
        f5 = self.layer4(f4)
        x = self.dec4(f5, f4)
        x = self.dec3(x, f3)
        x = self.dec2(x, f2)
        x = self.dec1(x, f0)
        x = self.dec0(x)
        return self.head(x)
