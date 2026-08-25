"""
[M2] U-Net + ImageNet 预训练 ResNet 编码器（实验 10）
与 baseline 的最大区别：编码器不再从零训练，而是加载 torchvision 的
ImageNet 预训练 ResNet 权重，只微调解码器。这是「迁移学习」路线。

接口契约（与 unet_baseline 完全一致，可直接替换）:
    输入 x:      (Batch, 1, H, W), float32
    输出 logits: (Batch, 1, H, W), float32, 未经过激活函数

核心防错:
    最后一层 **绝对不加 Sigmoid**（同 unet_baseline）。
    灰度适配：ResNet 的 stem 卷积是 3 通道，对灰度图（1 通道）需要
    换成 1 通道并把 RGB 权重取均值作为初始化，其余层沿用预训练权重。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    resnet34,
    resnet50,
    ResNet34_Weights,
    ResNet50_Weights,
)


class DecoderBlock(nn.Module):
    """上采样 → 拼接 Skip → 双卷积（(Conv-BN-ReLU) × 2）"""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        """
        Args:
            in_channels: 上一层解码器输出的通道数
            skip_channels: 跳跃连接的通道数（0 表示该层无 skip）
            out_channels: 本层输出通道数
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        if skip is not None:
            # 上采样到 skip 的尺寸（天然对齐，同时兼容奇数尺寸）
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        else:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return self.conv(x)


class UNetResNet(nn.Module):
    """
    U-Net with ImageNet-pretrained ResNet encoder.

    编码器: ResNet34（torchvision 预训练，冻结与否由训练脚本决定）
    解码器: 5 层上采样块 + Skip Connection，输出 raw logits

    Args:
        in_channels: 输入通道数（灰度图 = 1）
        out_channels: 输出通道数（二值分割 = 1）
        encoder_name: 编码器名称（resnet34 / resnet50）
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        encoder_name: str = "resnet34",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # ── 构建 ImageNet 预训练编码器 ──
        if encoder_name == "resnet34":
            encoder = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        elif encoder_name == "resnet50":
            encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unknown encoder: {encoder_name}")

        # ── 灰度适配：3 通道 stem 卷积 → in_channels，权重取 RGB 均值 ──
        if in_channels != 3:
            old_weight = encoder.conv1.weight.data  # (64, 3, 7, 7)
            encoder.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            with torch.no_grad():
                encoder.conv1.weight.data = old_weight.mean(dim=1, keepdim=True)

        # 拆出各编码阶段作为 Skip Connection
        # ResNet34 stem = conv1(stride2 → H/2) + maxpool(stride2 → H/4)
        self.conv1 = encoder.conv1
        self.bn1 = encoder.bn1
        self.relu = encoder.relu
        self.maxpool = encoder.maxpool
        self.layer1 = encoder.layer1  # 64,  H/4
        self.layer2 = encoder.layer2  # 128, H/8
        self.layer3 = encoder.layer3  # 256, H/16
        self.layer4 = encoder.layer4  # 512, H/32

        # ── 解码器（5 层，从 H/32 上采样回 H）──
        # 原实现写死 resnet34 通道；为支持 resnet50，按编码器自适应。
        # (stem, layer1, layer2, layer3, layer4) 的输出通道：
        encoder_channels = {
            'resnet34': (64, 64, 128, 256, 512),
            'resnet50': (64, 256, 512, 1024, 2048),
        }
        stem_ch, l1_ch, l2_ch, l3_ch, l4_ch = encoder_channels[encoder_name]
        self.dec4 = DecoderBlock(l4_ch, l3_ch, l3_ch)  # layer4 + layer3 → 256
        self.dec3 = DecoderBlock(l3_ch, l2_ch, l2_ch)  # + layer2        → 128
        self.dec2 = DecoderBlock(l2_ch, l1_ch, l1_ch)    # + layer1        → 64
        self.dec1 = DecoderBlock(l1_ch, stem_ch, 32)     # + stem          → 32
        self.dec0 = DecoderBlock(32, 0, 16)      # 无 skip          → 16

        # 最终 1×1 卷积 → 目标通道数，⚠️ 不加 Sigmoid
        self.head = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, 1, H, W), float32

        Returns:
            logits: (Batch, 1, H, W), float32, 未经过激活函数
        """
        f0 = self.relu(self.bn1(self.conv1(x)))  # (B, 64,  H/2)
        f1 = self.maxpool(f0)                    # (B, 64,  H/4)
        f2 = self.layer1(f1)                     # (B, 64,  H/4)
        f3 = self.layer2(f2)                     # (B, 128, H/8)
        f4 = self.layer3(f3)                     # (B, 256, H/16)
        f5 = self.layer4(f4)                     # (B, 512, H/32)

        x = self.dec4(f5, f4)   # (B, 256, H/16)
        x = self.dec3(x, f3)    # (B, 128, H/8)
        x = self.dec2(x, f2)    # (B, 64,  H/4)
        x = self.dec1(x, f0)    # (B, 32,  H/2)
        x = self.dec0(x)        # (B, 16,  H)

        logits = self.head(x)   # (B, out_channels, H)
        return logits
