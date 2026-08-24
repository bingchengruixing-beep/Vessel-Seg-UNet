"""面向脑血管细结构的多尺度融合分割模型。"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet34_Weights, resnet34


def _group_count(channels: int) -> int:
    """选择能整除通道数的 GroupNorm 分组数量。"""
    for groups in (16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Module):
    """卷积、归一化和激活组合。"""

    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SqueezeExcitation(nn.Module):
    """以全局上下文重新标定通道响应。"""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.projection(self.pool(x))


class GatedSkip(nn.Module):
    """根据解码器上下文抑制跳连中的背景纹理，同时保留基础响应。"""

    def __init__(self, gate_channels: int, skip_channels: int, hidden_channels: int):
        super().__init__()
        self.gate = nn.Conv2d(gate_channels, hidden_channels, 1, bias=False)
        self.skip = nn.Conv2d(skip_channels, hidden_channels, 1, bias=False)
        self.norm = nn.GroupNorm(_group_count(hidden_channels), hidden_channels)
        self.score = nn.Conv2d(hidden_channels, 1, 1)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        score = self.norm(self.gate(gate) + self.skip(skip))
        score = torch.sigmoid(self.score(F.silu(score, inplace=True)))
        return skip * (0.5 + score)


class MultiScaleRefine(nn.Module):
    """并行膨胀卷积捕获不同直径的血管，再以残差方式细化。"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        branch_channels = max(out_channels // 3, 16)
        self.branches = nn.ModuleList(
            [ConvNormAct(in_channels, branch_channels, dilation=value) for value in (1, 2, 3)]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_channels * 3, out_channels, 1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )
        self.residual = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.attention = SqueezeExcitation(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused = self.fuse(torch.cat([branch(x) for branch in self.branches], dim=1))
        return self.attention(fused + self.residual(x))


class FusionDecoderBlock(nn.Module):
    """上采样、门控跳连和多尺度细化模块。"""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.gated_skip = GatedSkip(in_channels, skip_channels, max(out_channels // 2, 16))
        self.refine = MultiScaleRefine(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.refine(torch.cat([x, self.gated_skip(x, skip)], dim=1))


class VesselFusion(nn.Module):
    """ResNet34、ASPP、门控跳连和多尺度细化组成的血管分割网络。"""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        pretrained: bool = True,
        deep_supervision: bool = True,
    ):
        super().__init__()
        try:
            encoder = resnet34(
                weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            )
        except (OSError, RuntimeError) as exc:
            if not pretrained:
                raise
            warnings.warn(f"ImageNet 权重加载失败，将使用随机初始化: {exc}")
            encoder = resnet34(weights=None)

        if in_channels != 3:
            old_weight = encoder.conv1.weight.data
            encoder.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                if in_channels == 1:
                    encoder.conv1.weight.copy_(old_weight.mean(dim=1, keepdim=True))
                else:
                    encoder.conv1.weight.copy_(old_weight[:, :in_channels])

        self.conv1 = encoder.conv1
        self.bn1 = encoder.bn1
        self.relu = encoder.relu
        self.maxpool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.aspp = nn.Sequential(
            ConvNormAct(512, 192, dilation=1),
            ConvNormAct(192, 192, dilation=3),
            SqueezeExcitation(192),
        )
        self.dec3 = FusionDecoderBlock(192, 256, 160)
        self.dec2 = FusionDecoderBlock(160, 128, 112)
        self.dec1 = FusionDecoderBlock(112, 64, 80)
        self.dec0 = FusionDecoderBlock(80, 64, 64)
        self.final_refine = nn.Sequential(
            ConvNormAct(64, 64, dilation=1),
            ConvNormAct(64, 64, dilation=2),
        )
        self.head = nn.Conv2d(64, out_channels, 1)
        self.aux_head_1 = nn.Conv2d(80, out_channels, 1)
        self.aux_head_2 = nn.Conv2d(112, out_channels, 1)
        self.deep_supervision = deep_supervision

    @staticmethod
    def _up_to(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=reference.shape[2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor):
        f0 = self.relu(self.bn1(self.conv1(x)))
        f1 = self.layer1(self.maxpool(f0))
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        bottleneck = self.aspp(f4)

        d3 = self.dec3(bottleneck, f3)
        d2 = self.dec2(d3, f2)
        d1 = self.dec1(d2, f1)
        d0 = self.dec0(d1, f0)
        main_logits = self.head(self._up_to(self.final_refine(d0), x))
        if not self.deep_supervision:
            return main_logits
        aux_1 = self.aux_head_1(self._up_to(d1, x))
        aux_2 = self.aux_head_2(self._up_to(d2, x))
        return main_logits, aux_1, aux_2
