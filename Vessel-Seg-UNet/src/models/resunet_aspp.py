"""ResNet34 + ASPP + U-Net++ 风格解码器的脑血管分割模型。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet34_Weights, resnet34


class ConvBlock(nn.Module):
    """嵌套解码路径使用的两层卷积块。"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ASPP(nn.Module):
    """使用多种膨胀率提取不同尺度的血管特征。"""

    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()
        branch_channels = 64
        dilations = (1, 3, 5, 7)
        branches = []
        for dilation in dilations:
            kernel_size = 1 if dilation == 1 else 3
            padding = 0 if dilation == 1 else dilation
            branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        branch_channels,
                        kernel_size,
                        padding=padding,
                        dilation=dilation,
                        bias=False,
                    ),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.branches = nn.ModuleList(branches)
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, branch_channels, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(branch_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[2:]
        features = [branch(x) for branch in self.branches]
        global_feature = F.interpolate(
            self.global_branch(x),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return self.project(torch.cat(features + [global_feature], dim=1))


class ResUNetASPP(nn.Module):
    """ResNet34 编码器、ASPP 瓶颈和 U-Net++ 风格嵌套解码器。"""

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
            # 无网络或本地没有权重时仍允许离线启动，网页可继续训练随机初始化模型。
            import warnings
            warnings.warn(f"ImageNet 权重加载失败，将使用随机初始化: {exc}")
            encoder = resnet34(weights=None)

        # 将 ResNet 的 RGB 第一层改成灰度输入，并平均 RGB 初始化权重。
        if in_channels != 3:
            old_weight = encoder.conv1.weight.data
            encoder.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
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

        self.aspp = ASPP(512, 256)

        # 同一尺度保留多级融合结果，形成 U-Net++ 风格的嵌套跳连。
        self.x3_1 = ConvBlock(256 + 256, 256)
        self.x2_1 = ConvBlock(128 + 256, 128)
        self.x2_2 = ConvBlock(128 + 128 + 256, 128)
        self.x1_1 = ConvBlock(64 + 128, 64)
        self.x1_2 = ConvBlock(64 + 64 + 128, 64)
        self.x1_3 = ConvBlock(64 + 64 + 64 + 128, 64)
        self.x0_1 = ConvBlock(64 + 64, 32)
        self.x0_2 = ConvBlock(64 + 32 + 64, 32)

        self.head = nn.Conv2d(32, out_channels, 1)
        self.aux_head_1 = nn.Conv2d(64, out_channels, 1)
        self.aux_head_2 = nn.Conv2d(128, out_channels, 1)
        self.deep_supervision = deep_supervision

    @staticmethod
    def _up_to(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x,
            size=reference.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, x: torch.Tensor):
        f0 = self.relu(self.bn1(self.conv1(x)))
        f1 = self.layer1(self.maxpool(f0))
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        bottleneck = self.aspp(f4)

        x3_1 = self.x3_1(torch.cat([f3, self._up_to(bottleneck, f3)], dim=1))
        x2_1 = self.x2_1(torch.cat([f2, self._up_to(f3, f2)], dim=1))
        x2_2 = self.x2_2(torch.cat([f2, x2_1, self._up_to(x3_1, f2)], dim=1))
        x1_1 = self.x1_1(torch.cat([f1, self._up_to(f2, f1)], dim=1))
        x1_2 = self.x1_2(torch.cat([f1, x1_1, self._up_to(x2_1, f1)], dim=1))
        x1_3 = self.x1_3(
            torch.cat([f1, x1_1, x1_2, self._up_to(x2_2, f1)], dim=1)
        )
        x0_1 = self.x0_1(torch.cat([f0, self._up_to(f1, f0)], dim=1))
        x0_2 = self.x0_2(torch.cat([f0, x0_1, self._up_to(x1_3, f0)], dim=1))

        main_logits = self.head(self._up_to(x0_2, x))
        if not self.deep_supervision:
            return main_logits
        aux_1 = self.aux_head_1(self._up_to(x1_3, x))
        aux_2 = self.aux_head_2(self._up_to(x2_2, x))
        return main_logits, aux_1, aux_2
