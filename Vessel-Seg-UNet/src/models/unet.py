"""
[M2] U-Net 基础模型
经典 4 层编码器-解码器架构 + Skip Connection。
通道数: [64, 128, 256, 512, 1024]

接口契约:
    输入 x:      (Batch, 1, H, W), float32
    输出 logits: (Batch, 1, H, W), float32, 未经过激活函数

核心防错:
    最后一层 **绝对不加 Sigmoid**。
    PyTorch 的 BCEWithLogitsLoss 内置了更数值稳定的 Sigmoid 计算，
    模型只需输出 Logits（实数范围的得分）。
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv2d → BN → ReLU) × 2"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Down(nn.Module):
    """MaxPool2d → DoubleConv (编码器下采样块)"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """上采样 → 拼接 Skip → DoubleConv (解码器上采样块)"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: 来自解码器上一层的特征图（需上采样）
            x2: 来自编码器的 Skip Connection 特征图
        """
        x1 = self.up(x1)

        # 处理输入尺寸不为 2 的幂次时的 padding 对齐
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = nn.functional.pad(
            x1, [diff_x // 2, diff_x - diff_x // 2,
                 diff_y // 2, diff_y - diff_y // 2]
        )

        # 沿通道维拼接 Skip Connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class FiLMBlock(nn.Module):
    """
    FiLM 相位条件化: 对编码器各级特征做逐通道缩放+平移 (1+dγ)·x + dβ。

    相位经 Embedding → 小 MLP 产生每级 2C 维修正量; dγ/dβ 零初始化,
    因此 phase=None 或训练初期等价于恒等映射(无条件网络)。
    """

    def __init__(self, num_classes: int, channels: list, emb_dim: int = 32):
        super().__init__()
        self.num_classes = num_classes
        self.emb = nn.Embedding(num_classes, emb_dim)
        self.layers = nn.ModuleList()
        for channels_stage in channels:
            layer = nn.Linear(emb_dim, channels_stage * 2)
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.layers.append(layer)

    def forward(self, x: torch.Tensor, phase: torch.Tensor, stage: int) -> torch.Tensor:
        if phase is None:
            return x
        h = self.layers[stage](self.emb(phase))          # (B, 2C)
        dgamma, dbeta = h.chunk(2, dim=1)                 # 各 (B, C)
        return x * (1.0 + dgamma.view(x.shape[0], -1, 1, 1)) + dbeta.view(x.shape[0], -1, 1, 1)


class UNetBaseline(nn.Module):
    """
    经典 U-Net：4 层编码 + 4 层解码 + Skip Connection

    通道结构: 1 → 64 → 128 → 256 → 512 → 1024 (bottleneck)
                                                  ↓
              1 ← 64 ← 128 ← 256 ← 512 ←────────┘

    Args:
        in_channels: 输入通道数（灰度图 = 1）
        out_channels: 输出通道数（二值分割 = 1）
        bilinear: 是否使用双线性上采样（True 节省显存）
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        bilinear: bool = True,
        phase_classes: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.phase_classes = int(phase_classes)

        # 编码器
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # 解码器
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # 最终 1×1 卷积映射到目标通道数
        # ⚠️ 不加 Sigmoid，输出 raw logits
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

        # 相位条件化(可选)
        self.film = None
        self.phase_encoder = None
        if self.phase_classes > 0:
            self.film = FiLMBlock(self.phase_classes, [64, 128, 256, 512])
            # 独立相位编码器: 直接吃原始图像。骨干在 FiLM 训练下会学成相位不变
            # 特征, 接在骨干上的分类头无特征可学; 专用分支与 FiLM 完全解耦。
            # 各时相全局统计接近, 差异在局部对比模式 → 需要多层卷积感受野;
            # 用 GroupNorm 替代 BN: 小 batch 微调时无 running-stats 不稳定问题。
            self.phase_encoder = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(4, 16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(4, 32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, 64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, self.phase_classes),
            )

    def forward(
        self,
        x: torch.Tensor,
        phase: torch.Tensor | None = None,
    ):
        """
        前向传播。

        Args:
            x: (Batch, 1, H, W), float32
            phase: (Batch,) LongTensor 相位 id; None = 不施加条件(FiLM 恒等)

        Returns:
            phase_classes == 0: logits (B, 1, H, W)
            phase_classes > 0: (logits, phase_logits (B, C))
        """
        x_input = x
        # 编码器路径（保存 skip features）
        x1 = self.inc(x)       # (B, 64, H, W)
        if self.film is not None:
            x1 = self.film(x1, phase, 0)
        x2 = self.down1(x1)    # (B, 128, H/2, W/2)
        if self.film is not None:
            x2 = self.film(x2, phase, 1)
        x3 = self.down2(x2)    # (B, 256, H/4, W/4)
        if self.film is not None:
            x3 = self.film(x3, phase, 2)
        x4 = self.down3(x3)    # (B, 512, H/8, W/8)
        if self.film is not None:
            x4 = self.film(x4, phase, 3)
        x5 = self.down4(x4)    # (B, 1024/factor, H/16, W/16) — bottleneck

        # 解码器路径（使用 skip connections）
        x = self.up1(x5, x4)   # (B, 512, H/8, W/8)
        x = self.up2(x, x3)    # (B, 256, H/4, W/4)
        x = self.up3(x, x2)    # (B, 128, H/2, W/2)
        x = self.up4(x, x1)    # (B, 64, H, W)

        logits = self.outc(x)  # (B, 1, H, W) — raw logits
        if self.phase_encoder is not None:
            return logits, self.phase_encoder(x_input)
        return logits
