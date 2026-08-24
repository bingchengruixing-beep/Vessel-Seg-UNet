"""
[M2] Attention U-Net
在标准 U-Net 的 Skip Connection 上引入注意力门控机制，
增强对末梢细支血管的捕捉能力。

接口契约:
    输入 x:      (Batch, 1, H, W), float32
    输出 logits: (Batch, 1, H, W), float32, 未经过激活函数
"""

import torch
import torch.nn as nn

from src.models.unet import DoubleConv, Down


class AttentionGate(nn.Module):
    """
    注意力门控模块 (Additive Attention)

    将解码器的门控信号 g 与编码器的跳跃连接 x 对齐，
    生成空间注意力权重图，抑制无关区域、突出血管结构。

    Args:
        F_g: 门控信号的通道数（来自解码器上一层）
        F_l: 跳跃连接的通道数（来自编码器）
        F_int: 中间层通道数（压缩维度）
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: 门控信号 (B, F_g, H, W)
            x: 跳跃连接 (B, F_l, H, W)

        Returns:
            加权后的跳跃连接 (B, F_l, H, W)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUp(nn.Module):
    """上采样 → 注意力门控 → 拼接 → DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.attn = AttentionGate(
            F_g=in_channels // 2,
            F_l=in_channels // 2,
            F_int=in_channels // 4,
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # Padding 对齐
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = nn.functional.pad(
            x1, [diff_x // 2, diff_x - diff_x // 2,
                 diff_y // 2, diff_y - diff_y // 2]
        )

        # 注意力加权
        x2 = self.attn(g=x1, x=x2)

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class AttentionUNet(nn.Module):
    """
    Attention U-Net：在每层 Skip Connection 上加 Attention Gate。

    Args:
        in_channels: 输入通道数（灰度 = 1）
        out_channels: 输出通道数（二值分割 = 1）
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()

        # 编码器
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # 解码器（带注意力门控）
        self.up1 = AttentionUp(1024, 512)
        self.up2 = AttentionUp(512, 256)
        self.up3 = AttentionUp(256, 128)
        self.up4 = AttentionUp(128, 64)

        # ⚠️ 不加 Sigmoid，输出 raw logits
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, 1, H, W), float32
        Returns:
            logits: (Batch, 1, H, W), float32, 未经过激活函数
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits
