"""
[M3] 损失函数定义
解决脑血管造影中前景像素极度稀疏（<5%~10%）的类别不平衡问题。
采用 BCE + Dice 混合损失，避免模型退化为全背景预测。

接口契约:
    logits: 未经 Sigmoid 的原始得分 (B, 1, H, W)
    targets: {0.0, 1.0} 二值张量 (B, 1, H, W)
"""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Soft Dice Loss

    L_dice = 1 - (2 * sum(p * y) + smooth) / (sum(p) + sum(y) + smooth)

    Args:
        smooth: 平滑因子，防止分母为零（默认 1e-6）
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, H, W) 未经 Sigmoid
            targets: (B, 1, H, W) {0.0, 1.0}

        Returns:
            标量 Dice Loss
        """
        preds = torch.sigmoid(logits)
        intersection = (preds * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            preds.sum() + targets.sum() + self.smooth
        )
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """
    BCE + Dice 混合损失

    L_total = bce_weight * L_bce + dice_weight * L_dice

    Args:
        bce_weight: BCE 损失权重（默认 0.5）
        dice_weight: Dice 损失权重（默认 0.5）
        dice_smooth: Dice 平滑因子
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        dice_smooth: float = 1e-6,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_fn = nn.BCEWithLogitsLoss()
        self.dice_fn = DiceLoss(smooth=dice_smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, H, W) 未经 Sigmoid
            targets: (B, 1, H, W) {0.0, 1.0}

        Returns:
            标量混合损失
        """
        bce_loss = self.bce_fn(logits, targets)
        dice_loss = self.dice_fn(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
