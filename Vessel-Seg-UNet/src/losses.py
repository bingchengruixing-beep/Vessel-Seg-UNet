"""
[M3] 损失函数定义
解决脑血管造影中前景像素极度稀疏（<5%~10%）的类别不平衡问题。
采用 BCE + Dice 混合损失，ResUNet-ASPP 可额外加入拓扑 clDice，避免模型退化为全背景预测。

接口契约:
    logits: 未经 Sigmoid 的原始得分 (B, 1, H, W)
    targets: {0.0, 1.0} 二值张量 (B, 1, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice Loss

    L_dice = 1 - mean_b (2 * sum(p_b * y_b) + smooth) / (sum(p_b) + sum(y_b) + smooth)

    逐样本计算 Dice 后在 batch 上取平均，避免大前景样本主导梯度。

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
        preds_flat = preds.flatten(start_dim=1)
        targets_flat = targets.flatten(start_dim=1)
        intersection = (preds_flat * targets_flat).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            preds_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth
        )
        return (1.0 - dice).mean()


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


def _soft_skeletonize(x: torch.Tensor, iterations: int = 5) -> torch.Tensor:
    """用可微形态学操作近似提取血管骨架。"""
    skeleton = torch.zeros_like(x)
    current = x
    for _ in range(iterations):
        eroded = -F.max_pool2d(-current, kernel_size=3, stride=1, padding=1)
        opened = F.max_pool2d(eroded, kernel_size=3, stride=1, padding=1)
        delta = F.relu(current - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
        current = eroded
    return skeleton


class ClDiceLoss(nn.Module):
    """拓扑感知的 clDice 损失，约束血管骨架连通性。"""

    def __init__(self, smooth: float = 1e-6, iterations: int = 5):
        super().__init__()
        self.smooth = smooth
        self.iterations = iterations

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = torch.sigmoid(logits)
        target_skeleton = _soft_skeletonize(targets, self.iterations)
        prediction_skeleton = _soft_skeletonize(predictions, self.iterations)
        precision = ((prediction_skeleton * targets).sum() + self.smooth) / (
            prediction_skeleton.sum() + self.smooth
        )
        recall = ((target_skeleton * predictions).sum() + self.smooth) / (
            target_skeleton.sum() + self.smooth
        )
        cl_dice = (2 * precision * recall + self.smooth) / (
            precision + recall + self.smooth
        )
        return 1.0 - cl_dice


class BCEDiceClDiceLoss(nn.Module):
    """BCE、区域 Dice 和拓扑 clDice 的组合损失。"""

    def __init__(
        self,
        bce_weight: float = 0.3,
        dice_weight: float = 0.55,
        cldice_weight: float = 0.15,
        dice_smooth: float = 1e-6,
        skeleton_iterations: int = 5,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.cldice_weight = cldice_weight
        self.bce_fn = nn.BCEWithLogitsLoss()
        self.dice_fn = DiceLoss(smooth=dice_smooth)
        self.cldice_fn = ClDiceLoss(smooth=dice_smooth, iterations=skeleton_iterations)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.bce_weight * self.bce_fn(logits, targets)
            + self.dice_weight * self.dice_fn(logits, targets)
            + self.cldice_weight * self.cldice_fn(logits, targets)
        )
