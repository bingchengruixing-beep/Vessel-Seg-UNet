"""
[M3] 损失函数定义
解决脑血管造影中前景像素极度稀疏（<5%~10%）的类别不平衡问题。

提供:
    - BCE + Dice 混合损失(默认,避免模型退化为全背景预测)
    - Focal Tversky Loss(α>β 加重细支血管漏检惩罚)
    - clDice 拓扑保持损失(中心线监督,需传入金标准骨架)
    - CombinedVesselLoss(主损失 + λ·clDice)

接口契约:
    logits: 未经 Sigmoid 的原始得分 (B, 1, H, W)
    targets: {0.0, 1.0} 二值张量 (B, 1, H, W)
    skeleton: {0.0, 1.0} 金标准骨架 (B, 1, H, W),仅 clDice 使用
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

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        skeleton: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, H, W) 未经 Sigmoid
            targets: (B, 1, H, W) {0.0, 1.0}
            skeleton: 可选,仅为了统一 criterion 调用签名,此处不使用

        Returns:
            标量混合损失
        """
        bce_loss = self.bce_fn(logits, targets)
        dice_loss = self.dice_fn(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss (Abraham & Khan 2019)

    Tversky = (TP + s) / (TP + alpha*FN + beta*FP + s)
    L       = mean_b (1 - Tversky_b) ** gamma

    alpha > beta 时加重漏检(FN)惩罚,适合细支血管前景稀疏的场景。

    Args:
        alpha: FN 权重(默认 0.7)
        beta:  FP 权重(默认 0.3)
        gamma: 焦点参数(默认 0.75)
        smooth: 平滑因子
    """

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 0.75,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        skeleton: torch.Tensor | None = None,
    ) -> torch.Tensor:
        preds = torch.sigmoid(logits)
        preds_flat = preds.flatten(start_dim=1)
        targets_flat = targets.flatten(start_dim=1)
        tp = (preds_flat * targets_flat).sum(dim=1)
        fn = ((1.0 - preds_flat) * targets_flat).sum(dim=1)
        fp = (preds_flat * (1.0 - targets_flat)).sum(dim=1)
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fn + self.beta * fp + self.smooth
        )
        return ((1.0 - tversky) ** self.gamma).mean()


def soft_skel(x: torch.Tensor, iters: int = 5) -> torch.Tensor:
    """
    可微的软骨架化(Shit et al. 2021, clDice)。

    用形态学腐蚀(min-pool 取反再取反)与轮廓剪枝近似骨架,
    全程由 max_pool2d 构成,梯度可回传。
    """
    for _ in range(iters):
        min_pool = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
        contour = F.relu(F.max_pool2d(min_pool, kernel_size=3, stride=1, padding=1) - min_pool)
        x = F.relu(x - contour)
    return x


class CLDiceLoss(nn.Module):
    """
    clDice 拓扑保持损失(Shit et al. 2021)

    Tprec  = |skel(P) ∩ T| / |skel(P)|   (预测骨架的精确率)
    Tsens  = |P ∩ skel(T)| / |skel(T)|   (金标准骨架的召回率)
    L      = 1 - 2*Tprec*Tsens / (Tprec + Tsens)

    其中 skel(P) 用可微软骨架,skel(T) 为预计算的金标准骨架。
    """

    def __init__(self, smooth: float = 1e-6, iters: int = 5):
        super().__init__()
        self.smooth = smooth
        self.iters = iters

    def forward(self, logits: torch.Tensor, skeleton: torch.Tensor) -> torch.Tensor:
        preds = torch.sigmoid(logits)
        skel_pred = soft_skel(preds, self.iters)
        tprec = (skel_pred * skeleton).sum() + self.smooth
        tprec = tprec / (skel_pred.sum() + self.smooth)
        tsens = (preds * skeleton).sum() + self.smooth
        tsens = tsens / (skeleton.sum() + self.smooth)
        cldice = 2.0 * tprec * tsens / (tprec + tsens + self.smooth)
        return 1.0 - cldice


class CombinedVesselLoss(nn.Module):
    """主分割损失 + 可选 clDice 中心线监督的组合损失。"""

    def __init__(self, main_loss: nn.Module, cldice_loss: CLDiceLoss, cldice_weight: float):
        super().__init__()
        self.main_loss = main_loss
        self.cldice_loss = cldice_loss
        self.cldice_weight = cldice_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        skeleton: torch.Tensor | None = None,
    ) -> torch.Tensor:
        total = self.main_loss(logits, targets)
        if skeleton is not None:
            total = total + self.cldice_weight * self.cldice_loss(logits, skeleton)
        return total
