"""
[M4] 评价指标计算
保证评测体系的科学性：Dice、IoU、Precision、Recall。

接口契约:
    所有函数的 preds_binary 必须是经过 Sigmoid 且阈值化后的 {0, 1} 张量。
    targets 为 {0.0, 1.0} 二值张量。
"""

import torch


def calculate_dice(
    preds_binary: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    """
    计算单批次的 Dice 系数 (F1-Score)。

    Dice = 2 * |P ∩ T| / (|P| + |T| + eps)

    Args:
        preds_binary: (B, 1, H, W) 二值预测 {0, 1}
        targets: (B, 1, H, W) 金标准 {0.0, 1.0}
        eps: 平滑因子

    Returns:
        Dice 系数（标量）
    """
    preds_flat = preds_binary.view(-1)
    targets_flat = targets.view(-1)

    intersection = (preds_flat * targets_flat).sum()
    dice = (2.0 * intersection + eps) / (
        preds_flat.sum() + targets_flat.sum() + eps
    )
    return dice.item()


def calculate_iou(
    preds_binary: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    """
    计算单批次的 IoU (Jaccard Index)。

    IoU = |P ∩ T| / |P ∪ T| + eps

    Args:
        preds_binary: (B, 1, H, W) 二值预测 {0, 1}
        targets: (B, 1, H, W) 金标准 {0.0, 1.0}
        eps: 平滑因子

    Returns:
        IoU 值（标量）
    """
    preds_flat = preds_binary.view(-1)
    targets_flat = targets.view(-1)

    intersection = (preds_flat * targets_flat).sum()
    union = preds_flat.sum() + targets_flat.sum() - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.item()


def calculate_precision(
    preds_binary: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    """
    计算 Precision (精确率)。

    Precision = TP / (TP + FP + eps)

    Args:
        preds_binary: (B, 1, H, W) 二值预测
        targets: (B, 1, H, W) 金标准

    Returns:
        Precision 值（标量）
    """
    preds_flat = preds_binary.view(-1)
    targets_flat = targets.view(-1)

    tp = (preds_flat * targets_flat).sum()
    fp = (preds_flat * (1 - targets_flat)).sum()
    precision = (tp + eps) / (tp + fp + eps)
    return precision.item()


def calculate_recall(
    preds_binary: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    """
    计算 Recall (召回率)。

    Recall = TP / (TP + FN + eps)

    Args:
        preds_binary: (B, 1, H, W) 二值预测
        targets: (B, 1, H, W) 金标准

    Returns:
        Recall 值（标量）
    """
    preds_flat = preds_binary.view(-1)
    targets_flat = targets.view(-1)

    tp = (preds_flat * targets_flat).sum()
    fn = ((1 - preds_flat) * targets_flat).sum()
    recall = (tp + eps) / (tp + fn + eps)
    return recall.item()
