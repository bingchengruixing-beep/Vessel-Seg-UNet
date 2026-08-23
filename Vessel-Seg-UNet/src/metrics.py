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
    # eps 只出现在分母：空预测时返回 0 而不是 0.5。
    precision = tp / (tp + fp + eps)
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
    # eps 只出现在分母：空目标时返回 0 而不是 0.5。
    recall = tp / (tp + fn + eps)
    return recall.item()


class MetricAccumulator:
    """全数据集像素级统计累加器，用于与 batch 划分无关的稳定指标。

    逐 batch 平均指标会让每个 batch 权重相同（无论其内容多少）；
    累加原始计数后在数据集层面一次性计算，得到与指标定义一致的
    全局 Dice / IoU / Precision / Recall。
    """

    def __init__(self) -> None:
        self.intersection = 0.0
        self.pred_sum = 0.0
        self.target_sum = 0.0

    def update(self, preds_binary: torch.Tensor, targets: torch.Tensor) -> None:
        """累积一个 batch 的二值预测与金标准统计量。"""
        self.intersection += float((preds_binary * targets).sum().item())
        self.pred_sum += float(preds_binary.sum().item())
        self.target_sum += float(targets.sum().item())

    def dice(self, eps: float = 1e-6) -> float:
        """全局 Dice = 2|P∩T| / (|P| + |T| + eps)；P、T 均空时约定为 1.0。"""
        return (2.0 * self.intersection + eps) / (
            self.pred_sum + self.target_sum + eps
        )

    def iou(self, eps: float = 1e-6) -> float:
        """全局 IoU = |P∩T| / (|P∪T| + eps)。"""
        union = self.pred_sum + self.target_sum - self.intersection
        return (self.intersection + eps) / (union + eps)

    def precision(self, eps: float = 1e-6) -> float:
        """全局 Precision；空预测时为 0。"""
        fp = self.pred_sum - self.intersection
        return self.intersection / (self.intersection + fp + eps)

    def recall(self, eps: float = 1e-6) -> float:
        """全局 Recall；空目标时为 0。"""
        fn = self.target_sum - self.intersection
        return self.intersection / (self.intersection + fn + eps)
