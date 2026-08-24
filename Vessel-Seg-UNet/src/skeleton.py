"""Zhang-Suen 骨架化(细化)算法,用于 clDice 中心线监督。

纯 numpy 实现,无额外依赖。输入 (H, W) 二值掩膜 {0, 1},输出单像素宽骨架。
"""

from __future__ import annotations

import numpy as np

# 8 邻域顺序: P2 P3 P4 P5 P6 P7 P8 P9 (顺时针, P2 在正上方)
_NEIGHBOR_SLICES = (
    (slice(None, -2), slice(1, -1)),   # P2 上
    (slice(None, -2), slice(2, None)), # P3 右上
    (slice(1, -1), slice(2, None)),    # P4 右
    (slice(2, None), slice(2, None)),  # P5 右下
    (slice(2, None), slice(1, -1)),    # P6 下
    (slice(2, None), slice(None, -2)), # P7 左下
    (slice(1, -1), slice(None, -2)),   # P8 左
    (slice(None, -2), slice(None, -2)),# P9 左上
)


def _transition_count(ring: np.ndarray) -> np.ndarray:
    """统计每个像素的 8 邻域环上 0→1 跳变次数 (Zhang-Suen 的 S(P1))。"""
    # ring: (8, H, W) 的 0/1 邻居张量
    return sum(
        (ring[i] == 0) & (ring[(i + 1) % 8] == 1)
        for i in range(8)
    )


def zhang_suen_thinning(binary: np.ndarray) -> np.ndarray:
    """对二值掩膜做 Zhang-Suen 细化,返回 bool 骨架 (H, W)。

    Args:
        binary: (H, W) 数组,前景为 True / >0.5 的值

    Returns:
        (H, W) bool 数组,True 表示骨架像素
    """
    if binary.ndim != 2:
        raise ValueError("zhang_suen_thinning expects a 2D binary mask")
    img = (np.asarray(binary) > 0.5).astype(np.uint8)
    if not img.any():
        return img.astype(bool)

    while True:
        deleted_any = False
        for step in (0, 1):
            padded = np.pad(img, 1, mode="constant")
            ring = np.stack([
                padded[s0, s1] for s0, s1 in _NEIGHBOR_SLICES
            ])  # (8, H, W)

            n_neighbors = ring.sum(axis=0)
            transitions = _transition_count(ring)

            p2, p4, p6, p8 = ring[0], ring[2], ring[4], ring[6]

            condition_a = (n_neighbors >= 2) & (n_neighbors <= 6)
            condition_b = transitions == 1
            if step == 0:
                condition_cd = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                condition_cd = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)

            to_delete = (img == 1) & condition_a & condition_b & condition_cd
            if to_delete.any():
                img[to_delete] = 0
                deleted_any = True
        if not deleted_any:
            break
    return img.astype(bool)
