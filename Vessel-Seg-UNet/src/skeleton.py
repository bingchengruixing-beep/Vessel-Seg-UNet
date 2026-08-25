"""Zhang-Suen 骨架化(细化)算法,用于 clDice 中心线监督。

纯 numpy 实现,无额外依赖。输入 (H, W) 二值掩膜 {0, 1},输出单像素宽骨架。
"""

from __future__ import annotations

import numpy as np

def zhang_suen_thinning(binary: np.ndarray) -> np.ndarray:
    """对二值掩膜做 Zhang-Suen 细化,返回 bool 骨架 (H, W)。

    Args:
        binary: (H, W) 数组,前景为 True / >0.5 的值

    Returns:
        (H, W) bool 数组,True 表示骨架像素
    """
    if binary.ndim != 2:
        raise ValueError("zhang_suen_thinning expects a 2D binary mask")
    source = np.asarray(binary) > 0.5
    result = np.zeros(source.shape, dtype=bool)
    positions = np.argwhere(source)
    if positions.size == 0:
        return result

    # 黑色背景不会影响细化，只处理前景包围盒可显著减少 1024 图像的临时数组。
    top, left = positions.min(axis=0)
    bottom, right = positions.max(axis=0) + 1
    img = source[top:bottom, left:right].astype(np.uint8)

    while True:
        deleted_any = False
        for step in (0, 1):
            padded = np.pad(img, 1, mode="constant")
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]

            n_neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = np.zeros_like(img, dtype=np.uint8)
            transitions += (p2 == 0) & (p3 == 1)
            transitions += (p3 == 0) & (p4 == 1)
            transitions += (p4 == 0) & (p5 == 1)
            transitions += (p5 == 0) & (p6 == 1)
            transitions += (p6 == 0) & (p7 == 1)
            transitions += (p7 == 0) & (p8 == 1)
            transitions += (p8 == 0) & (p9 == 1)
            transitions += (p9 == 0) & (p2 == 1)

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
    result[top:bottom, left:right] = img.astype(bool)
    return result
