"""
[M5] 后处理滤噪算法
使用连通域分析（Connected Component Analysis）去除网络预测中的
微小噪点斑块，保留真实的血管结构。

接口契约:
    输入/输出均为 (H, W) uint8 二值掩膜 {0, 255}
"""

import cv2
import numpy as np


def remove_small_components(
    binary_mask: np.ndarray,
    min_size: int = 50,
) -> np.ndarray:
    """
    去除二值掩膜中小于 min_size 个像素的独立连通域（噪点）。

    使用 OpenCV 的连通域分析标记每个独立区域，
    面积低于阈值的区域将被置零。

    Args:
        binary_mask: (H, W) uint8 二值掩膜 {0, 255}
        min_size: 最小保留面积（像素数），小于此值的连通域将被删除

    Returns:
        去噪后的 (H, W) uint8 二值掩膜 {0, 255}
    """
    # 确保输入是二值的
    _, binary_mask = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)

    # 连通域分析
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    # 构建去噪后的掩膜
    cleaned = np.zeros_like(binary_mask)

    for label_id in range(1, num_labels):  # 跳过背景 (label 0)
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_size:
            cleaned[labels == label_id] = 255

    return cleaned


def fill_small_holes(
    binary_mask: np.ndarray,
    max_hole_size: int = 100,
) -> np.ndarray:
    """
    填补二值掩膜中小于 max_hole_size 的内部孔洞。

    通过反转掩膜后进行连通域分析来检测孔洞。

    Args:
        binary_mask: (H, W) uint8 二值掩膜 {0, 255}
        max_hole_size: 最大孔洞面积（像素数），小于此值的孔洞将被填补

    Returns:
        填补后的 (H, W) uint8 二值掩膜 {0, 255}
    """
    _, binary_mask = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)

    # 反转掩膜，将孔洞变为前景
    inverted = cv2.bitwise_not(binary_mask)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        inverted, connectivity=8
    )

    result = binary_mask.copy()

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < max_hole_size:
            # 小孔洞 → 填充为前景
            result[labels == label_id] = 255

    return result


def postprocess_mask(
    binary_mask: np.ndarray,
    min_component_size: int = 50,
    max_hole_size: int = 100,
    morph_close_kernel: int = 3,
) -> np.ndarray:
    """
    完整后处理管线：形态学闭运算 → 去小连通域 → 填小孔洞。

    Args:
        binary_mask: (H, W) uint8 二值掩膜
        min_component_size: 连通域最小保留面积
        max_hole_size: 孔洞最大填充面积
        morph_close_kernel: 闭运算核大小

    Returns:
        后处理完成的 (H, W) uint8 二值掩膜
    """
    # 1. 轻度闭运算连接微小断点
    if morph_close_kernel > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_close_kernel, morph_close_kernel)
        )
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # 2. 去除小连通域噪点
    binary_mask = remove_small_components(binary_mask, min_component_size)

    # 3. 填补小孔洞
    binary_mask = fill_small_holes(binary_mask, max_hole_size)

    return binary_mask
