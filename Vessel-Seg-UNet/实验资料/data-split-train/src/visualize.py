"""
[M4] 结果可视化与热力图生成
将预测的张量转回肉眼可看的图像，对比 Ground Truth 生成叠加层，
直观分析模型的假阳性（误判）和假阴性（漏判）。

建议配色:
    - Ground Truth 轮廓: 绿色
    - Prediction 轮廓: 红色
    - 重叠区域（TP）: 黄色混合
"""

import os
import cv2
import numpy as np


def save_overlay_image(
    image: np.ndarray,
    mask_gt: np.ndarray,
    mask_pred: np.ndarray,
    save_path: str,
) -> None:
    """
    将原始造影图、金标准和预测结果拼合为一张对比图并保存。

    输出为三列拼接:
        [原图 + GT轮廓] | [原图 + Pred轮廓] | [误差热力图]

    配色规则:
        - GT 轮廓: 绿色 (0, 255, 0)
        - Pred 轮廓: 红色 (0, 0, 255)
        - 误差图: TP=白色, FP=红色, FN=蓝色

    Args:
        image: (H, W) uint8 灰度原始造影图
        mask_gt: (H, W) uint8 金标准二值掩膜 {0, 255}
        mask_pred: (H, W) uint8 预测二值掩膜 {0, 255}
        save_path: 输出图像保存路径
    """
    # 确保二值化
    _, mask_gt = cv2.threshold(mask_gt, 127, 255, cv2.THRESH_BINARY)
    _, mask_pred = cv2.threshold(mask_pred, 127, 255, cv2.THRESH_BINARY)

    # 灰度转 BGR 用于叠加彩色轮廓
    img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # ── Panel 1: 原图 + GT 轮廓（绿色）──
    panel_gt = img_bgr.copy()
    gt_contours, _ = cv2.findContours(
        mask_gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(panel_gt, gt_contours, -1, (0, 255, 0), 1)

    # ── Panel 2: 原图 + Pred 轮廓（红色）──
    panel_pred = img_bgr.copy()
    pred_contours, _ = cv2.findContours(
        mask_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(panel_pred, pred_contours, -1, (0, 0, 255), 1)

    # ── Panel 3: 误差热力图 ──
    # TP (True Positive) → 白色
    # FP (False Positive) → 红色
    # FN (False Negative) → 蓝色
    gt_bool = mask_gt > 127
    pred_bool = mask_pred > 127

    error_map = np.zeros((*image.shape[:2], 3), dtype=np.uint8)
    tp = gt_bool & pred_bool
    fp = (~gt_bool) & pred_bool
    fn = gt_bool & (~pred_bool)

    error_map[tp] = (255, 255, 255)   # 白色 = 正确
    error_map[fp] = (0, 0, 255)       # 红色 = 假阳性（误判）
    error_map[fn] = (255, 0, 0)       # 蓝色 = 假阴性（漏判）

    # ── 拼接三列 ──
    canvas = np.hstack([panel_gt, panel_pred, error_map])

    # 添加标题
    h = canvas.shape[0]
    header = np.zeros((30, canvas.shape[1], 3), dtype=np.uint8)
    w_panel = image.shape[1]
    cv2.putText(header, "GT (green)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(header, "Pred (red)", (w_panel + 10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(header, "Error: TP/FP/FN", (2 * w_panel + 10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    final = np.vstack([header, canvas])

    # 保存
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    success, buf = cv2.imencode('.png', final)
    if success:
        with open(save_path, 'wb') as f:
            f.write(buf.tobytes())


def save_prediction_grid(
    images: list,
    masks_gt: list,
    masks_pred: list,
    save_path: str,
    max_cols: int = 4,
) -> None:
    """
    将多张预测结果排列为网格图保存。

    每一行: [原图] [GT] [Pred]
    最多 max_cols 组并排。

    Args:
        images: 灰度图列表 [(H, W) uint8, ...]
        masks_gt: GT 掩膜列表
        masks_pred: 预测掩膜列表
        save_path: 输出路径
        max_cols: 每行最大组数
    """
    rows = []
    for i in range(0, len(images), max_cols):
        row_panels = []
        for j in range(i, min(i + max_cols, len(images))):
            img_bgr = cv2.cvtColor(images[j], cv2.COLOR_GRAY2BGR)
            gt_bgr = cv2.cvtColor(masks_gt[j], cv2.COLOR_GRAY2BGR)
            pred_bgr = cv2.cvtColor(masks_pred[j], cv2.COLOR_GRAY2BGR)
            triplet = np.hstack([img_bgr, gt_bgr, pred_bgr])
            row_panels.append(triplet)

        # Pad if needed
        if len(row_panels) < max_cols:
            h, w = row_panels[0].shape[:2]
            for _ in range(max_cols - len(row_panels)):
                row_panels.append(np.zeros((h, w, 3), dtype=np.uint8))

        rows.append(np.hstack(row_panels))

    grid = np.vstack(rows)

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    success, buf = cv2.imencode('.png', grid)
    if success:
        with open(save_path, 'wb') as f:
            f.write(buf.tobytes())
