"""探测新加入 dataset 文件夹的图像/掩膜格式。"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("E:/DSCA/Coding/Vessel-Seg-UNet/dataset")


def imread_gray(path):
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def imread_color(path):
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


groups = [
    ("dataset1/2~3s", "normal", "masks"),
    ("dataset1/4s", "normal", "masks"),
    ("dataset1/5~6s", "normal", "masks"),
    ("dataset2（4s）", "normal", "masks"),
]
for group, img_sub, mask_sub in groups:
    img_dir = ROOT / group / img_sub
    mask_dir = ROOT / group / mask_sub
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() == ".png")
    masks = sorted(p for p in mask_dir.iterdir() if p.suffix.lower() == ".png" and p.stem != "segmentation_result")
    print(f"=== {group}: {len(imgs)} imgs / {len(masks)} masks ===")
    img = imread_gray(imgs[0])
    mask_gray = imread_gray(masks[0])
    mask_bgr = imread_color(masks[0])
    if mask_bgr is not None:
        # 彩色掩膜: 三个通道是否一致?
        same = (mask_bgr[:, :, 0] == mask_bgr[:, :, 1]).all() and (mask_bgr[:, :, 1] == mask_bgr[:, :, 2]).all()
        print("  mask is color PNG, channels equal:", same)
    vals, counts = np.unique(mask_gray, return_counts=True)
    print("  img:", img.shape, img.dtype, "min/max", img.min(), img.max())
    print("  mask:", mask_gray.shape, mask_gray.dtype, "unique:", dict(zip(vals.tolist(), counts.tolist())))
    print("  mask fg ratio: %.4f" % (float((mask_gray > 127).sum()) / mask_gray.size))
    # 检查所有图像尺寸是否一致
    sizes = {}
    for p in imgs:
        s = imread_gray(p).shape
        sizes[s] = sizes.get(s, 0) + 1
    print("  img size distribution:", sizes)
