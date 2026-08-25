"""分析软标注掩膜的像素值分布。"""
import numpy as np
import cv2
from pathlib import Path

ROOT = Path("E:/DSCA/Coding/Vessel-Seg-UNet/dataset")
samples = [
    ROOT / "dataset1/4s/masks/26.png",
    ROOT / "dataset1/4s/masks/30.png",
    ROOT / "dataset2（4s）/masks/38.png",
    ROOT / "dataset2（4s）/masks/50.png",
    ROOT / "dataset2（4s）/masks/66.png",
]
for p in samples:
    buf = np.fromfile(str(p), dtype=np.uint8)
    m = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    # 分布: 0 占比 / 255 占比 / 中间值占比
    zero = float((m == 0).mean())
    full = float((m == 255).mean())
    mid = float(((m > 0) & (m < 255)).mean())
    mid_mean = m[(m > 0) & (m < 255)].mean() if mid > 0 else 0
    mid_std = m[(m > 0) & (m < 255)].std() if mid > 0 else 0
    print(f"{p.parent.name}/{p.name}: 0:{zero:.3f} 255:{full:.3f} mid:{mid:.4f} (mid mean {mid_mean:.0f}±{mid_std:.0f})")
    # 中间值是否集中在 0-127 还是 128-255
    if mid > 0:
        low = float(((m > 0) & (m < 128)).mean())
        high = float(((m >= 128) & (m < 255)).mean())
        print(f"   mid split: [1,127]={low:.4f} [128,254]={high:.4f}")
