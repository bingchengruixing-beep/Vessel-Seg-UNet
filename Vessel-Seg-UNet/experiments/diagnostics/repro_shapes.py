"""对每个样本重复随机训练增强, 捕获非 512 输出。"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.transforms import get_train_transforms

img_dir = Path("data/train/images")
mask_dir = Path("data/train/masks")
transform = get_train_transforms(512, True)

found = 0
for p in sorted(img_dir.glob("*.png")):
    ib = np.fromfile(str(p), dtype=np.uint8)
    mb = np.fromfile(str(mask_dir / p.name), dtype=np.uint8)
    img = cv2.imdecode(ib, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imdecode(mb, cv2.IMREAD_GRAYSCALE)
    for trial in range(30):
        out = transform(image=img, mask=mask)
        oi, om = out["image"].shape, out["mask"].shape
        if tuple(oi) != (1, 512, 512):
            print(f"{p.name} trial {trial}: img {img.shape} mask {mask.shape} -> out image {tuple(oi)}")
            found += 1
            break
    if found >= 5:
        break
print("done, found:", found)
