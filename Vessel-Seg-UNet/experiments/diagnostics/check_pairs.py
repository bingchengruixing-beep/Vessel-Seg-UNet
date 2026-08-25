"""检查 image/mask 配对尺寸是否一致, 并用 train transform 复现 collate 失败。"""
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

mismatch = 0
bad_out = []
for p in sorted(img_dir.glob("*.png")):
    ib = np.fromfile(str(p), dtype=np.uint8)
    mb = np.fromfile(str(mask_dir / p.name), dtype=np.uint8)
    img = cv2.imdecode(ib, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imdecode(mb, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        print("UNREADABLE:", p.name)
        continue
    if img.shape != mask.shape:
        mismatch += 1
        if mismatch <= 10:
            print(f"MISMATCH {p.name}: img {img.shape} mask {mask.shape}")
    out = transform(image=img, mask=mask)
    oi, om = out["image"].shape, out["mask"].shape
    if tuple(oi) != (1, 512, 512) or tuple(om) != (1, 512, 512):
        bad_out.append((p.name, img.shape, mask.shape, tuple(oi), tuple(om)))
print("mismatch pairs:", mismatch)
print("non-512 outputs:", len(bad_out))
for row in bad_out[:15]:
    print(" ", row)
