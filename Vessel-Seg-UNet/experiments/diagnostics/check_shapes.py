"""找出经过 val 变换后尺寸不为 512x512 的样本。"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.transforms import get_val_transforms

img_dir = Path("data/train/images")
transform = get_val_transforms(512, True)
bad = []
for p in sorted(img_dir.glob("*.png")):
    buf = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        bad.append((p.name, "unreadable"))
        continue
    out = transform(image=img)["image"]
    if tuple(out.shape) != (1, 512, 512):
        bad.append((p.name, f"src {img.shape} -> out {tuple(out.shape)}"))
print("bad count:", len(bad))
for name, why in bad[:20]:
    print(" ", name, "|", why)
