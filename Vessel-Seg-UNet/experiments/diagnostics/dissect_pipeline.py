"""对问题样本 d1_5-6s_1 逐步拆解增强管线, 定位掩膜尺寸异常。"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import albumentations as A

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

name = "d1_5-6s_1.png"
ib = np.fromfile(str(Path("data/train/images") / name), dtype=np.uint8)
mb = np.fromfile(str(Path("data/train/masks") / name), dtype=np.uint8)
img = cv2.imdecode(ib, cv2.IMREAD_GRAYSCALE)
mask = cv2.imdecode(mb, cv2.IMREAD_GRAYSCALE)
print("src:", img.shape, mask.shape)

steps = {
    "LMS": A.Compose([A.LongestMaxSize(max_size=512, interpolation=cv2.INTER_LINEAR)], is_check_shapes=False),
    "LMS+Pad": A.Compose([
        A.LongestMaxSize(max_size=512, interpolation=cv2.INTER_LINEAR),
        A.PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT, value=0),
    ], is_check_shapes=False),
}
for label, t in steps.items():
    out = t(image=img, mask=mask)
    print(f"{label}: img {out['image'].shape} mask {out['mask'].shape}")

# 完整 train 管线一次
from src.transforms import get_train_transforms
t = get_train_transforms(512, True)
for i in range(5):
    out = t(image=img, mask=mask)
    print(f"train[{i}]: img {out['image'].shape} mask {out['mask'].shape}")
