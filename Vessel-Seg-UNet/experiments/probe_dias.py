import glob
import os

import cv2
import numpy as np


def imread_gray(path):
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


base = "E:/DSCA/image/DSCA/开源数据集；DIAS"
for split in ("train", "val"):
    images = sorted(glob.glob(os.path.join(base, split, "images", "*.png")))
    masks = sorted(glob.glob(os.path.join(base, split, "masks", "*.png")))
    print(f"[{split}] images={len(images)} masks={len(masks)}")
    for p in images[:3]:
        img = imread_gray(p)
        print("  img", os.path.basename(p), img.shape, img.dtype, "min/max", img.min(), img.max())
    for p in masks[:3]:
        m = imread_gray(p)
        vals, counts = np.unique(m, return_counts=True)
        print("  mask", os.path.basename(p), m.shape, m.dtype, "unique:", dict(zip(vals.tolist(), counts.tolist())))
    img_names = {os.path.basename(p) for p in images}
    mask_names = {os.path.basename(p) for p in masks}
    print("  pairs ok:", img_names == mask_names)
    m = imread_gray(masks[0])
    print("  mask[0] fg ratio (vessel pixels): %.4f" % (float((m > 127).sum()) / m.size))
