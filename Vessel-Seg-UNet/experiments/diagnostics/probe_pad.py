"""直接测试 albumentations 1.4.7 PadIfNeeded 行为。"""
import cv2
import numpy as np
import albumentations as A

img = np.zeros((535, 489, 3), dtype=np.uint8)
for kwargs in [
    dict(min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT, value=0),
    dict(min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT),
]:
    try:
        t = A.Compose([A.PadIfNeeded(**kwargs)], is_check_shapes=False)
        out = t(image=img)["image"]
        print(kwargs, "->", out.shape)
    except Exception as exc:
        print(kwargs, "-> ERROR:", str(exc)[:150])

# 灰度 2D 输入
img2 = np.zeros((535, 489), dtype=np.uint8)
t = A.Compose([A.PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT, value=0)], is_check_shapes=False)
print("2D:", t(image=img2)["image"].shape)

# LongestMaxSize 对小图(最长边<512)的行为
t2 = A.Compose([A.LongestMaxSize(max_size=512, interpolation=cv2.INTER_LINEAR)], is_check_shapes=False)
print("LMS on 535x489:", t2(image=img2)["image"].shape)
print("LMS on 600x489:", t2(image=np.zeros((600, 489), np.uint8))["image"].shape)
