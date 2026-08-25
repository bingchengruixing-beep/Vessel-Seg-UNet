"""检查各时相图像的统计差异 + 编码器池化特征的可分性。"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.dataset import PHASE_PREFIXES
from src.models import build_model
from src.transforms import get_val_transforms

ROOT = Path(__file__).resolve().parent.parent.parent
IMG_DIR = ROOT / "data" / "train" / "images"
transform = get_val_transforms(512, True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model("unet_baseline", in_channels=1, out_channels=1, phase_classes=4).to(device)

feats = {}
for phase_id, prefix in enumerate(PHASE_PREFIXES):
    f = next(p for p in sorted(IMG_DIR.glob("*.png")) if p.name.startswith(prefix))
    buf = np.fromfile(str(f), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    out = transform(image=img)["image"]
    arr = out.numpy()
    x = torch.from_numpy(arr).unsqueeze(0).to(device)
    with torch.no_grad():
        pooled = model.phase_encoder[:-1](x)  # 到 Flatten 为止
        logits = model.phase_encoder(x)
    pooled_np = pooled.cpu().numpy().flatten()
    print(f"phase {phase_id} ({prefix}): img mean={arr.mean():.4f} std={arr.std():.4f} "
          f"p5={np.percentile(arr,5):.3f} p95={np.percentile(arr,95):.3f}")
    print(f"   pooled features: mean={pooled_np.mean():.3f} std={pooled_np.std():.3f} logits={logits.cpu().numpy().round(2).tolist()}")
    feats[phase_id] = pooled_np

# 各时相池化特征的两两距离
for i in range(4):
    for j in range(i + 1, 4):
        d = np.linalg.norm(feats[i] - feats[j])
        print(f"feat dist phase{i} vs phase{j}: {d:.3f}")
