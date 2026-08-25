"""按数据源分组评估: dataset1 三个时段 / dataset2 / 跨域对照。

用三个检查点(完整数据基线、完整数据+clDice、DIAS 训练+clDice)
分别在 data/val 的各子集上计算指标, 回答"不同数据集效果是否一致"。
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.checkpoints import checkpoint_model_config, load_checkpoint, load_model_state
from src.config import normalize_config
from src.metrics import MetricAccumulator
from src.models import build_model
from src.prediction import predictions_from_logits
from src.transforms import get_val_transforms

ROOT = Path(__file__).resolve().parent.parent.parent
VAL_IMG = ROOT / "data" / "val" / "images"
VAL_MASK = ROOT / "data" / "val" / "masks"

CHECKPOINTS = [
    ("完整数据-基线", ROOT / "checkpoints/exp5_own_baseline/best_model.pth"),
    ("完整数据+clDice", ROOT / "checkpoints/exp5_own_cl_dice/best_model.pth"),
    ("DIAS训练+clDice", ROOT / "checkpoints/exp3_cl_dice/best_model.pth"),
]
GROUPS = [
    ("dataset1-2~3s", "d1_2-3s_"),
    ("dataset1-4s", "d1_4s_"),
    ("dataset1-5~6s", "d1_5-6s_"),
    ("dataset2-4s", "d2_"),
]


def imread_gray(path):
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


files = sorted(VAL_IMG.glob("*.png"))
group_files = {g: [f for f in files if f.name.startswith(prefix)] for g, prefix in GROUPS}
print("验证集组成:", {g: len(v) for g, v in group_files.items()})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'模型':26s} {'子集':16s} {'n':>3s} {'Dice':>8s} {'IoU':>8s} {'P':>8s} {'R':>8s}")
print("-" * 90)
for ck_label, ck_path in CHECKPOINTS:
    ck = load_checkpoint(str(ck_path), map_location=device)
    cfg = normalize_config(ck.get("config") if isinstance(ck.get("config"), dict) else None)
    model_cfg = checkpoint_model_config(ck, cfg)
    model = build_model(
        model_cfg["name"],
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
    ).to(device)
    load_model_state(model, ck)
    model.eval()
    img_size = cfg["inference"]["img_size"] or cfg["dataset"]["img_size"]
    transform = get_val_transforms(img_size, cfg["dataset"]["keep_aspect_ratio"])
    for group, flist in group_files.items():
        acc = MetricAccumulator()
        for f in flist:
            img = imread_gray(VAL_IMG / f.name)
            mask = imread_gray(VAL_MASK / f.name)
            _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            augmented = transform(image=img, mask=mask_bin)
            x = augmented["image"].unsqueeze(0).to(device)
            with torch.no_grad():
                preds = predictions_from_logits(model(x), threshold=0.5)
            target = augmented["mask"]
            if target.dim() == 2:
                target = target.unsqueeze(0)
            target = (target.float() > 0.5).float().unsqueeze(0)
            acc.update(preds.cpu(), target)
        print(f"{ck_label:26s} {group:16s} {len(flist):3d} {acc.dice():8.4f} {acc.iou():8.4f} "
              f"{acc.precision():8.4f} {acc.recall():8.4f}")
