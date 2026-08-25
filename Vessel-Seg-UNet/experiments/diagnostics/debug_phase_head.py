"""诊断相位头: 检查训练集上头的输出分布与准确率。"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.checkpoints import checkpoint_model_config, load_checkpoint, load_model_state
from src.config import normalize_config
from src.dataset import phase_id_from_filename
from src.models import build_model
from src.transforms import get_val_transforms

ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINT = ROOT / "checkpoints" / (sys.argv[1] if len(sys.argv) > 1 else "exp6c_phase_head_fixed") / "best_model.pth"


def imread_gray(path):
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = load_checkpoint(str(CHECKPOINT), map_location=device)
    cfg = normalize_config(ck.get("config"))
    model_cfg = checkpoint_model_config(ck, cfg)
    model = build_model(
        model_cfg["name"],
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        phase_classes=model_cfg.get("phase_classes", 0),
    ).to(device)
    load_model_state(model, ck)
    model.eval()
    transform = get_val_transforms(
        cfg["inference"]["img_size"] or cfg["dataset"]["img_size"],
        cfg["dataset"]["keep_aspect_ratio"],
    )

    for split in ("train", "val"):
        img_dir = ROOT / "data" / split / "images"
        files = sorted(img_dir.glob("*.png"))
        correct = 0
        logits_all = []
        labels_all = []
        for f in files[:40]:
            img = imread_gray(img_dir / f.name)
            x = transform(image=img)["image"].unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(x, None)
            if not isinstance(out, tuple):
                print("model has no phase head")
                return
            _, phase_logits = out
            pred = int(phase_logits.argmax(dim=1).item())
            true = phase_id_from_filename(f.name)
            correct += int(pred == true)
            logits_all.append(phase_logits.cpu().numpy())
            labels_all.append(true)
        arr = np.concatenate(logits_all, axis=0)
        print(f"[{split}] head accuracy: {correct}/{len(files[:40])}")
        print(f"  logits per-class mean: {arr.mean(axis=0).round(3)}")
        print(f"  predicted distribution: {np.bincount(np.array(labels_all), minlength=4)} vs actual {np.bincount(arr.argmax(axis=1), minlength=4)}")


if __name__ == "__main__":
    main()
