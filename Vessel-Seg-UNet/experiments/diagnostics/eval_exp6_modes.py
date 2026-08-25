"""exp6 多模式评估: 真实相位 / 预测相位(部署) / 无条件(不提供相位) + 相位预判定准确率。"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.checkpoints import checkpoint_model_config, load_checkpoint, load_model_state
from src.config import normalize_config
from src.metrics import MetricAccumulator
from src.models import build_model
from src.prediction import predictions_from_logits
from src.transforms import get_val_transforms

ROOT = Path(__file__).resolve().parent.parent.parent
VAL_IMG = ROOT / "data" / "val" / "images"
VAL_MASK = ROOT / "data" / "val" / "masks"
CHECKPOINT = ROOT / "checkpoints" / (sys.argv[1] if len(sys.argv) > 1 else "exp6_phase_film") / "best_model.pth"
GROUPS = [
    ("dataset1-2~3s", "d1_2-3s_"),
    ("dataset1-4s", "d1_4s_"),
    ("dataset1-5~6s", "d1_5-6s_"),
    ("dataset2-4s", "d2_"),
]
PHASE_OF_PREFIX = {p: i for i, (_, p) in enumerate(GROUPS)}
MODES = ("true_phase", "predicted_phase", "no_phase")


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
    img_size = cfg["inference"]["img_size"] or cfg["dataset"]["img_size"]
    transform = get_val_transforms(img_size, cfg["dataset"]["keep_aspect_ratio"])

    files = sorted(VAL_IMG.glob("*.png"))
    group_files = {g: [f for f in files if f.name.startswith(p)] for g, p in GROUPS}

    accs = {mode: {g: MetricAccumulator() for g, _ in GROUPS} for mode in MODES}
    correct = {g: 0 for g, _ in GROUPS}
    total = {g: 0 for g, _ in GROUPS}
    confusion = np.zeros((4, 4), dtype=int)

    for group, flist in group_files.items():
        for f in flist:
            img = imread_gray(VAL_IMG / f.name)
            mask = imread_gray(VAL_MASK / f.name)
            _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            aug = transform(image=img, mask=mask_bin)
            x = aug["image"].unsqueeze(0).to(device)
            true_phase = PHASE_OF_PREFIX[next(p for _, p in GROUPS if f.name.startswith(p))]
            target = aug["mask"]
            if target.dim() == 2:
                target = target.unsqueeze(0)
            target = (target.float() > 0.5).float().unsqueeze(0)

            with torch.no_grad():
                # 无条件前向(同时拿到 phase logits 用于预判定)
                output = model(x, None)
                logits = output[0] if isinstance(output, tuple) else output
                phase_logits = output[1] if isinstance(output, tuple) else None
                predicted = int(phase_logits.argmax(dim=1).item()) if phase_logits is not None else None

                preds_no = predictions_from_logits(logits, threshold=0.5)
                accs["no_phase"][group].update(preds_no.cpu(), target)

                if phase_logits is not None:
                    out_pred = model(x, torch.tensor([predicted], device=device))
                    logits_pred = out_pred[0]
                    preds_pred = predictions_from_logits(logits_pred, threshold=0.5)
                    accs["predicted_phase"][group].update(preds_pred.cpu(), target)

                out_true = model(x, torch.tensor([true_phase], device=device))
                logits_true = out_true[0]
                preds_true = predictions_from_logits(logits_true, threshold=0.5)
                accs["true_phase"][group].update(preds_true.cpu(), target)

            if predicted is not None:
                correct[group] += int(predicted == true_phase)
                total[group] += 1
                confusion[true_phase, predicted] += 1

    print(f"{'模式':18s} {'子集':16s} {'n':>3s} {'Dice':>8s}")
    for mode in MODES:
        overall = []
        for group, _ in GROUPS:
            d = accs[mode][group].dice()
            overall.append((group, d))
        for group, d in overall:
            print(f"{mode:18s} {group:16s} {len(group_files[group]):3d} {d:8.4f}")

    print("\n相位预判定准确率:")
    for group, _ in GROUPS:
        if total[group]:
            print(f"  {group:16s} {correct[group]}/{total[group]} = {correct[group]/total[group]:.1%}")
    print("混淆矩阵(行=真实, 列=预测):")
    print(confusion)


if __name__ == "__main__":
    main()
