"""方案 B: 相位标定 —— 按数据源分组扫描阈值与后处理参数。

用现有检查点(默认 exp5_own_cl_dice), 对每个分组在验证子集上网格搜索
(threshold × min_component_size) 最大化 Dice, 输出相位标定表与 JSON。

用法: .venv/Scripts/python.exe -u experiments/phase_calibration.py
"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.checkpoints import checkpoint_model_config, load_checkpoint, load_model_state
from src.config import normalize_config
from src.metrics import MetricAccumulator
from src.models import build_model
from src.prediction import predictions_from_logits
from src.transforms import get_val_transforms

ROOT = Path(__file__).resolve().parent.parent
VAL_IMG = ROOT / "data" / "val" / "images"
VAL_MASK = ROOT / "data" / "val" / "masks"
CHECKPOINT = ROOT / "checkpoints" / "exp5_own_cl_dice" / "best_model.pth"

GROUPS = [
    ("dataset1-2~3s", "d1_2-3s_"),
    ("dataset1-4s", "d1_4s_"),
    ("dataset1-5~6s", "d1_5-6s_"),
    ("dataset2-4s", "d2_"),
]
THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(9)]  # 0.30 ~ 0.70
MIN_SIZES = [0, 5, 10, 20]


def imread_gray(path):
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = load_checkpoint(str(CHECKPOINT), map_location=device)
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
    pp_base = dict(cfg["inference"]["postprocess"])
    return model, transform, pp_base, device


def evaluate(model, transform, pp_base, device, flist, threshold, min_size):
    acc = MetricAccumulator()
    for f in flist:
        img = imread_gray(VAL_IMG / f.name)
        mask = imread_gray(VAL_MASK / f.name)
        _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        aug = transform(image=img, mask=mask_bin)
        x = aug["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            pp_cfg = dict(pp_base)
            pp_cfg["min_component_size"] = min_size if min_size > 0 else pp_base["min_component_size"]
            preds = predictions_from_logits(
                model(x), threshold=threshold,
                apply_postprocess=min_size > 0, postprocess_config=pp_cfg,
            )
        target = aug["mask"]
        if target.dim() == 2:
            target = target.unsqueeze(0)
        target = (target.float() > 0.5).float().unsqueeze(0)
        acc.update(preds.cpu(), target)
    return acc.dice()


def main():
    model, transform, pp_base, device = load_model()
    files = sorted(VAL_IMG.glob("*.png"))
    group_files = {g: [f for f in files if f.name.startswith(p)] for g, p in GROUPS}

    results = {}
    default_rows = {}
    for group, flist in group_files.items():
        base_dice = evaluate(model, transform, pp_base, device, flist, 0.5, 0)
        default_rows[group] = base_dice
        best = {"threshold": 0.5, "min_size": 0, "dice": base_dice}
        for th in THRESHOLDS:
            for ms in MIN_SIZES:
                dice = evaluate(model, transform, pp_base, device, flist, th, ms)
                if dice > best["dice"]:
                    best = {"threshold": th, "min_size": ms, "dice": dice}
        results[group] = {**best, "n": len(flist), "default_dice": round(base_dice, 4)}
        print(f"{group:16s} n={len(flist):2d} | 默认(0.5/off) {base_dice:.4f} | "
              f"最优 th={best['threshold']:.2f} min={best['min_size']:2d} -> {best['dice']:.4f} "
              f"(Δ {best['dice']-base_dice:+.4f})")

    # 部署视角: 各组用各自标定参数, 逐图 Dice 平均(与既有报告口径一致)
    per_image_dice = []
    for group, flist in group_files.items():
        cfg_row = results[group]
        for f in flist:
            img = imread_gray(VAL_IMG / f.name)
            mask = imread_gray(VAL_MASK / f.name)
            _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            aug = transform(image=img, mask=mask_bin)
            x = aug["image"].unsqueeze(0).to(device)
            with torch.no_grad():
                pp_cfg = dict(pp_base)
                if cfg_row["min_size"] > 0:
                    pp_cfg["min_component_size"] = cfg_row["min_size"]
                preds = predictions_from_logits(
                    model(x), threshold=cfg_row["threshold"],
                    apply_postprocess=cfg_row["min_size"] > 0, postprocess_config=pp_cfg,
                )
            target = aug["mask"]
            if target.dim() == 2:
                target = target.unsqueeze(0)
            target = (target.float() > 0.5).float().unsqueeze(0)
            acc = MetricAccumulator()
            acc.update(preds.cpu(), target)
            per_image_dice.append(acc.dice())
    overall = float(np.mean(per_image_dice))
    print(f"\n整体(各组标定参数, 逐图均值): Dice {overall:.4f}")

    out = {
        "checkpoint": str(CHECKPOINT),
        "default_overall_note": "默认参数(0.5/无后处理)整体逐图均值 0.7730(见 exp5 报告)",
        "per_group": results,
        "overall_with_calibration": round(overall, 4),
    }
    out_path = ROOT / "experiments" / "phase_calibration.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved:", out_path)


if __name__ == "__main__":
    main()
