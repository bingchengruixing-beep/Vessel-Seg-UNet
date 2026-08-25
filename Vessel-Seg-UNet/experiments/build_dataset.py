"""合并用户标注的自有 DSA 数据并划分 train/val。

来源(dataset 文件夹):
    dataset1/2~3s  dataset1/4s  dataset1/5~6s  dataset2(4s)
    各含 normal/ (图像) 与 masks/ (标注), 共 177 对。

输出(项目根下):
    data/train/{images,masks}   (85%, seed 42 随机划分)
    data/val/{images,masks}     (15%)
文件名加分组前缀避免重名冲突, 例如 4s_1.png。

用法: .venv/Scripts/python.exe experiments/build_dataset.py [--val-frac 0.15]
"""
import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = Path("E:/DSCA/Coding/Vessel-Seg-UNet/dataset")

GROUPS = [
    ("dataset1/2~3s", "normal", "masks", "d1_2-3s"),
    ("dataset1/4s", "normal", "masks", "d1_4s"),
    ("dataset1/5~6s", "normal", "masks", "d1_5-6s"),
    ("dataset2（4s）", "normal", "masks", "d2"),
]


def imread_gray(path):
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def collect_pairs():
    pairs = []
    for group, img_sub, mask_sub, prefix in GROUPS:
        img_dir = SRC_ROOT / group / img_sub
        mask_dir = SRC_ROOT / group / mask_sub
        if not img_dir.exists():
            print(f"[skip] {group}: not found")
            continue
        count = 0
        for img in sorted(img_dir.glob("*.png")):
            mask = mask_dir / img.name
            if not mask.exists():
                print(f"[warn] no mask for {group}/{img.name}")
                continue
            pairs.append((prefix, img, mask))
            count += 1
        print(f"[ok] {group}: {count} pairs")
    return pairs


def check_mask(mask_path):
    m = imread_gray(mask_path)
    if m is None:
        return "unreadable"
    vals = set(np.unique(m).tolist())
    if vals <= {0, 255}:
        return "binary"
    return "soft"  # 概率图, 复制时统一阈值 127 二值化


def binarize_and_write(src_img, src_mask, dst):
    """掩膜对齐图像尺寸(含转置修正)后阈值二值化写出(imencode 支持中文路径)。"""
    img = imread_gray(src_img)
    m = imread_gray(src_mask)
    if m is None or img is None:
        raise IOError(f"unreadable: {src_img} / {src_mask}")
    if m.shape != img.shape:
        if m.shape == img.shape[::-1]:
            m = cv2.transpose(m)  # 转置导出的掩膜
        else:
            m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    _, binary = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    ok, buf = cv2.imencode(".png", binary)
    if not ok:
        raise IOError(f"failed to encode {src_mask}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        f.write(buf.tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="清空已有 data/ 输出")
    args = parser.parse_args()

    if args.force:
        import shutil as _shutil
        out_root = ROOT / "data"
        if out_root.exists():
            _shutil.rmtree(out_root)
        print("[clean] removed existing data/")

    pairs = collect_pairs()
    print(f"total pairs: {len(pairs)}")

    # 校验掩膜类型并统计
    kinds = {}
    for _, _, mask in pairs:
        kind = check_mask(mask)
        kinds[kind] = kinds.get(kind, 0) + 1
    print("mask kinds:", kinds)
    if "unreadable" in kinds:
        print("ERROR: unreadable masks present")
        return

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    n_val = max(1, int(round(len(pairs) * args.val_frac)))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    print(f"split: train {len(train_pairs)} / val {len(val_pairs)} (seed {args.seed})")

    for split, split_pairs in (("train", train_pairs), ("val", val_pairs)):
        out_img = ROOT / "data" / split / "images"
        out_mask = ROOT / "data" / split / "masks"
        out_img.mkdir(parents=True, exist_ok=True)
        out_mask.mkdir(parents=True, exist_ok=True)
        for prefix, img, mask in split_pairs:
            new_name = f"{prefix}_{img.name}"
            shutil.copy2(img, out_img / new_name)
            binarize_and_write(img, mask, out_mask / new_name)
        print(f"[done] data/{split}: {len(split_pairs)} pairs")



if __name__ == "__main__":
    main()
