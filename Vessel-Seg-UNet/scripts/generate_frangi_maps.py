"""批量生成 Frangi vesselness 图，用于训练前预处理。

为指定目录下的所有图像生成 Frangi 血管增强图，
保存到对应的输出目录中，供双通道训练使用。

用法:
    python scripts/generate_frangi_maps.py
    python scripts/generate_frangi_maps.py --image-dir 训练集/normal --output-dir 训练集/frangi
    python scripts/generate_frangi_maps.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# 将项目根目录加入 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.frangi import vesselness


SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def process_directory(
    image_dir: str | Path,
    output_dir: str | Path,
    sigmas: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0),
    method: str = "hessian",
    beta: float = 0.5,
    c: float | None = None,
) -> int:
    """处理目录下所有图像，生成血管增强图。

    Args:
        image_dir: 原始图像所在目录
        output_dir: 增强图输出目录
        sigmas: 多尺度高斯核参数
        method: 增强方法，"hessian"（推荐DSA）或 "frangi"
        beta: Frangi blobness 抑制参数（仅 frangi 方法使用）
        c: Frangi 背景噪声抑制参数，None 时自动计算（仅 frangi 方法使用）

    Returns:
        处理的图像数量
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(SUPPORTED_EXTS)
    ])

    if not files:
        print(f"警告: 目录 {image_dir} 中没有找到支持的图像文件")
        return 0

    count = 0
    for fname in tqdm(files, desc=f"处理 {image_dir.name}"):
        image_path = image_dir / fname
        output_path = output_dir / fname

        buf = np.fromfile(str(image_path), dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"  跳过无法读取的图像: {fname}")
            continue

        vesselness_map = vesselness(image, sigmas=sigmas, method=method, beta=beta, c=c)
        vesselness_16u = (vesselness_map * 65535.0).astype(np.uint16)
        # 使用 imencode + tofile 支持中文路径
        success, encoded = cv2.imencode(".png", vesselness_16u)
        if success:
            encoded.tofile(str(output_path))
            count += 1
        else:
            print(f"  保存失败: {fname}")

    return count


def main():
    parser = argparse.ArgumentParser(
        description="批量生成 Frangi 血管增强图"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="原始图像目录（与 --output-dir 配合使用）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Frangi 图输出目录",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="从 YAML 配置读取数据路径（自动处理训练集和验证集）",
    )
    parser.add_argument(
        "--sigmas",
        type=str,
        default="1.0,2.0,3.0,4.0,5.0",
        help="多尺度 sigma 参数，逗号分隔（默认 1.0,2.0,3.0,4.0,5.0）",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="blobness 抑制参数（仅 frangi 方法，默认 0.5，DSA 建议 1.5~3.0）",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=None,
        help="背景噪声抑制参数（仅 frangi 方法，默认自动计算）",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="hessian",
        choices=["hessian", "frangi"],
        help="血管增强方法：hessian（推荐DSA，-λ2简化版）或 frangi（经典方法）",
    )
    args = parser.parse_args()

    sigmas = tuple(float(s) for s in args.sigmas.split(","))

    total = 0

    if args.config:
        from src.config import load_config, resolve_data_path

        config = load_config(args.config)
        dataset_cfg = config["dataset"]

        # 使用配置中的 frangi 输出目录，或默认在图像目录同级创建 frangi 子目录
        frangi_cfg = dataset_cfg.get("frangi", {})
        train_frangi_dir = frangi_cfg.get("train_frangi_dir", "")
        val_frangi_dir = frangi_cfg.get("val_frangi_dir", "")

        # 训练集
        train_image_dir = resolve_data_path(
            dataset_cfg["train_image_dir"], PROJECT_ROOT
        )
        if not train_frangi_dir:
            train_frangi_dir = train_image_dir.parent / "frangi"
        print(f"训练集: {train_image_dir} → {train_frangi_dir}")
        total += process_directory(
            train_image_dir, train_frangi_dir,
            sigmas=sigmas, method=args.method, beta=args.beta, c=args.c,
        )

        # 验证集
        val_image_dir = resolve_data_path(
            dataset_cfg["val_image_dir"], PROJECT_ROOT
        )
        if not val_frangi_dir:
            val_frangi_dir = val_image_dir.parent / "frangi"
        print(f"验证集: {val_image_dir} → {val_frangi_dir}")
        total += process_directory(
            val_image_dir, val_frangi_dir,
            sigmas=sigmas, method=args.method, beta=args.beta, c=args.c,
        )

    elif args.image_dir and args.output_dir:
        total = process_directory(
            args.image_dir, args.output_dir,
            sigmas=sigmas, method=args.method, beta=args.beta, c=args.c,
        )
    else:
        parser.error("请指定 --config 或同时指定 --image-dir 和 --output-dir")

    print(f"\n完成！共处理 {total} 张图像")


if __name__ == "__main__":
    main()