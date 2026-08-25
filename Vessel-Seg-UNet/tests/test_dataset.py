import cv2
import numpy as np
import pytest

from src.dataset import VesselDataset
from src.transforms import get_val_transforms


def _write_png(path, arr):
    ok, buf = cv2.imencode(".png", arr)
    assert ok
    with open(path, "wb") as f:
        f.write(buf.tobytes())


def test_dataset_aligns_mismatched_mask_to_image_size(tmp_path):
    """图像与掩膜尺寸不一致(源数据导出错位)时, __getitem__ 必须输出同尺寸。"""
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()

    # 图像 80x64, 掩膜 60x64(高度少 20) —— 模拟 PadIfNeeded 尺寸漂移场景
    image = np.zeros((80, 64), dtype=np.uint8)
    image[10:70, 10:54] = 128
    mask = np.zeros((60, 64), dtype=np.uint8)
    mask[8:52, 8:56] = 255
    _write_png(img_dir / "a.png", image)
    _write_png(mask_dir / "a.png", mask)

    dataset = VesselDataset(
        image_dir=str(img_dir),
        mask_dir=str(mask_dir),
        transform=get_val_transforms(32, keep_aspect_ratio=True),
    )
    out_image, out_mask = dataset[0]

    assert out_image.shape == out_mask.shape
    assert out_image.shape == (1, 32, 32)
    # 掩膜仍是二值
    assert set(torch_unique(out_mask)) <= {0.0, 1.0}


def torch_unique(tensor):
    import torch
    return set(torch.unique(tensor).tolist())


def test_dataset_returns_skeleton_triple_when_enabled(tmp_path):
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    image = np.zeros((64, 64), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[28:36, 10:54] = 255  # 横线
    _write_png(img_dir / "a.png", image)
    _write_png(mask_dir / "a.png", mask)

    dataset = VesselDataset(
        image_dir=str(img_dir),
        mask_dir=str(mask_dir),
        transform=get_val_transforms(64, keep_aspect_ratio=True),
        return_skeleton=True,
        skeleton_size=64,
    )
    out_image, out_mask, skeleton = dataset[0]
    assert out_image.shape == out_mask.shape == skeleton.shape == (1, 64, 64)
    assert float(skeleton.sum()) > 0  # 横线有骨架
