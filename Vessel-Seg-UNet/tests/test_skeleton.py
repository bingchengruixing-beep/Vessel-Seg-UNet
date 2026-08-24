import numpy as np
import pytest

from src.skeleton import zhang_suen_thinning


def test_empty_mask_returns_empty_skeleton():
    empty = np.zeros((32, 32), dtype=np.uint8)
    skeleton = zhang_suen_thinning(empty)
    assert skeleton.shape == (32, 32)
    assert not skeleton.any()


def test_thick_line_thins_to_single_pixel():
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[18:21, 5:35] = 1  # 3 像素宽的横线, 长 30

    skeleton = zhang_suen_thinning(mask)

    rows, cols = np.nonzero(skeleton)
    # Zhang-Suen 会在端点修剪 1~2 像素, 长度允许少量缩短但主体必须保留
    assert len(rows) >= 25
    # 宽度压缩为 1: 只剩中心行, 且落在原线范围内
    assert set(rows) == {19}
    # 每个列位置最多一个骨架像素, 且落在原线范围内
    assert (np.bincount(cols, minlength=40) <= 1).all()
    assert cols.min() >= 5 and cols.max() <= 34


def test_full_block_erodes_to_thin_skeleton():
    mask = np.zeros((41, 41), dtype=np.uint8)
    mask[8:33, 8:33] = 1  # 25x25 实心块

    skeleton = zhang_suen_thinning(mask)

    # 骨架必须全部落在原块内部
    assert skeleton[8:33, 8:33].sum() > 0
    assert skeleton[:8].sum() == 0 and skeleton[33:].sum() == 0
    assert skeleton[:, :8].sum() == 0 and skeleton[:, 33:].sum() == 0
    # 骨架是单像素宽的: 不存在 2x2 全前景块
    thick = (
        skeleton[1:, 1:]
        & skeleton[:-1, 1:]
        & skeleton[1:, :-1]
        & skeleton[:-1, :-1]
    )
    assert not thick.any()


def test_input_dtype_flexibility():
    mask_float = np.zeros((20, 20), dtype=np.float32)
    mask_float[9, 5:15] = 0.9
    skeleton = zhang_suen_thinning(mask_float)
    assert skeleton.dtype == bool
    assert skeleton[9, 5:15].all()


def test_rejects_3d_input():
    with pytest.raises(ValueError):
        zhang_suen_thinning(np.zeros((2, 20, 20), dtype=np.uint8))
