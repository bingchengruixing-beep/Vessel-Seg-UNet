import numpy as np

from src.postprocess import fill_small_holes, postprocess_mask, remove_small_components


def test_remove_small_components_keeps_large_blob_and_drops_noise():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:30, 10:30] = 255   # 400 像素的大连通域
    mask[50, 50] = 255         # 1 像素噪点
    mask[51, 50] = 255         # 2 像素噪点

    cleaned = remove_small_components(mask, min_size=50)

    assert cleaned[50, 50] == 0
    assert cleaned[51, 50] == 0
    assert cleaned[10:30, 10:30].min() == 255


def test_fill_small_holes_fills_interior_but_not_background():
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[5:35, 5:35] = 255     # 30x30 前景块
    mask[13:21, 13:21] = 0     # 8x8=64 像素的内部孔洞 (< max_hole_size=100)

    filled = fill_small_holes(mask, max_hole_size=100)

    assert filled[13:21, 13:21].min() == 255   # 孔洞被填补
    assert filled[0, 0] == 0                   # 外部背景不受影响
    assert filled[5, 5] == 255                 # 前景保留


def test_postprocess_pipeline_runs_end_to_end():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:30, 10:30] = 255   # 大块前景
    mask[13:21, 13:21] = 0     # 内部孔洞
    mask[60, 60] = 255         # 孤立噪点

    cleaned = postprocess_mask(
        mask, min_component_size=50, max_hole_size=100, morph_close_kernel=3
    )

    assert cleaned[13:21, 13:21].min() == 255   # 孔洞填平
    assert cleaned[10:30, 10:30].min() == 255   # 前景块保留
    assert cleaned[60, 60] == 0                 # 噪点被去除
