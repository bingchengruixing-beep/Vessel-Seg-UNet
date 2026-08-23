import numpy as np
import torch

from inference import VesselSegmentor, restore_original_geometry
from src.prediction import predictions_from_logits


def test_logits_are_thresholded_consistently_without_postprocessing():
    logits = torch.tensor([[[[-1.0, 0.0, 1.0]]]])
    predictions = predictions_from_logits(logits, threshold=0.5)
    assert torch.equal(predictions, torch.tensor([[[[0.0, 0.0, 1.0]]]]))


def test_letterboxed_prediction_restores_original_aspect_ratio():
    segmentor = VesselSegmentor.__new__(VesselSegmentor)
    segmentor.img_size = 8
    segmentor.keep_aspect_ratio = True
    letterboxed = torch.zeros((8, 8), dtype=torch.uint8).numpy()
    letterboxed[2:6, :] = 255  # 2x4 image was resized to 4x8 and padded vertically.

    restored = segmentor._restore_original_geometry(letterboxed, original_h=2, original_w=4)
    assert restored.shape == (2, 4)
    assert restored.min() == 255 and restored.max() == 255


def test_restore_original_geometry_removes_centered_padding():
    # 300x200 原图在 img_size=512 下: scale=512/300, resized=512x341, left=(512-341)//2=85
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[:, 85:85 + 341] = 255

    restored = restore_original_geometry(
        mask, original_h=300, original_w=200, img_size=512, keep_aspect_ratio=True
    )

    assert restored.shape == (300, 200)
    assert restored.min() == 255


def test_restore_original_geometry_plain_resize_without_aspect_ratio():
    mask = np.full((512, 512), 255, dtype=np.uint8)

    restored = restore_original_geometry(
        mask, original_h=128, original_w=64, img_size=512, keep_aspect_ratio=False
    )

    assert restored.shape == (128, 64)
    assert restored.min() == 255
