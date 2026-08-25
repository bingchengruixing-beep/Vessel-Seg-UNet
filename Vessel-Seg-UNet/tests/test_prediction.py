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


def test_temporal_model_repeats_a_single_grayscale_image():
    segmentor = VesselSegmentor.__new__(VesselSegmentor)
    segmentor.in_channels = 3
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)

    prepared, height, width = segmentor._prepare_input(image)

    assert prepared.shape == (3, 4, 3)
    assert height == 3 and width == 4
    assert np.array_equal(prepared[..., 0], image)
    assert np.array_equal(prepared[..., 1], image)
    assert np.array_equal(prepared[..., 2], image)


def test_grayscale_model_uses_current_frame_from_temporal_input():
    segmentor = VesselSegmentor.__new__(VesselSegmentor)
    segmentor.in_channels = 1
    temporal = np.stack([
        np.zeros((2, 3), dtype=np.uint8),
        np.ones((2, 3), dtype=np.uint8),
        np.full((2, 3), 2, dtype=np.uint8),
    ], axis=-1)

    prepared, _, _ = segmentor._prepare_input(temporal)

    assert np.array_equal(prepared, temporal[..., 1])
