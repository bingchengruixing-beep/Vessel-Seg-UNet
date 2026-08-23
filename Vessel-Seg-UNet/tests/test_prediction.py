import torch

from inference import VesselSegmentor
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
