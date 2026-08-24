"""Shared logits-to-mask conversion for evaluation and deployment."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch

from src.postprocess import postprocess_mask


def binarize_logits(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Apply sigmoid and a common binary threshold to model logits."""
    return (torch.sigmoid(logits) > threshold).to(dtype=torch.float32)


def postprocess_predictions(
    predictions: torch.Tensor,
    postprocess_config: Mapping[str, int | bool],
) -> torch.Tensor:
    """Apply the deployment postprocessing pipeline to a Bx1xHxW binary tensor."""
    if predictions.ndim != 4 or predictions.shape[1] != 1:
        raise ValueError("predictions must have shape (B, 1, H, W)")
    if not postprocess_config.get("enabled", True):
        return predictions

    device = predictions.device
    arrays = predictions.detach().cpu().numpy()
    processed: list[np.ndarray] = []
    for prediction in arrays:
        mask = (prediction[0] > 0.5).astype(np.uint8) * 255
        cleaned = postprocess_mask(
            mask,
            min_component_size=int(postprocess_config["min_component_size"]),
            max_hole_size=int(postprocess_config["max_hole_size"]),
            morph_close_kernel=int(postprocess_config["morph_close_kernel"]),
        )
        processed.append((cleaned > 127).astype(np.float32))
    return torch.from_numpy(np.stack(processed)[:, None]).to(device)


def main_logits_from_output(output: torch.Tensor | tuple | list) -> torch.Tensor:
    """兼容普通模型和深监督模型，只取主输出用于指标与推理。"""
    if isinstance(output, (tuple, list)):
        if not output:
            raise ValueError("Model output sequence is empty")
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError("Model output must be a tensor or a non-empty sequence")
    return output


def predictions_from_logits(
    logits: torch.Tensor | tuple | list,
    *,
    threshold: float = 0.5,
    apply_postprocess: bool = False,
    postprocess_config: Mapping[str, int | bool] | None = None,
) -> torch.Tensor:
    """Produce binary masks using the same optional postprocessing as inference."""
    predictions = binarize_logits(main_logits_from_output(logits), threshold)
    if apply_postprocess:
        if postprocess_config is None:
            raise ValueError("postprocess_config is required when apply_postprocess=True")
        return postprocess_predictions(predictions, postprocess_config)
    return predictions
