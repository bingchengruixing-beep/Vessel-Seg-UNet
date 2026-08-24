"""Versioned, safe-to-load checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch


CHECKPOINT_FORMAT_VERSION = 2


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be safely used by this application."""


def build_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    epoch: int,
    best_dice: float,
    metrics: Mapping[str, float],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a self-describing checkpoint used by all project entry points."""
    checkpoint: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "epoch": epoch,
        "best_dice": float(best_dice),
        "metrics": dict(metrics),
        "config": dict(config),
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    return checkpoint


def save_checkpoint(path: str | Path, **kwargs: Any) -> None:
    """Persist a versioned checkpoint, creating the target directory if necessary."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(build_checkpoint(**kwargs), target)


def load_checkpoint(path: str | Path, map_location: Any = "cpu") -> dict[str, Any]:
    """Load state dictionaries only; reject arbitrary pickled Python objects."""
    target = Path(path)
    try:
        payload = torch.load(target, map_location=map_location, weights_only=True)
    except TypeError as exc:
        raise CheckpointError(
            "PyTorch with torch.load(..., weights_only=True) is required to load checkpoints safely."
        ) from exc
    except Exception as exc:  # Torch wraps unsafe/unreadable checkpoints in varied errors.
        raise CheckpointError(f"Unable to safely load checkpoint: {target}") from exc

    if not isinstance(payload, Mapping):
        raise CheckpointError("Checkpoint must be a state-dict mapping")
    payload = dict(payload)
    # Backward compatibility for historical raw state_dict checkpoints.
    if "model_state_dict" not in payload:
        if not payload or not all(isinstance(value, torch.Tensor) for value in payload.values()):
            raise CheckpointError("Checkpoint has no model_state_dict")
        payload = {"format_version": 1, "model_state_dict": payload}
    if not isinstance(payload["model_state_dict"], Mapping):
        raise CheckpointError("model_state_dict must be a mapping")
    return payload


def load_model_state(model: torch.nn.Module, checkpoint: Mapping[str, Any]) -> None:
    """Load model weights from either a current or legacy normalized checkpoint."""
    model.load_state_dict(checkpoint["model_state_dict"])


def infer_legacy_model_name(checkpoint: Mapping[str, Any]) -> str | None:
    """根据旧版裸 state_dict 的层名推断真实模型结构。"""
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        return None
    keys = tuple(str(key) for key in state_dict.keys())
    if any(key.startswith("final_refine.") for key in keys):
        return "vessel_fusion"
    if any(key.startswith("aspp.") or key.startswith("x3_1.") for key in keys):
        return "resunet_aspp"
    if any(key.startswith("dec4.") for key in keys):
        return "unet_resnet"
    return None


def checkpoint_model_config(
    checkpoint: Mapping[str, Any], fallback_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Prefer saved model settings, preventing a later config edit from changing a model."""
    saved_config = checkpoint.get("config")
    if isinstance(saved_config, Mapping) and isinstance(saved_config.get("model"), Mapping):
        return dict(saved_config["model"])
    return dict(fallback_config["model"])
