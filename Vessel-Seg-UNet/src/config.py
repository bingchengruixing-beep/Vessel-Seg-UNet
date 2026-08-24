"""Configuration loading, migration, validation, and path safety helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "dataset": {
        "img_size": 512,
        "num_workers": 0,
        "pin_memory": True,
        "keep_aspect_ratio": True,
        "augmentation": {
            "elastic_transform": False,
        },
        "train_image_dir": "data/train/images",
        "train_mask_dir": "data/train/masks",
        "val_image_dir": "data/val/images",
        "val_mask_dir": "data/val/masks",
    },
    "model": {
        "name": "unet_baseline",
        "in_channels": 1,
        "out_channels": 1,
        "encoder_name": "resnet34",
        "pretrained": True,
        "deep_supervision": True,
    },
    "training": {
        "batch_size": 1,
        "epochs": 100,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "use_amp": True,
        "loss": {
            "name": "BCEDiceLoss",
            "bce_weight": 0.5,
            "dice_weight": 0.5,
            "cldice_weight": 0.0,
            "dice_smooth": 1e-6,
            "skeleton_iterations": 5,
        },
        "deep_supervision_weights": [0.3, 0.2],
        "early_stopping": {"patience": 10},
        "checkpoint": {
            "save_dir": "checkpoints",
            "save_best_only": True,
            "save_interval": 10,
        },
    },
    "evaluation": {
        "threshold": 0.5,
        "apply_postprocess": False,
    },
    "inference": {
        "threshold": 0.5,
        # null means "use dataset.img_size" and keeps training/inference aligned.
        "img_size": None,
        "postprocess": {
            "enabled": True,
            "min_component_size": 50,
            "max_hole_size": 100,
            "morph_close_kernel": 3,
        },
    },
}


class ConfigError(ValueError):
    """Raised when configuration is malformed or unsafe."""


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def migrate_legacy_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Convert the repository's former top-level config keys to the new schema."""
    migrated = deepcopy(dict(config))
    training = migrated.setdefault("training", {})
    if not isinstance(training, dict):
        raise ConfigError("'training' must be a mapping")

    # Pre-refactor CLI looked for these at the root whereas the Web UI wrote them
    # under training. Accept both locations so existing config files continue to work.
    if "loss" in migrated and "loss" not in training:
        training["loss"] = migrated.pop("loss")
    if "checkpoint" in migrated and "checkpoint" not in training:
        training["checkpoint"] = migrated.pop("checkpoint")

    legacy_checkpoint_keys = ("save_dir", "save_best_only", "save_interval")
    legacy_values = {
        key: training.pop(key)
        for key in legacy_checkpoint_keys
        if key in training
    }
    if legacy_values:
        training["checkpoint"] = _deep_merge(
            legacy_values, training.get("checkpoint", {})
        )
    return migrated


def normalize_config(config: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """Merge user values with defaults and return a canonical config dictionary."""
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ConfigError("Configuration root must be a mapping")
    normalized = _deep_merge(DEFAULT_CONFIG, migrate_legacy_config(config))
    validate_config(normalized)
    return normalized


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML configuration and normalize it to the canonical schema."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    return normalize_config(loaded)


def save_config(config_path: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and persist a canonical configuration. Returns the saved value."""
    normalized = normalize_config(config)
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(normalized, file, allow_unicode=True, sort_keys=False)
    return normalized


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate values that affect model shape, training behavior, and file safety."""
    dataset = config.get("dataset")
    model = config.get("model")
    training = config.get("training")
    evaluation = config.get("evaluation")
    inference = config.get("inference")
    if not all(isinstance(section, Mapping) for section in (dataset, model, training, evaluation, inference)):
        raise ConfigError("dataset, model, training, evaluation, and inference must be mappings")

    _positive_int(dataset.get("img_size"), "dataset.img_size")
    _nonnegative_int(dataset.get("num_workers"), "dataset.num_workers")
    if not isinstance(dataset.get("keep_aspect_ratio"), bool):
        raise ConfigError("dataset.keep_aspect_ratio must be boolean")
    augmentation = dataset.get("augmentation")
    if not isinstance(augmentation, Mapping) or not isinstance(augmentation.get("elastic_transform"), bool):
        raise ConfigError("dataset.augmentation.elastic_transform must be boolean")
    _positive_int(model.get("in_channels"), "model.in_channels")
    _positive_int(model.get("out_channels"), "model.out_channels")
    if model.get("name") not in {"unet_baseline", "attention_unet", "unet_resnet", "resunet_aspp"}:
        raise ConfigError("model.name must be unet_baseline, attention_unet, unet_resnet, or resunet_aspp")
    if model.get("name") == "unet_resnet" and model.get("encoder_name") not in {"resnet34", "resnet50"}:
        raise ConfigError("model.encoder_name must be resnet34 or resnet50")
    for field in ("pretrained", "deep_supervision"):
        if not isinstance(model.get(field), bool):
            raise ConfigError(f"model.{field} must be boolean")

    _positive_int(training.get("batch_size"), "training.batch_size")
    _positive_int(training.get("epochs"), "training.epochs")
    _positive_float(training.get("learning_rate"), "training.learning_rate")
    _nonnegative_float(training.get("weight_decay"), "training.weight_decay")
    if str(training.get("optimizer", "")).lower() not in {"adam", "adamw", "sgd"}:
        raise ConfigError("training.optimizer must be adam, adamw, or sgd")
    if str(training.get("scheduler", "")).lower() not in {"cosine", "plateau", "step", "none"}:
        raise ConfigError("training.scheduler must be cosine, plateau, step, or none")

    loss = training.get("loss")
    early_stopping = training.get("early_stopping")
    checkpoint = training.get("checkpoint")
    postprocess = inference.get("postprocess")
    if not all(isinstance(section, Mapping) for section in (loss, early_stopping, checkpoint, postprocess)):
        raise ConfigError("training loss/checkpoint settings and inference postprocess must be mappings")
    bce_weight = _nonnegative_float(loss.get("bce_weight"), "training.loss.bce_weight")
    dice_weight = _nonnegative_float(loss.get("dice_weight"), "training.loss.dice_weight")
    if bce_weight + dice_weight <= 0:
        raise ConfigError("At least one loss weight must be positive")
    _positive_float(loss.get("dice_smooth"), "training.loss.dice_smooth")
    _nonnegative_float(loss.get("cldice_weight", 0.0), "training.loss.cldice_weight")
    _positive_int(loss.get("skeleton_iterations", 5), "training.loss.skeleton_iterations")
    deep_supervision_weights = training.get("deep_supervision_weights", [0.3, 0.2])
    if not isinstance(deep_supervision_weights, list) or any(
        not isinstance(value, (int, float)) or float(value) < 0 for value in deep_supervision_weights
    ):
        raise ConfigError("training.deep_supervision_weights must be a list of non-negative numbers")
    _positive_int(early_stopping.get("patience"), "training.early_stopping.patience")
    _positive_int(checkpoint.get("save_interval"), "training.checkpoint.save_interval")
    if not isinstance(checkpoint.get("save_best_only"), bool):
        raise ConfigError("training.checkpoint.save_best_only must be boolean")
    _safe_relative_path(str(checkpoint.get("save_dir", "")), "training.checkpoint.save_dir")

    for name, value in (("evaluation.threshold", evaluation.get("threshold")),
                        ("inference.threshold", inference.get("threshold"))):
        if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            raise ConfigError(f"{name} must be between 0 and 1")
    if inference.get("img_size") is not None:
        _positive_int(inference["img_size"], "inference.img_size")
    _positive_int(postprocess.get("min_component_size"), "inference.postprocess.min_component_size")
    _positive_int(postprocess.get("max_hole_size"), "inference.postprocess.max_hole_size")
    _positive_int(postprocess.get("morph_close_kernel"), "inference.postprocess.morph_close_kernel")


def resolve_checkpoint_dir(config: Mapping[str, Any], project_root: str | Path) -> Path:
    """Return a checkpoint directory guaranteed to remain inside ``project_root``."""
    relative = config["training"]["checkpoint"]["save_dir"]
    root = Path(project_root).resolve()
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ConfigError("Checkpoint directory must stay inside the project root") from exc
    return target


def resolve_data_path(path_value: str, project_root: str | Path) -> Path:
    """Resolve data paths; absolute external dataset paths remain supported."""
    path = Path(path_value)
    return path if path.is_absolute() else (Path(project_root) / path).resolve()


def _safe_relative_path(value: str, field: str) -> None:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ConfigError(f"{field} must be a non-empty project-relative path")


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ConfigError(f"{field} must be a non-negative integer")
    return value


def _positive_float(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) <= 0:
        raise ConfigError(f"{field} must be positive")
    return float(value)


def _nonnegative_float(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) < 0:
        raise ConfigError(f"{field} must be non-negative")
    return float(value)
