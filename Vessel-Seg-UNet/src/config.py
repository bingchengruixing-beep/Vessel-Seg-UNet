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
        # clDice 金标准骨架的计算分辨率(最长边),拓扑损失对分辨率不敏感,256 可大幅提速
        "skeleton_size": 256,
        "train_image_dir": "data/train/images",
        "train_mask_dir": "data/train/masks",
        "val_image_dir": "data/val/images",
        "val_mask_dir": "data/val/masks",
    },
    "model": {
        "name": "unet_baseline",
        "in_channels": 1,
        "out_channels": 1,
    },
    "training": {
        "batch_size": 1,
        "epochs": 100,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "use_amp": True,
        "seed": 42,
        "warmup_epochs": 5,
        "grad_clip": 1.0,
        "ema_decay": 0.999,
        # FiLM 相位条件化与相位分类辅助损失
        "phase_condition": False,
        "phase_loss_weight": 0.0,
        "loss": {
            "name": "BCEDiceLoss",
            "bce_weight": 0.5,
            "dice_weight": 0.5,
            "dice_smooth": 1e-6,
            # > 0 时启用 clDice 中心线监督(数据集会额外返回骨架)
            "cl_dice_weight": 0.0,
            "focal_tversky": {
                "alpha": 0.7,
                "beta": 0.3,
                "gamma": 0.75,
            },
        },
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
    _positive_int(dataset.get("skeleton_size"), "dataset.skeleton_size")
    _positive_int(model.get("in_channels"), "model.in_channels")
    _positive_int(model.get("out_channels"), "model.out_channels")
    if model.get("name") not in {"unet_baseline", "attention_unet"}:
        raise ConfigError("model.name must be 'unet_baseline' or 'attention_unet'")
    if model.get("phase_classes") is not None:
        _positive_int(model["phase_classes"], "model.phase_classes")

    _positive_int(training.get("batch_size"), "training.batch_size")
    _positive_int(training.get("epochs"), "training.epochs")
    _positive_float(training.get("learning_rate"), "training.learning_rate")
    _nonnegative_float(training.get("weight_decay"), "training.weight_decay")
    _nonnegative_int(training.get("seed"), "training.seed")
    _nonnegative_int(training.get("warmup_epochs"), "training.warmup_epochs")
    _nonnegative_float(training.get("grad_clip"), "training.grad_clip")
    if not isinstance(training.get("ema_decay"), (int, float)) or not 0.0 <= float(training["ema_decay"]) < 1.0:
        raise ConfigError("training.ema_decay must be in [0, 1)")
    if not isinstance(training.get("phase_condition"), bool):
        raise ConfigError("training.phase_condition must be boolean")
    _nonnegative_float(training.get("phase_loss_weight"), "training.phase_loss_weight")
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
    if loss.get("name") not in {"BCEDiceLoss", "FocalTverskyLoss"}:
        raise ConfigError("training.loss.name must be 'BCEDiceLoss' or 'FocalTverskyLoss'")
    bce_weight = _nonnegative_float(loss.get("bce_weight"), "training.loss.bce_weight")
    dice_weight = _nonnegative_float(loss.get("dice_weight"), "training.loss.dice_weight")
    if bce_weight + dice_weight <= 0:
        raise ConfigError("At least one loss weight must be positive")
    _positive_float(loss.get("dice_smooth"), "training.loss.dice_smooth")
    _nonnegative_float(loss.get("cl_dice_weight"), "training.loss.cl_dice_weight")
    focal_tversky = loss.get("focal_tversky")
    if not isinstance(focal_tversky, Mapping):
        raise ConfigError("training.loss.focal_tversky must be a mapping")
    _positive_float(focal_tversky.get("alpha"), "training.loss.focal_tversky.alpha")
    _positive_float(focal_tversky.get("beta"), "training.loss.focal_tversky.beta")
    _nonnegative_float(focal_tversky.get("gamma"), "training.loss.focal_tversky.gamma")
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
