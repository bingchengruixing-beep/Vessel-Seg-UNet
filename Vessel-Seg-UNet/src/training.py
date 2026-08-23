"""Shared construction of loss, optimizer, and learning-rate scheduler."""

from __future__ import annotations

import torch

from src.losses import BCEDiceLoss


def build_criterion(config: dict) -> BCEDiceLoss:
    loss_cfg = config["training"]["loss"]
    return BCEDiceLoss(
        bce_weight=loss_cfg["bce_weight"],
        dice_weight=loss_cfg["dice_weight"],
        dice_smooth=loss_cfg["dice_smooth"],
    )


def build_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    train_cfg = config["training"]
    name = train_cfg["optimizer"].lower()
    
    if hasattr(model, 'get_param_groups'):
        encoder_lr_scale = config.get("model", {}).get("encoder_lr_scale", 0.1)
        params = model.get_param_groups(train_cfg["learning_rate"], encoder_lr_scale)
        kwargs = {"weight_decay": train_cfg["weight_decay"]}
    else:
        params = model.parameters()
        kwargs = {
            "lr": train_cfg["learning_rate"],
            "weight_decay": train_cfg["weight_decay"],
        }

    if name == "adamw":
        return torch.optim.AdamW(params, **kwargs)
    if name == "adam":
        return torch.optim.Adam(params, **kwargs)
    if name == "sgd":
        return torch.optim.SGD(params, momentum=0.9, **kwargs)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict):
    train_cfg = config["training"]
    name = train_cfg["scheduler"].lower()
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_cfg["epochs"]
        )
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    raise ValueError(f"Unknown scheduler: {name}")
