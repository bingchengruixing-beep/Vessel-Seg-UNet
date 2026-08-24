"""Shared construction of loss, optimizer, and learning-rate scheduler."""

from __future__ import annotations

import torch

from src.losses import BCEDiceClDiceLoss, BCEDiceLoss


def build_criterion(config: dict) -> torch.nn.Module:
    loss_cfg = config["training"]["loss"]
    if config["model"]["name"] == "resunet_aspp":
        return BCEDiceClDiceLoss(
            bce_weight=loss_cfg["bce_weight"],
            dice_weight=loss_cfg["dice_weight"],
            cldice_weight=loss_cfg.get("cldice_weight", 0.0),
            dice_smooth=loss_cfg["dice_smooth"],
            skeleton_iterations=loss_cfg.get("skeleton_iterations", 5),
        )
    return BCEDiceLoss(
        bce_weight=loss_cfg["bce_weight"],
        dice_weight=loss_cfg["dice_weight"],
        dice_smooth=loss_cfg["dice_smooth"],
    )


def build_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    train_cfg = config["training"]
    name = train_cfg["optimizer"].lower()
    kwargs = {
        "lr": train_cfg["learning_rate"],
        "weight_decay": train_cfg["weight_decay"],
    }
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), momentum=0.9, **kwargs)
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
