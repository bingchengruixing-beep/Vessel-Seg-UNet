"""Shared construction of loss, optimizer, and learning-rate scheduler."""

from __future__ import annotations

import random

import numpy as np
import torch

from src.losses import BCEDiceLoss, CLDiceLoss, CombinedVesselLoss, FocalTverskyLoss


def set_seed(seed: int) -> None:
    """Set Python/NumPy/PyTorch seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_criterion(config: dict) -> torch.nn.Module:
    """Build the main segmentation loss plus optional clDice supervision.

    The returned module accepts ``(logits, targets, skeleton=None)``:
    skeleton is required when ``training.loss.cl_dice_weight > 0``.
    """
    loss_cfg = config["training"]["loss"]
    name = loss_cfg["name"]
    if name == "FocalTverskyLoss":
        ft = loss_cfg["focal_tversky"]
        main_loss = FocalTverskyLoss(
            alpha=float(ft["alpha"]),
            beta=float(ft["beta"]),
            gamma=float(ft["gamma"]),
            smooth=float(loss_cfg["dice_smooth"]),
        )
    else:
        main_loss = BCEDiceLoss(
            bce_weight=loss_cfg["bce_weight"],
            dice_weight=loss_cfg["dice_weight"],
            dice_smooth=loss_cfg["dice_smooth"],
        )
    cldice_weight = float(loss_cfg.get("cl_dice_weight", 0.0))
    if cldice_weight > 0:
        return CombinedVesselLoss(main_loss, CLDiceLoss(), cldice_weight)
    return main_loss


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
    warmup_epochs = int(train_cfg.get("warmup_epochs", 0))
    epochs = int(train_cfg["epochs"])
    if name == "none":
        return None
    if name == "cosine":
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs - warmup_epochs)
        )
        if warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-3, total_iters=warmup_epochs
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
            )
        return cosine
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    raise ValueError(f"Unknown scheduler: {name}")
