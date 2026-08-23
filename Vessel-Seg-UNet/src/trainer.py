"""Shared training loop used by both the CLI and the local Web interface."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.checkpoints import save_checkpoint
from src.metrics import MetricAccumulator
from src.prediction import predictions_from_logits


logger = logging.getLogger(__name__)
EpochCallback = Callable[[Dict[str, float]], None]
StopCallback = Callable[[], bool]


class Trainer:
    """Train a segmentation model and emit structured progress for any UI."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        config: Optional[dict] = None,
        checkpoint_dir: Optional[str | Path] = None,
        on_epoch_end: Optional[EpochCallback] = None,
        should_stop: Optional[StopCallback] = None,
        device: Optional[str | torch.device] = None,
    ):
        self.config = config or {}
        train_cfg = self.config["training"]
        checkpoint_cfg = train_cfg["checkpoint"]

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested device {self.device} but CUDA is not available")
        logger.info("Using device: %s", self.device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = train_cfg["epochs"]
        self.use_amp = bool(train_cfg["use_amp"]) and self.device.type == "cuda"
        self.patience = train_cfg["early_stopping"]["patience"]
        self.save_best_only = checkpoint_cfg["save_best_only"]
        self.save_interval = checkpoint_cfg["save_interval"]
        self.save_dir = Path(checkpoint_dir or checkpoint_cfg["save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.on_epoch_end = on_epoch_end
        self.should_stop = should_stop or (lambda: False)

        self.scaler = GradScaler(enabled=self.use_amp)
        self.best_dice = float("-inf")
        self.epochs_no_improve = 0
        self.current_epoch = 0

    def _is_stop_requested(self) -> bool:
        return bool(self.should_stop())

    def train_one_epoch(self) -> Optional[float]:
        """Execute one epoch. ``None`` means a stop was requested."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.epochs} [Train]",
            leave=False,
        )
        for images, masks in progress:
            if self._is_stop_requested():
                return None
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, masks)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
            num_batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")
        if num_batches == 0:
            raise RuntimeError("Training DataLoader produced no batches")
        return total_loss / num_batches

    @torch.no_grad()
    def validate(self) -> Optional[Dict[str, float]]:
        """Evaluate one epoch with the same optional postprocessing as deployment."""
        self.model.eval()
        total_loss = 0.0
        accumulator = MetricAccumulator()
        num_batches = 0
        evaluation_cfg = self.config["evaluation"]
        inference_cfg = self.config["inference"]
        progress = tqdm(
            self.val_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.epochs} [Val]",
            leave=False,
        )
        for images, masks in progress:
            if self._is_stop_requested():
                return None
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, masks)
            predictions = predictions_from_logits(
                logits,
                threshold=evaluation_cfg["threshold"],
                apply_postprocess=evaluation_cfg["apply_postprocess"],
                postprocess_config=inference_cfg["postprocess"],
            )
            total_loss += loss.item()
            accumulator.update(predictions, masks)
            num_batches += 1
        if num_batches == 0:
            raise RuntimeError("Validation DataLoader produced no batches")
        return {
            "val_loss": total_loss / num_batches,
            # 全数据集像素级聚合，避免小 batch 与大 batch 权重相同带来的偏差。
            "dice": accumulator.dice(),
            "iou": accumulator.iou(),
        }

    def _save_checkpoint(self, filename: str, metrics: Dict[str, float]) -> None:
        path = self.save_dir / filename
        save_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch + 1,
            best_dice=self.best_dice,
            metrics=metrics,
            config=self.config,
        )
        logger.info("Checkpoint saved: %s", path)

    def run(self) -> Dict[str, float | bool]:
        """Run training, checkpointing, early stopping, and optional progress callbacks."""
        logger.info("Starting training for %s epochs (AMP: %s)", self.epochs, self.use_amp)
        last_metrics: Optional[Dict[str, float]] = None
        stopped = False

        for epoch in range(self.epochs):
            self.current_epoch = epoch
            train_loss = self.train_one_epoch()
            if train_loss is None:
                stopped = True
                break
            val_metrics = self.validate()
            if val_metrics is None:
                stopped = True
                break

            last_metrics = val_metrics
            val_dice = val_metrics["dice"]
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_dice)
                else:
                    self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            if val_dice > self.best_dice:
                self.best_dice = val_dice
                self.epochs_no_improve = 0
                self._save_checkpoint("best_model.pth", val_metrics)
                logger.info("New best Dice: %.4f", val_dice)
            else:
                self.epochs_no_improve += 1

            epoch_metrics: Dict[str, float] = {
                "epoch": float(epoch + 1),
                "train_loss": train_loss,
                "val_loss": val_metrics["val_loss"],
                "dice": val_dice,
                "iou": val_metrics["iou"],
                "best_dice": self.best_dice,
                "lr": current_lr,
            }
            logger.info(
                "Epoch [%s/%s] Train Loss: %.4f | Val Loss: %.4f | Dice: %.4f | IoU: %.4f | LR: %.2e",
                epoch + 1, self.epochs, train_loss, val_metrics["val_loss"], val_dice,
                val_metrics["iou"], current_lr,
            )
            if self.on_epoch_end:
                self.on_epoch_end(epoch_metrics)

            if not self.save_best_only and (epoch + 1) % self.save_interval == 0:
                self._save_checkpoint(f"model_epoch_{epoch + 1}.pth", val_metrics)
            if self.epochs_no_improve >= self.patience:
                logger.info("Early stopping after %s epochs", epoch + 1)
                break

        if last_metrics is not None and not self.save_best_only:
            self._save_checkpoint("last_model.pth", last_metrics)
        return {
            "best_dice": self.best_dice if self.best_dice != float("-inf") else 0.0,
            "stopped": stopped,
            "completed_epochs": float(self.current_epoch + (0 if stopped else 1)),
        }
