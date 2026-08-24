"""
[M3] 训练与验证循环逻辑
拼装 M1 的数据和 M2 的模型，执行前向、反向传播、梯度更新及断点保存。
支持自动混合精度（AMP）以应对游戏本显存不足的场景。

接口契约:
    Trainer.train_one_epoch() -> float (平均 Loss)
    Trainer.validate() -> dict (含 val_loss, dice, iou 等指标)
    Trainer.run() -> None (完整训练循环，含早停与模型保存)
"""

import os
import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.metrics import dice_per_sample, iou_per_sample

logger = logging.getLogger(__name__)


class Trainer:
    """
    统一的训练管理器。

    Args:
        model: 分割模型 (nn.Module)
        train_loader: 训练 DataLoader
        val_loader: 验证 DataLoader
        criterion: 损失函数 (nn.Module)
        optimizer: 优化器
        scheduler: 学习率调度器（可选）
        config: 全局配置字典
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        config: dict = None,
    ):
        self.config = config or {}
        train_cfg = self.config.get('training', {})
        ckpt_cfg = self.config.get('checkpoint', {})

        # 设备自动检测（严禁写死 'cuda:0'）
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        logger.info(f"Using device: {self.device}")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        # 训练参数
        self.epochs = train_cfg.get('epochs', 100)
        self.use_amp = train_cfg.get('use_amp', True) and self.device.type == 'cuda'

        # 早停参数
        self.patience = ckpt_cfg.get('early_stopping_patience', 15)
        self.save_best_only = ckpt_cfg.get('save_best_only', True)
        self.save_dir = ckpt_cfg.get('save_dir', 'checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)

        # AMP 混合精度
        self.scaler = GradScaler(enabled=self.use_amp)

        # 训练状态
        self.best_dice = 0.0
        self.epochs_no_improve = 0
        self.current_epoch = 0

    def train_one_epoch(self) -> float:
        """
        执行一轮训练。

        Returns:
            平均训练 Loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.epochs} [Train]",
            leave=False,
        )

        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    @torch.no_grad()
    def validate(self) -> dict:
        """
        执行一轮验证，计算损失和评估指标。

        Returns:
            字典: {val_loss, dice, iou}
        """
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        num_batches = 0
        num_samples = 0

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.epochs} [Val]",
            leave=False,
        )

        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            # 阈值化预测 (Sigmoid → >0.5 → binary)
            preds_binary = (torch.sigmoid(logits) > 0.5).float()

            total_loss += loss.item()
            # 逐样本 Dice/IoU 后再平均，与 evaluate.py 口径一致（与 batch 划分无关）
            total_dice += dice_per_sample(preds_binary, masks).sum().item()
            total_iou += iou_per_sample(preds_binary, masks).sum().item()
            num_batches += 1
            num_samples += masks.size(0)

        n_batches = max(num_batches, 1)
        n_samples = max(num_samples, 1)
        return {
            'val_loss': total_loss / n_batches,
            'dice': total_dice / n_samples,
            'iou': total_iou / n_samples,
        }

    def _save_checkpoint(self, filename: str, metrics: dict):
        """保存模型检查点。"""
        path = os.path.join(self.save_dir, filename)
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_dice': self.best_dice,
            'metrics': metrics,
        }, path)
        logger.info(f"Checkpoint saved: {path}")

    def run(self):
        """
        主训练循环。

        包含:
            - 逐轮训练 + 验证
            - 学习率调度
            - 早停机制（基于验证 Dice）
            - 最佳模型自动保存
        """
        logger.info(f"Starting training for {self.epochs} epochs")
        logger.info(f"AMP: {'ON' if self.use_amp else 'OFF'}")

        for epoch in range(self.epochs):
            self.current_epoch = epoch

            # 训练
            train_loss = self.train_one_epoch()

            # 验证
            val_metrics = self.validate()
            val_loss = val_metrics['val_loss']
            val_dice = val_metrics['dice']
            val_iou = val_metrics['iou']

            # 学习率调度
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_dice)
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']

            # 日志
            logger.info(
                f"Epoch [{epoch + 1}/{self.epochs}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Dice: {val_dice:.4f} | "
                f"IoU: {val_iou:.4f} | "
                f"LR: {current_lr:.2e}"
            )

            # 最佳模型保存 + 早停
            if val_dice > self.best_dice:
                self.best_dice = val_dice
                self.epochs_no_improve = 0
                self._save_checkpoint('best_model.pth', val_metrics)
                logger.info(f"★ New best Dice: {val_dice:.4f}")
            else:
                self.epochs_no_improve += 1
                logger.info(
                    f"No improvement for {self.epochs_no_improve}/{self.patience} epochs"
                )

            # 早停判断
            if self.epochs_no_improve >= self.patience:
                logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs. "
                    f"Best Dice: {self.best_dice:.4f}"
                )
                break

        # 保存最终模型
        self._save_checkpoint('last_model.pth', val_metrics)
        logger.info(f"Training complete. Best Dice: {self.best_dice:.4f}")
