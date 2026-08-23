"""
主训练入口脚本
读取 YAML 配置 → 构建 DataLoader → 构建模型 → 构建损失/优化器/调度器 → 启动 Trainer

使用方式:
    python train.py                          # 使用默认配置
    python train.py --config configs/custom.yaml  # 使用自定义配置
"""

import os
import sys
import argparse
import logging

import yaml
import torch

# 将项目根目录加入 sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from src.dataset import get_dataloaders
from src.models import build_model
from src.losses import BCEDiceLoss
from src.trainer import Trainer

# ── 日志配置 ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger('train')


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def build_optimizer(model, config: dict) -> torch.optim.Optimizer:
    """根据配置构建优化器"""
    train_cfg = config['training']
    lr = train_cfg['learning_rate']
    wd = train_cfg.get('weight_decay', 1e-4)
    name = train_cfg.get('optimizer', 'adamw').lower()

    if name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == 'sgd':
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=wd
        )
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, config: dict):
    """根据配置构建学习率调度器"""
    train_cfg = config['training']
    name = train_cfg.get('scheduler', 'cosine').lower()
    epochs = train_cfg['epochs']

    if name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
    elif name == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
    elif name == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )
    else:
        logger.warning(f"Unknown scheduler: {name}, using None")
        return None


def main():
    parser = argparse.ArgumentParser(description='Vessel-Seg-UNet Training')
    parser.add_argument(
        '--config', type=str,
        default=os.path.join(ROOT_DIR, 'configs', 'default.yaml'),
        help='Path to YAML config file',
    )
    args = parser.parse_args()

    # ── 加载配置 ──
    config = load_config(args.config)

    logger.info("=" * 60)
    logger.info("  Vessel-Seg-UNet Training")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config}")

    # ── 构建 DataLoader (M1) ──
    logger.info("Building dataloaders...")
    train_loader, val_loader = get_dataloaders(config)
    logger.info(
        f"Train: {len(train_loader.dataset)} images, "
        f"Val: {len(val_loader.dataset)} images"
    )

    # ── 构建模型 (M2) ──
    model_cfg = config['model']
    model = build_model(
        model_name=model_cfg['name'],
        in_channels=model_cfg.get('in_channels', 1),
        out_channels=model_cfg.get('out_channels', 1),
    )
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model: {model_cfg['name']} | "
        f"Total params: {total_params:,} | "
        f"Trainable: {trainable_params:,}"
    )

    # ── 构建损失函数 (M3) ──
    loss_cfg = config.get('loss', {})
    criterion = BCEDiceLoss(
        bce_weight=loss_cfg.get('bce_weight', 0.5),
        dice_weight=loss_cfg.get('dice_weight', 0.5),
        dice_smooth=loss_cfg.get('dice_smooth', 1e-6),
    )
    logger.info(
        f"Loss: BCEDiceLoss (BCE={loss_cfg.get('bce_weight', 0.5)}, "
        f"Dice={loss_cfg.get('dice_weight', 0.5)})"
    )

    # ── 构建优化器和调度器 (M3) ──
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    logger.info(
        f"Optimizer: {config['training'].get('optimizer', 'adamw')} | "
        f"LR: {config['training']['learning_rate']}"
    )

    # ── 启动训练 (M3) ──
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
    )

    trainer.run()


if __name__ == '__main__':
    main()
