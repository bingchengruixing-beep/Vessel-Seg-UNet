"""
[M4] 独立评估脚本
加载训练好的模型检查点，在测试/验证集上计算完整评估指标，
并生成叠加对比可视化图像。

使用方式:
    python evaluate.py --checkpoint checkpoints/best_model.pth
    python evaluate.py --checkpoint checkpoints/best_model.pth --visualize
"""

import os
import sys
import argparse
import logging

import yaml
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from src.dataset import VesselDataset
from src.transforms import get_val_transforms
from src.models import build_model
from src.losses import BCEDiceLoss
from src.metrics import calculate_dice, calculate_iou, calculate_precision, calculate_recall
from src.visualize import save_overlay_image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger('evaluate')


def main():
    parser = argparse.ArgumentParser(description='Vessel-Seg-UNet Evaluation')
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint (.pth)',
    )
    parser.add_argument(
        '--config', type=str,
        default=os.path.join(ROOT_DIR, 'configs', 'default.yaml'),
        help='Path to YAML config file',
    )
    parser.add_argument(
        '--visualize', action='store_true',
        help='Generate overlay visualization images',
    )
    parser.add_argument(
        '--output-dir', type=str, default='results/eval',
        help='Directory to save evaluation results',
    )
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    data_cfg = config['dataset']
    model_cfg = config['model']

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # 构建模型并加载权重
    model = build_model(
        model_name=model_cfg['name'],
        in_channels=model_cfg.get('in_channels', 1),
        out_channels=model_cfg.get('out_channels', 1),
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', '?')
        best_dice = checkpoint.get('best_dice', '?')
        logger.info(f"Loaded checkpoint from epoch {epoch}, best dice: {best_dice}")
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # 构建验证数据集
    img_size = data_cfg['img_size']
    val_transform = get_val_transforms(img_size)
    val_dataset = VesselDataset(
        image_dir=data_cfg['val_image_dir'],
        mask_dir=data_cfg['val_mask_dir'],
        transform=val_transform,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    logger.info(f"Evaluating on {len(val_dataset)} images...")

    # ── 逐样本评估 ──
    all_dice = []
    all_iou = []
    all_precision = []
    all_recall = []

    os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (images, masks) in enumerate(tqdm(val_loader, desc="Evaluating")):
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            preds_binary = (torch.sigmoid(logits) > 0.5).float()

            # 计算指标
            dice = calculate_dice(preds_binary, masks)
            iou = calculate_iou(preds_binary, masks)
            precision = calculate_precision(preds_binary, masks)
            recall = calculate_recall(preds_binary, masks)

            all_dice.append(dice)
            all_iou.append(iou)
            all_precision.append(precision)
            all_recall.append(recall)

            # 可视化
            if args.visualize:
                img_np = (images[0, 0].cpu().numpy() * 255).astype(np.uint8)
                gt_np = (masks[0, 0].cpu().numpy() * 255).astype(np.uint8)
                pred_np = (preds_binary[0, 0].cpu().numpy() * 255).astype(np.uint8)

                fname = val_dataset.filenames[idx]
                save_path = os.path.join(args.output_dir, f"overlay_{fname}")
                save_overlay_image(img_np, gt_np, pred_np, save_path)

    # ── 汇总报告 ──
    mean_dice = np.mean(all_dice)
    mean_iou = np.mean(all_iou)
    mean_prec = np.mean(all_precision)
    mean_rec = np.mean(all_recall)

    logger.info("=" * 50)
    logger.info("  Evaluation Results")
    logger.info("=" * 50)
    logger.info(f"  Dice Coefficient : {mean_dice:.4f} ± {np.std(all_dice):.4f}")
    logger.info(f"  IoU (Jaccard)    : {mean_iou:.4f} ± {np.std(all_iou):.4f}")
    logger.info(f"  Precision        : {mean_prec:.4f} ± {np.std(all_precision):.4f}")
    logger.info(f"  Recall           : {mean_rec:.4f} ± {np.std(all_recall):.4f}")
    logger.info(f"  Samples          : {len(all_dice)}")
    logger.info("=" * 50)

    # 保存报告
    report_path = os.path.join(args.output_dir, 'eval_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Samples: {len(all_dice)}\n\n")
        f.write(f"Dice:      {mean_dice:.4f} ± {np.std(all_dice):.4f}\n")
        f.write(f"IoU:       {mean_iou:.4f} ± {np.std(all_iou):.4f}\n")
        f.write(f"Precision: {mean_prec:.4f} ± {np.std(all_precision):.4f}\n")
        f.write(f"Recall:    {mean_rec:.4f} ± {np.std(all_recall):.4f}\n")

    logger.info(f"Report saved to: {report_path}")


if __name__ == '__main__':
    main()
