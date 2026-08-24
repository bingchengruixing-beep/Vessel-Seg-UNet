"""
[M1] 数据集与 DataLoader 定义
负责图像读取、掩膜二值化清洗、与增强管线对接。

接口契约:
    VesselDataset.__getitem__ 输出:
        image: (1, H, W), float32, 值域 [0, 1]
        mask:  (1, H, W), float32, 值域 {0.0, 1.0}
        skeleton: (1, H, W), float32, 值域 {0.0, 1.0} —— 仅 return_skeleton=True 时返回

    get_dataloaders(config) -> (DataLoader, DataLoader)
    cl_dice_weight > 0 时 dataloader 自动输出 (image, mask, skeleton) 三元组。

核心防错:
    - Mask 必须是 {0.0, 1.0} 的 FloatTensor，绝不是 0~255 的 ByteTensor
    - Windows 下 num_workers 建议设为 0，避免多进程僵死
"""

import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple

from src.config import resolve_data_path
from src.skeleton import zhang_suen_thinning
from src.transforms import get_train_transforms, get_val_transforms


class VesselDataset(Dataset):
    """
    脑血管造影分割数据集。

    读取 image_dir 下的灰度造影图像和 mask_dir 下的对应掩膜，
    经 Albumentations 增强后输出标准化的 PyTorch Tensor。

    Args:
        image_dir: 原始造影图像所在目录
        mask_dir: 二值掩膜所在目录（文件名需与 image_dir 一一对应）
        transform: Albumentations Compose 增强管线（可选）
    """

    SUPPORTED_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    patch_size: Optional[int] = None
    foreground_probability: float = 0.0
    min_foreground_ratio: float = 0.0

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform=None,
        return_skeleton: bool = False,
        patch_size: Optional[int] = None,
        foreground_probability: float = 0.0,
        min_foreground_ratio: float = 0.0,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.return_skeleton = return_skeleton
        self.patch_size = int(patch_size) if patch_size else None
        self.foreground_probability = float(foreground_probability)
        self.min_foreground_ratio = float(min_foreground_ratio)

        # 收集所有支持格式的图像文件名（仅保留在 mask_dir 中也存在的）
        self.filenames = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(self.SUPPORTED_EXTS)
            and os.path.exists(os.path.join(mask_dir, f))
        ])

        if len(self.filenames) == 0:
            raise RuntimeError(
                f"No matched image-mask pairs found in\n"
                f"  image_dir: {image_dir}\n"
                f"  mask_dir:  {mask_dir}"
            )

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        读取并返回第 idx 对 (image, mask)。

        Returns:
            image: (1, H, W), float32, 值域 [0, 1]（经归一化）
            mask:  (1, H, W), float32, 值域 {0.0, 1.0}
        """
        fname = self.filenames[idx]
        img_path = os.path.join(self.image_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)

        # ── 读取灰度图像（使用 np.fromfile 支持中文路径）──
        img_buf = np.fromfile(img_path, dtype=np.uint8)
        image = cv2.imdecode(img_buf, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise IOError(f"Failed to read image: {img_path}")

        # ── 读取掩膜并二值化 ──
        mask_buf = np.fromfile(mask_path, dtype=np.uint8)
        mask = cv2.imdecode(mask_buf, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"Failed to read mask: {mask_path}")

        if mask.shape != image.shape:
            # 个别标注文件与原图尺寸不同，先用最近邻插值对齐，避免改变掩膜类别值。
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 硬阈值二值化：>127 → 255, 其余 → 0
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        if self.patch_size:
            image, mask = self._sample_patch(image, mask)

        # ── 数据增强 ──
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']   # (1, H, W) float32
            mask = augmented['mask']     # (H, W) uint8 or float
        else:
            # 手动转 Tensor（无增强时的 fallback）
            image = torch.from_numpy(image).unsqueeze(0).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        # ── 确保 mask 维度和值域正确 ──
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # (H, W) → (1, H, W)

        # 强制 {0.0, 1.0} 二值化（防御性处理）
        mask = (mask > 0.5).float()

        if self.return_skeleton:
            # 骨架必须从增强后的掩膜计算，保证与 mask 空间一致
            skeleton_np = zhang_suen_thinning(mask[0].cpu().numpy())
            skeleton = torch.from_numpy(skeleton_np.astype(np.float32)).unsqueeze(0)
            return image, mask, skeleton

        return image, mask

    def _sample_patch(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """从原始分辨率图像裁剪固定大小 patch，优先保留含血管区域。"""
        size = self.patch_size
        if size is None:
            return image, mask
        height, width = image.shape[:2]
        pad_height = max(0, size - height)
        pad_width = max(0, size - width)
        if pad_height or pad_width:
            image = np.pad(image, ((0, pad_height), (0, pad_width)), mode="constant")
            mask = np.pad(mask, ((0, pad_height), (0, pad_width)), mode="constant")
            height, width = image.shape[:2]
        max_top = height - size
        max_left = width - size
        foreground = mask > 127
        use_foreground = (
            random.random() < self.foreground_probability
            and float(foreground.mean()) >= self.min_foreground_ratio
            and bool(foreground.any())
        )
        if use_foreground:
            ys, xs = np.where(foreground)
            center_index = random.randrange(len(ys))
            top = min(max(int(ys[center_index]) - size // 2, 0), max_top)
            left = min(max(int(xs[center_index]) - size // 2, 0), max_left)
        else:
            top = random.randint(0, max_top) if max_top else 0
            left = random.randint(0, max_left) if max_left else 0
        return image[top:top + size, left:left + size], mask[top:top + size, left:left + size]


def get_dataloaders(
    config: dict,
    project_root: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    根据全局配置字典创建训练和验证 DataLoader。

    Args:
        config: 从 configs/default.yaml 加载的完整配置字典
        project_root: 相对数据路径的解析根目录；默认使用当前工作目录

    Returns:
        (train_loader, val_loader)
    """
    data_cfg = config['dataset']
    train_cfg = config['training']

    img_size = data_cfg['img_size']
    batch_size = train_cfg['batch_size']
    num_workers = data_cfg.get('num_workers', 0)
    pin_memory = data_cfg.get('pin_memory', True)

    # 构建增强管线
    keep_aspect_ratio = data_cfg.get('keep_aspect_ratio', True)
    augmentation_cfg = data_cfg.get("augmentation", {})
    patch_cfg = data_cfg.get("patch", {})
    use_patches = bool(patch_cfg.get("enabled", False))
    train_size = int(patch_cfg.get("size", img_size)) if use_patches else img_size
    train_transform = get_train_transforms(
        train_size,
        keep_aspect_ratio,
        elastic_transform=bool(augmentation_cfg.get("elastic_transform", False)),
    )
    val_transform = get_val_transforms(img_size, keep_aspect_ratio)

    # clDice 监督需要数据集同时返回金标准骨架
    return_skeleton = float(
        config['training']['loss'].get('cl_dice_weight', 0.0)
    ) > 0

    root = project_root or os.getcwd()

    # 构建数据集
    train_dataset = VesselDataset(
        image_dir=str(resolve_data_path(data_cfg['train_image_dir'], root)),
        mask_dir=str(resolve_data_path(data_cfg['train_mask_dir'], root)),
        transform=train_transform,
        return_skeleton=return_skeleton,
        patch_size=train_size if use_patches else None,
        foreground_probability=float(patch_cfg.get("foreground_probability", 0.0)),
        min_foreground_ratio=float(patch_cfg.get("min_foreground_ratio", 0.0)),
    )
    val_dataset = VesselDataset(
        image_dir=str(resolve_data_path(data_cfg['val_image_dir'], root)),
        mask_dir=str(resolve_data_path(data_cfg['val_mask_dir'], root)),
        transform=val_transform,
        return_skeleton=return_skeleton,
    )

    # 构建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        # Do not silently produce zero train batches for a small dataset.
        drop_last=len(train_dataset) >= batch_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader
