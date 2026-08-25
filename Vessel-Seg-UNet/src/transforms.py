"""
[M1] 数据增强流程
使用 Albumentations 构建训练与验证增强管线。
所有增强操作同步作用于 image 和 mask，确保空间对齐。

接口契约:
    - get_train_transforms(img_size) -> A.Compose
    - get_val_transforms(img_size) -> A.Compose
    - 输出 image: (C, H, W) float32 [0, 1]（C=1 或 2）
    - 输出 mask:  (1, H, W) float32 {0.0, 1.0}
"""

import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2


def _apply_clahe_safe(image: np.ndarray, **kwargs) -> np.ndarray:
    """安全应用 CLAHE，支持 1/2/3 通道图像。

    Albumentations 的 A.CLAHE 仅支持 1 或 3 通道。
    Frangi 双通道输入时为 (H, W, 2)，CLAHE 只应在原始图像通道上执行，
    Frangi vesselness 通道保持原样。

    Args:
        image: (H, W) 或 (H, W, C) uint8 图像

    Returns:
        同形状的 CLAHE 增强后图像
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if image.ndim == 2:
        return clahe.apply(image)
    channels = image.shape[-1]
    if channels == 2:
        # 双通道：仅对第一个通道（原始图像）做 CLAHE，Frangi 通道不变
        return np.dstack([clahe.apply(image[..., 0]), image[..., 1]])
    # 3 通道：逐通道 CLAHE（标准做法）
    return np.dstack([clahe.apply(image[..., i]) for i in range(channels)])


def _resize_or_pad(img_size: int, keep_aspect_ratio: bool) -> list:
    """Build a spatial normalization pipeline shared by training and inference."""
    if not keep_aspect_ratio:
        return [A.Resize(img_size, img_size)]
    return [
        A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR),
        A.PadIfNeeded(
            min_height=img_size,
            min_width=img_size,
            border_mode=cv2.BORDER_CONSTANT,
        ),
    ]


def get_train_transforms(
    img_size: int = 512,
    keep_aspect_ratio: bool = True,
    elastic_transform: bool = False,
) -> A.Compose:
    """
    返回训练阶段的 Albumentations 增强管线。

    包含:
        - 几何变换: 随机翻转、旋转(±30°)、弹性形变
        - 对比度/亮度扰动: CLAHE、RandomBrightnessContrast
        - 尺寸统一: 等比例缩放 + 补边（可选退回强制 Resize）
        - 归一化: [0, 255] → [0, 1]
        - ToTensorV2: numpy → torch.Tensor

    Args:
        img_size: 输出图像的边长（正方形）
        keep_aspect_ratio: 是否等比例缩放并补边，避免拉伸血管形态

    Returns:
        Albumentations Compose 对象
    """
    transforms = [
        *_resize_or_pad(img_size, keep_aspect_ratio),
        # ── 几何变换 ──
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, border_mode=0, p=0.5),
        # ── 对比度/亮度扰动 ──
        A.Lambda(image=_apply_clahe_safe, p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.3
        ),
        # ── 归一化 + Tensor ──
        A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ]
    if elastic_transform:
        transforms.insert(5, A.ElasticTransform(alpha=120, sigma=6, border_mode=0, p=0.3))
    return A.Compose(transforms, is_check_shapes=False)


def get_val_transforms(
    img_size: int = 512,
    keep_aspect_ratio: bool = True,
) -> A.Compose:
    """
    返回验证/测试阶段的 Albumentations 增强管线。
    仅做 Resize + 归一化，不做随机增强。

    Args:
        img_size: 输出图像的边长（正方形）
        keep_aspect_ratio: 是否等比例缩放并补边，避免拉伸血管形态

    Returns:
        Albumentations Compose 对象
    """
    return A.Compose([
        *_resize_or_pad(img_size, keep_aspect_ratio),
        A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ], is_check_shapes=False)
