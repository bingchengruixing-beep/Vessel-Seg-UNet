"""
[M1] 数据增强流程
使用 Albumentations 构建训练与验证增强管线。
所有增强操作同步作用于 image 和 mask，确保空间对齐。

接口契约:
    - get_train_transforms(img_size, keep_aspect_ratio) -> A.Compose
    - get_val_transforms(img_size, keep_aspect_ratio) -> A.Compose
    - 输出 image: (1, H, W) float32 [0, 1]
    - 输出 mask:  (1, H, W) float32 {0.0, 1.0}
"""

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


def _resize_or_pad(img_size: int, keep_aspect_ratio: bool) -> list:
    """构建空间归一化管线：等比例缩放 + 补边（或退回强制 Resize）。

    keep_aspect_ratio=True 时，先按最长边缩放到 img_size，再补零到正方形，
    避免把非正方形 DSA 硬拉伸导致血管形态失真。
    """
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
    return A.Compose([
        *_resize_or_pad(img_size, keep_aspect_ratio),
        # ── 几何变换 ──
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, border_mode=0, p=0.5),
        A.ElasticTransform(
            alpha=120, sigma=6,
            border_mode=0, p=0.3
        ),
        # ── 对比度/亮度扰动 ──
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.3
        ),
        # ── 归一化 + Tensor ──
        A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ], is_check_shapes=False)


def get_val_transforms(
    img_size: int = 512,
    keep_aspect_ratio: bool = True,
) -> A.Compose:
    """
    返回验证/测试阶段的 Albumentations 增强管线。
    仅做尺寸统一 + 归一化，不做随机增强。

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
