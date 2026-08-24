"""
[M1] 数据增强流程
使用 Albumentations 构建训练与验证增强管线。
所有增强操作同步作用于 image 和 mask，确保空间对齐。

接口契约:
    - get_train_transforms(img_size) -> A.Compose
    - get_val_transforms(img_size) -> A.Compose
    - 输出 image: (1, H, W) float32 [0, 1]
    - 输出 mask:  (1, H, W) float32 {0.0, 1.0}
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size: int = 512, strong_aug: bool = False) -> A.Compose:
    """
    返回训练阶段的 Albumentations 增强管线。

    包含:
        - 几何变换: 随机翻转、旋转(±30°)、弹性形变
        - 对比度/亮度扰动: CLAHE、RandomBrightnessContrast
        - 尺寸统一: Resize → img_size x img_size
        - 归一化: [0, 255] → [0, 1]
        - ToTensorV2: numpy → torch.Tensor

    Args:
        img_size: 输出图像的边长（正方形）

    Returns:
        Albumentations Compose 对象
    """
    transforms = [
        A.Resize(img_size, img_size),
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
    ]
    if strong_aug:
        transforms += [
            # ── 强增强（开关控制，对抗小数据过拟合）──
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, border_mode=0, p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.3, border_mode=0, p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        ]
    transforms += [
        # ── 归一化 + Tensor ──
        A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ]
    return A.Compose(transforms, is_check_shapes=False)


def get_val_transforms(img_size: int = 512) -> A.Compose:
    """
    返回验证/测试阶段的 Albumentations 增强管线。
    仅做 Resize + 归一化，不做随机增强。

    Args:
        img_size: 输出图像的边长（正方形）

    Returns:
        Albumentations Compose 对象
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ], is_check_shapes=False)
