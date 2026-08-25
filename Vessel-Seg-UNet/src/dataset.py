"""
[M1] 数据集与 DataLoader 定义
负责图像读取、掩膜二值化清洗、与增强管线对接。

接口契约:
    VesselDataset.__getitem__ 输出:
        image: (C, H, W), float32, 值域 [0, 1]
            - C=1 时仅含原始造影图像
            - C=2 时含原始造影图像 + Frangi vesselness 通道
        mask:  (1, H, W), float32, 值域 {0.0, 1.0}
        skeleton: (1, H, W), float32, 值域 {0.0, 1.0} —— 仅 return_skeleton=True 时返回

    get_dataloaders(config) -> (DataLoader, DataLoader)
    cl_dice_weight > 0 时 dataloader 自动输出 (image, mask, skeleton) 三元组。

核心防错:
    - Mask 必须是 {0.0, 1.0} 的 FloatTensor，绝不是 0~255 的 ByteTensor
    - Windows 下 num_workers 建议设为 0，避免多进程僵死
    - Frangi 通道仅在 dataset.frangi.enabled=true 且对应 frangi 目录存在时加载
"""

import os
import random
import re
from collections import OrderedDict, defaultdict
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Iterable, Optional, Sequence, Tuple

from src.config import resolve_data_path
from src.skeleton import zhang_suen_thinning
from src.transforms import get_train_transforms, get_val_transforms


_DATASET1_PATTERN = re.compile(
    r"^dataset1_(2-3s|4s|5-6s)_(\d+)$", re.IGNORECASE
)
_SINGLE_DOMAIN_PATTERN = re.compile(
    r"^(dataset2_4s|dias_train|dias_val)_(\d+)$", re.IGNORECASE
)


def sample_group_key(filename: str) -> str:
    """从统一数据集文件名解析患者/序列组，防止同序列跨折。"""
    stem = os.path.splitext(os.path.basename(filename))[0]
    match = _DATASET1_PATTERN.match(stem)
    if match:
        return f"dataset1_{match.group(2)}"
    match = _SINGLE_DOMAIN_PATTERN.match(stem)
    if match:
        return f"{match.group(1).lower()}_{match.group(2)}"
    return stem.lower()


def sample_domain(filename: str) -> str:
    """返回分折时使用的数据域名称。"""
    stem = os.path.splitext(os.path.basename(filename))[0].lower()
    if stem.startswith("dias_"):
        return "dias"
    if stem.startswith("dataset1_"):
        return "dataset1"
    if stem.startswith("dataset2_"):
        return "dataset2"
    return "other"


def temporal_phase(filename: str) -> int:
    """返回 dataset1 三时相的时间顺序；其他样本只有中心帧。"""
    stem = os.path.splitext(os.path.basename(filename))[0]
    match = _DATASET1_PATTERN.match(stem)
    if not match:
        return 1
    return {"2-3s": 0, "4s": 1, "5-6s": 2}[match.group(1).lower()]


def grouped_kfold_split(
    filenames: Sequence[str],
    num_folds: int,
    fold_index: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    """按域均衡、按序列分组生成一个 K 折训练/验证划分。"""
    if not 3 <= num_folds <= 5:
        raise ValueError("num_folds 必须在 3 到 5 之间")
    if not 0 <= fold_index < num_folds:
        raise ValueError("fold_index 必须小于 num_folds")

    groups_by_domain: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for filename in filenames:
        groups_by_domain[sample_domain(filename)][sample_group_key(filename)].append(filename)

    folds: list[list[str]] = [[] for _ in range(num_folds)]
    for domain in sorted(groups_by_domain):
        groups = list(groups_by_domain[domain].values())
        random.Random(f"{seed}:{domain}").shuffle(groups)
        groups.sort(key=len, reverse=True)
        domain_sizes = [0] * num_folds
        for group in groups:
            destination = min(range(num_folds), key=lambda index: (domain_sizes[index], index))
            folds[destination].extend(group)
            domain_sizes[destination] += len(group)

    validation = sorted(folds[fold_index])
    validation_set = set(validation)
    training = sorted(filename for filename in filenames if filename not in validation_set)
    if not training or not validation:
        raise RuntimeError("分组 K 折产生了空训练集或空验证集")
    return training, validation


def domain_balanced_weights(
    filenames: Sequence[str],
    target_prefixes: Iterable[str],
    target_probability: float,
) -> list[float]:
    """计算使目标域期望抽样占比等于指定概率的逐样本权重。"""
    prefixes = tuple(value.lower() for value in target_prefixes)
    target_flags = [os.path.basename(name).lower().startswith(prefixes) for name in filenames]
    target_count = sum(target_flags)
    source_count = len(target_flags) - target_count
    if target_count == 0 or source_count == 0:
        raise RuntimeError("目标域平衡采样需要同时包含目标域和非目标域样本")
    target_weight = target_probability / target_count
    source_weight = (1.0 - target_probability) / source_count
    return [target_weight if is_target else source_weight for is_target in target_flags]


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
        frangi_dir: str = "",
        filenames: Optional[Sequence[str]] = None,
        temporal_2_5d: bool = False,
        cache_size: int = 0,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.return_skeleton = return_skeleton
        self.patch_size = int(patch_size) if patch_size else None
        self.foreground_probability = float(foreground_probability)
        self.min_foreground_ratio = float(min_foreground_ratio)
        self.frangi_dir = frangi_dir
        self.use_frangi = bool(frangi_dir) and os.path.isdir(frangi_dir)
        self.temporal_2_5d = bool(temporal_2_5d)
        self.cache_size = max(0, int(cache_size))
        self._decode_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        if self.temporal_2_5d and self.use_frangi:
            raise ValueError("2.5D 时相输入不能与 Frangi 通道同时启用")

        # 收集所有支持格式的图像文件名（仅保留在 mask_dir 中也存在的）
        available_filenames = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(self.SUPPORTED_EXTS)
            and os.path.exists(os.path.join(mask_dir, f))
        ])
        if filenames is None:
            self.filenames = available_filenames
        else:
            available_set = set(available_filenames)
            missing = sorted(set(filenames) - available_set)
            if missing:
                raise RuntimeError(f"指定的数据文件不存在或缺少同名掩膜: {missing[0]}")
            self.filenames = sorted(filenames)

        if len(self.filenames) == 0:
            raise RuntimeError(
                f"No matched image-mask pairs found in\n"
                f"  image_dir: {image_dir}\n"
                f"  mask_dir:  {mask_dir}"
            )
        self.temporal_neighbors = self._build_temporal_neighbors()

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

        # 2.5D 直接读取三时相，避免先额外解码一次中心帧。
        image = self.load_input_image(fname) if self.temporal_2_5d else self._read_gray(img_path)

        # ── 读取掩膜并二值化 ──
        mask = self._read_gray(mask_path)

        if mask.shape != image.shape:
            # 个别标注文件与原图尺寸不同，先用最近邻插值对齐，避免改变掩膜类别值。
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 硬阈值二值化：>127 → 255, 其余 → 0
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # ── 读取 Frangi vesselness 通道 ──
        frangi = None
        if self.use_frangi:
            frangi_path = os.path.join(self.frangi_dir, fname)
            if os.path.exists(frangi_path):
                frangi_buf = np.fromfile(frangi_path, dtype=np.uint8)
                frangi = cv2.imdecode(frangi_buf, cv2.IMREAD_UNCHANGED)
                if frangi is None:
                    raise IOError(f"Failed to read Frangi map: {frangi_path}")
                if frangi.shape != image.shape:
                    frangi = cv2.resize(
                        frangi, (image.shape[1], image.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                # 归一化到 [0, 255] uint8，便于与原始图像堆叠后统一增强
                if frangi.dtype == np.uint16:
                    frangi = (frangi.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)
                elif frangi.max() > 1.0:
                    pass  # 已是 uint8 范围
                else:
                    frangi = (frangi.astype(np.float32) * 255.0).astype(np.uint8)

        # ── Patch 裁剪 ──
        if self.patch_size:
            if self.temporal_2_5d:
                image, mask = self._sample_patch_rgb(image, mask)
            elif frangi is not None:
                # 将 frangi 与 image 堆叠为 (H, W, 2) 同步裁剪
                stacked = np.stack([image, frangi], axis=-1)
                stacked, mask = self._sample_patch_rgb(stacked, mask)
                image = stacked[..., 0]
                frangi = stacked[..., 1]
            else:
                image, mask = self._sample_patch(image, mask)

        # ── 数据增强 ──
        if self.transform is not None:
            if self.temporal_2_5d:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            elif frangi is not None:
                # 双通道增强：堆叠为 (H, W, 2) uint8，Albumentations 对各通道做相同几何变换
                stacked = np.stack([image, frangi], axis=-1)  # (H, W, 2) uint8
                augmented = self.transform(image=stacked, mask=mask)
                image = augmented['image']   # (2, H, W) float32 [0, 1]
                mask = augmented['mask']     # (H, W)
            else:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']   # (1, H, W) float32
                mask = augmented['mask']     # (H, W)
        else:
            # 手动转 Tensor（无增强时的 fallback）
            if image.ndim == 3:
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            else:
                image = torch.from_numpy(image).unsqueeze(0).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float()
            if frangi is not None:
                frangi_tensor = torch.from_numpy(frangi).unsqueeze(0).float() / 255.0
                image = torch.cat([image, frangi_tensor], dim=0)  # (2, H, W)

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

    def _read_gray(self, path: str) -> np.ndarray:
        """读取中文路径灰度图，并在 worker 内维护受控 LRU 缓存。"""
        if path in self._decode_cache:
            image = self._decode_cache.pop(path)
            self._decode_cache[path] = image
            return image.copy()
        buffer = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise IOError(f"Failed to read image: {path}")
        if self.cache_size > 0:
            self._decode_cache[path] = image
            while len(self._decode_cache) > self.cache_size:
                self._decode_cache.popitem(last=False)
            return image.copy()
        return image

    def load_input_image(self, filename: str) -> np.ndarray:
        """读取单通道图像，或按配置读取前/中/后三时相三通道输入。"""
        if not self.temporal_2_5d:
            return self._read_gray(os.path.join(self.image_dir, filename))
        neighbors = self.temporal_neighbors[filename]
        return np.stack(
            [self._read_gray(os.path.join(self.image_dir, name)) for name in neighbors],
            axis=-1,
        )

    def _build_temporal_neighbors(self) -> dict[str, tuple[str, str, str]]:
        """为每个样本建立前/当前/后时相；缺失位置重复边界时相。"""
        result = {name: (name, name, name) for name in self.filenames}
        if not self.temporal_2_5d:
            return result
        groups: dict[str, list[str]] = defaultdict(list)
        for name in self.filenames:
            groups[sample_group_key(name)].append(name)
        for members in groups.values():
            ordered = sorted(members, key=lambda name: (temporal_phase(name), name))
            if len(ordered) == 1:
                continue
            for index, name in enumerate(ordered):
                result[name] = (
                    ordered[max(index - 1, 0)],
                    name,
                    ordered[min(index + 1, len(ordered) - 1)],
                )
        return result

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

    def _sample_patch_rgb(
        self, stacked: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """与 _sample_patch 相同的裁剪逻辑，适用于 (H, W, C) 多通道图像。"""
        size = self.patch_size
        if size is None:
            return stacked, mask
        height, width = stacked.shape[:2]
        pad_height = max(0, size - height)
        pad_width = max(0, size - width)
        if pad_height or pad_width:
            stacked = np.pad(
                stacked, ((0, pad_height), (0, pad_width), (0, 0)), mode="constant"
            )
            mask = np.pad(mask, ((0, pad_height), (0, pad_width)), mode="constant")
            height, width = stacked.shape[:2]
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
        return (
            stacked[top:top + size, left:left + size],
            mask[top:top + size, left:left + size],
        )


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

    # Frangi 双通道配置
    frangi_cfg = data_cfg.get("frangi", {})
    use_frangi = bool(frangi_cfg.get("enabled", False))
    train_frangi_dir = ""
    val_frangi_dir = ""
    if use_frangi:
        train_frangi_raw = frangi_cfg.get("train_frangi_dir", "")
        val_frangi_raw = frangi_cfg.get("val_frangi_dir", "")
        if train_frangi_raw:
            train_frangi_dir = str(resolve_data_path(train_frangi_raw, root))
        else:
            # 默认在训练图像同级创建 frangi 子目录
            train_frangi_dir = str(
                resolve_data_path(data_cfg["train_image_dir"], root).parent / "frangi"
            )
        if val_frangi_raw:
            val_frangi_dir = str(resolve_data_path(val_frangi_raw, root))
        else:
            val_frangi_dir = str(
                resolve_data_path(data_cfg["val_image_dir"], root).parent / "frangi"
            )

    train_image_dir = str(resolve_data_path(data_cfg['train_image_dir'], root))
    train_mask_dir = str(resolve_data_path(data_cfg['train_mask_dir'], root))
    all_train_filenames = sorted([
        name for name in os.listdir(train_image_dir)
        if name.lower().endswith(VesselDataset.SUPPORTED_EXTS)
        and os.path.exists(os.path.join(train_mask_dir, name))
    ])
    cross_validation_cfg = data_cfg.get("cross_validation", {})
    use_cross_validation = bool(cross_validation_cfg.get("enabled", False))
    train_filenames: Optional[list[str]] = None
    val_filenames: Optional[list[str]] = None
    if use_cross_validation:
        train_filenames, val_filenames = grouped_kfold_split(
            all_train_filenames,
            int(cross_validation_cfg.get("num_folds", 3)),
            int(cross_validation_cfg.get("fold_index", 0)),
            int(train_cfg.get("seed", 42)),
        )

    temporal_enabled = bool(data_cfg.get("temporal_2_5d", {}).get("enabled", False))
    loader_cfg = data_cfg.get("loader", {})
    cache_size = int(loader_cfg.get("cache_size", 0))

    # 构建数据集
    train_dataset = VesselDataset(
        image_dir=train_image_dir,
        mask_dir=train_mask_dir,
        transform=train_transform,
        return_skeleton=return_skeleton,
        patch_size=train_size if use_patches else None,
        foreground_probability=float(patch_cfg.get("foreground_probability", 0.0)),
        min_foreground_ratio=float(patch_cfg.get("min_foreground_ratio", 0.0)),
        frangi_dir=train_frangi_dir,
        filenames=train_filenames,
        temporal_2_5d=temporal_enabled,
        cache_size=cache_size,
    )
    validation_image_dir = (
        train_image_dir if use_cross_validation
        else str(resolve_data_path(data_cfg['val_image_dir'], root))
    )
    validation_mask_dir = (
        train_mask_dir if use_cross_validation
        else str(resolve_data_path(data_cfg['val_mask_dir'], root))
    )
    val_dataset = VesselDataset(
        image_dir=validation_image_dir,
        mask_dir=validation_mask_dir,
        transform=val_transform,
        return_skeleton=return_skeleton,
        frangi_dir=train_frangi_dir if use_cross_validation else val_frangi_dir,
        filenames=val_filenames,
        temporal_2_5d=temporal_enabled,
        cache_size=cache_size,
    )

    sampler = None
    domain_balance_cfg = data_cfg.get("domain_balance", {})
    if domain_balance_cfg.get("enabled", False):
        weights = domain_balanced_weights(
            train_dataset.filenames,
            domain_balance_cfg.get("target_prefixes", ["dias_train_"]),
            float(domain_balance_cfg.get("target_probability", 0.4)),
        )
        samples_per_epoch = int(domain_balance_cfg.get("samples_per_epoch", 0))
        generator = torch.Generator().manual_seed(int(train_cfg.get("seed", 42)))
        sampler = WeightedRandomSampler(
            weights,
            num_samples=samples_per_epoch or len(train_dataset),
            replacement=True,
            generator=generator,
        )

    # 构建 DataLoader
    worker_options = {}
    if num_workers > 0:
        worker_options = {
            "persistent_workers": bool(loader_cfg.get("persistent_workers", True)),
            "prefetch_factor": int(loader_cfg.get("prefetch_factor", 2)),
        }
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        # Do not silently produce zero train batches for a small dataset.
        drop_last=len(train_dataset) >= batch_size,
        **worker_options,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        **worker_options,
    )

    return train_loader, val_loader
