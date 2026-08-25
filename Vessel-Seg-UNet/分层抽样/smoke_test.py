"""冒烟测试：验证依赖导入、CUDA、数据集加载、模型前向"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 1. 关键库导入
import torch, torchvision
import numpy as np
import cv2
import albumentations as A
import yaml
import scipy
from skimage import measure

print("=== 库版本 ===")
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("albumentations:", A.__version__)
print("cv2:", cv2.__version__)
print("numpy:", np.__version__)
print("scipy:", scipy.__version__)

# 2. CUDA 检测
print("\n=== CUDA ===")
print("available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("显存:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), "GB")

# 3. 数据集加载
from src.dataset import VesselDataset, get_dataloaders
from src.transforms import get_train_transforms, get_val_transforms

config = yaml.safe_load(open("configs/default.yaml", encoding="utf-8"))
data_cfg = config["data"]

val_ds = VesselDataset(
    data_cfg["val_image_dir"], data_cfg["val_mask_dir"],
    transform=get_val_transforms(data_cfg["img_size"]),
)
img, mask = val_ds[0]
print("\n=== 数据集 ===")
print("val 样本数:", len(val_ds))
print("image shape:", tuple(img.shape), "dtype:", img.dtype, "范围:", round(img.min().item(), 3), "~", round(img.max().item(), 3))
print("mask shape:", tuple(mask.shape), "dtype:", mask.dtype, "唯一值:", mask.unique().tolist())

train_ds = VesselDataset(
    data_cfg["train_image_dir"], data_cfg["train_mask_dir"],
    transform=get_train_transforms(data_cfg["img_size"]),
)
print("train 样本数:", len(train_ds))

# 4. 模型构建 + 前向
from src.models import build_model
model = build_model("unet_baseline", in_channels=1, out_channels=1)
x = torch.randn(1, 1, 512, 512)
with torch.no_grad():
    out = model(x)
print("\n=== 模型 ===")
print("unet_baseline 输出 shape:", tuple(out.shape))
total = sum(p.numel() for p in model.parameters())
print("参数量:", f"{total:,}")

print("\n=== 冒烟测试通过 ===")
