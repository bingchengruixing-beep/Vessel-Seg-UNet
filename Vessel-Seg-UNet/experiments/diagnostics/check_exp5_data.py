"""exp5 数据流检查: dataloader + 骨架(256 降采样)耗时估算。"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.dataset import get_dataloaders

cfg = load_config("configs/experiments/exp5_own_cl_dice.yaml")
train_loader, val_loader = get_dataloaders(cfg, project_root=".")
print("train batches:", len(train_loader), "| val batches:", len(val_loader))
print("train images:", len(train_loader.dataset), "| val images:", len(val_loader.dataset))

sample = next(iter(train_loader))
print("sample:", [tuple(t.shape) for t in sample], "| skeleton mean:", float(sample[2].mean()))

# 估算每 epoch 骨架开销: 计时 8 个样本
ds = train_loader.dataset
t0 = time.perf_counter()
for i in range(8):
    ds[i]
elapsed = time.perf_counter() - t0
per_image = elapsed / 8
n_total = len(ds) + len(val_loader.dataset)
print(f"per-item: {per_image*1000:.0f} ms | est skeleton+io per epoch: {n_total*per_image:.1f} s")
