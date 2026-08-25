"""复现失败并打印具体文件。第一部分: 迭代 loader 至多 200 个 batch。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.dataset import get_dataloaders

cfg = load_config("configs/experiments/exp5_own_baseline.yaml")
train_loader, _ = get_dataloaders(cfg, project_root=".")
ds = train_loader.dataset

seen = 0
for batch_idx, batch in enumerate(train_loader):
    shapes = [tuple(t.shape) for t in batch]
    if any(s != (4, 1, 512, 512) for s in shapes):
        print(f"FAIL at batch {batch_idx}: {shapes}")
        sys.exit(1)
    seen += 1
    if seen >= 200:
        break
print("loader OK for 200 batches")
