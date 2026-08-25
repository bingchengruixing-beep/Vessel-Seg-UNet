"""给 __getitem__ 加探针, 定位哪个文件/哪个分支产生异常尺寸。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.dataset import VesselDataset, get_dataloaders

cfg = load_config("configs/experiments/exp5_own_baseline.yaml")
train_loader, _ = get_dataloaders(cfg, project_root=".")
ds = train_loader.dataset
orig_getitem = VesselDataset.__getitem__


def traced_getitem(self, idx):
    out = orig_getitem(self, idx)
    shapes = tuple(tuple(t.shape) for t in out)
    if any(s != (1, 512, 512) for s in shapes):
        print(f"BAD {self.filenames[idx]} -> {shapes}")
    return out


VesselDataset.__getitem__ = traced_getitem
for batch_idx, batch in enumerate(train_loader):
    if batch_idx >= 60:
        break
print("done")
