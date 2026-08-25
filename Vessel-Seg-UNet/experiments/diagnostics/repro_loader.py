"""用真实 VesselDataset + 基线配置复现 collate 失败。"""
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.dataset import get_dataloaders

cfg = load_config("configs/experiments/exp5_own_baseline.yaml")
train_loader, val_loader = get_dataloaders(cfg, project_root=".")
print("train:", len(train_loader.dataset), "val:", len(val_loader.dataset))

try:
    for batch_idx, batch in enumerate(train_loader):
        shapes = [tuple(t.shape) for t in batch]
        bad = [s for s in shapes if s != (4, 1, 512, 512)]
        if bad or len(shapes) != 2:
            print(f"batch {batch_idx}: {shapes}")
        if batch_idx >= 40:
            break
    print("train loader fully OK")
except Exception:
    traceback.print_exc()

try:
    for batch_idx, batch in enumerate(val_loader):
        shapes = [tuple(t.shape) for t in batch]
        bad = [s for s in shapes if s != (4, 1, 512, 512)]
        if bad or len(shapes) != 2:
            print(f"val batch {batch_idx}: {shapes}")
    print("val loader fully OK")
except Exception:
    traceback.print_exc()
