"""exp6 相位条件管线冒烟测试。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from src.config import load_config
from src.dataset import get_dataloaders
from src.models import build_model
from src.training import build_criterion, set_seed

cfg = load_config("configs/experiments/exp6_phase_film.yaml")
set_seed(42)
train_loader, val_loader = get_dataloaders(cfg, project_root=".")
print("train batches:", len(train_loader), "| val batches:", len(val_loader))

batch = next(iter(train_loader))
print("batch len:", len(batch), [tuple(t.shape) for t in batch])

model = build_model(
    cfg["model"]["name"], in_channels=1, out_channels=1,
    phase_classes=cfg["model"]["phase_classes"],
).cuda()
out = model(batch[0].cuda(), batch[-1].cuda())
print("output type:", type(out).__name__)
print("logits:", tuple(out[0].shape), "| phase_logits:", tuple(out[1].shape))

criterion = build_criterion(cfg)
loss = criterion(out[0], batch[1].cuda(), batch[2].cuda())
print("main loss:", float(loss))

# 相位分类损失
phase_loss = torch.nn.CrossEntropyLoss()(out[1], batch[-1].cuda())
print("phase loss:", float(phase_loss))
print("PHASE PIPELINE OK")
