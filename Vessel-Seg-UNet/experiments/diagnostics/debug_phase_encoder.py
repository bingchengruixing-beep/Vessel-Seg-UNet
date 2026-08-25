"""诊断: 独立相位编码器为何不收敛。检查标签分布 + 更激进训练。"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import load_config
from src.dataset import get_dataloaders
from src.models import build_model

ROOT = Path(__file__).resolve().parent.parent.parent
cfg = load_config(str(ROOT / "configs/experiments/exp6c_phase_head_fixed.yaml"))
train_loader, val_loader = get_dataloaders(cfg, project_root=str(ROOT))

# 标签分布检查
all_labels = []
for batch in train_loader:
    all_labels.extend(batch[-1].tolist())
print("train label counts:", np.bincount(all_labels, minlength=4))
batch = next(iter(train_loader))
print("batch shapes:", [tuple(t.shape) for t in batch], "| labels:", batch[-1].tolist())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model("unet_baseline", in_channels=1, out_channels=1, phase_classes=4).to(device)
# 只训 phase_encoder
for name, p in model.named_parameters():
    p.requires_grad_(name.startswith("phase_encoder"))
opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
criterion = torch.nn.CrossEntropyLoss()

for step in range(300):
    try:
        batch = next(iter(train_loader))
    except StopIteration:
        break
    images, phase = batch[0].to(device), batch[-1].to(device)
    opt.zero_grad()
    _, phase_logits = model(images, None)
    loss = criterion(phase_logits, phase)
    loss.backward()
    opt.step()
    if (step + 1) % 60 == 0:
        with torch.no_grad():
            acc = (phase_logits.argmax(1) == phase).float().mean().item()
        print(f"step {step+1}: loss={loss.item():.4f} batch_acc={acc:.2f}")

correct = total = 0
with torch.no_grad():
    for batch in val_loader:
        images, phase = batch[0].to(device), batch[-1].to(device)
        _, phase_logits = model(images, None)
        correct += (phase_logits.argmax(1) == phase).sum().item()
        total += phase.numel()
print(f"val accuracy: {correct}/{total} = {correct/total:.1%}")
