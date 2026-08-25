"""诊断相位头梯度: ① 单步梯度范数 ② 冻结骨干单独训头 100 步。"""
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.checkpoints import checkpoint_model_config, load_checkpoint, load_model_state
from src.config import load_config, normalize_config
from src.dataset import get_dataloaders
from src.models import build_model

ROOT = Path(__file__).resolve().parent.parent.parent
CK_DIR = sys.argv[1] if len(sys.argv) > 1 else "exp6c_phase_head_fixed"
CK = ROOT / "checkpoints" / CK_DIR / "best_model.pth"
CFG = ROOT / "configs" / "experiments" / f"{CK_DIR}.yaml"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = load_checkpoint(str(CK), map_location=device)
cfg = normalize_config(ck.get("config"))
model_cfg = checkpoint_model_config(ck, cfg)
model = build_model(
    model_cfg["name"], in_channels=1, out_channels=1,
    phase_classes=model_cfg.get("phase_classes", 0),
).to(device)
load_model_state(model, ck)
model.train()

cfg_yaml = load_config(str(CFG))
train_loader, val_loader = get_dataloaders(cfg_yaml, project_root=str(ROOT))
batch = next(iter(train_loader))
images, phase = batch[0].to(device), batch[-1].to(device)

criterion = torch.nn.CrossEntropyLoss()

# ① 梯度检查
model.zero_grad(set_to_none=True)
out = model(images, phase)
_, phase_logits = out
loss = criterion(phase_logits, phase)
loss.backward()
head_grad_norm = sum(p.grad.norm().item() for p in model.phase_head.parameters())
emb_grad_norm = model.film.emb.weight.grad.norm().item() if model.film.emb.weight.grad is not None else 0
print(f"① loss={loss.item():.4f} | phase_head grad={head_grad_norm:.4f} | film.emb grad={emb_grad_norm:.4f}")

# ② 冻结除 phase_head 外的一切, 单独训头
for p in model.parameters():
    p.requires_grad_(False)
for p in model.phase_head.parameters():
    p.requires_grad_(True)
opt = torch.optim.Adam(model.phase_head.parameters(), lr=1e-3)
for step in range(100):
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

correct = total = 0
for batch in train_loader:
    images, phase = batch[0].to(device), batch[-1].to(device)
    with torch.no_grad():
        _, phase_logits = model(images, None)
    correct += (phase_logits.argmax(1) == phase).sum().item()
    total += phase.numel()
print(f"② 冻结骨干训头后 train 准确率: {correct}/{total} = {correct/total:.1%}")
