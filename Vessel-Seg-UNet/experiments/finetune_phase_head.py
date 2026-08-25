"""冻结骨干, 微调独立相位编码器(修复预判定头, 无需全量重训)。

用法: .venv/Scripts/python.exe -u experiments/finetune_phase_head.py [base_ckpt_dir] [out_dir]
默认: 基于 exp6c_phase_head_fixed 骨干, 输出 checkpoints/exp6d_phase_encoder。
"""
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.checkpoints import checkpoint_model_config, load_checkpoint, save_checkpoint
from src.config import load_config, normalize_config
from src.dataset import get_dataloaders
from src.models import build_model

ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = sys.argv[1] if len(sys.argv) > 1 else "exp6c_phase_head_fixed"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "exp6d_phase_encoder"
BASE_CK = ROOT / "checkpoints" / BASE_DIR / "best_model.pth"
CFG = ROOT / "configs" / "experiments" / f"{BASE_DIR}.yaml"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = load_checkpoint(str(BASE_CK), map_location=device)
saved_cfg = normalize_config(ck.get("config"))
model_cfg = checkpoint_model_config(ck, saved_cfg)
model = build_model(
    model_cfg["name"], in_channels=1, out_channels=1,
    phase_classes=model_cfg.get("phase_classes", 0),
).to(device)
# 新分支(phase_encoder)在基检查点中不存在, 用 strict=False 加载其余权重
missing, unexpected = model.load_state_dict(ck["model_state_dict"], strict=False)
print("missing (new branch, to train):", missing)
print("unexpected:", unexpected)

for name, p in model.named_parameters():
    p.requires_grad_(name.startswith("phase_encoder"))

cfg = load_config(str(CFG))
train_loader, val_loader = get_dataloaders(cfg, project_root=str(ROOT))
optimizer = torch.optim.Adam(
    [p for p in model.phase_encoder.parameters() if p.requires_grad], lr=1e-3
)
criterion = torch.nn.CrossEntropyLoss()

# 关键: 骨干保持 eval(冻结 BN running stats, 防止微调污染骨干统计量),
# 仅 phase_encoder 处于训练模式(GroupNorm 无 running stats, 天然稳定)。
model.eval()
model.phase_encoder.train()
for step in range(600):
    try:
        batch = next(iter(train_loader))
    except StopIteration:
        break
    images, phase = batch[0].to(device), batch[-1].to(device)
    optimizer.zero_grad()
    _, phase_logits = model(images, None)
    loss = criterion(phase_logits, phase)
    loss.backward()
    optimizer.step()
    if (step + 1) % 50 == 0:
        print(f"step {step+1}/200 loss={loss.item():.4f}")

model.eval()
correct = total = 0
with torch.no_grad():
    for batch in val_loader:
        images, phase = batch[0].to(device), batch[-1].to(device)
        _, phase_logits = model(images, None)
        correct += (phase_logits.argmax(1) == phase).sum().item()
        total += phase.numel()
print(f"val phase accuracy: {correct}/{total} = {correct/total:.1%}")

out_dir = ROOT / "checkpoints" / OUT_DIR
out_dir.mkdir(parents=True, exist_ok=True)
save_checkpoint(
    out_dir / "best_model.pth",
    model=model,
    optimizer=None,
    scheduler=None,
    epoch=int(ck.get("epoch", 0)),
    best_dice=float(ck.get("best_dice", 0.0)),
    metrics={"note": "backbone from " + BASE_DIR + ", phase encoder fine-tuned (frozen backbone)"},
    config=saved_cfg,
)
print("saved:", out_dir / "best_model.pth")
