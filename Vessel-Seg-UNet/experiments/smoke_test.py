"""GPU 冒烟测试: 验证 clDice 三元组数据流 + 新损失 + EMA + 梯度裁剪全链路。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import load_config
from src.dataset import get_dataloaders
from src.models import build_model
from src.training import build_criterion, build_optimizer, build_scheduler, set_seed

cfg = load_config("configs/experiments/exp3_cl_dice.yaml")
set_seed(int(cfg["training"]["seed"]))
print("cl_dice_weight:", cfg["training"]["loss"]["cl_dice_weight"])

train_loader, val_loader = get_dataloaders(cfg, project_root=".")
print("train batches:", len(train_loader), "| val batches:", len(val_loader))

sample = next(iter(train_loader))
print("sample tuple len:", len(sample), [t.shape for t in sample])
skel = sample[2] if len(sample) == 3 else None
if skel is not None:
    print("skeleton fg ratio: %.5f" % float(skel.mean()))

model = build_model(
    cfg["model"]["name"],
    in_channels=cfg["model"]["in_channels"],
    out_channels=cfg["model"]["out_channels"],
).cuda()
criterion = build_criterion(cfg)
optimizer = build_optimizer(model, cfg)
scheduler = build_scheduler(optimizer, cfg)
print("criterion:", type(criterion).__name__)

images, masks = sample[0].cuda(), sample[1].cuda()
skel = skel.cuda() if skel is not None else None
logits = model(images)
loss = criterion(logits, masks, skel)
print("logits:", tuple(logits.shape), "| loss:", float(loss))
loss.backward()
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
print("backward OK | grad_norm:", float(grad_norm), "| lr:", optimizer.param_groups[0]["lr"])

# 验证 EMA + 全周期训练入口可实例化
from src.trainer import Trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    config=cfg,
    checkpoint_dir="checkpoints/_smoke",
    device="cuda",
)
print("Trainer OK | device:", trainer.device, "| use_ema:", trainer.use_ema,
      "| grad_clip:", trainer.grad_clip, "| amp:", trainer.use_amp)
print("SMOKE TEST PASSED")
