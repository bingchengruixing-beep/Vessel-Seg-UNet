# Vessel-Seg-UNet

脑血管造影图像分割 —— 基于 U-Net 的血管分割模型。对比了 **3 种架构 × 2 种数据策略** 的 8 组消融实验，找到综合最优方案。

## 环境

- Python 3.9+
- PyTorch ≥ 2.0（本机实测 `2.6.0+cu124`，CUDA 可用）
- 其余依赖见 `requirements.txt`

```bash
pip install -r requirements.txt
```

## 目录结构

```
src/           核心代码（dataset / models / losses / trainer / metrics / transforms / postprocess / visualize）
configs/       配置文件（exp1~8 训练配置 + eval_* 评估配置 + hang/ 旧实验配置）
reports/       实验报告（experiments.md 详细记录 + summary.md 汇总与文件索引）
weights/       最优权重（deploy_exp6_resnet_mixed_fp16.pth）
train.py       训练入口
evaluate.py    评估脚本
inference.py   推理脚本
smoke_test.py  冒烟测试
```

## 模型

| 模型 | 参数量 | 说明 |
|------|--------|------|
| `unet_baseline` | 13.39M | 标准 U-Net |
| `attention_unet` | 31.39M | 带注意力门 U-Net |
| `unet_resnet` | 24.43M / 73.16M | ImageNet 预训练 ResNet34 / ResNet50 编码器 U-Net |

## 实验结论（8 组消融）

完整数据见 [`reports/summary.md`](reports/summary.md)。核心结论：

1. **混合训练（own + DIAS-train）是提升外部泛化最有效的手段**——三个模型的外部 DIAS-val 都上升，其中 resnet 受益最大（+2.9 点）。
2. **最优方案 = 实验 6**：`unet_resnet`（ResNet34）+ own+DIAS 混合训练，own_val Dice **0.7688**、外部 DIAS-val **0.7182** 双双第一。
3. 更深编码器（resnet50，3 倍参数）无实质提升；盲目调参反而变差——**瓶颈在数据量，不在模型容量**。

| 实验 | 模型 | 数据 | own_val Dice | DIAS-val Dice |
|------|------|------|-------------|---------------|
| 1 | baseline | own 142 | 0.7649 | 0.6952 |
| 2 | baseline | own+DIAS 172 | 0.7563 | 0.7029 |
| 3 | attention | own 142 | 0.7621 | 0.6933 |
| 4 | attention | own+DIAS 172 | 0.7639 | 0.7181 |
| 5 | resnet34 | own 142 | 0.7673 | 0.6893 |
| 6 | resnet34 | own+DIAS 172 | **0.7688** | **0.7182** |
| 7 | resnet50 | own+DIAS 172 | 0.7631 | 0.7221 |
| 8 | resnet34+tune | own+DIAS 172 | 0.7562 | 0.7049 |

> DIAS-val 为纯未见外部盲测集（20 张）的 Dice，是衡量泛化能力的诚实指标。

## 快速开始

### 训练

```bash
python train.py --config configs/exp6_resnet_mixed.yaml
```

> ⚠️ 配置里的数据路径是**绝对路径**（`C:/Users/.../数据集/...`），使用前请改成你自己的数据目录。

### 评估

```bash
python evaluate.py --checkpoint <权重路径> --config configs/exp6_resnet_mixed.yaml
```

### 推理

```bash
python inference.py --model <权重路径> --input <图片或目录> --output <输出目录> --model-name unet_resnet
```

### 冒烟测试

```bash
python smoke_test.py
```

## 加载最优权重

`weights/deploy_exp6_resnet_mixed_fp16.pth` 是实验 6 的 **fp16 精简权重**（仅 `model_state_dict`，46.7MB），加载时需把模型转为半精度：

```python
import torch
from src.models import build_model

model = build_model('unet_resnet', in_channels=1, out_channels=1, encoder_name='resnet34')
ckpt = torch.load('weights/deploy_exp6_resnet_mixed_fp16.pth', map_location='cpu')
model = model.half()                      # fp16 权重需半精度
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
# ckpt 里还存了 best_dice / epoch / model_name 等元信息
```

## 数据划分

- **own 数据**：按 4 个时长子集（2~3s / 4s / 5~6s / ds2-4s）分层 80/20 切分（seed=42），训练 142 / 验证 35，两者完全互斥。
- **DIAS**：开源数据集，作为外部测试集（DIAS-all 50 = train 30 + val 20）。
- 划分明细：`own_split/split_record.csv`（本地）。
