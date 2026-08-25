# Vessel-Seg-UNet

基于 PyTorch 的脑血管造影 U-Net 分割项目。

## 项目概述

本项目针对脑血管 DSA（数字减影血管造影）图像，使用 U-Net 及其变体进行血管区域的自动语义分割。

### 核心特性

- **U-Net Baseline**: 经典 4 层编码器-解码器 + Skip Connection
- **Attention U-Net**: 注意力门控增强末梢细支血管捕捉
- **多种损失**: BCE + Dice 混合、Focal Tversky、clDice 中心线拓扑监督(Zhang-Suen 骨架)
- **现代训练策略**: warmup + 余弦退火、梯度裁剪、EMA 权重、固定随机种子
- **AMP 混合精度训练**: 支持显存受限的游戏本环境
- **完整后处理**: 连通域分析滤噪 + 孔洞填充

## 目录结构

```
Vessel-Seg-UNet/
├── data/                       # 数据目录（.gitignore）
├── checkpoints/                # 模型权重（.gitignore）
├── results/                    # 可视化输出
├── configs/
│   └── default.yaml            # 全局超参数配置
├── src/
│   ├── dataset.py              # [M1] 数据集与 DataLoader
│   ├── transforms.py           # [M1] 数据增强管线
│   ├── models/
│   │   ├── __init__.py         # [M2] 模型工厂
│   │   ├── unet.py             # [M2] U-Net Baseline
│   │   └── attention_unet.py   # [M2] Attention U-Net
│   ├── losses.py               # [M3] BCE+Dice / Focal Tversky / clDice 损失函数
│   ├── skeleton.py             # Zhang-Suen 骨架化(clDice 监督用)
│   ├── trainer.py              # [M3] 训练循环 (AMP + 早停)
│   ├── training.py             # 损失、优化器与调度器工厂
│   ├── config.py               # 配置归一化、校验与路径安全
│   ├── checkpoints.py          # 版本化检查点读写
│   ├── prediction.py           # 共享预测/后处理流程
│   ├── metrics.py              # [M4] Dice / IoU / Precision / Recall
│   ├── visualize.py            # [M4] 叠加对比图生成
│   └── postprocess.py          # [M5] 连通域去噪
├── train.py                    # 主训练入口
├── evaluate.py                 # 独立评估脚本
├── inference.py                # 推理 API
├── requirements.txt            # 依赖清单
├── docs/REFACTORING.md         # 重构与迁移说明
└── README.md
```

## 快速开始

### 1. 环境安装

```bash
pip install -r requirements.txt
```

> **注意**: PyTorch 需根据 CUDA 版本单独安装，请参考 [PyTorch 官网](https://pytorch.org/get-started/locally/)

### 2. 配置数据路径

编辑 `configs/default.yaml`，设置训练和验证数据的路径：

```yaml
dataset:
  train_image_dir: "D:/datasets/dsa/train/images"
  train_mask_dir: "D:/datasets/dsa/train/masks"
  val_image_dir: "D:/datasets/dsa/val/images"
  val_mask_dir: "D:/datasets/dsa/val/masks"
```

相对路径以项目根目录为基准。配置结构、旧配置迁移、检查点兼容性与 Web 安全边界见 [重构说明](docs/REFACTORING.md)。

### 3. 开始训练

```bash
python train.py
# 或指定自定义配置
python train.py --config configs/custom.yaml
# 指定设备(默认自动)
python train.py --device cuda:0
```

### 3.1 全量对比实验

`configs/experiments/` 提供 4 组对照配置(基线 / Focal Tversky / +clDice / Attention+组合),一键运行:

```powershell
powershell -File experiments/run_experiments.ps1
```

每组训练后自动做原始与后处理评估,日志在 `experiments/logs/`,结果在 `results/experiments/`。

### 4. 评估模型

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth --visualize
```

### 5. 推理预测

```bash
python inference.py --model checkpoints/best_model.pth --input path/to/image.png
```

### 6. 本地 Web 面板

```bash
python web_server.py
```

打开 `http://127.0.0.1:5001`。该服务仅供本机可信用户使用，请勿暴露到局域网或公网。

或在代码中使用：

```python
from inference import VesselSegmentor

seg = VesselSegmentor('checkpoints/best_model.pth')
mask = seg.predict('path/to/angiogram.png')
```

## 接口契约

| 模块 | 张量 | 形状 | 类型 | 值域 |
|------|------|------|------|------|
| Dataset 输出 image | `(1, H, W)` | float32 | `[0, 1]` |
| Dataset 输出 mask | `(1, H, W)` | float32 | `{0.0, 1.0}` |
| Dataset 输出 skeleton | `(1, H, W)` | float32 | `{0.0, 1.0}`,仅在 `cl_dice_weight > 0` 时返回 |
| 模型输入 | `(B, 1, H, W)` | float32 | 任意 |
| 模型输出 (logits) | `(B, 1, H, W)` | float32 | 任意 (未经 Sigmoid) |
| 损失函数输入 targets | `(B, 1, H, W)` | float32 | `{0.0, 1.0}` |

## 团队分工

| 角色 | 负责模块 | 核心文件 |
|------|----------|----------|
| M1 数据工程师 | 数据加载与增强 | `dataset.py`, `transforms.py` |
| M2 模型架构师 | 网络搭建 | `models/unet.py`, `models/attention_unet.py` |
| M3 训练工程师 | 损失/训练循环 | `losses.py`, `trainer.py` |
| M4 评测工程师 | 指标与可视化 | `metrics.py`, `visualize.py` |
| M5 部署工程师 | 推理与后处理 | `postprocess.py`, `inference.py` |

## united 分支整合功能

本分支以最新 `main` 为基础，整合了当前项目的模型与网页功能，并保留高级损失和训练稳定性改进：

- 新增 `unet_resnet`、`resunet_aspp` 和 `vessel_fusion` 模型，兼容普通输出、深监督输出及旧版裸权重。
- 支持前景 Patch 训练和重叠滑窗推理。网页仅在 `ResUNet-ASPP` 或 `VesselFusion` 模型下显示 Patch 设置，普通模型不会显示无关选项。
- 支持弹性形变开关、每批次训练日志、自动查找多个 checkpoint 目录、单图和批量推理，以及验证集阈值扫描。
- 保留 Focal Tversky、clDice、Zhang-Suen 骨架监督、EMA、Warmup、梯度裁剪和固定随机种子。
- `分层抽样/` 收纳成员分层抽样实验的配置、报告、源码和部署权重，仅作为可复现实验资料，不覆盖主项目核心代码。

### 目标域平衡、分组 K 折与 2.5D

- Web 数据配置默认开启 **DIAS 目标域平衡采样**，每轮总采样次数保持不变，`dias_train_*` 的期望占比为 40%。可在页面改为 30%～50%。
- 开启 **按序列分组 K 折** 后，`dataset1` 同编号的 2～3s、4s、5～6s 三个时相始终位于同一折。当前折权重自动保存到 `checkpoints/fold_1_of_3/` 等目录，不覆盖其他折。
- 完成每一折训练后，在推理页按住 Ctrl 多选 1～5 个 checkpoint。单图、批量推理和验证集阈值扫描都会先恢复各模型的原图概率，再做概率平均。
- 开启 **2.5D 前/中/后三时相** 后，模型输入自动切换为 3 通道。dataset1 使用同编号三时相；dataset2、DIAS 等缺少配对时相的样本自动重复当前图。
- 2.5D checkpoint 做 Web 推理时，可以选择普通单图（自动重复三通道），也可以切换到三时相输入并按前、当前、后顺序上传 3 张图。
- Web 数据配置提供 DataLoader 常驻 worker、预取 batch 数和每 worker 解码缓存。Windows 多 worker 训练建议保持默认的常驻 worker、预取 2 batch、缓存 32 项；`num_workers=0` 时这些并行选项会自动忽略。

推荐的 3 折操作顺序：

1. 开启分组 K 折，设置 `K=3`、当前折 `1`，保存配置并训练。
2. 依次把当前折改为 `2`、`3`，分别训练完成。
3. 在推理页多选 `fold_1_of_3/best_model.pth`、`fold_2_of_3/best_model.pth`、`fold_3_of_3/best_model.pth`，执行概率集成或阈值扫描。

默认端口仍为 5001，也可以通过环境变量启动其他端口：

```powershell
$env:VESSEL_WEB_PORT = "5002"
python web_server.py
```

阈值扫描结果用于验证集部署校准；不同数据划分或外部测试集的 Dice 不应直接混用比较。
