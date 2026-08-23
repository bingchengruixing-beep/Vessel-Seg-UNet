# Vessel-Seg-UNet: 脑血管语义分割框架

Vessel-Seg-UNet 是一个基于 PyTorch 构建的语义分割项目，专为极高正负样本不平衡的脑血管造影（DSA）图像分割设计。本项目具有高度模块化、源码清晰、并且集成了从训练、验证、推断到网页 UI 部署的完整生命周期管理。

---

## 一、 核心代码架构分析

项目严格遵循领域驱动设计，按照职责进行了模块化切分（位于 `src/` 目录下）：

### 1. 数据处理管道 (`src/dataset.py` & `src/transforms.py`)
- **数据集抽象**：继承 `torch.utils.data.Dataset` 的血管数据集读取器，自动完成图像与金标准（Mask）的严格对齐配对读取。
- **动态数据增强**：基于 `albumentations` 库。提供了缩放（Resize）、自动补边（PadIfNeeded）、随机翻转、随机旋转、对比度增强等操作，极大提高模型在跨域数据上的泛化能力。
- **等比缩放保护**：针对医学图像的特性，支持长边缩放 + 补边（Letterbox）机制，防止血管在预处理阶段发生形变。

### 2. 模型动物园 (`src/models/`)
工厂模式驱动（`__init__.py: build_model`），所有模型接口均保证输入输出张量形状契合 `(B, 1, H, W)` 灰度输入和无激活的原始 Logits 输出。
- **U-Net Baseline (`unet.py`)**：纯粹从零搭建的标准 4 层下采样编码器-解码器架构，附带双卷积块和特征级联跳跃连接（Skip Connection）。
- **Attention U-Net (`attention_unet.py`)**：在跳跃连接中嵌入注意力门控（Attention Gate），使模型自动抑制背景特征并聚焦于末梢微小血管。
- **U-Net-ResNet (迁移学习, `unet_resnet.py`)**：基于 `torchvision` 预训练权重的混合架构。编码器复用 ImageNet 预训练的 ResNet34/ResNet50 权重（针对单通道灰度图进行了底层算子通道压缩适配），并通过 `get_param_groups()` 实现了**差分学习率**——编码器以低学习率微调，解码器全速学习。

### 3. 损失函数与优化体系 (`src/losses.py` & `src/training.py`)
- **BCEDiceLoss**：为攻克血管分割中前景像素占比极低（通常 < 5%）的难题，代码将逐像素分类交叉熵（BCE）与区域重合度惩罚（Soft Dice Loss）结合，确保梯度既关注局部也关注全局结构。
- **优化器抽象**：内置 `AdamW`、`Adam`、`SGD` 等主流优化器。
- **动态学习率**：内置 `CosineAnnealingLR` (余弦退火)、`ReduceLROnPlateau` (自适应降低)、`StepLR`。

### 4. 训练引擎 (`src/trainer.py`)
- **自动混合精度 (AMP)**：使用 `torch.cuda.amp.autocast` 与 `GradScaler`，将显存占用减半，支持在民用级显卡上训练大 Batch Size 数据。
- **早停机制 (Early Stopping)**：基于验证集 Dice 系数的早停机制，持续追踪 `best_dice` 自动保存最佳模型。

### 5. 评估与后处理 (`src/metrics.py` & `src/postprocess.py`)
- **无偏评估体系**：严格的全样本混淆矩阵累加，避免基于 Batch 均值计算 Dice / IoU / Precision / Recall 时产生的统计学偏差。
- **连通域滤噪算法**：基于 `cv2.connectedComponentsWithStats`，过滤网络预测出来的孤立微小斑块噪点（如 `<50` 像素点）。

---

## 二、 交互接口与功能特性

### 1. 命令行接口 (CLI)
- **模型训练 (`train.py`)**：解析 YAML 配置文件，初始化数据流与模型，并挂载 `Trainer` 执行训练循环。
- **批量评估 (`evaluate.py`)**：加载检查点（Checkpoints）和验证数据集，报告全局四维指标（Dice/IoU/Pre/Rec），并支持生成预测叠加对比图。
- **部署推理 (`inference.py`)**：提供纯后端的预测管道，支持单张图片推断。自动撤销几何缩放（还原回原图大小），并可将结果保存为二值化图片。

### 2. 本地 Web 面板 (`web_server.py`)
为避免繁琐的命令行调参，项目基于 Flask 编写了本地可视化 UI 面板 (`http://127.0.0.1:5001`)：
- **实时配置编辑器**：可视化修改 `default.yaml` 的各项参数（选择模型架构、调整 Batch Size、指定数据目录等）。
- **动态训练控制**：在网页端点击启动训练，实时查阅后端打印的 Loss/Dice 曲线流与当前 Epoch 信息，并可强制中断训练。
- **在线推理测试**：支持拖拽图片上传，调用后端缓存的模型实时返回血管分割高亮蒙版。

---

## 三、 YAML 配置系统 (`configs/default.yaml`)

项目依靠 `src/config.py` 解析并校验全局 YAML 配置。以下是控制整个管线的关键字段抽象：

* `data`: 数据集相对/绝对路径映射，以及数据流拉取设置 (`num_workers`)。
* `model`: `name` (选择模型)，以及输入输出通道定义。
* `training`: 整合 `batch_size`、`learning_rate`、`optimizer` 等超参。
* `loss`: `bce_weight` 和 `dice_weight` 权重分配比例。
* `checkpoint`: 早停容忍度 (`early_stopping_patience`)，输出目录定义。
* `postprocess` & `visualization`: 后处理连通域阈值限制与验证集叠加图像开关。

---

## 四、 快速使用指令指南

* **环境要求**：请确保系统已安装满足 `requirements.txt` 的包，且 `PyTorch` 为支持本地显卡的 CUDA 版本。
* **启动 Web 界面** (推荐初学者)：
  ```bash
  python web_server.py
  ```
* **纯命令行训练**：
  ```bash
  python train.py --config configs/default.yaml
  ```
* **测试单张图像推理**：
  ```bash
  python inference.py --model checkpoints/best_model.pth --input test.png
  ```
