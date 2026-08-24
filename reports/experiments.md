# 训练与评估报告

> 每次训练 + 评估的结果都汇总记录于此，便于对比不同配置/数据集/模型的效果。

## 实验汇总

| # | 配置 | 模型 | 训练/验证 | Best Dice(val) | Eval Dice | Eval IoU | Precision | Recall |
|---|------|------|-----------|----------------|-----------|----------|-----------|--------|
| 1 | exp1_baseline_own | unet_baseline | 142 / 35 | 0.7645 | 0.7649 | 0.6311 | 0.7777 | 0.7685 |
| 2 | exp2_baseline_mixed | unet_baseline | 172 / 35 | 0.7569 | 0.7563 | 0.6173 | 0.7635 | 0.7625 |
| 3 | exp3_attention_own | attention_unet | 142 / 35 | 0.7618 | 0.7621 | 0.6265 | 0.7998 | 0.7418 |
| 4 | exp4_attention_mixed | attention_unet | 172 / 35 | 0.7643 | 0.7639 | 0.6291 | 0.7774 | 0.7677 |
| 5 | exp5_resnet_own | unet_resnet | 142 / 35 | 0.7683 | 0.7673 | 0.6329 | 0.7573 | 0.7917 |
| 6 | exp6_resnet_mixed | unet_resnet | 172 / 35 | 0.7693 | 0.7688 | 0.6353 | 0.7727 | 0.7781 |
| 7 | exp7_resnet50_mixed | unet_resnet(resnet50) | 172 / 35 | 0.7644 | 0.7631 | 0.6276 | 0.7314 | 0.8135 |
| 8 | exp8_resnet_mixed_tuned | unet_resnet | 172 / 35 | 0.7564 | 0.7562 | 0.6179 | 0.7654 | 0.7608 |

---

## 实验 1：exp1_baseline_own（unet_baseline）

### 配置
- 配置文件：`configs/exp1_baseline_own.yaml`
- 模型：`unet_baseline`（13,390,209 参数）
- 数据：own_train **142** / own_val **35**（自有数据按 4 个时长子集分层 80/20 切分，seed=42）
- 超参：batch_size 4 · epochs 100 · AdamW lr 1e-4 · cosine 调度 · AMP on · 早停 patience 15

### 数据划分
| 子集 | train | val |
|------|-------|-----|
| sub1_23s (2~3s) | 33 | 8 |
| sub2_4s (4s) | 33 | 8 |
| sub3_56s (5~6s) | 33 | 8 |
| sub4_ds2_4s (ds2) | 43 | 11 |
| **合计** | **142** | **35** |

划分明细：`C:/Users/13091/Desktop/数据集/own_split/split_record.csv`

### 训练结果
- 早停：第 71 轮触发（patience 15）
- 最佳验证 Dice：**0.7645**（第 55 轮）
- 权重：`checkpoints_exp1_data/best_model.pth`、`checkpoints_exp1_data/last_model.pth`（各 153.4 MB）
- 训练日志：`train_own.log`

### 评估结果（own_val 35 张）
| 指标 | 均值 ± 标准差 |
|------|--------------|
| Dice | 0.7649 ± 0.118 |
| IoU | 0.6311 ± 0.125 |
| Precision | 0.7777 ± 0.121 |
| Recall | 0.7685 ± 0.135 |

评估产物：`results/exp1_own_val/eval_report.txt` + 35 张 overlay 叠加图

### 外部盲测（DIAS-all，50 张）

> DIAS 数据未参与该模型训练，作为外部测试集评估泛化能力。

| 指标 | 均值 ± 标准差 |
|------|--------------|
| Dice | 0.7038 ± 0.046 |
| IoU | 0.5449 ± 0.055 |
| Precision | 0.6573 ± 0.079 |
| Recall | 0.7760 ± 0.091 |

- 测试集：`C:/Users/13091/Desktop/数据集/DIAS-all/`（DIAS train 30 + val 20 合并，文件名加 train_/val_ 前缀）
- 评估产物：`results/exp1_dias_all/eval_report.txt` + 50 张 overlay
- 对比 own_val（Dice 0.7649）：外部数据 Dice 下降约 0.06，主要来自 Precision（0.778→0.657），Recall 基本持平，提示存在领域差异（domain gap）

### 外部盲测（DIAS-val，20 张，纯未见）
> 用 DIAS-val（不含 DIAS-train）做诚实外部评估，避免混入训练集造成泄漏。

| 指标 | 均值 ± 标准差 |
|------|--------------|
| Dice | 0.6952 ± 0.047 |
| IoU | 0.5348 ± 0.055 |
| Precision | 0.6504 ± 0.094 |
| Recall | 0.7729 ± 0.097 |

评估产物：`results/exp1_dias_val/eval_report.txt`

---

## 实验 2：exp2_baseline_mixed（unet_baseline）—— 消融：训练集加入 DIAS-train

### 配置
- 配置文件：`configs/exp2_baseline_mixed.yaml`
- 模型：`unet_baseline`（13,390,209 参数）
- 数据：训练集 **172**（own_train 142 + DIAS-train 30，多目录合并）；验证集 **35**（own_val，与实验 1 完全一致）
- 超参：与实验 1 相同（batch 4 · epochs 100 · AdamW lr 1e-4 · cosine · AMP · patience 15）

### 训练结果
- 早停：第 32 轮触发
- 最佳验证 Dice：**0.7569**
- 权重：`checkpoints_exp2_dias_mixed/best_model.pth`、`last_model.pth`
- 训练日志：`train_own_dias_mixed.log`

### 评估结果
| 测试集 | Dice | IoU | Precision | Recall |
|--------|------|-----|-----------|--------|
| own_val (35) | 0.7563 ± 0.105 | 0.6173 ± 0.111 | 0.7635 ± 0.119 | 0.7625 ± 0.123 |
| DIAS-all (50，外部) | 0.7174 ± 0.046 | 0.5613 ± 0.057 | 0.6962 ± 0.072 | 0.7516 ± 0.078 |

评估产物：`results/exp2_own_val/`、`results/exp2_dias_all/`

### 外部盲测（DIAS-val，20 张，纯未见）
| 指标 | 均值 ± 标准差 |
|------|--------------|
| Dice | 0.7029 ± 0.047 |
| IoU | 0.5439 ± 0.055 |
| Precision | 0.6887 ± 0.082 |
| Recall | 0.7358 ± 0.092 |

评估产物：`results/exp2_dias_val/eval_report.txt`

### 消融结论（vs 实验 1）
| 指标 | 实验 1（own） | 实验 2（own+DIAS） | 变化 |
|------|--------------|-------------------|------|
| own_val Dice | 0.7649 | 0.7563 | **-0.009** |
| DIAS-all Dice | 0.7038 | 0.7174 | **+0.014** |
| DIAS-all Precision | 0.6573 | 0.6962 | **+0.039** |
| DIAS-all Recall | 0.7760 | 0.7516 | -0.024 |

加入 DIAS-train 后：外部 DIAS 泛化提升（Dice +1.4 点、Precision +3.9 点，误报减少），代价是自有域性能小幅回落（own_val Dice -0.9 点）。符合「混合训练缩小 domain gap」的预期。

---

## 实验 3：exp3_attention_own（attention_unet）

### 配置
- 配置文件：`configs/exp3_attention_own.yaml`
- 模型：`attention_unet`（31,388,013 参数，约为 baseline 的 2.3 倍）
- 数据：own_train 142 / own_val 35（与实验 1 完全一致）
- 超参：与实验 1 相同（batch 4 · epochs 100 · AdamW lr 1e-4 · cosine · AMP · patience 15）

### 训练结果
- 早停：第 73 轮触发
- 最佳验证 Dice：**0.7618**
- 权重：`checkpoints_exp3_attention/best_model.pth`、`last_model.pth`（各 359.4 MB）
- 训练日志：`train_own_attention.log`

### 评估结果（own_val 35 张）
| 指标 | 均值 ± 标准差 |
|------|--------------|
| Dice | 0.7621 ± 0.113 |
| IoU | 0.6265 ± 0.120 |
| Precision | 0.7998 ± 0.124 |
| Recall | 0.7418 ± 0.130 |

评估产物：`results/exp3_own_val/eval_report.txt` + 35 张 overlay 叠加图

### 外部盲测（DIAS-all，50 张）
> DIAS 数据未参与该模型训练，作为外部测试集评估泛化能力。

| 指标 | 均值 ± 标准差 |
|------|--------------|
| Dice | 0.6975 ± 0.054 |
| IoU | 0.5381 ± 0.062 |
| Precision | 0.6574 ± 0.105 |
| Recall | 0.7768 ± 0.116 |

- 测试集：`C:/Users/13091/Desktop/数据集/DIAS-all/`（DIAS train 30 + val 20 合并）
- 评估产物：`results/exp3_dias_all/eval_report.txt` + 50 张 overlay
- 对比 own_val（Dice 0.7621）：外部数据 Dice 下降约 0.06，主要来自 Precision（0.800→0.657），提示存在领域差异（domain gap）

### 外部盲测（DIAS-val，20 张，纯未见）
| 指标 | 均值 ± 标准差 |
|------|--------------|
| Dice | 0.6933 ± 0.058 |
| IoU | 0.5335 ± 0.066 |
| Precision | 0.6574 ± 0.117 |
| Recall | 0.7702 ± 0.109 |

评估产物：`results/exp3_dias_val/eval_report.txt`

### 对比（vs 实验 1 unet_baseline）
| 指标 | unet_baseline | attention_unet | 变化 |
|------|---------------|----------------|------|
| own_val Dice | 0.7649 | 0.7621 | -0.003 |
| DIAS-all Dice | 0.7038 | 0.6975 | -0.006 |
| 参数量 | 13.39M | 31.39M | +134% |

结论：attention_unet 在参数量翻倍（13.4M→31.4M）、训练更慢的情况下，own_val 与 DIAS-all 均未超过 baseline，反而略低（Dice -0.3~-0.6 点）。注意力机制对该任务无明显收益。

---

## 实验 4：exp4_attention_mixed（attention_unet）

### 配置
- 配置文件：`configs/exp4_attention_mixed.yaml`
- 模型：`attention_unet`（31,388,013 参数）
- 数据：训练集 **172**（own_train 142 + DIAS-train 30，多目录合并）；验证集 **35**（own_val，与实验 2 完全一致）
- 超参：与实验 2 相同（batch 4 · epochs 100 · AdamW lr 1e-4 · cosine · AMP · patience 15）

### 训练结果
- 早停：第 77 轮触发
- 最佳验证 Dice：**0.7643**
- 权重：`checkpoints_exp4_dias_mixed_attention/best_model.pth`、`last_model.pth`（各 359.4 MB）
- 训练日志：`train_own_dias_mixed_attention.log`

### 评估结果
| 测试集 | Dice | IoU | Precision | Recall |
|--------|------|-----|-----------|--------|
| own_val (35) | 0.7639 ± 0.113 | 0.6291 ± 0.121 | 0.7774 ± 0.126 | 0.7677 ± 0.131 |
| DIAS-all (50，外部) | 0.7331 ± 0.049 | 0.5810 ± 0.059 | 0.7646 ± 0.061 | 0.7169 ± 0.095 |

评估产物：`results/exp4_own_val/`、`results/exp4_dias_all/`

### 外部盲测（DIAS-val，20 张，纯未见）
| 指标 | 均值 ± 标准差 |
|------|--------------|
| Dice | 0.7181 ± 0.053 |
| IoU | 0.5626 ± 0.060 |
| Precision | 0.7574 ± 0.066 |
| Recall | 0.7003 ± 0.109 |

评估产物：`results/exp4_dias_val/eval_report.txt`

### 消融结论（vs 实验 2 unet_baseline 同数据）
| 指标 | 实验 2（baseline） | 实验 4（attention） | 变化 |
|------|-------------------|--------------------|------|
| own_val Dice | 0.7563 | 0.7639 | +0.008 |
| DIAS-all Dice | 0.7174 | 0.7331 | +0.016 |
| DIAS-val Dice | 0.7029 | 0.7181 | +0.015 |

结论：与实验 3（纯 own 数据下 attention 略差于 baseline）不同，**加入 DIAS-train 后（172 张），attention_unet 全面超过 baseline**（own_val +0.8 点、外部 DIAS-val +1.5 点）。说明注意力机制在数据更多、领域更多样时开始发挥优势（容量更大 + 数据足够支撑）。实验 4 成为当前外部泛化最优（DIAS-val 0.7181）。

---

## 实验 5：exp5_resnet_own（unet_resnet）

### 配置
- 配置文件：`configs/exp5_resnet_own.yaml`
- 模型：`unet_resnet`（24,429,969 参数，ImageNet 预训练 ResNet34 编码器）
- 数据：own_train 142 / own_val 35（自有数据按 4 个时长子集分层 80/20 切分，seed=42）
- 超参：与实验 1 相同（batch 4 · epochs 100 · AdamW lr 1e-4 · cosine · AMP · patience 15）

### 数据划分
| 子集 | train | val |
|------|-------|-----|
| sub1_23s (2~3s) | 33 | 8 |
| sub2_4s (4s) | 33 | 8 |
| sub3_56s (5~6s) | 33 | 8 |
| sub4_ds2_4s (ds2) | 43 | 11 |
| **合计** | **142** | **35** |

划分明细：`C:/Users/13091/Desktop/数据集/own_split/split_record.csv`

### 训练结果
- 早停：未触发（跑满 100 轮）
- 最佳验证 Dice：**0.7683**（5 个实验中最高）
- 权重：`checkpoints_exp5_resnet/best_model.pth`、`last_model.pth`（各 279.9 MB）
- 训练日志：`train_own_resnet.log`

### 评估结果（own_val 35 张）
| 指标 | 均值 ± 标准差 |
|------|--------------|
| Dice | 0.7673 ± 0.110 |
| IoU | 0.6329 ± 0.118 |
| Precision | 0.7573 ± 0.116 |
| Recall | 0.7917 ± 0.131 |

评估产物：`results/exp5_own_val/eval_report.txt`

### 外部盲测（DIAS-all，50 张）
> DIAS 数据未参与该模型训练，作为外部测试集评估泛化能力。

| 指标 | 均值 ± 标准差 |
|------|--------------|
| Dice | 0.7007 ± 0.041 |
| IoU | 0.5409 ± 0.050 |
| Precision | 0.6205 ± 0.069 |
| Recall | 0.8169 ± 0.067 |

评估产物：`results/exp5_dias_all/eval_report.txt`

### 外部盲测（DIAS-val，20 张，纯未见）
| 指标 | 均值 ± 标准差 |
|------|--------------|
| Dice | 0.6893 ± 0.036 |
| IoU | 0.5270 ± 0.042 |
| Precision | 0.6027 ± 0.058 |
| Recall | 0.8150 ± 0.065 |

评估产物：`results/exp5_dias_val/eval_report.txt`

### 结论（vs 实验 1）
| 指标 | 实验 1（baseline） | 实验 5（resnet） | 变化 |
|------|-------------------|-----------------|------|
| own_val Dice | 0.7649 | 0.7673 | +0.002 |
| DIAS-val Dice | 0.6952 | 0.6893 | -0.006 |

预训练 ResNet 编码器带来**自有数据小幅提升（own_val Dice 0.7649→0.7673，全场最高）**，但**外部 DIAS 泛化反而全场最差（DIAS-val 0.6893）**。Precision 在外部数据上明显下降（0.757→0.603），Recall 偏高（0.815），说明预训练编码器过拟合自有域、在外部数据上过分割。符合「ImageNet 预训练对自然图像有效、对医学造影泛化有限」的预期。

---

## 实验 6：exp6_resnet_mixed（unet_resnet）

### 配置
- 配置文件：`configs/exp6_resnet_mixed.yaml`
- 模型：`unet_resnet`（24,429,969 参数，ImageNet 预训练 ResNet34 编码器）
- 数据：训练集 **172**（own_train 142 + DIAS-train 30，多目录合并）；验证集 **35**（own_val，与实验 2/4 完全一致）
- 超参：与实验 2/4 相同（batch 4 · epochs 100 · AdamW lr 1e-4 · cosine · AMP · patience 15）

### 训练结果
- 早停：第 88 轮触发
- 最佳验证 Dice：**0.7693**（6 个实验中最高）
- 权重：`checkpoints_exp6_dias_mixed_resnet/best_model.pth`、`last_model.pth`（各 279.9 MB）
- 训练日志：`train_own_dias_mixed_resnet.log`

### 评估结果
| 测试集 | Dice | IoU | Precision | Recall |
|--------|------|-----|-----------|--------|
| own_val (35) | 0.7688 ± 0.112 | 0.6353 ± 0.120 | 0.7727 ± 0.118 | 0.7781 ± 0.128 |
| DIAS-all (50，外部) | 0.7497 ± 0.047 | 0.6018 ± 0.060 | 0.7568 ± 0.054 | 0.7499 ± 0.079 |

评估产物：`results/exp6_own_val/`、`results/exp6_dias_all/`

### 外部盲测（DIAS-val，20 张，纯未见）
| 指标 | 均值 ± 标准差 |
|------|--------------|
| Dice | 0.7182 ± 0.040 |
| IoU | 0.5618 ± 0.048 |
| Precision | 0.7386 ± 0.069 |
| Recall | 0.7148 ± 0.099 |

评估产物：`results/exp6_dias_val/eval_report.txt`

### 消融结论（vs 实验 5 unet_resnet 纯 own）
| 指标 | 实验 5（纯 own） | 实验 6（own+DIAS） | 变化 |
|------|-----------------|-------------------|------|
| own_val Dice | 0.7673 | 0.7688 | +0.002 |
| DIAS-val Dice | 0.6893 | 0.7182 | **+0.029** |
| DIAS-val Precision | 0.6027 | 0.7386 | **+0.136** |

结论：加入 DIAS-train 对 unet_resnet 的效果**最显著**——外部 DIAS-val Dice 从 0.6893 飙到 0.7182（+2.9 点），Precision 从 0.603 涨到 0.739（+13.6 点，过分割被大幅纠正），同时 own_val 仍保持全场最高（0.7688）。预训练编码器在纯 own 数据下过拟合自有域、外部泛化最差；混合训练恰好补上了这个短板。实验 6 成为**当前综合最优**：own_val 与外部 DIAS-val 双双第一。

---

## 实验 7：exp7_resnet50_mixed（unet_resnet + resnet50）

### 配置
- 配置文件：`configs/exp7_resnet50_mixed.yaml`
- 模型：`unet_resnet`（73,162,641 参数，ImageNet 预训练 ResNet50 编码器）
- 数据：训练集 **172**（own_train 142 + DIAS-train 30，多目录合并）；验证集 **35**（own_val，与实验 2/4/6 完全一致）
- 超参：与实验 6 相同（batch 4 · epochs 100 · AdamW lr 1e-4 · cosine · AMP · patience 15）

### 训练结果
- 早停：第 83 轮触发
- 最佳验证 Dice：**0.7644**
- 权重：`checkpoints_exp7_dias_mixed_resnet50/best_model.pth`、`last_model.pth`（各 837.8 MB）
- 训练日志：`train_own_dias_mixed_resnet50.log`

### 评估结果
| 测试集 | Dice | IoU | Precision | Recall |
|--------|------|-----|-----------|--------|
| own_val (35) | 0.7631 ± 0.111 | 0.6276 ± 0.119 | 0.7314 ± 0.116 | 0.8135 ± 0.135 |
| DIAS-all (50，外部) | 0.7451 ± 0.040 | 0.5954 ± 0.052 | 0.7003 ± 0.058 | 0.8033 ± 0.065 |

评估产物：`results/exp7_own_val/`、`results/exp7_dias_all/`

### 外部盲测（DIAS-val，20 张，纯未见）
| 指标 | 均值 ± 标准差 |
|------|--------------|
| Dice | 0.7221 ± 0.033 |
| IoU | 0.5661 ± 0.040 |
| Precision | 0.6844 ± 0.064 |
| Recall | 0.7772 ± 0.079 |

评估产物：`results/exp7_dias_val/eval_report.txt`

### 消融结论（vs 实验 6 resnet34 同数据）
| 指标 | 实验 6（resnet34） | 实验 7（resnet50） | 变化 |
|------|-------------------|-------------------|------|
| 参数量 | 24.43M | 73.16M | +199% |
| own_val Dice | 0.7688 | 0.7631 | -0.006 |
| DIAS-val Dice | 0.7182 | 0.7221 | +0.004 |

结论：更深编码器（resnet50，3 倍参数、3 倍权重体积、训练更慢）**没有带来实质提升**——own_val 反而略降（-0.6 点），DIAS-val 仅微升 +0.4 点（20 张样本上属噪声量级）。印证「瓶颈在数据量、不在模型容量」。resnet34（实验 6）仍是性价比最优。

---

## 实验 8：exp8_resnet_mixed_tuned（超参调优 + 强增强）

### 配置
- 配置文件：`configs/exp8_resnet_mixed_tuned.yaml`
- 模型：`unet_resnet`（resnet34，24,429,969 参数）
- 数据：训练集 **172**（own_train 142 + DIAS-train 30）；验证集 **35**（own_val）
- 相对实验 6 的改动：loss `bce/dice = 0.3/0.7`、`dice_smooth 1e-6→1.0`、`lr 1e-4→3e-4`、开启强增强（ShiftScaleRotate/GridDistortion/RandomGamma）

### 训练结果
- 早停：第 40 轮触发
- 最佳验证 Dice：**0.7564**
- 权重：`checkpoints_exp8_dias_mixed_resnet_tuned/best_model.pth`、`last_model.pth`（各 279.9 MB）
- 训练日志：`train_own_dias_mixed_resnet_tuned.log`

### 评估结果
| 测试集 | Dice | IoU | Precision | Recall |
|--------|------|-----|-----------|--------|
| own_val (35) | 0.7562 ± 0.108 | 0.6179 ± 0.115 | 0.7654 ± 0.123 | 0.7608 ± 0.121 |
| DIAS-all (50，外部) | 0.7201 ± 0.040 | 0.5642 ± 0.050 | 0.6867 ± 0.062 | 0.7653 ± 0.070 |

评估产物：`results/exp8_own_val/`、`results/exp8_dias_all/`

### 外部盲测（DIAS-val，20 张，纯未见）
| 指标 | 均值 ± 标准差 |
|------|--------------|
| Dice | 0.7049 ± 0.036 |
| IoU | 0.5455 ± 0.043 |
| Precision | 0.6776 ± 0.066 |
| Recall | 0.7467 ± 0.081 |

评估产物：`results/exp8_dias_val/eval_report.txt`

### 结论（vs 实验 6）
| 指标 | 实验 6（原超参） | 实验 8（调优） | 变化 |
|------|----------------|---------------|------|
| own_val Dice | 0.7688 | 0.7562 | -0.013 |
| DIAS-val Dice | 0.7182 | 0.7049 | -0.013 |

结论：这组调优（4 项同时改）**全面变差**，三个测试集 Dice 均下降约 1.3 点。由于一次同时改了 loss 平滑/权重、lr、强增强 4 个变量，无法归因是哪一项导致退化；但整体说明「在最优实验 6 基础上盲目调参」没有正向收益，实验 6 仍是当前最优。
