# Vessel-Seg-UNet 优化与全量实验工作报告

> 时间：2026-08-23 ~ 2026-08-24
> 机器：NVIDIA RTX 4060 Laptop GPU(8GB)/ Windows / Python 3.11
> 数据集：DIAS 脑血管造影(训练 30 张 / 验证 20 张,800×800 灰度,血管前景占比约 6%)

---

## 1. 任务背景

用户先后提出三条需求,本报告按四个阶段记录完整过程:

1. **分析代码架构** —— 通读全部源码,输出架构评审结论;
2. **解决问题** —— 修复评审中发现的 8 类问题;
3. **进行优化并全量实验** —— 实现模型优化并在本机 RTX 4060 上完成 4 组对照实验。

---

## 2. 阶段一：代码架构评审与问题修复（08-23）

### 2.1 架构结论

项目是一个基于 PyTorch 的脑血管 DSA 二值分割项目,分层清晰:

```text
入口层:  train.py / evaluate.py / inference.py / web_server.py
核心层:  src/{config,dataset,transforms,models,losses,training,trainer,
              checkpoints,prediction,postprocess,metrics,visualize}.py
```

核心设计亮点:**单一事实来源**(预测管线、训练循环、配置 schema 各只有一份,CLI 与 Web 共用)、
**安全防线完整**(路径防穿越、`weights_only` 加载、仅监听本机)、**工厂 + 注册表 + 回调钩子**的扩展模式。

### 2.2 修复的 8 个问题

| # | 问题 | 修复 |
|---|------|------|
| 1 | `requirements.txt` 含未使用的 scikit-image / scipy | 移除 |
| 2 | `inference.py` 用 `cv2.imread/imwrite`,Windows 中文路径失败 | 改 `np.fromfile + imdecode` / `imencode + open(wb)` |
| 3 | 早停指标按 batch 平均,受 batch 划分影响 | 新增 `MetricAccumulator` 做全数据集像素级聚合 |
| 4 | 空预测时 Precision/Recall 因 eps 返回 0.5 | eps 只留分母,空预测返回 0 |
| 5 | DiceLoss 全 batch 求和,大前景样本主导梯度 | 改逐样本平均 |
| 6 | 无法指定训练设备 | `train/evaluate/inference.py` 新增 `--device`,`Trainer` 支持设备参数 |
| 7 | Web 并发竞态(推理缓存、配置读写无锁) | 新增 `CONFIG_LOCK` / `INFERENCE_LOCK`,训练中改配置返回提示 |
| 8 | 测试覆盖薄弱 | 新增 postprocess / metrics / losses / 几何还原测试 |

同时产出架构评审文档 `docs/ARCHITECTURE_REVIEW.md`。

---

## 3. 阶段二：模型优化实现（08-23）

### 3.1 新增能力

| 优化 | 实现方式 | 文件 |
|------|----------|------|
| **Focal Tversky 损失** | α=0.7(漏检惩罚)/ β=0.3 / γ=0.75,逐样本计算 | `src/losses.py` |
| **clDice 中心线监督** | ① Zhang-Suen 骨架化(纯 numpy,无新依赖)提取金标准骨架;② 预测侧可微软骨架(max-pool 形态学近似);③ 主损失 + λ·clDice 组合 | `src/skeleton.py`、`src/losses.py` |
| **warmup + 余弦退火** | `SequentialLR`(LinearLR 前 5 epoch + CosineAnnealingLR) | `src/training.py` |
| **梯度裁剪** | AMP 下先 `unscale_` 再 `clip_grad_norm_` | `src/trainer.py` |
| **EMA 权重** | 指数滑动平均(带 warmup 衰减);验证与保存均用 EMA 权重,保证"验证的模型 = 部署的模型" | `src/trainer.py` |
| **随机种子** | seed 42,Python/NumPy/PyTorch/cuDNN 全固定 | `src/training.py` |
| **数据流扩展** | `cl_dice_weight > 0` 时 Dataset 自动返回 (image, mask, skeleton) 三元组 | `src/dataset.py` |

### 3.2 设计原则

- **向后兼容**:所有新配置键都有默认值;损失函数签名统一为 `(logits, targets, skeleton=None)`;
  旧检查点可正常加载(缺省配置自动补齐,cl_dice_weight 默认 0);
- **配置校验**:`validate_config` 覆盖全部新键(损失名、α/β/γ、cl_dice 权重、ema_decay∈[0,1) 等)。

### 3.3 测试

新增 `tests/test_skeleton.py`(Zhang-Suen 5 例)与损失新用例(Focal Tversky / clDice / 组合损失)。
**最终:27 个测试全部通过**(含全部历史用例)。

---

## 4. 阶段三：环境搭建（08-23，耗时最长）

本机环境原本一无所有(无 torch、无数据索引、网络受限),依次踩过 4 个坑:

| 坑 | 现象 | 解法 |
|----|------|------|
| **沙箱 TLS 失效** | curl/.NET 所有 HTTPS 报 `SEC_E_NO_CREDENTIALS`(schannel 拿不到证书库凭据) | 换 Python 验证:OpenSSL 不走 schannel,**不受影响** |
| **pip 走注册表代理卡死** | pip 自动读取 Windows 注册表里的 Clash 代理(127.0.0.1:7897),大文件下载永久停滞 | `$env:NO_PROXY="*"` 强制直连 |
| **直连国外被墙** | download.pytorch.org / pypi.org 直连不通 | 阿里云镜像:`mirrors.aliyun.com/pypi`(直连可用)+ `pytorch-wheels` 目录(有 CUDA 轮子) |
| **albumentations 需要 C 编译器** | 1.4.8+ 依赖 albucore→stringzilla(C 扩展),本机无 MSVC,源码构建必失败 | 锁定 `albumentations<1.4.8`(1.4.7 无该依赖),并给 venv 打补丁注释掉 import 时的联网版本检查 |

**最终环境**:torch 2.10.0+cu128 / torchvision 0.25.0+cu128 / albumentations 1.4.7 / numpy 2.4.6 / opencv 5.0.0,CUDA 正常识别 RTX 4060。

完整记录见 `docs/ENVIRONMENT.md`。

数据发现:本机 `E:\DSCA\image\DSCA\开源数据集；DIAS` 有现成的 train(30 对)/ val(20 对)图像+掩膜,
恰好是项目期望的目录布局,直接作为实验数据集(掩膜为干净的 {0,255} 二值图,前景占比 ~6%,符合血管分割场景)。

---

## 5. 阶段四：全量对比实验（08-23 深夜 ~ 08-24）

### 5.1 实验设计(控制变量)

训练策略四组完全一致:AdamW lr=1e-4、warmup 5 + cosine、grad clip 1.0、EMA 0.999、seed 42、AMP、batch 2、早停 patience 10、img_size 512。

| 实验 | 模型 | 损失 | 目的 |
|------|------|------|------|
| exp1 | UNet(13.4M) | BCEDice | 基线 |
| exp2 | UNet | FocalTversky | 损失对比 |
| exp3 | UNet | BCEDice + clDice(λ=0.5) | 中心线监督 |
| exp4 | AttentionUNet(31.4M) | FocalTversky + clDice | 组合 |

每组训练后分别做**原始**与**后处理**两次评估。一键执行:`powershell -File experiments/run_experiments.ps1`。

### 5.2 实验结果(验证集,EMA 权重)

| 实验 | Dice | IoU | Precision | Recall | 训练最佳Dice | 完成Epoch |
|------|------|-----|-----------|--------|--------------|-----------|
| exp1 UNet + BCEDice | 0.6861 | 0.5277 | 0.7477 | 0.6609 | 0.6823 | 45(早停) |
| exp2 UNet + FocalTversky | 0.6855 | 0.5242 | 0.6441 | **0.7577** | 0.6802 | 51(早停) |
| **exp3 UNet + BCEDice + clDice** | **0.7029** | **0.5436** | 0.7056 | 0.7159 | **0.6995** | 52(早停) |
| exp4 Attention + Focal + clDice | 0.6699 | 0.5057 | 0.6133 | 0.7617 | 0.6660 | 51(被休眠中断) |

### 5.3 六条结论

1. **clDice 中心线监督是本轮最大赢家**:不改模型架构,Dice 提升 **+1.7pp** 至 0.7029,
   Precision/Recall 最均衡(0.706/0.716),逐样本标准差最小(0.045),稳定性最好;
2. **Focal Tversky 按设计工作**:Recall 提升 **+9.7pp**(0.661→0.758),Precision 相应下降——"少漏血管"场景适用;
3. **小数据上"大模型 + 组合损失"不占优**:exp4 的 31.4M 参数在 30 张训练图上易过拟合;
4. **默认后处理对细血管数据集有害**:`min_component_size=50` 删除真实细支,Dice 全线下滑 2~2.5pp,
   建议 DIAS 上 ≤10 或关闭,部署前按血管口径重新标定;
5. **训练成本**:clDice 的逐 epoch Zhang-Suen 计算使每 epoch 从 ~7s 涨到 ~55s;
   后续可降采样骨架计算(拓扑损失对分辨率不敏感)或预计算+几何同步;
6. **环境意外**:训练期间笔记本多次休眠导致墙钟时间膨胀,exp4 被中断于 51/80(其检查点为 epoch 42 最佳);
   重跑请保持通电/关闭睡眠。

---

## 6. 全部产出物清单

### 6.1 代码

- `src/losses.py` — 新增 FocalTverskyLoss / CLDiceLoss / soft_skel / CombinedVesselLoss;Dice 逐样本
- `src/skeleton.py` — 新增:Zhang-Suen 骨架化
- `src/metrics.py` — MetricAccumulator;Precision/Recall 空预测语义修正
- `src/trainer.py` — EMA、梯度裁剪、设备参数、骨架传递、torch.amp 新 API
- `src/training.py` — 损失工厂、warmup 调度器、set_seed
- `src/dataset.py` — 骨架三元组输出
- `src/config.py` / `configs/default.yaml` — 新配置键 + 校验
- `src/transforms.py` — PadIfNeeded 显式 value=0(1.4.7 兼容)
- `train.py` / `evaluate.py` / `inference.py` — --device、set_seed、中文路径读写
- `web_server.py` — 配置/推理锁、训练中改配置提示
- `requirements.txt` — 移除 scikit-image/scipy、albumentations<1.4.8

### 6.2 测试(27 passed)

`tests/test_skeleton.py`(新)、`tests/test_metrics.py`(新)、`tests/test_losses.py`(新+扩展)、
`tests/test_postprocess.py`(新)、`tests/test_prediction.py`(几何还原扩展)

### 6.3 实验资产

- `configs/experiments/exp1~exp4.yaml` — 4 组实验配置
- `experiments/run_experiments.ps1` — 一键训练+评估
- `experiments/summarize_results.py` — 结果汇总脚本
- `experiments/smoke_test.py` — GPU 全链路冒烟测试
- `experiments/results_summary.md` — **实验结果总报告**
- `experiments/logs/` — 全部训练/评估/环境日志
- `checkpoints/exp1~exp4/best_model.pth` — 4 个 EMA 权重检查点
- `results/experiments/` — 8 份评估报告 + 可视化

### 6.4 文档

- `docs/ARCHITECTURE_REVIEW.md` — 架构评审与修复总结
- `docs/ENVIRONMENT.md` — 环境搭建与踩坑记录
- `docs/REFACTORING.md` — 重构说明(增补优化批次)
- `README.md` — 核心特性/接口契约/实验用法更新

---

## 7. 遗留事项与下一步建议

1. **重跑 exp4**(可选):保持通电运行 `powershell -File experiments/run_experiments.ps1`,或仅重跑 exp4 训练;
2. **后处理参数标定**:按 DIAS 血管口径把 `min_component_size` 从 50 降到 ≤10,或增加"骨架长度"维度的滤噪;
3. **clDice 提速**:骨架降采样到 256 计算,或预计算+与几何变换同步,把每 epoch 从 ~55s 拉回 ~10s;
4. **下一步优化**(按预期收益排序):
   - 深监督(解码器侧输出)+ 残差块;
   - Patch 训练(高分辨率保留细支)+ 滑窗推理 + 翻转 TTA;
   - 评估补充 clDice 指标与按血管粗细分层报告;
   - 把 clDice 方案(exp3)固化为默认配置,并在自有病例数据(E:\DSCA 下的 0710/1~14/29~41 3D 病例)上做 2D 切片验证。
