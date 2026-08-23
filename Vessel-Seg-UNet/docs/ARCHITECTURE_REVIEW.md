# Vessel-Seg-UNet 架构分析与问题修复总结

> 日期：2026-08-23
> 范围：全代码库架构评审、7 类问题修复、模型优化路线建议
> 测试状态：11 个改动文件通过只读 AST 语法校验；完整 pytest 需在装有 torch / opencv 的训练机上执行

---

## 1. 架构总览

项目目标：对脑血管 DSA（数字减影血管造影）图像做二值语义分割，提供 U-Net 基线与 Attention U-Net 两种模型。核心设计理念是**消除 CLI 训练与 Web 训练两套实现的漂移**——所有入口共享同一套核心逻辑。

### 1.1 分层结构

```text
┌────────────────────────── 表示层 (入口点) ──────────────────────────┐
│  train.py        evaluate.py      inference.py       web_server.py │
│  (CLI 训练)      (CLI 评估)       (推理 API/CLI)     (Flask 本机UI) │
└──────────────┬────────────────┬──────────────────┬─────────────────┘
               │                │                  │
┌──────────────▼────────────────▼──────────────────▼─────────────────┐
│                          领域层 (src/)                              │
│                                                                     │
│  config.py        配置归一化 / 旧配置迁移 / 校验 / 路径安全          │
│  dataset.py       VesselDataset + get_dataloaders                   │
│  transforms.py    Albumentations 训练/验证增强管线                  │
│  models/          模型工厂 + UNetBaseline + AttentionUNet           │
│  losses.py        BCE + Dice 混合损失                               │
│  training.py      损失/优化器/调度器工厂                            │
│  trainer.py       共享训练循环 (AMP + 早停 + 回调)                   │
│  checkpoints.py   版本化、安全加载的检查点                          │
│  prediction.py    唯一 logits→mask 入口 (训练/评估/推理共用)         │
│  postprocess.py   连通域分析滤噪 + 孔洞填充                         │
│  metrics.py       Dice / IoU / Precision / Recall                   │
│  visualize.py     叠加对比图 / 网格图生成                           │
└─────────────────────────────────────────────────────────────────────┘
```

依赖方向严格单向：**入口点 → src 核心 → PyTorch/OpenCV**。`web_server.py` 不复制训练逻辑，只把 Web 的开始/停止操作翻译成 `Trainer` 调用，通过 `on_epoch_end` / `should_stop` 两个回调与训练循环交互。

### 1.2 张量契约

| 环节 | 形状 | 类型 | 值域 |
|------|------|------|------|
| Dataset 输出 image | `(1, H, W)` | float32 | `[0, 1]` |
| Dataset 输出 mask | `(1, H, W)` | float32 | `{0.0, 1.0}` |
| 模型输入 | `(B, 1, H, W)` | float32 | 任意 |
| 模型输出 (logits) | `(B, 1, H, W)` | float32 | 任意（**不加 Sigmoid**） |
| 损失函数输入 targets | `(B, 1, H, W)` | float32 | `{0.0, 1.0}` |

### 1.3 原有优点（评审确认）

1. **单一事实来源**：预测管线、训练循环、配置 schema 各只有一份，四类入口共享；
2. **安全防线完整**：检查点路径防穿越（`resolve_checkpoint_dir`）、`torch.load(weights_only=True)`、Base64 严格校验、像素数上限、仅监听 127.0.0.1；
3. **可维护性**：模型注册表 + 工厂函数 + 回调钩子，扩展成本低；
4. **工程细节**：中文路径读取（`np.fromfile`）、非 2 幂尺寸 padding 对齐、小数据集 `drop_last` 处理、AMP 显存优化、掩膜值域防御性二值化。

---

## 2. 已修复的问题

| # | 问题 | 修复 | 涉及文件 |
|---|------|------|----------|
| 1 | 依赖清单含未使用的 `scikit-image`、`scipy` | 移除 | `requirements.txt` |
| 2 | 中文路径读写不一致：`inference.py` 用 `cv2.imread`/`cv2.imwrite`，Windows 中文路径会失败 | 读取改 `np.fromfile + cv2.imdecode`，写入改 `cv2.imencode + open(wb)`，与 `dataset.py` 统一 | `inference.py` |
| 3 | 早停指标按 batch 平均，小 batch 与大 batch 权重相同，结果随 batch 划分波动 | 新增 `MetricAccumulator` 做全数据集像素级聚合，早停依据改为全局 Dice/IoU | `src/metrics.py`、`src/trainer.py` |
| 4 | 空预测时 Precision/Recall 因 eps 在分子上返回 0.5 而非 0 | eps 只保留在分母；P、T 均空时 Dice/IoU 约定为 1.0（文档化） | `src/metrics.py` |
| 5 | DiceLoss 全 batch 像素求和，大前景样本主导梯度 | 改为逐样本 Dice 后取平均 | `src/losses.py` |
| 6 | 训练/评估/推理无法指定设备 | `train.py`、`evaluate.py`、`inference.py` 新增 `--device`（如 `cuda:1`、`cpu`），`Trainer` 支持显式设备，非法设备或 CUDA 不可用启动即报错 | `train.py`、`evaluate.py`、`inference.py`、`src/trainer.py` |
| 7 | Web 并发竞态：`inference_cache` 与 `segmentor.threshold` 无锁；训练中改配置无提示 | 新增 `CONFIG_LOCK`（配置读写互斥）与 `INFERENCE_LOCK`（推理串行化）；训练进行中保存配置返回 `note` 提示下次生效 | `web_server.py` |
| 8 | 测试覆盖薄弱（仅 config/checkpoints/prediction 少量用例） | 新增 `test_postprocess.py`、`test_metrics.py`、`test_losses.py`；几何还原逻辑抽为纯函数 `restore_original_geometry` 并补充 2 个用例 | `tests/`、`inference.py` |

### 2.1 ⚠️ 行为变更提醒

1. **DiceLoss 改为逐样本平均**：训练损失数值与旧版不可直接对比；旧检查点权重加载不受影响，仍可用于推理；
2. **早停 Dice 改为全局像素级聚合**：与旧的 batch 平均有微小数值差异，历史最优 Dice 记录仅供参考；
3. **Precision/Recall 空预测语义**：空预测/空目标场景从 0.5 变为 0（更符合指标定义）。

---

## 3. 验证情况

- ✅ 11 个改动文件通过只读 AST 语法校验；
- ✅ 残留引用检查：确认无遗漏的 `cv2.imread/imwrite`（inference）、`load_config(CONFIG_PATH)` 裸调用（web_server，仅 `_load_config` 内部保留）、`calculate_dice` 在 `evaluate.py` 的逐样本报告用途属有意保留；
- ⏳ **待执行**：在装有依赖的训练机上运行：

```bash
pip install -r requirements-dev.txt
pytest -q
```

新增测试共 12 个用例（postprocess 3、metrics 5、losses 3、几何还原 3，含原有用例扩充）。

---

## 4. 后续模型优化路线（按优先级）

### 4.1 损失函数（收益最大，先行）

- **Focal Tversky Loss**：β 参数调大以惩罚细支血管漏检，比 BCE+Dice 对不平衡 + 细结构更友好；
- **骨架辅助监督（clDice）**：从掩膜提取血管中心线作为辅助标签，专门引导网络保留末梢细支连通性——脑血管分割的关键痛点；
- 保留当前 BCE+Dice 作为基线对比。

### 4.2 数据层面

- 中心线/骨架标签辅助通道；
- **随机缩放 + Patch 裁块训练**：避免全图 512 缩放导致细支退化；推理滑窗拼接 + 翻转 TTA；
- 强度增强加强：`RandomGamma`、`GridDistortion`、调优 CLAHE 参数；
- 类别采样保证 batch 内含足够前景，防全背景 batch 梯度噪声。

### 4.3 模型架构

- **残差块 + 深监督**：`DoubleConv` 换残差块，解码器各级加侧输出（deep supervision）；
- 下一步可试 **U-Net++**（密集跳跃连接）或 nnU-Net 式配置；数据量充足后再考虑 TransUNet / Swin-UNet；
- 显存受限（游戏本）：加 gradient checkpointing、`channels_last` 内存格式，保留现有双线性上采样 + AMP。

### 4.4 训练策略

- Warmup + Cosine Restart 替代纯 cosine；推理用 **EMA 权重**；
- **梯度裁剪**（`clip_grad_norm_`）防 AMP + 极端不平衡下的 loss 尖峰；
- 固定种子保证可复现（config 加 `seed` 并 `torch.manual_seed`）。

### 4.5 后处理

- 从像素级连通域升级为**拓扑级**：基于骨架端点配对连接断裂血管段；
- 阈值小网格搜索或自适应阈值（Otsu / 预测直方图）。

### 4.6 评测

- 补充 **clDice**（骨架 Dice）、灵敏度/特异度、95% Hausdorff 距离；
- 按血管粗细分层报告（细/中/粗），更贴合临床关注点。

### 4.7 部署

- ONNX → TensorRT FP16 推理；
- Web 端可换 FastAPI 或 Flask + waitress；多用户场景用进程池按检查点缓存 Segmentor。

**建议顺序**：骨架/clDice 辅助监督 + Focal Tversky 损失先行（不改架构、收益明确）→ 验证后再动架构（Patch 训练 + 残差/深监督）。
