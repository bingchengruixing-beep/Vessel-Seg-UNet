# 重构说明（2026-08-23）

## 目标

本次重构消除了命令行训练与 Web 训练两套实现的漂移，统一了配置、检查点、预测与评估行为，并把 Web 服务限定为本机使用。

## 新架构

```text
configs/default.yaml
        │
        ├── src/config.py ───────────── 配置归一化、旧配置迁移、路径校验
        ├── src/training.py ─────────── 损失 / 优化器 / 调度器构建
        ├── src/trainer.py ──────────── 训练、验证、早停、检查点、进度回调
        ├── src/checkpoints.py ──────── 版本化、安全加载的检查点
        └── src/prediction.py ───────── Logits→阈值→可选后处理
              ▲                 ▲                 ▲
           train.py        evaluate.py       inference.py / web_server.py
```

`web_server.py` 不再复制反向传播、优化器、调度器与验证逻辑；它只负责把 Web 的开始/停止操作转换成 `Trainer` 调用，并通过回调显示训练状态。

## 配置迁移

唯一支持的配置层级如下：

```yaml
training:
  loss: {bce_weight: 0.5, dice_weight: 0.5, dice_smooth: 0.000001}
  early_stopping: {patience: 10}
  checkpoint:
    save_dir: checkpoints
    save_best_only: true
    save_interval: 10
evaluation:
  threshold: 0.5
  apply_postprocess: false
inference:
  threshold: 0.5
  img_size: null
  postprocess: {enabled: true, min_component_size: 50, max_hole_size: 100, morph_close_kernel: 3}
```

旧版顶层 `loss`、顶层 `checkpoint`，以及 `training.save_dir`、`training.save_best_only`、`training.save_interval` 会在读取时自动迁移。保存配置时会写回新结构。

数据路径可以是绝对路径或相对于项目根目录的路径；检查点目录必须是项目内的相对路径，禁止 `..` 和绝对路径。

`dataset.keep_aspect_ratio` 默认为 `true`：图像和掩膜先按最长边等比例缩放，再补零到正方形。推理会移除补边后还原到原图大小，避免原实现把非正方形 DSA 拉伸变形。

## 检查点与兼容性

新产生的 `.pth` 包含格式版本、epoch、最佳 Dice、指标、完整配置、模型状态、优化器状态和调度器状态。推理与评估优先使用检查点内保存的模型架构和推理设置，避免后来修改 YAML 后错误加载旧模型。

历史的纯 `state_dict` 与旧版包含 `model_state_dict` 的检查点仍可读取；纯 `state_dict` 请在命令行提供 `--config` 和必要时的 `--model-name`。加载使用 `torch.load(weights_only=True)`，拒绝不安全或非权重格式的序列化对象。

## 评估与推理一致性

`src/prediction.py` 是唯一的 Logits→Sigmoid→阈值→后处理入口。默认训练/评估不做后处理，保持对原始模型能力的度量；运行：

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth --postprocess
```

可得到与部署后处理策略一致的评估。`inference.img_size: null` 表示继承训练的 `dataset.img_size`，防止原来的 Web 推理固定 512 而训练尺寸可变的问题。

## Web 服务边界

- 服务只监听 `127.0.0.1:5001`，不再监听所有网卡。
- 已移除不必要的跨域访问和任意项目文件预览接口。
- 配置提交会校验数值和检查点目录；删除和加载只接受检查点目录内的 `.pth` 文件名。
- 推理请求限制为 10 MB，检查图片像素数，并对 Base64 输入和阈值校验。

Web 服务面向本机可信用户。不要通过端口转发、反向代理或公网暴露它。

## 使用方式

```bash
pip install -r requirements.txt
python train.py --config configs/default.yaml
python evaluate.py --checkpoint checkpoints/best_model.pth --visualize
python inference.py --model checkpoints/best_model.pth --input path/to/image.png
python web_server.py
```

`train.py`、`evaluate.py`、`inference.py` 均支持 `--device` 参数（如 `cuda:1`、`cpu`），默认自动选择。

运行测试：

```bash
pip install -r requirements-dev.txt
pytest -q
```

## 已知行为变更

- 默认数据目录不再写入某台机器的绝对路径；请先配置 `dataset.*_dir`。
- `save_best_only: true` 时仅保存 `best_model.pth`；设为 `false` 才会按照 `save_interval` 保存 epoch 模型并写出 `last_model.pth`。
- 推理输出统一保存为 `.png`，不再出现文件内容为 PNG、扩展名却是 JPG 的情况。

## 2026-08-23 修正批次

本次修正解决了架构评审中发现的细节问题：

- **损失函数**：`DiceLoss` 从"全 batch 像素求和"改为逐样本计算后取平均，避免大前景样本主导梯度（`BCEDiceLoss` 组合方式不变）。
- **训练期指标**：早停依据的 Dice/IoU 从"逐 batch 平均"改为全数据集像素级聚合（新增 `src/metrics.py` 的 `MetricAccumulator`），与 batch 划分无关、更稳定。
- **Precision/Recall 语义**：eps 只保留在分母，空预测/空目标时返回 0 而不是 0.5；P、T 均为空时 Dice/IoU 约定为 1.0。
- **中文路径**：`inference.py` 的读写改用 `np.fromfile` + `cv2.imdecode` 与 `cv2.imencode` + 二进制写入，与 `dataset.py` 行为一致，修复 Windows 中文路径失败问题。
- **设备选择**：`train.py` / `evaluate.py` / `inference.py` 新增 `--device`，`Trainer` 支持显式设备参数，非法设备与 CUDA 不可用会在启动时报错。
- **Web 并发**：配置读写共用 `CONFIG_LOCK`；推理请求用 `INFERENCE_LOCK` 串行化，消除 `inference_cache` 与 `segmentor.threshold` 的竞态；训练进行中修改配置会返回提示（下次启动训练生效）。
- **依赖清理**：移除未使用的 `scikit-image` 与 `scipy`。
- **可测试性**：几何还原逻辑抽为 `inference.restore_original_geometry` 纯函数；新增 `tests/test_postprocess.py`、`tests/test_metrics.py`、`tests/test_losses.py` 并扩充几何还原测试。

## 2026-08-23 模型优化批次

本轮为模型优化 + 全量实验所做的功能扩展：

- **Focal Tversky Loss**（`src/losses.py`）：`training.loss.name: FocalTverskyLoss`，`focal_tversky.{alpha,beta,gamma}` 可调，α>β 加重漏检惩罚；
- **clDice 中心线监督**（`src/losses.py` + `src/skeleton.py`）：`training.loss.cl_dice_weight > 0` 时启用。数据集额外返回金标准骨架（Zhang-Suen 细化，纯 numpy 实现），预测侧用可微软骨架（max-pool 形态学近似）计算 clDice；
- **训练策略升级**（`src/training.py` + `src/trainer.py`）：
  - `training.seed` 固定随机种子（Python/NumPy/PyTorch/cuDNN）；
  - `training.warmup_epochs` 线性 warmup + 余弦退火（SequentialLR）；
  - `training.grad_clip` 梯度裁剪（AMP 下先 unscale 再 clip）；
  - `training.ema_decay` 指数滑动平均权重，验证与保存均使用 EMA 权重，保证"验证的模型 = 部署的模型"；
- **实验配置与一键脚本**：`configs/experiments/exp1~exp4.yaml` + `experiments/run_experiments.ps1`，依次训练 4 组对照并分别做原始/后处理评估；
- 新增 `tests/test_skeleton.py` 与损失函数新用例（Focal Tversky / clDice / 组合损失）。
- **环境适配**：`albumentations` 锁定 `<1.4.8`（1.4.8+ 依赖需要 C 编译器的 stringzilla）；`PadIfNeeded` 显式 `value=0`（1.4.7 pydantic 校验要求）；AMP 迁移到 `torch.amp` 新 API（torch≥2.3）。完整环境搭建与踩坑记录见 [ENVIRONMENT.md](ENVIRONMENT.md)。

**兼容性**：所有新配置键均有默认值；旧检查点加载不受影响（`cl_dice_weight` 默认 0、损失签名向后兼容）。`train.py`、`evaluate.py`、`inference.py` 均支持 `--device`。

## 2026-08-25 相位优化批次

针对完整数据集按时相(2~3s / 4s / 5~6s / dataset2)效果不一致的问题：

- **方案 B(相位标定)**：`experiments/phase_calibration.py` 按分组网格搜索阈值与后处理参数，输出 `phase_calibration.json`。结论:显影越淡的时相需要越低阈值(5~6s 最优 0.35、4s 最优 0.65),后处理在所有分组均为负收益;
- **方案 A(FiLM 相位条件化)**：
  - `src/models/unet.py`:新增 `FiLMBlock`(Embedding→MLP 产生逐通道 (1+dγ)·x+dβ,零初始化保证恒等起点),`UNetBaseline(phase_classes>0)` 时在编码器四级施加条件并新增相位分类头,前向返回 `(logits, phase_logits)`;
  - `src/dataset.py`:新增 `PHASE_PREFIXES` 与 `phase_id_from_filename`,config `training.phase_condition` 开启时数据集返回相位 id(四元组);
  - `src/trainer.py`:相位传递 + `phase_loss_weight × CrossEntropy` 辅助损失(顺带训练"相位预判定"能力);
  - 全链路适配:train / evaluate / inference(`predict_phase` 方法) / web_server;
  - 新配置键:`model.phase_classes`、`training.phase_condition`、`training.phase_loss_weight`,均有校验;
  - 实验:`configs/experiments/exp6_phase_film.yaml`(clDice 配方 + 相位条件),新增 `tests/test_phase.py`(5 例,总计 34 passed)。
