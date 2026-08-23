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

运行测试：

```bash
pip install -r requirements-dev.txt
pytest -q
```

## 已知行为变更

- 默认数据目录不再写入某台机器的绝对路径；请先配置 `dataset.*_dir`。
- `save_best_only: true` 时仅保存 `best_model.pth`；设为 `false` 才会按照 `save_interval` 保存 epoch 模型并写出 `last_model.pth`。
- 推理输出统一保存为 `.png`，不再出现文件内容为 PNG、扩展名却是 JPG 的情况。
