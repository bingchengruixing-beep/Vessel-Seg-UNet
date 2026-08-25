"""Local-only Flask UI for the shared Vessel-Seg-UNet training pipeline."""

import base64
import binascii
import copy
import logging
import os
import threading
from collections import OrderedDict
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import cv2
import torch
import yaml
from flask import Flask, abort, jsonify, request, send_from_directory
from PIL import Image

from inference import VesselSegmentor
from src.config import ConfigError, load_config, resolve_checkpoint_dir, resolve_data_path, save_config
from src.dataset import get_dataloaders
from src.dataset import VesselDataset
from src.frangi import vesselness
from src.metrics import calculate_dice, calculate_iou, calculate_precision, calculate_recall
from src.models import build_model_from_config
from src.prediction import main_logits_from_output, postprocess_predictions
from src.trainer import Trainer
from src.training import build_criterion, build_optimizer, build_scheduler
from src.transforms import get_val_transforms


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
STATIC_FOLDER = PROJECT_ROOT / "web_static"
LOG_DIR = PROJECT_ROOT / "logs"

app = Flask(__name__, static_folder=str(STATIC_FOLDER), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 16_000_000

STATE_LOCK = threading.Lock()
CONFIG_LOCK = threading.Lock()
INFERENCE_LOCK = threading.Lock()
DIRECTORY_DIALOG_LOCK = threading.Lock()
training_state = {
    "running": False,
    "stop_requested": False,
    "epoch": 0,
    "total_epochs": 0,
    "train_loss": 0.0,
    "val_loss": 0.0,
    "dice": 0.0,
    "iou": 0.0,
    "lr": 0.0,
    "best_dice": 0.0,
    "history": [],
    "log_path": "",
    "message": "就绪",
    "frangi_running": False,
    "frangi_stop_requested": False,
    "frangi_progress": "",
    "frangi_train_done": 0,
    "frangi_train_total": 0,
    "frangi_val_done": 0,
    "frangi_val_total": 0,
    "frangi_result": None,
}
inference_cache: OrderedDict[tuple[str, int], VesselSegmentor] = OrderedDict()
MAX_CACHED_MODELS = 5


def _snapshot_state():
    with STATE_LOCK:
        return copy.deepcopy(training_state)


def _load_config():
    """线程安全的配置读取：与 _save_config 共用 CONFIG_LOCK 防止读写交错。"""
    with CONFIG_LOCK:
        return load_config(CONFIG_PATH)


def _save_config(payload):
    with CONFIG_LOCK:
        return save_config(CONFIG_PATH, payload)


def _safe_checkpoint_name(filename: str) -> str:
    normalized = str(filename).replace("\\", "/")
    candidate = Path(normalized)
    if (
        not normalized
        or candidate.is_absolute()
        or candidate.suffix.lower() != ".pth"
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ValueError("Invalid checkpoint filename")
    return normalized


def _checkpoint_path(filename: str) -> Path:
    safe_name = _safe_checkpoint_name(filename)
    directories = _checkpoint_directories(_load_config())
    for directory in directories:
        candidate = (directory / safe_name).resolve()
        try:
            candidate.relative_to(directory.resolve())
        except ValueError:
            continue
        if candidate.is_file():
            return candidate
    return directories[0] / safe_name


def _checkpoint_directories(config: dict) -> list[Path]:
    """返回配置目录及本项目已有权重目录，供网页列表和推理共用。"""
    directories = [resolve_checkpoint_dir(config, PROJECT_ROOT)]
    for name in ("checkpoints_b", "checkpoints_web"):
        directory = PROJECT_ROOT / name
        if directory not in directories:
            directories.append(directory)
    return directories


def _stop_requested() -> bool:
    with STATE_LOCK:
        return training_state["stop_requested"]


def _on_epoch_end(metrics):
    with STATE_LOCK:
        training_state.update(metrics)
        training_state["epoch"] = int(metrics["epoch"])
        training_state["history"].append(copy.deepcopy(metrics))
        training_state["message"] = "训练中"
        log_path_value = training_state.get("log_path", "")
    log_path = PROJECT_ROOT / log_path_value if log_path_value else None
    _append_training_log(
        log_path,
        "Epoch [{epoch:.0f}/{total}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Dice: {dice:.4f}, IoU: {iou:.4f}".format(
            total=training_state.get("total_epochs", 0), **metrics
        ),
    )


def _on_batch_end(metrics):
    """按固定间隔记录训练批次进度，避免日志写入影响训练速度。"""
    _append_training_log(
        _training_log_path(),
        "Epoch [{epoch:.0f}] Batch [{batch:.0f}/{total_batches:.0f}] - Loss: {batch_loss:.4f}, Elapsed: {elapsed_seconds:.1f}s, LR: {lr:.2e}".format(**metrics),
    )


def _training_log_path() -> Path | None:
    with STATE_LOCK:
        log_path_value = training_state.get("log_path", "")
    return PROJECT_ROOT / log_path_value if log_path_value else None


def _append_training_log(log_path: Path | None, message: str) -> None:
    """把训练配置和每轮结果写入独立 UTF-8 日志文件。"""
    if log_path is None:
        return
    try:
        with log_path.open("a", encoding="utf-8") as file:
            file.write(f"{message}\n")
    except OSError:
        logger.exception("Unable to write training log")


def run_training():
    log_path = None
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        with STATE_LOCK:
            training_state["log_path"] = str(log_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        _append_training_log(log_path, "训练线程已启动")
        config = _load_config()
        _append_training_log(log_path, "训练配置:")
        _append_training_log(log_path, yaml.safe_dump(config, allow_unicode=True, sort_keys=False).rstrip())
        with STATE_LOCK:
            training_state["total_epochs"] = config["training"]["epochs"]
            training_state["message"] = "准备数据和模型..."

        train_loader, val_loader = get_dataloaders(config, project_root=PROJECT_ROOT)
        model_cfg = config["model"]
        model = build_model_from_config(model_cfg)
        optimizer = build_optimizer(model, config)
        checkpoint_dir = resolve_checkpoint_dir(config, PROJECT_ROOT)
        cross_validation_cfg = config["dataset"].get("cross_validation", {})
        if cross_validation_cfg.get("enabled", False):
            checkpoint_dir = checkpoint_dir / (
                f"fold_{int(cross_validation_cfg['fold_index']) + 1}"
                f"_of_{int(cross_validation_cfg['num_folds'])}"
            )
            _append_training_log(log_path, f"当前折权重目录: {checkpoint_dir}")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=build_criterion(config),
            optimizer=optimizer,
            scheduler=build_scheduler(optimizer, config),
            config=config,
            checkpoint_dir=checkpoint_dir,
            on_epoch_end=_on_epoch_end,
            on_batch_end=_on_batch_end,
            should_stop=_stop_requested,
        )
        with STATE_LOCK:
            training_state["message"] = "数据和模型已就绪，正在训练第一轮..."
        result = trainer.run()
        with STATE_LOCK:
            training_state["best_dice"] = result["best_dice"]
            training_state["message"] = "训练已由用户终止" if result["stopped"] else "训练完成"
    except Exception as exc:
        logger.exception("Training failed")
        _append_training_log(log_path, f"训练异常: {exc}")
        with STATE_LOCK:
            training_state["message"] = f"错误: {exc}"
    finally:
        _append_training_log(log_path, "训练线程已结束")
        with STATE_LOCK:
            training_state["running"] = False
            training_state["stop_requested"] = False


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/config", methods=["GET"])
def get_config():
    return jsonify(_load_config())


@app.route("/api/config", methods=["POST"])
def update_config():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"success": False, "message": "配置必须是 JSON 对象"}), 400
    try:
        saved = _save_config(payload)
        with STATE_LOCK:
            running = training_state["running"]
        response = {"success": True, "config": saved}
        if running:
            response["note"] = "训练进行中：配置已保存，将在下一次启动训练时生效。"
        return jsonify(response)
    except ConfigError as exc:
        return jsonify({"success": False, "message": str(exc)}), 400


@app.route("/api/select-directory", methods=["POST"])
def select_directory():
    """在运行 Web 服务的本机打开系统目录选择窗口。"""
    try:
        import tkinter as tk
        from tkinter import filedialog

        with DIRECTORY_DIALOG_LOCK:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            selected = filedialog.askdirectory(title="选择数据文件夹", mustexist=True)
            root.destroy()
        if not selected:
            return jsonify({"success": False, "cancelled": True})
        return jsonify({"success": True, "path": str(Path(selected).resolve())})
    except (ImportError, RuntimeError, OSError) as exc:
        logger.exception("打开系统目录选择窗口失败")
        return jsonify({"success": False, "message": f"无法打开系统目录选择窗口：{exc}"}), 500
    except Exception as exc:
        logger.exception("目录选择失败")
        return jsonify({"success": False, "message": str(exc)}), 500


@app.route("/api/dataset/info", methods=["GET"])
def dataset_info():
    config = _load_config()
    dataset_cfg = config["dataset"]

    def image_count(path_value):
        path = resolve_data_path(path_value, PROJECT_ROOT)
        if not path.is_dir():
            return 0
        return sum(item.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"} for item in path.iterdir())

    train_count = image_count(dataset_cfg["train_image_dir"])
    val_count = image_count(dataset_cfg["val_image_dir"])
    split_strategy = "使用独立训练集和验证集"
    cross_validation_cfg = dataset_cfg.get("cross_validation", {})
    if cross_validation_cfg.get("enabled", False):
        train_loader, val_loader = get_dataloaders(config, project_root=PROJECT_ROOT)
        train_count = len(train_loader.dataset)
        val_count = len(val_loader.dataset)
        split_strategy = (
            f"按序列分组 {cross_validation_cfg['num_folds']} 折，"
            f"当前第 {int(cross_validation_cfg['fold_index']) + 1} 折"
        )
    domain_balance_cfg = dataset_cfg.get("domain_balance", {})
    if domain_balance_cfg.get("enabled", False):
        split_strategy += (
            f"；每轮目标域期望占比 "
            f"{float(domain_balance_cfg['target_probability']) * 100:.0f}%"
        )
    if dataset_cfg.get("temporal_2_5d", {}).get("enabled", False):
        split_strategy += "；2.5D 前/中/后三时相输入"
    return jsonify({
        "train": {"count": train_count, "path": dataset_cfg["train_image_dir"]},
        "val": {"count": val_count, "path": (
            dataset_cfg["train_image_dir"]
            if cross_validation_cfg.get("enabled", False)
            else dataset_cfg["val_image_dir"]
        )},
        "split_strategy": split_strategy,
    })


@app.route("/api/train/start", methods=["POST"])
def start_training():
    with STATE_LOCK:
        if training_state["running"]:
            return jsonify({"success": False, "message": "训练已经在运行中"}), 409
        training_state.update({
            "running": True,
            "stop_requested": False,
            "epoch": 0,
            "total_epochs": 0,
            "train_loss": 0.0,
            "val_loss": 0.0,
            "dice": 0.0,
            "iou": 0.0,
            "lr": 0.0,
            "best_dice": 0.0,
            "history": [],
            "log_path": "",
            "message": "训练正在启动...",
        })
    threading.Thread(target=run_training, daemon=True).start()
    return jsonify({"success": True, "message": "训练已开始"})


@app.route("/api/train/stop", methods=["POST"])
def stop_training():
    with STATE_LOCK:
        if not training_state["running"]:
            return jsonify({"success": False, "message": "目前没有正在运行的训练"}), 409
        training_state["stop_requested"] = True
    return jsonify({"success": True, "message": "已发送停止信号"})


@app.route("/api/train/status", methods=["GET"])
def train_status():
    return jsonify(_snapshot_state())


@app.route("/api/model/info", methods=["GET"])
def model_info():
    config = _load_config()
    model_cfg = config["model"]
    model = build_model_from_config(model_cfg)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return jsonify({"success": True, "data": {
        "name": model_cfg["name"],
        "architecture": model.__class__.__name__,
        "total_params": parameter_count,
        "trainable_params": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }})


@app.route("/api/checkpoints", methods=["GET"])
def list_checkpoints():
    checkpoints = []
    seen_names = set()
    for directory in _checkpoint_directories(_load_config()):
        if not directory.exists():
            continue
        for file in directory.rglob("*.pth"):
            relative_name = file.relative_to(directory).as_posix()
            if file.is_file() and file.suffix.lower() == ".pth" and relative_name not in seen_names:
                checkpoints.append({
                    "name": relative_name,
                    "size": file.stat().st_size,
                    "modified": datetime.fromtimestamp(file.stat().st_mtime).isoformat(),
                })
                seen_names.add(relative_name)
    return jsonify(sorted(checkpoints, key=lambda item: item["modified"], reverse=True))


@app.route("/api/checkpoints/<path:filename>", methods=["DELETE"])
def delete_checkpoint(filename):
    try:
        path = _checkpoint_path(filename)
    except (ConfigError, ValueError):
        abort(400)
    if not path.is_file():
        abort(404)
    path.unlink()
    return jsonify({"success": True, "message": f"已删除 {path.name}"})


@app.route("/api/inference", methods=["POST"])
def run_inference():
    # 串行化推理请求：缓存更新与 segmentor.threshold 的修改需要互斥。
    with INFERENCE_LOCK:
        return _run_inference()


def _run_inference():
    payload = request.get_json(silent=True) or {}
    try:
        checkpoint_names = _requested_checkpoint_names(payload)
        threshold = float(payload.get("threshold", 0.5))
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        image = _decode_inference_input(payload)
        probability, segmentor = _predict_ensemble_probability(image, checkpoint_names)
        _apply_processing_options(segmentor, payload)
        mask = _mask_from_probability(probability, threshold, segmentor.postprocess_config)
        buffer = BytesIO()
        Image.fromarray(mask).save(buffer, format="PNG")
        return jsonify({
            "mask_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "checkpoints": checkpoint_names,
        })
    except KeyError:
        return jsonify({"error": "未提供 image_base64 或 temporal_images"}), 400
    except (ValueError, binascii.Error, OSError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Inference failed")
        return jsonify({"error": str(exc)}), 500


def _requested_checkpoint_names(payload: dict) -> list[str]:
    """解析单权重或最多五个集成权重。"""
    raw_names = payload.get("checkpoints")
    if raw_names is None:
        raw_names = [payload.get("checkpoint", "best_model.pth")]
    if not isinstance(raw_names, list) or not raw_names or len(raw_names) > 5:
        raise ValueError("checkpoints 必须包含 1 到 5 个权重")
    names = []
    for value in raw_names:
        name = _safe_checkpoint_name(str(value))
        if name not in names:
            names.append(name)
    return names


def _decode_base64_gray(encoded: str) -> np.ndarray:
    """解码网页上传的单张灰度图。"""
    image_bytes = base64.b64decode(encoded, validate=True)
    image = Image.open(BytesIO(image_bytes)).convert("L")
    if image.width * image.height > Image.MAX_IMAGE_PIXELS:
        raise ValueError("image is too large")
    return np.asarray(image)


def _decode_inference_input(payload: dict) -> np.ndarray:
    """读取单图，或读取前/当前/后三张时相图。"""
    temporal_images = payload.get("temporal_images")
    if temporal_images is None:
        return _decode_base64_gray(payload["image_base64"])
    if not isinstance(temporal_images, list) or len(temporal_images) != 3:
        raise ValueError("temporal_images 必须恰好包含前、当前、后三张图像")
    frames = [_decode_base64_gray(str(value)) for value in temporal_images]
    if any(frame.shape != frames[1].shape for frame in frames):
        raise ValueError("前、当前、后三张时相图尺寸必须一致")
    return np.stack(frames, axis=-1)


def _get_cached_segmentor(checkpoint_name: str) -> VesselSegmentor:
    """按检查点修改时间维护最多五个模型的 LRU 缓存。"""
    model_path = _checkpoint_path(checkpoint_name)
    if not model_path.is_file():
        raise FileNotFoundError(f"检查点未找到: {checkpoint_name}")
    mtime_ns = model_path.stat().st_mtime_ns
    cache_key = (str(model_path), mtime_ns)
    segmentor = inference_cache.get(cache_key)
    if segmentor is None:
        stale_keys = [key for key in inference_cache if key[0] == str(model_path)]
        for key in stale_keys:
            del inference_cache[key]
        segmentor = VesselSegmentor(str(model_path), config=_load_config())
        inference_cache[cache_key] = segmentor
        while len(inference_cache) > MAX_CACHED_MODELS:
            inference_cache.popitem(last=False)
    else:
        inference_cache.move_to_end(cache_key)
    return segmentor


def _predict_ensemble_probability(
    image: np.ndarray,
    checkpoint_names: list[str],
) -> tuple[np.ndarray, VesselSegmentor]:
    """逐模型恢复到原图后平均概率，兼容不同输入尺寸与通道数。"""
    probability_sum = None
    reference_segmentor = None
    for checkpoint_name in checkpoint_names:
        segmentor = _get_cached_segmentor(checkpoint_name)
        probability = segmentor.predict_probability_array(image)
        if probability_sum is None:
            probability_sum = probability.astype(np.float32, copy=True)
            reference_segmentor = segmentor
        else:
            if probability.shape != probability_sum.shape:
                raise ValueError("集成权重恢复后的概率图尺寸不一致")
            probability_sum += probability
    return probability_sum / len(checkpoint_names), reference_segmentor


def _mask_from_probability(
    probability: np.ndarray,
    threshold: float,
    postprocess_config: dict,
) -> np.ndarray:
    """对集成概率统一阈值化，并只执行一次后处理。"""
    prediction = torch.from_numpy((probability > threshold).astype(np.float32))[None, None]
    if postprocess_config.get("enabled", False):
        prediction = postprocess_predictions(prediction, postprocess_config)
    return prediction[0, 0].numpy().astype(np.uint8) * 255


def _apply_processing_options(segmentor: VesselSegmentor, payload: dict) -> None:
    config = _load_config()["inference"]["postprocess"]
    segmentor.postprocess_config = dict(config)
    mode = payload.get("processing", "config")
    if mode == "off":
        segmentor.postprocess_config["enabled"] = False
    elif mode == "light":
        segmentor.postprocess_config.update({
            "enabled": True,
            "min_component_size": 10,
            "max_hole_size": 30,
            "morph_close_kernel": 3,
        })
    elif mode == "strong":
        segmentor.postprocess_config.update({
            "enabled": True,
            "min_component_size": 100,
            "max_hole_size": 200,
            "morph_close_kernel": 5,
        })
    elif mode == "custom":
        min_component_size = int(payload.get("min_component_size", config["min_component_size"]))
        max_hole_size = int(payload.get("max_hole_size", config["max_hole_size"]))
        morph_close_kernel = int(payload.get("morph_close_kernel", config["morph_close_kernel"]))
        if min_component_size < 0 or max_hole_size < 0 or morph_close_kernel < 1 or morph_close_kernel % 2 == 0:
            raise ValueError("后处理参数无效：连通域和孔洞不能为负，闭运算核必须为正奇数")
        segmentor.postprocess_config.update({
            "enabled": True,
            "min_component_size": min_component_size,
            "max_hole_size": max_hole_size,
            "morph_close_kernel": morph_close_kernel,
        })
    elif mode != "config":
        raise ValueError("未知的后处理方式")


@app.route("/api/inference-batch", methods=["POST"])
def run_batch_inference():
    """对网页上传的多张图像使用相同权重集成批量推理。"""
    with INFERENCE_LOCK:
        payload = request.get_json(silent=True) or {}
        try:
            checkpoint_names = _requested_checkpoint_names(payload)
            threshold = float(payload.get("threshold", 0.5))
            images = payload.get("images")
            if not isinstance(images, list) or not images or len(images) > 32:
                raise ValueError("images 必须包含 1 到 32 张图像")
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("threshold must be between 0 and 1")
            results = []
            for item in images:
                if not isinstance(item, dict) or "image_base64" not in item:
                    raise ValueError("每个图像项都必须包含 image_base64")
                image = _decode_base64_gray(item["image_base64"])
                probability, segmentor = _predict_ensemble_probability(image, checkpoint_names)
                _apply_processing_options(segmentor, payload)
                mask = _mask_from_probability(
                    probability, threshold, segmentor.postprocess_config
                )
                buffer = BytesIO()
                Image.fromarray(mask).save(buffer, format="PNG")
                results.append({
                    "name": str(item.get("name", f"image_{len(results) + 1}")),
                    "mask_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
                })
            return jsonify({
                "results": results,
                "count": len(results),
                "checkpoints": checkpoint_names,
            })
        except (ValueError, binascii.Error, OSError) as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            logger.exception("Batch inference failed")
            return jsonify({"error": str(exc)}), 500


@app.route("/api/threshold-scan", methods=["POST"])
def threshold_scan():
    """在当前验证集上批量比较多个二值化阈值。"""
    with STATE_LOCK:
        if training_state["running"]:
            return jsonify({"success": False, "error": "训练正在运行，请先停止训练再扫描阈值"}), 409
    try:
        payload = request.get_json(silent=True) or {}
        checkpoint_names = _requested_checkpoint_names(payload)
        adaptive_scan = bool(payload.get("adaptive_scan", False))

        def threshold_grid(start: float, end: float, step: float) -> list[float]:
            """生成包含端点的稳定浮点阈值网格。"""
            values = []
            index = 0
            while start + index * step <= end + 1e-9:
                values.append(round(start + index * step, 2))
                index += 1
            if not values or values[-1] < end - 1e-9:
                values.append(round(end, 2))
            return sorted(set(values))

        if adaptive_scan:
            scan_start = float(payload.get("scan_start", 0.30))
            scan_end = float(payload.get("scan_end", 0.80))
            coarse_step = float(payload.get("coarse_step", 0.05))
            fine_step = float(payload.get("fine_step", 0.01))
            if not 0.0 <= scan_start < scan_end <= 1.0:
                raise ValueError("粗扫范围必须满足 0 <= 起点 < 终点 <= 1")
            if not 0.0 < fine_step <= coarse_step <= 1.0:
                raise ValueError("精扫步长必须大于 0 且不大于粗扫步长")
            thresholds = threshold_grid(scan_start, scan_end, coarse_step)
        else:
            raw_thresholds = payload.get("thresholds", [round(0.30 + index * 0.01, 2) for index in range(51)])
            if not isinstance(raw_thresholds, list):
                raise ValueError("thresholds 必须是数字列表")
            thresholds = sorted({float(value) for value in raw_thresholds})
            if not thresholds or len(thresholds) > 101 or any(not 0.0 <= value <= 1.0 for value in thresholds):
                raise ValueError("阈值数量必须为 1 到 101 个，且每个值在 0 到 1 之间")

        config = _load_config()
        evaluation_split = str(payload.get("evaluation_split", "dias_external"))
        if evaluation_split not in {"dias_external", "fold_validation"}:
            raise ValueError("evaluation_split 必须是 dias_external 或 fold_validation")
        evaluation_config = copy.deepcopy(config)
        if evaluation_split == "dias_external":
            # 阈值可比性优先使用独立 DIAS 验证集，不受当前 K 折开关影响。
            evaluation_config["dataset"]["cross_validation"]["enabled"] = False
            evaluation_label = "DIAS 外部验证集"
        else:
            evaluation_label = "当前 K 折验证集"
        _, val_loader = get_dataloaders(evaluation_config, project_root=PROJECT_ROOT)
        dataset = val_loader.dataset
        totals = {
            threshold: {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
            for threshold in thresholds
        }
        sample_count = 0

        def accumulate_scores(probability, target, target_totals, target_thresholds):
            nonlocal sample_count
            for threshold in target_thresholds:
                prediction = (probability > threshold).float()
                if segmentor.postprocess_config["enabled"]:
                    prediction = postprocess_predictions(prediction, segmentor.postprocess_config)
                target_totals[threshold]["dice"] += calculate_dice(prediction, target)
                target_totals[threshold]["iou"] += calculate_iou(prediction, target)
                target_totals[threshold]["precision"] += calculate_precision(prediction, target)
                target_totals[threshold]["recall"] += calculate_recall(prediction, target)

        cached_samples = []
        with torch.inference_mode():
            for filename in dataset.filenames:
                image = dataset.load_input_image(filename)
                mask_buffer = np.fromfile(Path(dataset.mask_dir) / filename, dtype=np.uint8)
                mask = cv2.imdecode(mask_buffer, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise OSError(f"无法读取验证掩膜: {filename}")
                probability_array, segmentor = _predict_ensemble_probability(
                    image, checkpoint_names
                )
                _apply_processing_options(segmentor, payload)
                probability = torch.from_numpy(probability_array).float()[None, None]
                target = torch.from_numpy((mask > 127).astype(np.float32))[None, None]
                accumulate_scores(probability, target, totals, thresholds)
                if adaptive_scan:
                    # 只缓存自适应扫描所需的概率和掩膜，避免第二次占用 GPU 推理。
                    cached_samples.append((probability_array.astype(np.float16), (mask > 127).astype(np.uint8)))
                sample_count += 1
        divisor = max(sample_count, 1)
        coarse_results = [
            {"threshold": threshold, **{name: values[name] / divisor for name in values}}
            for threshold, values in totals.items()
        ]
        fine_results = []
        fine_start = fine_end = None
        if adaptive_scan:
            coarse_best = max(coarse_results, key=lambda item: item["dice"])
            fine_start = max(scan_start, float(coarse_best["threshold"]) - coarse_step)
            fine_end = min(scan_end, float(coarse_best["threshold"]) + coarse_step)
            fine_thresholds = threshold_grid(fine_start, fine_end, fine_step)
            fine_totals = {
                threshold: {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
                for threshold in fine_thresholds
            }
            for probability_array, mask_array in cached_samples:
                probability = torch.from_numpy(probability_array.astype(np.float32))[None, None]
                target = torch.from_numpy(mask_array.astype(np.float32))[None, None]
                accumulate_scores(probability, target, fine_totals, fine_thresholds)
            fine_results = [
                {"threshold": threshold, **{name: value / divisor for name, value in metrics.items()}}
                for threshold, metrics in fine_totals.items()
            ]

        result_by_threshold = {item["threshold"]: item for item in coarse_results}
        result_by_threshold.update({item["threshold"]: item for item in fine_results})
        results = sorted(result_by_threshold.values(), key=lambda item: item["threshold"])
        best = max(results, key=lambda item: item["dice"])
        return jsonify({
            "success": True,
            "checkpoints": checkpoint_names,
            "device": str(segmentor.device),
            "samples": sample_count,
            "results": results,
            "best_threshold": best["threshold"],
            "best_dice": best["dice"],
            "scan_mode": "coarse_fine" if adaptive_scan else "custom",
            "coarse_best_threshold": max(coarse_results, key=lambda item: item["dice"])["threshold"],
            "fine_range": ([round(fine_start, 2), round(fine_end, 2)] if adaptive_scan else None),
            "evaluation_split": evaluation_split,
            "evaluation_label": evaluation_label,
            "evaluation_path": str(Path(dataset.image_dir).resolve()),
        })
    except (ConfigError, ValueError, OSError) as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Threshold scan failed")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/frangi/status", methods=["GET"])
def frangi_status():
    with STATE_LOCK:
        return jsonify({
            "running": training_state.get("frangi_running", False),
            "progress": training_state.get("frangi_progress", ""),
            "train_done": training_state.get("frangi_train_done", 0),
            "train_total": training_state.get("frangi_train_total", 0),
            "val_done": training_state.get("frangi_val_done", 0),
            "val_total": training_state.get("frangi_val_total", 0),
            "result": training_state.get("frangi_result"),
        })


@app.route("/api/system", methods=["GET"])
def system_info():
    import platform
    cuda_available = torch.cuda.is_available()
    return jsonify({
        "gpu_available": cuda_available,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else "无 GPU",
        "gpu_memory": f"{torch.cuda.get_device_properties(0).total_memory // (1024 ** 2):,} MB" if cuda_available else "",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "os": f"{platform.system()} {platform.release()}",
    })


@app.errorhandler(413)
def payload_too_large(_error):
    return jsonify({"error": "上传图片超过 10 MB 限制"}), 413


@app.route("/api/frangi/generate", methods=["POST"])
def generate_frangi_maps():
    """在服务端为训练集和验证集批量生成 Frangi vesselness 增强图（后台线程）。"""
    with STATE_LOCK:
        if training_state.get("frangi_running"):
            return jsonify({"success": False, "message": "Frangi 生成正在进行中"}), 409
        training_state["frangi_running"] = True
        training_state["frangi_stop_requested"] = False
        training_state["frangi_progress"] = "准备中..."
        training_state["frangi_train_done"] = 0
        training_state["frangi_train_total"] = 0
        training_state["frangi_val_done"] = 0
        training_state["frangi_val_total"] = 0

    payload = request.get_json(silent=True) or {}
    config = _load_config()
    if payload:
        config = payload

    def _run():
        try:
            dataset_cfg = config["dataset"]
            frangi_cfg = dataset_cfg.get("frangi", {})
            method = str(frangi_cfg.get("method", "hessian"))
            sigmas = tuple(float(s) for s in frangi_cfg.get("sigmas", [1.0, 2.0, 3.0, 4.0, 5.0]))
            beta = float(frangi_cfg.get("beta", 0.5))
            c_raw = frangi_cfg.get("c", 0)
            c = float(c_raw) if c_raw and float(c_raw) > 0 else None

            train_count = 0
            val_count = 0

            for key in ("train", "val"):
                image_dir = resolve_data_path(
                    dataset_cfg[f"{key}_image_dir"], PROJECT_ROOT
                )
                frangi_dir_raw = frangi_cfg.get(f"{key}_frangi_dir", "")
                if frangi_dir_raw:
                    frangi_dir = resolve_data_path(frangi_dir_raw, PROJECT_ROOT)
                else:
                    frangi_dir = image_dir.parent / "frangi"
                frangi_dir.mkdir(parents=True, exist_ok=True)

                if not image_dir.is_dir():
                    continue

                items = sorted([
                    item for item in image_dir.iterdir()
                    if item.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
                ])
                total = len(items)

                with STATE_LOCK:
                    training_state["frangi_progress"] = f"正在处理{key}集..."
                    if key == "train":
                        training_state["frangi_train_total"] = total
                    else:
                        training_state["frangi_val_total"] = total

                for i, item in enumerate(items):
                    # 检查停止请求
                    with STATE_LOCK:
                        if training_state.get("frangi_stop_requested"):
                            training_state["frangi_progress"] = f"已停止（{key}集处理了 {i}/{total} 张）"
                            return
                    buf = np.fromfile(str(item), dtype=np.uint8)
                    image = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        continue
                    vesselness_map = vesselness(image, sigmas=sigmas, method=method, beta=beta, c=c)
                    vesselness_16u = (vesselness_map * 65535.0).astype(np.uint16)
                    out_path = frangi_dir / item.name
                    # 使用 imencode + tofile 支持中文路径
                    success, encoded = cv2.imencode(".png", vesselness_16u)
                    if success:
                        encoded.tofile(str(out_path))
                        if key == "train":
                            train_count += 1
                        else:
                            val_count += 1
                    with STATE_LOCK:
                        if key == "train":
                            training_state["frangi_train_done"] = i + 1
                        else:
                            training_state["frangi_val_done"] = i + 1

            with STATE_LOCK:
                training_state["frangi_progress"] = f"完成！训练集 {train_count} 张，验证集 {val_count} 张"
                training_state["frangi_result"] = {
                    "train_count": train_count,
                    "val_count": val_count,
                }
        except Exception as exc:
            logger.exception("Frangi generation failed")
            with STATE_LOCK:
                training_state["frangi_progress"] = f"失败: {exc}"
        finally:
            with STATE_LOCK:
                training_state["frangi_running"] = False

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"success": True, "message": "Frangi 生成已开始"})


@app.route("/api/frangi/stop", methods=["POST"])
def stop_frangi_generation():
    """请求停止正在进行的 Frangi 生成任务。"""
    with STATE_LOCK:
        if not training_state.get("frangi_running"):
            return jsonify({"success": False, "message": "没有正在进行的 Frangi 生成任务"}), 409
        training_state["frangi_stop_requested"] = True
        training_state["frangi_progress"] = "正在停止..."
    return jsonify({"success": True, "message": "停止请求已发送，正在安全停止..."})


@app.route("/api/frangi/clear", methods=["POST"])
def clear_frangi_maps():
    """清空 Frangi 增强图目录中的所有图像。"""
    config = _load_config()
    payload = request.get_json(silent=True) or {}
    if payload:
        config = payload

    with STATE_LOCK:
        if training_state.get("frangi_running"):
            return jsonify({"success": False, "message": "Frangi 生成正在进行中，请先停止"}), 409

    dataset_cfg = config["dataset"]
    frangi_cfg = dataset_cfg.get("frangi", {})
    deleted_train = 0
    deleted_val = 0

    for key in ("train", "val"):
        image_dir = resolve_data_path(dataset_cfg[f"{key}_image_dir"], PROJECT_ROOT)
        frangi_dir_raw = frangi_cfg.get(f"{key}_frangi_dir", "")
        if frangi_dir_raw:
            frangi_dir = resolve_data_path(frangi_dir_raw, PROJECT_ROOT)
        else:
            frangi_dir = image_dir.parent / "frangi"

        if not frangi_dir.is_dir():
            continue

        for item in frangi_dir.iterdir():
            if item.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                try:
                    item.unlink()
                    if key == "train":
                        deleted_train += 1
                    else:
                        deleted_val += 1
                except OSError:
                    pass

    return jsonify({
        "success": True,
        "message": f"已清空：训练集 {deleted_train} 张，验证集 {deleted_val} 张",
        "train_deleted": deleted_train,
        "val_deleted": deleted_val,
    })


if __name__ == "__main__":
    port = int(os.environ.get("VESSEL_WEB_PORT", "5001"))
    logger.info("Starting local Vessel-Seg-UNet server at http://127.0.0.1:%s", port)
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
