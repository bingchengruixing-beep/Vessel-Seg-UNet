"""Local-only Flask UI for the shared Vessel-Seg-UNet training pipeline."""

import base64
import binascii
import copy
import logging
import os
import threading
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import yaml
from flask import Flask, abort, jsonify, request, send_from_directory
from PIL import Image

from inference import VesselSegmentor
from src.config import ConfigError, load_config, resolve_checkpoint_dir, resolve_data_path, save_config
from src.dataset import get_dataloaders
from src.dataset import VesselDataset
from src.metrics import calculate_dice, calculate_iou, calculate_precision, calculate_recall
from src.models import build_model_from_config
from src.prediction import main_logits_from_output
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
}
inference_cache = {"path": None, "mtime_ns": None, "segmentor": None}


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
    candidate = Path(filename)
    if not filename or candidate.name != filename or candidate.suffix.lower() != ".pth":
        raise ValueError("Invalid checkpoint filename")
    return filename


def _checkpoint_path(filename: str) -> Path:
    return resolve_checkpoint_dir(_load_config(), PROJECT_ROOT) / _safe_checkpoint_name(filename)


def _stop_requested() -> bool:
    with STATE_LOCK:
        return training_state["stop_requested"]


def _on_epoch_end(metrics):
    with STATE_LOCK:
        training_state.update(metrics)
        training_state["epoch"] = int(metrics["epoch"])
        training_state["history"].append(copy.deepcopy(metrics))
        training_state["message"] = "训练中"


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
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=build_criterion(config),
            optimizer=optimizer,
            scheduler=build_scheduler(optimizer, config),
            config=config,
            checkpoint_dir=resolve_checkpoint_dir(config, PROJECT_ROOT),
            on_epoch_end=_on_epoch_end,
            should_stop=_stop_requested,
        )
        result = trainer.run()
        for metrics in training_state["history"]:
            _append_training_log(
                log_path,
                "Epoch [{epoch:.0f}/{total}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Dice: {dice:.4f}, IoU: {iou:.4f}".format(
                    total=config["training"]["epochs"], **metrics
                ),
            )
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


@app.route("/api/dataset/info", methods=["GET"])
def dataset_info():
    config = _load_config()
    dataset_cfg = config["dataset"]

    def image_count(path_value):
        path = resolve_data_path(path_value, PROJECT_ROOT)
        if not path.is_dir():
            return 0
        return sum(item.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"} for item in path.iterdir())

    return jsonify({
        "train": {"count": image_count(dataset_cfg["train_image_dir"]), "path": dataset_cfg["train_image_dir"]},
        "val": {"count": image_count(dataset_cfg["val_image_dir"]), "path": dataset_cfg["val_image_dir"]},
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
    directory = resolve_checkpoint_dir(_load_config(), PROJECT_ROOT)
    if not directory.exists():
        return jsonify([])
    checkpoints = [
        {"name": file.name, "size": file.stat().st_size, "modified": datetime.fromtimestamp(file.stat().st_mtime).isoformat()}
        for file in directory.iterdir() if file.is_file() and file.suffix.lower() == ".pth"
    ]
    return jsonify(sorted(checkpoints, key=lambda item: item["modified"], reverse=True))


@app.route("/api/checkpoints/<filename>", methods=["DELETE"])
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
        image_b64 = payload["image_base64"]
        checkpoint_name = _safe_checkpoint_name(payload.get("checkpoint", "best_model.pth"))
        threshold = float(payload.get("threshold", 0.5))
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        image_bytes = base64.b64decode(image_b64, validate=True)
        image = Image.open(BytesIO(image_bytes)).convert("L")
        if image.width * image.height > Image.MAX_IMAGE_PIXELS:
            raise ValueError("image is too large")
        model_path = _checkpoint_path(checkpoint_name)
        if not model_path.is_file():
            return jsonify({"error": f"检查点未找到: {checkpoint_name}"}), 404

        mtime_ns = model_path.stat().st_mtime_ns
        if inference_cache["path"] != model_path or inference_cache["mtime_ns"] != mtime_ns:
            inference_cache.update({
                "path": model_path,
                "mtime_ns": mtime_ns,
                "segmentor": VesselSegmentor(str(model_path), config=_load_config()),
            })
        segmentor = inference_cache["segmentor"]
        segmentor.threshold = threshold
        mask = segmentor.predict_array(np.asarray(image))
        buffer = BytesIO()
        Image.fromarray(mask).save(buffer, format="PNG")
        return jsonify({"mask_base64": base64.b64encode(buffer.getvalue()).decode("ascii")})
    except KeyError:
        return jsonify({"error": "未提供 image_base64"}), 400
    except (ValueError, binascii.Error, OSError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Inference failed")
        return jsonify({"error": str(exc)}), 500


def _get_cached_segmentor(checkpoint_name: str) -> VesselSegmentor:
    """按检查点修改时间缓存模型，确保不同架构权重不会混用。"""
    model_path = _checkpoint_path(checkpoint_name)
    if not model_path.is_file():
        raise FileNotFoundError(f"检查点未找到: {checkpoint_name}")
    mtime_ns = model_path.stat().st_mtime_ns
    if inference_cache["path"] != model_path or inference_cache["mtime_ns"] != mtime_ns:
        inference_cache.update({
            "path": model_path,
            "mtime_ns": mtime_ns,
            "segmentor": VesselSegmentor(str(model_path), config=_load_config()),
        })
    return inference_cache["segmentor"]


@app.route("/api/inference-batch", methods=["POST"])
def run_batch_inference():
    """对网页上传的多张图像使用同一检查点批量推理。"""
    with INFERENCE_LOCK:
        payload = request.get_json(silent=True) or {}
        try:
            checkpoint_name = _safe_checkpoint_name(payload.get("checkpoint", "best_model.pth"))
            threshold = float(payload.get("threshold", 0.5))
            images = payload.get("images")
            if not isinstance(images, list) or not images or len(images) > 32:
                raise ValueError("images 必须包含 1 到 32 张图像")
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("threshold must be between 0 and 1")
            segmentor = _get_cached_segmentor(checkpoint_name)
            segmentor.threshold = threshold
            results = []
            for item in images:
                if not isinstance(item, dict) or "image_base64" not in item:
                    raise ValueError("每个图像项都必须包含 image_base64")
                image = Image.open(BytesIO(base64.b64decode(item["image_base64"], validate=True))).convert("L")
                mask = segmentor.predict_array(np.asarray(image))
                buffer = BytesIO()
                Image.fromarray(mask).save(buffer, format="PNG")
                results.append({
                    "name": str(item.get("name", f"image_{len(results) + 1}")),
                    "mask_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
                })
            return jsonify({"results": results, "count": len(results)})
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
        checkpoint_name = _safe_checkpoint_name(payload.get("checkpoint", "best_model.pth"))
        raw_thresholds = payload.get("thresholds", [round(0.30 + index * 0.05, 2) for index in range(9)])
        if not isinstance(raw_thresholds, list):
            raise ValueError("thresholds 必须是数字列表")
        thresholds = sorted({float(value) for value in raw_thresholds})
        if not thresholds or len(thresholds) > 21 or any(not 0.0 <= value <= 1.0 for value in thresholds):
            raise ValueError("阈值数量必须为 1 到 21 个，且每个值在 0 到 1 之间")

        config = _load_config()
        dataset_cfg = config["dataset"]
        dataset = VesselDataset(
            image_dir=str(resolve_data_path(dataset_cfg["val_image_dir"], PROJECT_ROOT)),
            mask_dir=str(resolve_data_path(dataset_cfg["val_mask_dir"], PROJECT_ROOT)),
            transform=get_val_transforms(dataset_cfg["img_size"], dataset_cfg["keep_aspect_ratio"]),
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=max(int(config["training"]["batch_size"]), 1),
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        segmentor = _get_cached_segmentor(checkpoint_name)
        segmentor.model.eval()
        totals = {
            threshold: {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
            for threshold in thresholds
        }
        sample_count = 0
        with torch.inference_mode():
            for images, masks in loader:
                images = images.to(segmentor.device)
                masks = masks.to(segmentor.device)
                probabilities = torch.sigmoid(main_logits_from_output(segmentor.model(images)))
                for index in range(images.shape[0]):
                    probability = probabilities[index:index + 1]
                    target = masks[index:index + 1]
                    for threshold in thresholds:
                        prediction = (probability > threshold).float()
                        totals[threshold]["dice"] += calculate_dice(prediction, target)
                        totals[threshold]["iou"] += calculate_iou(prediction, target)
                        totals[threshold]["precision"] += calculate_precision(prediction, target)
                        totals[threshold]["recall"] += calculate_recall(prediction, target)
                    sample_count += 1
        divisor = max(sample_count, 1)
        results = [
            {"threshold": threshold, **{name: values[name] / divisor for name in values}}
            for threshold, values in totals.items()
        ]
        best = max(results, key=lambda item: item["dice"])
        return jsonify({
            "success": True,
            "checkpoint": checkpoint_name,
            "device": str(segmentor.device),
            "samples": sample_count,
            "results": results,
            "best_threshold": best["threshold"],
            "best_dice": best["dice"],
        })
    except (ConfigError, ValueError, OSError) as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Threshold scan failed")
        return jsonify({"success": False, "error": str(exc)}), 500


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


if __name__ == "__main__":
    logger.info("Starting local Vessel-Seg-UNet server at http://127.0.0.1:5001")
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)
